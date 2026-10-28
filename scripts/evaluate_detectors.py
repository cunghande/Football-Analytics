"""Evaluate three detector checkpoints with one four-class COCO-style protocol.

Example:
  python scripts/evaluate_detectors.py --data-root datasets/football_4class_clip_split \
    --baseline yolo26x.pt --full runs/modal/detection_player_full_v1/weights/best.pt \
    --freeze runs/modal/detection_player_freeze10_v1/weights/best.pt \
    --output-dir runs/evaluation_b --device 0

The COCO-pretrained baseline maps person -> player and sports ball -> ball.
It has no separate goalkeeper or referee output. All models are evaluated
under the same image preprocessing, confidence threshold, NMS and COCOeval.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import ultralytics


NAMES = ("ball", "goalkeeper", "player", "referee")
BASELINE_CLASSES = {0: 2, 32: 0}  # COCO person -> player, sports ball -> ball.
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
MIN_CONFIDENCE = 0.001
FIXED_CONFIDENCE = 0.25
NMS_IOU = 0.7
MAX_DET = 300
EVAL_MAX_DET = 100  # Standard COCO bbox evaluation cap.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--freeze", type=str, required=True)
    parser.add_argument("--full", type=str, required=True)
    parser.add_argument("--candidate", type=str,
                        help="Optional fourth four-class checkpoint to evaluate")
    parser.add_argument("--candidate-name", type=str, default="candidate",
                        help="Result key used for --candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--split", choices=("train", "valid", "test", "both"), default="both"
    )
    return parser.parse_args()


def load_ground_truth(root: Path, split: str) -> tuple[COCO, list[Path]]:
    paths = sorted((root / split / "images").glob("*.jpg"))
    if not paths:
        raise ValueError(f"No JPG images in {root / split / 'images'}")
    records = {"info": {}, "licenses": [],
               "images": [], "annotations": [],
               "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(NAMES)]}
    annotation_id = 1
    for image_id, path in enumerate(paths, 1):
        with Image.open(path) as image:
            width, height = image.size
        records["images"].append({"id": image_id, "file_name": path.name,
                                   "width": width, "height": height})
        labels = root / split / "labels" / f"{path.stem}.txt"
        if not labels.is_file():
            raise FileNotFoundError(labels)
        for line in labels.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Bad YOLO label: {labels}: {line}")
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            if cls not in range(4) or bw <= 0 or bh <= 0:
                raise ValueError(f"Invalid YOLO label: {labels}: {line}")
            box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                   bw * width, bh * height]
            records["annotations"].append({"id": annotation_id, "image_id": image_id,
                                           "category_id": cls + 1, "bbox": box,
                                           "area": box[2] * box[3], "iscrowd": 0})
            annotation_id += 1
    coco = COCO()
    coco.dataset = records
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, paths


def mapped_class(result_class: int, kind: str) -> int | None:
    if kind == "baseline":
        return BASELINE_CLASSES.get(result_class)
    return result_class if result_class in range(4) else None


def predict(model: YOLO, paths: list[Path], kind: str, device: str) -> list[dict]:
    kwargs = {"imgsz": 640, "rect": False, "device": device,
              "conf": MIN_CONFIDENCE, "iou": NMS_IOU,
              "max_det": MAX_DET, "verbose": False}
    if kind == "baseline":
        kwargs["classes"] = list(BASELINE_CLASSES)
    detections = []
    # A list source may be treated as one large batch despite the `batch` argument.
    # Explicit chunks bound GPU memory while preserving image-ID order.
    for start in range(0, len(paths), 8):
        chunk = paths[start:start + 8]
        results = model.predict(source=[str(path) for path in chunk], **kwargs)
        for offset, (path, result) in enumerate(zip(chunk, results, strict=True)):
            if Path(result.path).name != path.name:
                raise RuntimeError(f"Prediction order mismatch: {result.path} != {path}")
            image_id = start + offset + 1
            for box in result.boxes:
                class_id = mapped_class(int(box.cls[0]), kind)
                if class_id is None:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
                detections.append({"image_id": image_id, "category_id": class_id + 1,
                                   "bbox": [x1, y1, x2 - x1, y2 - y1],
                                   "score": float(box.conf[0])})
    return detections


def average_valid(array: np.ndarray) -> float:
    valid = array[array > -1]
    return float(valid.mean()) if valid.size else 0.0


def fixed_threshold_counts(gt: COCO, detections: list[dict]) -> dict:
    """Greedy, confidence-sorted class matching at IoU >= 0.5."""
    gt_by_key: dict[tuple[int, int], list[list[float]]] = {}
    pred_by_key: dict[tuple[int, int], list[dict]] = {}
    for ann in gt.dataset["annotations"]:
        gt_by_key.setdefault((ann["image_id"], ann["category_id"]), []).append(ann["bbox"])
    for det in detections:
        if det["score"] >= FIXED_CONFIDENCE:
            pred_by_key.setdefault((det["image_id"], det["category_id"]), []).append(det)
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in NAMES}
    for key in gt_by_key.keys() | pred_by_key.keys():
        image_id, category_id = key
        del image_id
        name = NAMES[category_id - 1]
        labels = gt_by_key.get(key, [])
        predictions = sorted(pred_by_key.get(key, []), key=lambda item: item["score"], reverse=True)
        matched: set[int] = set()
        for pred in predictions:
            px, py, pw, ph = pred["bbox"]
            best_iou, best_index = 0.0, None
            for index, (gx, gy, gw, gh) in enumerate(labels):
                if index in matched:
                    continue
                intersection = max(0.0, min(px + pw, gx + gw) - max(px, gx)) * \
                               max(0.0, min(py + ph, gy + gh) - max(py, gy))
                union = pw * ph + gw * gh - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_iou >= 0.5 and best_index is not None:
                matched.add(best_index)
                counts[name]["tp"] += 1
            else:
                counts[name]["fp"] += 1
        counts[name]["fn"] += len(labels) - len(matched)
    for values in counts.values():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        values["precision"] = tp / (tp + fp) if tp + fp else 0.0
        values["recall"] = tp / (tp + fn) if tp + fn else 0.0
    return counts


def evaluate(gt: COCO, detections: list[dict]) -> dict:
    if detections:
        with redirect_stdout(io.StringIO()):
            dt = gt.loadRes(detections)
    else:
        dt = COCO()
        dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"],
                      "annotations": []}
        with redirect_stdout(io.StringIO()):
            dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.iouThrs = IOU_THRESHOLDS
    evaluator.params.maxDets = [1, 10, EVAL_MAX_DET]
    with redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]  # IoU, recall, class, area, maxDet.
    per_class = {}
    for idx, name in enumerate(NAMES):
        per_class[name] = {
            "ap50": average_valid(precision[0, :, idx, 0, 2]),
            "ap50_95": average_valid(precision[:, :, idx, 0, 2]),
        }
    fixed = fixed_threshold_counts(gt, detections)
    return {
        "map50": average_valid(precision[0, :, :, 0, 2]),
        "map50_95": average_valid(precision[:, :, :, 0, 2]),
        "precision_macro_conf025_iou05": float(np.mean([v["precision"] for v in fixed.values()])),
        "recall_macro_conf025_iou05": float(np.mean([v["recall"] for v in fixed.values()])),
        "per_class": per_class,
        "fixed_threshold_per_class": fixed,
        "num_predictions_conf0001": len(detections),
    }


def main() -> None:
    args = parse_args()
    splits = ("valid", "test") if args.split == "both" else (args.split,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {"baseline": args.baseline, "freeze10": args.freeze, "full": args.full}
    if args.candidate:
        if args.candidate_name in models or not args.candidate_name.strip():
            raise ValueError("--candidate-name must be a non-empty, unique result key")
        models[args.candidate_name] = args.candidate
    summary = {"protocol": {
        "dataset": str(args.data_root), "splits": list(splits), "class_names": list(NAMES),
        "baseline_mapping": {"person": "player", "sports ball": "ball",
                             "goalkeeper": "no baseline output", "referee": "no baseline output"},
        "imgsz": 640, "rect": False, "batch": 8, "inference_confidence": MIN_CONFIDENCE,
        "nms_iou": NMS_IOU, "inference_max_det": MAX_DET, "eval_max_det": EVAL_MAX_DET,
        "ap_implementation": "pycocotools COCOeval bbox, maxDets=100, area=all",
        "precision_recall": "macro across 4 classes, conf>=0.25, greedy IoU>=0.5",
        "ultralytics_version": ultralytics.__version__,
    }, "results": {}}
    for kind, checkpoint in models.items():
        model = YOLO(checkpoint)
        names = model.names
        if kind == "baseline":
            if names.get(0) != "person" or names.get(32) != "sports ball":
                raise ValueError(f"Baseline COCO class IDs changed: {names.get(0)}, {names.get(32)}")
        elif [names.get(idx) for idx in range(4)] != list(NAMES):
            raise ValueError(f"Four-class checkpoint names do not match: {names}")
        path = Path(checkpoint)
        summary["results"][kind] = {"checkpoint": checkpoint,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "splits": {}}
        for split in splits:
            print(f"Evaluating {kind} on {split}...", flush=True)
            gt, paths = load_ground_truth(args.data_root, split)
            detections = predict(model, paths, kind, args.device)
            split_result = evaluate(gt, detections)
            split_result["num_images"] = len(paths)
            split_result["num_ground_truth"] = len(gt.dataset["annotations"])
            summary["results"][kind]["splits"][split] = split_result
            (args.output_dir / f"predictions_{kind}_{split}.json").write_text(
                json.dumps(detections, separators=(",", ":")), encoding="utf-8")
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"  mAP50={split_result['map50']:.5f}; "
                  f"mAP50-95={split_result['map50_95']:.5f}; "
                  f"AP50 ball={split_result['per_class']['ball']['ap50']:.5f}", flush=True)
        del model
        gc.collect()
        if args.device != "cpu":
            import torch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
