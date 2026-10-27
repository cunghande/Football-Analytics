"""Generate report figures from the completed Modal training run.

Usage: python scripts/make_report_figures.py
Requires matplotlib, Pillow, PyYAML, ultralytics, and a local best.pt checkpoint.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "football_4class_clip_split"
RUN = ROOT / "runs" / "modal" / "detection_player_full_v1"
OUTPUT = ROOT / "report" / "figures"
CLASS_NAMES = ("ball", "goalkeeper", "player", "referee")
CLASS_COLORS = ("#E69F00", "#009E73", "#0072B2", "#CC79A7")
SPLITS = ("train", "valid", "test")
SPLIT_COLORS = ("#0072B2", "#009E73", "#D55E00")
EXAMPLES = (
    "1_mp4-16_jpg.rf.3aba0aa5b925eec6081a2fad787c1b5a.jpg",
    "2_mp4-50_jpg.rf.2aafaed312a0a8593bf2868862f97ad7.jpg",
)
PRED_CONFIDENCE = 0.25


def count_dataset() -> tuple[dict[str, int], dict[str, Counter]]:
    images = {}
    boxes = {}
    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"
        image_paths = sorted(image_dir.glob("*.jpg"))
        images[split] = len(image_paths)
        boxes[split] = Counter()
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    boxes[split][int(line.split()[0])] += 1
    return images, boxes


def read_training_results() -> tuple[list[int], dict[str, list[float]], dict[str, str]]:
    with (RUN / "results.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("results.csv is empty")
    columns = {key.strip(): key for key in rows[0]}
    wanted = (
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "train/box_loss", "val/box_loss",
        "train/cls_loss", "val/cls_loss",
    )
    series = {name: [float(row[columns[name]]) for row in rows] for name in wanted}
    epochs = [int(row[columns["epoch"]]) for row in rows]
    best_index = int(np.argmax(series["metrics/mAP50-95(B)"]))
    summary = {
        "best_epoch": str(epochs[best_index]),
        "epochs_completed": str(len(rows)),
        "precision": f"{series['metrics/precision(B)'][best_index]:.5f}",
        "recall": f"{series['metrics/recall(B)'][best_index]:.5f}",
        "mAP50": f"{series['metrics/mAP50(B)'][best_index]:.5f}",
        "mAP50_95": f"{series['metrics/mAP50-95(B)'][best_index]:.5f}",
    }
    return epochs, series, summary


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataset_figure(images: dict[str, int], boxes: dict[str, Counter]) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 4.7), layout="constrained")
    x = np.arange(len(SPLITS))
    bars = left.bar(x, [images[s] for s in SPLITS], color=SPLIT_COLORS, width=0.62)
    left.bar_label(bars, padding=3, fontsize=9)
    left.set_xticks(x, ["Train", "Validation", "Test"])
    left.set_ylim(0, max(images.values()) * 1.17)
    left.set_ylabel("Number of images")
    left.set_title("(a) Images by split")
    left.spines[["top", "right"]].set_visible(False)
    left.grid(axis="y", alpha=0.22)
    left.set_axisbelow(True)

    class_x = np.arange(len(CLASS_NAMES))
    width = 0.25
    for split_index, split in enumerate(SPLITS):
        values = [boxes[split][class_id] for class_id in range(len(CLASS_NAMES))]
        bar_set = right.bar(class_x + (split_index - 1) * width, values,
                            width=width, color=SPLIT_COLORS[split_index], label=split.title())
        right.bar_label(bar_set, labels=[f"{v:,}" for v in values], padding=2, rotation=90, fontsize=7)
    right.set_yscale("log")
    right.set_ylim(80, 40000)
    right.set_xticks(class_x, CLASS_NAMES, rotation=20, ha="right")
    right.set_ylabel("Labeled boxes (log scale)")
    right.set_title("(b) Class distribution")
    right.legend(frameon=False, ncol=3, loc="upper left")
    right.spines[["top", "right"]].set_visible(False)
    right.grid(axis="y", alpha=0.22)
    right.set_axisbelow(True)
    fig.suptitle("Detection-player v4: cleaned clip-group split", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.03, "Clip groups do not establish match-level independence.",
             ha="center", fontsize=9, color="#555555")
    save_figure(fig, "dataset_distribution")


def training_figure(epochs: list[int], series: dict[str, list[float]], best_epoch: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), layout="constrained")
    panels = (
        (axes[0, 0], "Validation mAP", ("metrics/mAP50(B)", "metrics/mAP50-95(B)"),
         ("mAP@0.5", "mAP@0.5:0.95")),
        (axes[0, 1], "Validation precision and recall", ("metrics/precision(B)", "metrics/recall(B)"),
         ("Precision", "Recall")),
        (axes[1, 0], "Box loss", ("train/box_loss", "val/box_loss"),
         ("Train", "Validation")),
        (axes[1, 1], "Classification loss", ("train/cls_loss", "val/cls_loss"),
         ("Train", "Validation")),
    )
    for ax, title, keys, labels in panels:
        for key, label, color in zip(keys, labels, ("#0072B2", "#D55E00")):
            ax.plot(epochs, series[key], label=label, color=color, linewidth=1.8)
        ax.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1,
                   label=f"Best epoch {best_epoch}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, max(epochs))
        ax.grid(alpha=0.23)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("Metric value")
    axes[0, 1].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 1].set_ylabel("Loss")
    fig.suptitle("YOLO26x fine-tuning on Modal (clip-group validation)", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "The dashed line marks the highest validation mAP@0.5:0.95; match IDs are unknown.",
             ha="center", fontsize=9, color="#555555")
    save_figure(fig, "training_curves")


def draw_gt(ax: plt.Axes, image_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ax.imshow(image)
    label_path = DATASET / "valid" / "labels" / f"{image_path.stem}.txt"
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, cx_text, cy_text, w_text, h_text = line.split()
        class_id = int(class_id_text)
        cx, cy, bw, bh = map(float, (cx_text, cy_text, w_text, h_text))
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        ax.add_patch(Rectangle((x1, y1), bw * width, bh * height,
                               fill=False, edgecolor=CLASS_COLORS[class_id],
                               linewidth=1.2 if class_id == 2 else 2.0))
    ax.set_axis_off()


def draw_predictions(ax: plt.Axes, image_path: Path, result) -> None:
    ax.imshow(Image.open(image_path).convert("RGB"))
    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=CLASS_COLORS[class_id],
                               linewidth=1.2 if class_id == 2 else 2.0,
                               linestyle="-"))
    ax.set_axis_off()


def qualitative_figure(checkpoint: Path) -> None:
    image_paths = [DATASET / "valid" / "images" / filename for filename in EXAMPLES]
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    model = YOLO(str(checkpoint))
    if set(model.names.values()) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected checkpoint classes: {model.names}")
    predictions = model.predict([str(path) for path in image_paths],
                                imgsz=640, conf=PRED_CONFIDENCE, iou=0.45,
                                device=0, verbose=False)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.91, bottom=0.10,
                        hspace=0.23, wspace=0.03)
    for row, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        draw_gt(axes[row, 0], image_path)
        draw_predictions(axes[row, 1], image_path, prediction)
        clip = image_path.name.split("_jpg", 1)[0]
        axes[row, 0].set_title(f"{clip}: ground truth", fontsize=10)
        axes[row, 1].set_title(f"{clip}: prediction (conf. ≥ {PRED_CONFIDENCE:.2f})", fontsize=10)
    fig.legend(handles=[Patch(facecolor="none", edgecolor=color, label=name)
                        for name, color in zip(CLASS_NAMES, CLASS_COLORS)],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Qualitative examples from the clip-group validation split",
                 fontsize=13, fontweight="bold", y=0.98)
    save_figure(fig, "qualitative_examples")
    ball_zoom_figure(image_paths, predictions)


def ball_zoom_figure(image_paths: list[Path], predictions) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.9))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.08,
                        hspace=0.28, wspace=0.16)
    crop_size = 96
    for row, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = DATASET / "valid" / "labels" / f"{image_path.stem}.txt"
        balls = [list(map(float, line.split()[1:])) for line in label_path.read_text().splitlines()
                 if line.strip() and int(line.split()[0]) == 0]
        if not balls:
            raise ValueError(f"Selected example has no labeled ball: {image_path}")
        cx, cy, bw, bh = balls[0]
        center_x, center_y = cx * width, cy * height
        crop_x = int(max(0, min(width - crop_size, center_x - crop_size / 2)))
        crop_y = int(max(0, min(height - crop_size, center_y - crop_size / 2)))
        crop = image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
        gt_left = (cx - bw / 2) * width - crop_x
        gt_top = (cy - bh / 2) * height - crop_y
        for col in (0, 1):
            ax = axes[row, col]
            ax.imshow(crop, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#888888")
        axes[row, 0].add_patch(Rectangle((gt_left, gt_top), bw * width, bh * height,
                                         fill=False, edgecolor=CLASS_COLORS[0], linewidth=2))
        found = 0
        for box in prediction.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            if x2 <= crop_x or x1 >= crop_x + crop_size or y2 <= crop_y or y1 >= crop_y + crop_size:
                continue
            axes[row, 1].add_patch(Rectangle((x1 - crop_x, y1 - crop_y), x2 - x1, y2 - y1,
                                             fill=False, edgecolor=CLASS_COLORS[0], linewidth=2))
            found += 1
        clip = image_path.name.split("_jpg", 1)[0]
        axes[row, 0].set_title(f"{clip}: labeled ball", fontsize=10)
        axes[row, 1].set_title(f"{clip}: {found} ball prediction(s)", fontsize=10)
    fig.suptitle("Ball regions from validation frames (96 × 96 source pixels)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025, f"Orange boxes; predictions shown at confidence ≥ {PRED_CONFIDENCE:.2f}. Crops share the same coordinates.",
             ha="center", fontsize=9, color="#555555")
    save_figure(fig, "ball_detail")


def main() -> None:
    checkpoint = RUN / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    images, boxes = count_dataset()
    epochs, series, summary = read_training_results()
    best_epoch = int(summary["best_epoch"])
    dataset_figure(images, boxes)
    training_figure(epochs, series, best_epoch)
    qualitative_figure(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = {
        "dataset": "detection-player v4, cleaned clip-group split",
        "image_counts": images,
        "box_counts": {split: {name: boxes[split][class_id] for class_id, name in enumerate(CLASS_NAMES)}
                       for split in SPLITS},
        "training_summary": summary,
        "checkpoint_sha256": digest,
        "qualitative_images": list(EXAMPLES),
        "prediction_confidence_for_figure_only": PRED_CONFIDENCE,
        "limitation": "Clip groups do not establish match-level independence.",
    }
    (OUTPUT / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved figures in {OUTPUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
