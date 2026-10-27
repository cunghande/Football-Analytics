"""Build a provisional, clip-disjoint YOLO dataset from detection-player v4.

This does not establish match-disjoint evaluation: source match IDs are unknown.
Usage: python scripts/prepare_clip_split.py
"""

from __future__ import annotations

import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "datasets" / "football_4class"
DESTINATION = PROJECT_ROOT / "datasets" / "football_4class_clip_split"
GROUP_SPLIT = {
    "Video-sans-titre-Realisee-avec-Clipchamp-1-": "train",
    "3-_-2": "train",
    "bolhi": "train",
    "courtoi_1": "train",
    "1": "valid",
    "2": "valid",
    "pelanty": "test",
}
NAMES = ["ball", "goalkeeper", "player", "referee"]
MAX_ALLOWED_OVERFLOW = 0.001


def source_group(image: Path) -> str:
    original_stem = image.stem.split("_jpg.rf.", 1)[0]
    if "_mp4-" in original_stem:
        return original_stem.rsplit("_mp4-", 1)[0]
    return original_stem.split("-", 1)[0]


def clean_labels(label: Path) -> tuple[str, Counter]:
    cleaned = []
    changes: Counter = Counter()
    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 YOLO columns in {label}:{line_number}")
        class_id = int(parts[0])
        if class_id not in range(4):
            raise ValueError(f"Invalid class ID in {label}:{line_number}: {class_id}")
        cx, cy, width, height = map(float, parts[1:])
        if width <= 0 or height <= 0:
            changes["dropped_zero_area_boxes"] += 1
            continue
        left, right = cx - width / 2, cx + width / 2
        top, bottom = cy - height / 2, cy + height / 2
        overflow = max(0, -left, -top, right - 1, bottom - 1)
        if overflow > MAX_ALLOWED_OVERFLOW:
            raise ValueError(f"Box exceeds allowed overflow in {label}:{line_number}: {overflow}")
        clipped_left = max(0.0, left)
        clipped_right = min(1.0, right)
        clipped_top = max(0.0, top)
        clipped_bottom = min(1.0, bottom)
        if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
            raise ValueError(f"Box has no area after clipping in {label}:{line_number}")
        if overflow > 0:
            changes["clipped_boxes"] += 1
        out_cx = (clipped_left + clipped_right) / 2
        out_cy = (clipped_top + clipped_bottom) / 2
        out_width = clipped_right - clipped_left
        out_height = clipped_bottom - clipped_top
        cleaned.append(f"{class_id} {out_cx:.10f} {out_cy:.10f} {out_width:.10f} {out_height:.10f}")
    return "\n".join(cleaned) + ("\n" if cleaned else ""), changes


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"Derived dataset already exists: {DESTINATION}")
    if not (SOURCE / "data.yaml").is_file():
        raise FileNotFoundError(f"Source dataset missing: {SOURCE}")
    metadata = json.loads((SOURCE / "source.json").read_text(encoding="utf-8"))
    if metadata.get("version") != 4 or metadata.get("source") != "https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4":
        raise ValueError("Unexpected source dataset; expected detection-player v4")

    by_digest: dict[str, tuple[str, Path, Path, str]] = {}
    stats: Counter = Counter()
    for old_split in ("train", "valid", "test"):
        for image in sorted((SOURCE / old_split / "images").iterdir()):
            if not image.is_file():
                continue
            group = source_group(image)
            if group not in GROUP_SPLIT:
                raise ValueError(f"Unrecognized clip group: {image.name} -> {group}")
            label = SOURCE / old_split / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
            digest = sha256(image.read_bytes()).hexdigest()
            raw_label = label.read_text(encoding="utf-8-sig")
            previous = by_digest.get(digest)
            if previous:
                if previous[0] != group or previous[3] != raw_label:
                    raise ValueError(f"Duplicate image has different group/label: {previous[1]} | {image}")
                stats["duplicate_images_removed"] += 1
                continue
            by_digest[digest] = (group, image, label, raw_label)

    DESTINATION.mkdir(parents=True)
    manifest_rows = []
    for digest, (group, image, label, _) in sorted(by_digest.items()):
        split = GROUP_SPLIT[group]
        image_target = DESTINATION / split / "images" / image.name
        label_target = DESTINATION / split / "labels" / label.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        if image_target.exists() or label_target.exists():
            raise FileExistsError(f"Duplicate output filename: {image.name}")
        cleaned, changes = clean_labels(label)
        shutil.copy2(image, image_target)
        label_target.write_text(cleaned, encoding="utf-8")
        stats.update(changes)
        stats[f"{split}_images"] += 1
        manifest_rows.append({
            "sha256": digest,
            "clip_group": group,
            "source_image": str(image.relative_to(SOURCE)),
            "derived_split": split,
            "derived_image": str(image_target.relative_to(DESTINATION)),
            "dropped_zero_area_boxes": changes["dropped_zero_area_boxes"],
            "clipped_boxes": changes["clipped_boxes"],
        })

    data_yaml = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 4\n"
        "names: [ball, goalkeeper, player, referee]\n"
    )
    (DESTINATION / "data.yaml").write_text(data_yaml, encoding="utf-8")
    with (DESTINATION / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    provenance = {
        "source": metadata,
        "split_method": "filename-derived clip groups; match identity unknown",
        "group_split": GROUP_SPLIT,
        "classes": NAMES,
        "stats": dict(stats),
        "limitation": "Clip-disjoint only; independent matches cannot be verified.",
    }
    (DESTINATION / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Created {DESTINATION}")
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
