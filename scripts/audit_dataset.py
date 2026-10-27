"""Audit a Roboflow YOLO detection export before training.

Usage: python scripts/audit_dataset.py datasets/football_4class
Requires: PyYAML (python -m pip install PyYAML)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED_CLASSES = {"ball", "goalkeeper", "player", "referee"}
BOUNDARY_EPSILON = 1e-4  # Tolerate rounding in exported normalized coordinates.


def read_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"Missing 'names' in {data_yaml}")
    raw = data["names"]
    if isinstance(raw, list):
        names = dict(enumerate(raw))
    elif isinstance(raw, dict):
        names = {int(key): value for key, value in raw.items()}
    else:
        raise ValueError("'names' must be a list or ID-to-name mapping")
    if set(names.values()) != EXPECTED_CLASSES or set(names) != set(range(4)):
        raise ValueError(f"Expected exactly {sorted(EXPECTED_CLASSES)}; found {names}")
    return names


def audit(root: Path) -> int:
    names = read_names(root / "data.yaml")
    seen_hashes: dict[str, tuple[str, Path]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    found_splits = 0

    for split in ("train", "valid", "val", "test"):
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            continue
        found_splits += 1
        labels_dir = root / split / "labels"
        images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        boxes = Counter()
        images_with_class: dict[int, set[Path]] = defaultdict(set)
        unlabeled = 0
        label_stems = set()

        for image in images:
            relative = image.relative_to(images_dir)
            label = labels_dir / relative.with_suffix(".txt")
            label_stems.add(label)
            if not label.exists():
                unlabeled += 1
                warnings.append(f"{image}: missing label file (valid only for a negative image)")
            else:
                for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    try:
                        if len(parts) != 5:
                            raise ValueError("expected 5 values: ID cx cy w h")
                        class_id = int(parts[0])
                        cx, cy, width, height = (float(value) for value in parts[1:])
                        if class_id not in names:
                            raise ValueError(f"class ID {class_id} is not in data.yaml")
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                            raise ValueError("coordinate/size outside normalized range")
                        if not (-BOUNDARY_EPSILON <= cx - width / 2 <= cx + width / 2 <= 1 + BOUNDARY_EPSILON):
                            raise ValueError("box exceeds horizontal image boundary")
                        if not (-BOUNDARY_EPSILON <= cy - height / 2 <= cy + height / 2 <= 1 + BOUNDARY_EPSILON):
                            raise ValueError("box exceeds vertical image boundary")
                    except ValueError as exc:
                        errors.append(f"{label}:{line_number}: {exc}")
                        continue
                    boxes[class_id] += 1
                    images_with_class[class_id].add(image)

            digest = sha256(image.read_bytes()).hexdigest()
            previous = seen_hashes.get(digest)
            if previous and previous[0] != split:
                errors.append(f"Identical image across {previous[0]} and {split}: {previous[1]} | {image}")
            else:
                seen_hashes[digest] = (split, image)

        if labels_dir.is_dir():
            for label in labels_dir.rglob("*.txt"):
                if label not in label_stems:
                    warnings.append(f"{label}: label has no matching image")

        print(f"\n[{split}] {len(images)} exported images; {unlabeled} missing label files")
        for class_id, name in sorted(names.items()):
            print(f"  {name}: {boxes[class_id]} boxes in {len(images_with_class[class_id])} images")

    if found_splits == 0:
        errors.append(f"No train/valid/val/test images found in {root}")
    if not (root / "train" / "images").is_dir():
        errors.append("Missing train/images")
    if not ((root / "valid" / "images").is_dir() or (root / "val" / "images").is_dir()):
        errors.append("Missing valid/images or val/images")

    for warning in warnings:
        print("WARNING:", warning)
    for error in errors:
        print("ERROR:", error)
    print("\nNote: inspect augmentation and near-duplicate frames using metadata/video provenance.")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Directory with data.yaml and train/valid/test")
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    if not root.is_dir():
        parser.error(f"Directory not found: {root}")
    if not (root / "data.yaml").is_file():
        parser.error(f"data.yaml not found in {root}")
    return audit(root)


if __name__ == "__main__":
    raise SystemExit(main())
