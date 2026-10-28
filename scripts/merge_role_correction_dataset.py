"""Merge manually corrected role labels into the detector training split.

The base validation and test splits are copied unchanged. All correction images
come from one video, so they are added only to train to avoid temporal leakage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import yaml


EXPECTED_NAMES = ["ball", "goalkeeper", "player", "referee"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Base YOLO dataset with train/valid/test splits")
    parser.add_argument("correction", type=Path, help="Roboflow YOLO export containing train labels")
    parser.add_argument("output", type=Path, help="New derived dataset directory")
    parser.add_argument("--prefix", default="role_v1__", help="Prefix for correction filenames")
    return parser.parse_args()


def read_names(dataset: Path) -> list[str]:
    yaml_path = dataset / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if names != EXPECTED_NAMES:
        raise ValueError(f"Expected classes {EXPECTED_NAMES}; found {names} in {yaml_path}")
    return names


def paired_files(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing {split} images/labels under {dataset}")
    images = {path.stem: path for path in image_dir.iterdir() if path.is_file()}
    labels = {path.stem: path for path in label_dir.glob("*.txt") if path.is_file()}
    if set(images) != set(labels):
        missing_labels = sorted(set(images) - set(labels))[:5]
        missing_images = sorted(set(labels) - set(images))[:5]
        raise ValueError(
            f"Unpaired files in {dataset / split}: "
            f"missing labels={missing_labels}, missing images={missing_images}"
        )
    return [(images[stem], labels[stem]) for stem in sorted(images)]


def validate_labels(pairs: list[tuple[Path, Path]]) -> Counter:
    counts: Counter = Counter()
    for _, label in pairs:
        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Expected 5 YOLO columns in {label}:{line_number}")
            class_id = int(parts[0])
            if class_id not in range(len(EXPECTED_NAMES)):
                raise ValueError(f"Invalid class ID in {label}:{line_number}: {class_id}")
            cx, cy, width, height = map(float, parts[1:])
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid normalized box in {label}:{line_number}: {parts[1:]}")
            counts[EXPECTED_NAMES[class_id]] += 1
    return counts


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    correction = args.correction.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    read_names(base)
    read_names(correction)

    base_pairs = {split: paired_files(base, split) for split in ("train", "valid", "test")}
    correction_pairs = paired_files(correction, "train")
    base_counts = {split: validate_labels(pairs) for split, pairs in base_pairs.items()}
    correction_counts = validate_labels(correction_pairs)

    shutil.copytree(base, output)
    for cache in output.rglob("*.cache"):
        cache.unlink()

    destination_images = output / "train" / "images"
    destination_labels = output / "train" / "labels"
    for image, label in correction_pairs:
        image_target = destination_images / f"{args.prefix}{image.name}"
        label_target = destination_labels / f"{args.prefix}{label.name}"
        if image_target.exists() or label_target.exists():
            raise FileExistsError(f"Correction filename collision: {image_target.name}")
        shutil.copy2(image, image_target)
        shutil.copy2(label, label_target)

    (output / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 4\n"
        "names: [ball, goalkeeper, player, referee]\n",
        encoding="utf-8",
    )
    provenance = {
        "base_dataset": str(base),
        "correction_dataset": str(correction),
        "policy": "correction video added to train only; base valid/test copied unchanged",
        "classes": EXPECTED_NAMES,
        "images": {
            "base_train": len(base_pairs["train"]),
            "correction_train": len(correction_pairs),
            "merged_train": len(base_pairs["train"]) + len(correction_pairs),
            "valid": len(base_pairs["valid"]),
            "test": len(base_pairs["test"]),
        },
        "base_box_counts": {split: dict(counts) for split, counts in base_counts.items()},
        "correction_box_counts": dict(correction_counts),
    }
    (output / "role_correction_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
