"""Plot the fixed-protocol comparison for the role-correction experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))["results"]
    labels = [
        "mAP@0.5", "mAP@0.5:0.95", "Ball AP50",
        "GK AP50", "Player AP50", "Referee AP50",
    ]
    metric_keys = ["map50", "map50_95"]
    classes = ["ball", "goalkeeper", "player", "referee"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    panels = (
        ("valid", "Validation (599 images)"),
        ("test", "Test: pelanty clip (244 images)"),
    )
    for axis, (split, title) in zip(axes, panels, strict=True):
        old = data["freeze10"]["splits"][split]
        new = data["role_ft_v1"]["splits"][split]
        old_values = [old[key] for key in metric_keys] + [
            old["per_class"][name]["ap50"] for name in classes
        ]
        new_values = [new[key] for key in metric_keys] + [
            new["per_class"][name]["ap50"] for name in classes
        ]
        x = np.arange(len(labels))
        width = 0.38
        axis.bar(x - width / 2, old_values, width,
                 label="freeze=10 (old)", color="#718096")
        axis.bar(x + width / 2, new_values, width,
                 label="role_ft_v1 (new)", color="#2B6CB0")
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="upper right")
    figure.suptitle("Fixed COCO-style evaluation: old checkpoint vs role fine-tune")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
