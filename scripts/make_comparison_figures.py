"""Create step-B report plots from a completed common-protocol evaluation.

Usage: python -X utf8 scripts/make_comparison_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "report" / "data"
EVALUATION = DATA / "comparison_b_summary.json"
OUTPUT = ROOT / "report" / "figures"
KINDS = ("baseline", "freeze10", "full")
LABELS = ("Pretrained\nCOCO mapping", "Freeze=10", "Full fine-tune")
COLORS = ("#7F7F7F", "#0072B2", "#D55E00")
CLASSES = ("ball", "goalkeeper", "player", "referee")


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def metric_bars(summary: dict) -> None:
    panels = (
        ("mAP@0.5", "map50"),
        ("mAP@0.5:0.95", "map50_95"),
        ("AP@0.5 of ball", "ball_ap50"),
        ("AP@0.5:0.95 of ball", "ball_ap50_95"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), layout="constrained")
    for ax, (title, key) in zip(axes.flat, panels):
        values = []
        for kind in KINDS:
            result = summary["results"][kind]["splits"]["valid"]
            if key.startswith("ball_"):
                values.append(result["per_class"]["ball"][key.removeprefix("ball_")])
            else:
                values.append(result[key])
        bars = ax.bar(LABELS, values, color=COLORS, width=0.64)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
        ax.set_ylim(0, max(0.1, max(values) * 1.24))
        ax.set_title(title)
        ax.set_ylabel("COCOeval score")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Four-class validation: shared COCOeval protocol", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "COCO baseline maps person to player and sports ball to ball; goalkeeper/referee are unavailable.",
             ha="center", fontsize=9, color="#555555")
    save(fig, "comparison_b_metrics")


def class_heatmap(summary: dict) -> None:
    scores = np.array([[summary["results"][kind]["splits"]["valid"]["per_class"][name]["ap50"]
                        for name in CLASSES] for kind in KINDS])
    fig, ax = plt.subplots(figsize=(8.5, 4.1), layout="constrained")
    image = ax.imshow(scores, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(3), LABELS)
    for row in range(3):
        for col in range(4):
            value = scores[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > 0.55 else "black", fontsize=10)
    fig.colorbar(image, ax=ax, label="AP@0.5", shrink=0.8)
    ax.set_title("Per-class AP@0.5 on clip-group validation", fontsize=12, fontweight="bold")
    save(fig, "comparison_b_per_class")


def training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), layout="constrained")
    for label, filename, color in (("Freeze=10", "freeze10_results.csv", COLORS[1]),
                                   ("Full fine-tune", "full_results.csv", COLORS[2])):
        with (DATA / filename).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        columns = {key.strip(): key for key in rows[0]}
        epochs = np.array([int(row[columns["epoch"]]) for row in rows])
        for ax, metric in zip(axes, ("metrics/mAP50(B)", "metrics/mAP50-95(B)")):
            values = np.array([float(row[columns[metric]]) for row in rows])
            ax.plot(epochs, values, color=color, label=label, linewidth=1.7)
            if metric == "metrics/mAP50-95(B)":
                best = int(np.argmax(values))
                ax.scatter([epochs[best]], [values[best]], color=color, s=35, zorder=4)
    for ax, title in zip(axes, ("mAP@0.5", "mAP@0.5:0.95")):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Ultralytics validation metric")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Training histories of the two transfer-learning settings", fontsize=12, fontweight="bold")
    fig.text(0.5, -0.04, "These curves come from training logs; the comparison table uses a separate shared COCOeval pass.",
             ha="center", fontsize=9, color="#555555")
    save(fig, "comparison_b_training_curves")


def main() -> None:
    summary = json.loads(EVALUATION.read_text(encoding="utf-8"))
    if set(summary["results"]) != set(KINDS):
        raise ValueError("Evaluation must include baseline, freeze10 and full")
    metric_bars(summary)
    class_heatmap(summary)
    training_curves()
    print(f"Saved step-B figures in {OUTPUT}")


if __name__ == "__main__":
    main()
