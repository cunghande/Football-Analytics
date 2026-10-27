"""Compare static vs temporal jersey clustering on visually reviewed samples.

The 200 evaluation crops in report/data/team_review_verified.csv were manually
checked as white or red/black kits. Dataset boxes, not detector predictions,
are used here so this measures team assignment conditional on object labels.
"""

from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path
import re
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.team_assigner import TeamAssigner
from src.team_assigner_baseline import TeamAssigner as StaticTeamAssigner

DATASET = ROOT / "datasets" / "football_4class_clip_split" / "valid"
REFERENCE = ROOT / "report" / "data" / "team_review_verified.csv"
OUTPUT = ROOT / "report" / "data"
FIGURES = ROOT / "report" / "figures"
FIRST_FRAME = 0
LAST_FRAME = 197
CALIBRATION_FRAME = 147
EVALUATION_FRAMES = range(148, 198)


def chosen_images() -> dict[int, Path]:
    paths = {}
    for path in sorted((DATASET / "images").glob("2_mp4-*.jpg")):
        match = re.match(r"2_mp4-(\d+)_", path.name)
        if match:
            paths.setdefault(int(match.group(1)), path)
    if set(range(FIRST_FRAME, LAST_FRAME + 1)) - set(paths):
        raise ValueError("Selected clip has missing frame IDs")
    return paths


def iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def load_sequence() -> tuple[list[np.ndarray], list[dict], dict[int, dict[str, int]]]:
    images = chosen_images()
    frames = []
    tracks = []
    role_counts = {}
    previous: dict[int, list[int]] = {}
    next_track_id = 1
    for frame_id in range(FIRST_FRAME, LAST_FRAME + 1):
        path = images[frame_id]
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read {path}")
        frames.append(frame)
        height, width = frame.shape[:2]
        labels = (DATASET / "labels" / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines()
        players = []
        counts = {"goalkeeper": 0, "referee": 0}
        for line_index, line in enumerate(labels):
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            if class_id == 1:
                counts["goalkeeper"] += 1
            elif class_id == 3:
                counts["referee"] += 1
            if class_id != 2:
                continue
            cx, cy, bw, bh = map(float, parts[1:])
            box = [int((cx - bw / 2) * width), int((cy - bh / 2) * height),
                   int((cx + bw / 2) * width), int((cy + bh / 2) * height)]
            players.append((line_index, box))
        role_counts[frame_id] = counts
        candidates = sorted(((iou(box, previous_box), line_index, old_id)
                             for line_index, box in players
                             for old_id, previous_box in previous.items()), reverse=True)
        assigned_lines = set()
        assigned_ids = set()
        ids_by_line = {}
        for overlap, line_index, old_id in candidates:
            if overlap < 0.25:
                break
            if line_index not in assigned_lines and old_id not in assigned_ids:
                ids_by_line[line_index] = old_id
                assigned_lines.add(line_index)
                assigned_ids.add(old_id)
        current = {}
        for line_index, box in players:
            track_id = ids_by_line.get(line_index)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
            current[track_id] = {"bbox": box, "line_index": line_index}
        tracks.append(current)
        previous = {track_id: track["bbox"] for track_id, track in current.items()}
    return frames, tracks, role_counts


def load_reference() -> list[dict]:
    with REFERENCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 202 or {int(row["frame_id"]) for row in rows} != set(range(147, 198)):
        raise ValueError("Expected four reviewed samples per evaluation frame and two for calibration")
    counts = Counter(int(row["frame_id"]) for row in rows)
    if counts[CALIBRATION_FRAME] != 2 or any(counts[frame_id] != 4 for frame_id in EVALUATION_FRAMES):
        raise ValueError("Unexpected number of reviewed players in a frame")
    if len({(row["frame_id"], row["line_index"]) for row in rows}) != len(rows):
        raise ValueError("A player annotation was selected more than once in the same frame")
    if any(row["manual_team"] not in ("white", "red_black") for row in rows):
        raise ValueError("Unexpected manual team label")
    return rows


def predict(assigner, frames: list, tracks: list, temporal: bool) -> dict[tuple[int, int], int]:
    assigner.fit_team_colors(frames, tracks)
    predictions = {}
    for frame_id, (frame, frame_tracks) in enumerate(zip(frames, tracks, strict=True)):
        if temporal:
            assigner.update_team_colors(frames, tracks, frame_id)
        for track_id, track in frame_tracks.items():
            team_id = assigner.get_player_team(frame, track["bbox"], track_id)
            predictions[(frame_id, track["line_index"])] = team_id
    return predictions


def calibration_mapping(reference: list[dict], predictions: dict, assigner) -> dict[int, str]:
    pairs = [(row["manual_team"], predictions[(CALIBRATION_FRAME, int(row["line_index"]))])
             for row in reference if int(row["frame_id"]) == CALIBRATION_FRAME]
    if len(pairs) != 2:
        raise ValueError("Missing calibration samples")
    white_id = next(team_id for name, team_id in pairs if name == "white")
    red_id = next(team_id for name, team_id in pairs if name == "red_black")
    if white_id in (1, 2) and red_id in (1, 2) and white_id != red_id:
        return {white_id: "white", red_id: "red_black", 0: "unknown"}
    # If the two calibration samples coincide, choose by center brightness.
    centers = assigner.team_colors
    white_id = max((1, 2), key=lambda team_id: centers[team_id].mean()) if centers else 1
    return {white_id: "white", 3 - white_id: "red_black", 0: "unknown"}


def score(reference: list[dict], predictions: dict, mapping: dict[int, str]) -> tuple[dict, dict]:
    eval_rows = [row for row in reference if row["purpose"] == "evaluation"]
    correct = 0
    unknown = 0
    per_frame = {}
    per_team = {name: {"correct": 0, "total": 0} for name in ("white", "red_black")}
    detailed = {}
    for row in eval_rows:
        frame_id, line_index = int(row["frame_id"]), int(row["line_index"])
        team_id = predictions[(frame_id, line_index)]
        predicted = mapping.get(team_id, "unknown")
        is_correct = predicted == row["manual_team"]
        correct += is_correct
        unknown += predicted == "unknown"
        per_frame.setdefault(frame_id, []).append(is_correct)
        per_team[row["manual_team"]]["correct"] += is_correct
        per_team[row["manual_team"]]["total"] += 1
        detailed[(frame_id, line_index)] = {"team_id": team_id, "predicted": predicted,
                                            "correct": is_correct}
    result = {"instances": len(eval_rows), "correct_instances": correct,
              "instance_accuracy": correct / len(eval_rows), "unknown_instances": unknown,
              "frames": len(per_frame), "all_correct_frames": sum(all(v) for v in per_frame.values()),
              "all_correct_frame_rate": sum(all(v) for v in per_frame.values()) / len(per_frame),
              "per_team": per_team, "calibration_mapping": {str(k): v for k, v in mapping.items()}}
    return result, detailed


def figure(scores: dict, detail: dict) -> None:
    names = ("legacy", "torso_vote", "temporal")
    labels = ("Original static", "Torso + vote", "Temporal refit")
    colors = ("#7F7F7F", "#0072B2", "#D55E00")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    for ax, key, title in zip(axes, ("instance_accuracy", "all_correct_frame_rate"),
                              ("Correct player samples (200)", "Frames with all four samples correct (50)")):
        values = [scores[name][key] for name in names]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Rate")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Team assignment on 50 reviewed validation frames", fontweight="bold")
    fig.text(0.5, -0.04, "Two visually reviewed field players per kit per frame; GT boxes isolate the color-assignment stage.",
             ha="center", fontsize=9, color="#555555")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"team_assignment_comparison.{suffix}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def frame_heatmap(evaluation_rows: list[dict]) -> None:
    names = ("legacy", "torso_vote", "temporal")
    counts = np.zeros((3, len(EVALUATION_FRAMES)), dtype=int)
    for row in evaluation_rows:
        column = int(row["frame_id"]) - min(EVALUATION_FRAMES)
        for index, name in enumerate(names):
            counts[index, column] += bool(row[f"{name}_correct"])
    fig, ax = plt.subplots(figsize=(12, 3.5), layout="constrained")
    image = ax.imshow(counts, cmap="YlGnBu", vmin=0, vmax=4, aspect="auto")
    ax.set_yticks(range(3), ("Original static", "Torso + vote", "Temporal refit"))
    ax.set_xticks(range(0, 50, 5), range(148, 198, 5))
    ax.set_xlabel("Source frame index in clip 2")
    ax.set_title("Correct team assignments among four reviewed players per frame", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Correct / 4", ticks=range(5), shrink=0.9)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"team_assignment_frame_heatmap.{suffix}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    frames, tracks, roles = load_sequence()
    reference = load_reference()
    configurations = {
        "legacy": StaticTeamAssigner(n_teams=2, color_sample_frames=10,
                                      min_confidence_threshold=0.15),
        "torso_vote": TeamAssigner(n_teams=2, color_sample_frames=10,
                                    min_confidence_threshold=0.15, recluster_interval=0),
        "temporal": TeamAssigner(n_teams=2, color_sample_frames=10,
                                  min_confidence_threshold=0.15, recluster_interval=30,
                                  recluster_window=90, team_vote_window=7),
    }
    scores = {}
    details = {}
    for name, assigner in configurations.items():
        predictions = predict(assigner, frames, tracks, temporal=name == "temporal")
        mapping = calibration_mapping(reference, predictions, assigner)
        scores[name], details[name] = score(reference, predictions, mapping)
        scores[name]["refresh_count"] = getattr(assigner, "refresh_count", 0)
        print(f"{name}: {scores[name]['correct_instances']}/200, "
              f"{scores[name]['all_correct_frames']}/50 frames, "
              f"unknown={scores[name]['unknown_instances']}")
    evaluation_rows = []
    for row in reference:
        if row["purpose"] != "evaluation":
            continue
        key = (int(row["frame_id"]), int(row["line_index"]))
        evaluation_rows.append({**row, **{f"{name}_{field}": details[name][key][field]
                                          for name in configurations
                                          for field in ("team_id", "predicted", "correct")}})
    with (OUTPUT / "team_assignment_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluation_rows[0]))
        writer.writeheader()
        writer.writerows(evaluation_rows)
    summary = {
        "source": "detection-player v4 clip 2, first sorted export for frame IDs 0..197",
        "reference": str(REFERENCE.relative_to(ROOT)),
        "evaluation_frames": [min(EVALUATION_FRAMES), max(EVALUATION_FRAMES)],
        "calibration_frame": CALIBRATION_FRAME,
        "protocol": "Four visually reviewed field-player crops per frame; GT YOLO boxes; simple adjacent-frame IoU IDs; one calibration frame excluded; unknown counts as incorrect.",
        "role_boxes_excluded_in_50_frames": {
            role: sum(roles[frame_id][role] for frame_id in EVALUATION_FRAMES)
            for role in ("goalkeeper", "referee")},
        "scores": scores,
    }
    (OUTPUT / "team_assignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure(scores, details)
    frame_heatmap(evaluation_rows)


if __name__ == "__main__":
    main()
