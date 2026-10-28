"""Conservative pass-candidate heuristic and explicitly labeled pass map."""

import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


EVENT_FIELDS = [
    "start_frame", "end_frame", "start_seconds", "end_seconds", "gap_frames",
    "team_id", "from_track_id", "to_track_id", "from_x_norm", "from_y_norm",
    "to_x_norm", "to_y_norm", "distance_norm", "distance_m_if_calibrated",
    "observed_ball_frames", "status",
]


def _runs(owners):
    if not owners:
        return []
    result = []
    start = 0
    for frame in range(1, len(owners) + 1):
        if frame == len(owners) or owners[frame] != owners[start]:
            result.append((owners[start], start, frame - 1))
            start = frame
    return result


def _ball_observed(frame_ball):
    return any(float(info.get("conf", 0.0)) > 0.0 for info in frame_ball.values())


def detect_pass_candidates(tracks, owner_ids, fps, visualizer,
                           min_stable_frames=2, max_gap_seconds=1.5,
                           min_distance_norm=0.03, max_distance_norm=0.75):
    """Find same-team owner transfers with observed ball at both endpoints.

    Ball may disappear during flight. Such events are marked as sparse-evidence
    candidates, never as verified passes. Direct owner changes are omitted.
    """
    players = tracks.get("players", [])
    balls = tracks.get("ball", [])
    if fps <= 0 or min_stable_frames < 1 or max_gap_seconds <= 0:
        raise ValueError("Invalid pass-candidate timing parameters")
    if not (0 <= min_distance_norm < max_distance_norm <= math.sqrt(2)):
        raise ValueError("Invalid distance gates")
    if len(players) != len(owner_ids) or len(balls) != len(owner_ids):
        raise ValueError("Track and owner frame counts must match")
    max_gap_frames = max(1, round(max_gap_seconds * fps))
    runs = _runs(owner_ids)
    events = []
    for left, middle, right in zip(runs, runs[1:], runs[2:]):
        sender, left_start, release = left
        unknown, gap_start, gap_end = middle
        receiver, receive, right_end = right
        gap = gap_end - gap_start + 1
        if (sender is None or unknown is not None or receiver is None
                or sender == receiver or gap < 1 or gap > max_gap_frames
                or max(release - left_start + 1,
                       right_end - receive + 1) < min_stable_frames):
            continue
        start_info = players[release].get(sender)
        end_info = players[receive].get(receiver)
        if start_info is None or end_info is None:
            continue
        team = start_info.get("team")
        if team not in (1, 2) or end_info.get("team") != team:
            continue
        # Check all observed owner frames, including a one-frame endpoint.
        if any(players[f].get(sender, {}).get("team") != team
               for f in range(left_start, release + 1)):
            continue
        if any(players[f].get(receiver, {}).get("team") != team
               for f in range(receive, right_end + 1)):
            continue
        observed = sum(_ball_observed(balls[f]) for f in range(release, receive + 1))
        if (not _ball_observed(balls[release])
                or not _ball_observed(balls[receive])):
            continue
        start = visualizer.normalized_position(start_info)
        end = visualizer.normalized_position(end_info)
        if start is None or end is None:
            continue
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        if not min_distance_norm <= distance <= max_distance_norm:
            continue
        meters = ""
        if visualizer.calibrated:
            meters = round(math.hypot((end[0] - start[0]) * visualizer.field_width,
                                      (end[1] - start[1]) * visualizer.field_height), 4)
        events.append({
            "start_frame": release,
            "end_frame": receive,
            "start_seconds": round(release / fps, 4),
            "end_seconds": round(receive / fps, 4),
            "gap_frames": gap,
            "team_id": team,
            "from_track_id": sender,
            "to_track_id": receiver,
            "from_x_norm": round(start[0], 6),
            "from_y_norm": round(start[1], 6),
            "to_x_norm": round(end[0], 6),
            "to_y_norm": round(end[1], 6),
            "distance_norm": round(distance, 6),
            "distance_m_if_calibrated": meters,
            "observed_ball_frames": observed,
            "status": ("candidate_unverified_visible_flight" if observed > 2 else
                       "candidate_unverified_sparse_ball"),
        })
    return events


def save_pass_map(events, visualizer, output_stem, frame_count, fps,
                  params=None, example_label=None):
    """Write events CSV, a two-team directed map, and protocol metadata."""
    parent = os.path.dirname(output_stem)
    if parent:
        os.makedirs(parent, exist_ok=True)
    paths = {
        "csv": output_stem + "_pass_candidates.csv",
        "png": output_stem + "_pass_map.png",
        "pdf": output_stem + "_pass_map.pdf",
        "metadata": output_stem + "_pass_map_metadata.json",
    }
    with open(paths["csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)

    grouped = defaultdict(list)
    for event in events:
        grouped[(event["team_id"], event["from_track_id"], event["to_track_id"])].append(event)
    edges = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, team, color in zip(axes, (1, 2), ("#2877b8", "#d35247")):
        ax.set_facecolor("#417747" if visualizer.calibrated else "#dedede")
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="white" if visualizer.calibrated else "#888888"))
        if visualizer.calibrated:
            ax.axvline(0.5, color="white", linewidth=0.7, alpha=0.8)
        team_edges = [(key, values) for key, values in edges if key[0] == team][:20]
        for (_, sender, receiver), values in team_edges:
            x1 = sum(float(event["from_x_norm"]) for event in values) / len(values)
            y1 = sum(float(event["from_y_norm"]) for event in values) / len(values)
            x2 = sum(float(event["to_x_norm"]) for event in values) / len(values)
            y2 = sum(float(event["to_y_norm"]) for event in values) / len(values)
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=12, linewidth=min(4, 1 + len(values)),
                                    color=color, alpha=0.9, shrinkA=8, shrinkB=8)
            ax.add_patch(arrow)
            ax.scatter([x1, x2], [y1, y2], c=color, s=40, edgecolors="white", zorder=3)
            ax.text(x1, y1 - 0.035, f"P{sender}", fontsize=8, ha="center", color="black")
            ax.text(x2, y2 - 0.035, f"P{receiver}", fontsize=8, ha="center", color="black")
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(len(values)),
                    fontsize=8, color="black", ha="center", va="center",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
        if not team_edges:
            ax.text(0.5, 0.5, "No candidates", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_box_aspect(visualizer.field_height / visualizer.field_width
                          if visualizer.calibrated else visualizer.frame_height / visualizer.frame_width)
        ax.set_title(f"Team {team}: {sum(event['team_id'] == team for event in events)} candidates")
        ax.set_xlabel("Field width (%)" if visualizer.calibrated else "Image width (%)")
        if team == 1:
            ax.set_ylabel("Field height (%)" if visualizer.calibrated else "Image height (%)")
        ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
        ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
    title = "Directed pass candidates — heuristic, not verified events"
    if example_label:
        title = f"{example_label} — {title}"
    fig.suptitle(title, fontsize=11)
    fig.savefig(paths["png"], dpi=200, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "coordinate_mode": visualizer.coordinate_mode,
        "frames": frame_count,
        "fps": fps,
        "candidate_count": len(events),
        "edge_count": len(edges),
        "displayed_edges_per_team_max": 20,
        "params": params or {},
        "example_label": example_label,
        "method": "Observed ball at both owners, at least one stable owner run, same known team, bounded gap and image distance. Flight ball may be missing.",
        "warning": "Candidate events only; nearest-foot ownership and track IDs are unverified. No pass precision/recall is claimed.",
    }
    with open(paths["metadata"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return paths
