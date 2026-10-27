"""Export reproducible track and ball-control aggregates without extra dependencies."""

import csv
import json
import math
import os
from collections import Counter, defaultdict


PLAYER_FIELDS = [
    "role", "track_id", "team_id", "team_assignment_changes", "first_frame",
    "last_frame", "observed_frames", "observed_seconds", "observed_path_px",
    "mean_observed_motion_px_s", "distance_m_estimate",
    "mean_speed_kmh_estimate", "max_speed_kmh_estimate", "possession_frames",
    "possession_seconds_estimate", "possession_pct_video",
    "pass_candidate_sent", "pass_candidate_received",
]
TEAM_FIELDS = [
    "team_id", "tracked_people", "distance_m_estimate", "mean_speed_kmh_estimate",
    "possession_frames", "possession_seconds_estimate", "possession_pct_video",
    "possession_pct_assigned_frames",
]
TIMELINE_FIELDS = ["frame", "time_seconds", "owner_track_id", "owner_team_id"]


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round(value):
    return round(float(value), 4)


def export_analysis_csv(tracks, possession_owner_ids, team_ball_control, fps,
                        output_stem, calibrated=False, meters_per_pixel=None,
                        pass_events=None):
    """Write per-person, per-team and frame-level CSV plus measurement metadata.

    Percent-of-video uses all frames, including unassigned frames. Percent-of-
    assigned uses only frames assigned to team 1 or 2. Team 0 is unknown.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = len(tracks.get("players", []))
    if len(possession_owner_ids) != frames or len(team_ball_control) != frames:
        raise ValueError("possession arrays must match the number of video frames")
    if len(tracks.get("goalkeepers", [])) not in (0, frames):
        raise ValueError("goalkeeper tracks must match the number of video frames")
    parent = os.path.dirname(output_stem)
    if parent:
        os.makedirs(parent, exist_ok=True)

    observations = defaultdict(list)
    team_distance = defaultdict(float)
    prior_distances = {}
    team_speed = defaultdict(list)
    for role in ("players", "goalkeepers"):
        for frame, frame_tracks in enumerate(tracks.get(role, [])):
            for track_id, info in frame_tracks.items():
                team = info.get("team", 0) if role == "players" else 0
                team = team if team in (1, 2) else 0
                observations[(role, track_id)].append((frame, team, info))
                if "speed" in info:
                    team_speed[team].append(float(info["speed"]))
                if "distance" in info:
                    key = (role, track_id)
                    distance = float(info["distance"])
                    previous = prior_distances.get(key, 0.0)
                    team_distance[team] += max(0.0, distance - previous)
                    prior_distances[key] = distance

    player_rows = []
    team_people = Counter()
    sent = Counter(int(event["from_track_id"]) for event in (pass_events or []))
    received = Counter(int(event["to_track_id"]) for event in (pass_events or []))
    for (role, track_id), items in sorted(observations.items(), key=lambda item: (item[0][0], item[0][1])):
        teams = [team for _, team, _ in items]
        known = Counter(team for team in teams if team in (1, 2))
        # A tied/unknown identity is kept outside both team totals.
        if not known or known[1] == known[2]:
            team = 0
        else:
            team = 1 if known[1] > known[2] else 2
        if role == "goalkeepers":
            team = 0
        team_people[team] += 1
        distances = [float(info["distance"]) for _, _, info in items if "distance" in info]
        speeds = [float(info["speed"]) for _, _, info in items if "speed" in info]
        possession = sum(bool(info.get("has_ball")) for _, _, info in items)
        observed_path_px = 0.0
        observed_steps = 0
        previous = None
        for frame, _, info in items:
            point = info.get("position")
            if point is None or len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
                previous = None
                continue
            if previous is not None and previous[0] == frame - 1:
                observed_path_px += math.hypot(float(point[0]) - previous[1][0],
                                               float(point[1]) - previous[1][1])
                observed_steps += 1
            previous = (frame, (float(point[0]), float(point[1])))
        player_rows.append({
            "role": "player" if role == "players" else "goalkeeper",
            "track_id": track_id,
            "team_id": team,
            "team_assignment_changes": sum(a != b for a, b in zip(teams, teams[1:])),
            "first_frame": items[0][0],
            "last_frame": items[-1][0],
            "observed_frames": len(items),
            "observed_seconds": _round(len(items) / fps),
            "observed_path_px": _round(observed_path_px) if observed_steps else "",
            "mean_observed_motion_px_s": _round(observed_path_px * fps / observed_steps)
            if observed_steps else "",
            "distance_m_estimate": _round(max(distances)) if distances else "",
            "mean_speed_kmh_estimate": _round(sum(speeds) / len(speeds)) if speeds else "",
            "max_speed_kmh_estimate": _round(max(speeds)) if speeds else "",
            "possession_frames": possession,
            "possession_seconds_estimate": _round(possession / fps),
            "possession_pct_video": _round(100 * possession / frames) if frames else 0.0,
            "pass_candidate_sent": sent[track_id] if role == "players" else 0,
            "pass_candidate_received": received[track_id] if role == "players" else 0,
        })

    possession_counts = Counter(team if team in (1, 2) else 0 for team in team_ball_control)
    assigned_frames = possession_counts[1] + possession_counts[2]
    team_rows = []
    for team in (1, 2, 0):
        speeds = team_speed[team]
        count = possession_counts[team]
        team_rows.append({
            "team_id": team,
            "tracked_people": team_people[team],
            "distance_m_estimate": _round(team_distance[team]),
            "mean_speed_kmh_estimate": _round(sum(speeds) / len(speeds)) if speeds else "",
            "possession_frames": count,
            "possession_seconds_estimate": _round(count / fps),
            "possession_pct_video": _round(100 * count / frames) if frames else 0.0,
            "possession_pct_assigned_frames": (
                _round(100 * count / assigned_frames) if assigned_frames and team in (1, 2) else ""
            ),
        })

    timeline_rows = [
        {"frame": frame, "time_seconds": _round(frame / fps),
         "owner_track_id": "" if owner is None else owner,
         "owner_team_id": team if team in (1, 2) else 0}
        for frame, (owner, team) in enumerate(zip(possession_owner_ids, team_ball_control))
    ]
    paths = {
        "players": output_stem + "_players.csv",
        "teams": output_stem + "_teams.csv",
        "timeline": output_stem + "_possession_timeline.csv",
        "metadata": output_stem + "_csv_metadata.json",
    }
    _write_csv(paths["players"], PLAYER_FIELDS, player_rows)
    _write_csv(paths["teams"], TEAM_FIELDS, team_rows)
    _write_csv(paths["timeline"], TIMELINE_FIELDS, timeline_rows)
    metadata = {
        "fps": fps,
        "frames": frames,
        "coordinate_method": "four_point_field_homography" if calibrated else "constant_meter_per_pixel_assumption",
        "meters_per_pixel_assumed": None if calibrated else meters_per_pixel,
        "distance_note": "Approximate observed track path only; gaps are not interpolated.",
        "possession_note": "Nearest player to detected ball per frame; unknown includes no ball or no assigned player.",
        "team_note": "Team 1/2 are anonymous kit-color clusters; team 0 is unknown and includes goalkeepers.",
        "seconds_note": "Frame count divided by video FPS; consecutive assignments are not validated ball possession events.",
        "observed_path_px_note": "Sum of consecutive-frame foot-position changes in image pixels; camera motion and perspective remain. Not measured running distance.",
        "pass_candidate_note": "Same-team owner transfer heuristic; sent/received counts are unverified candidates, not confirmed successful passes.",
    }
    with open(paths["metadata"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return paths
