"""Sanity-check metrics computed from an enriched tracks dict, for the smoke test's
tolerance-band assertions. Stdlib only — must stay importable without torch/ultralytics."""
import json
import os


def compute_summary_stats(tracks: dict, num_frames: int, observed_ball_frames=None) -> dict:
    player_frames = tracks.get("players", [])
    referee_frames = tracks.get("referees", [])
    ball_frames = tracks.get("ball", [])

    player_track_ids = set()
    speeds = []
    for frame_tracks in player_frames:
        for track_id, info in frame_tracks.items():
            player_track_ids.add(track_id)
            if "speed" in info:
                speeds.append(info["speed"])

    final_players = player_frames[-1] if player_frames else {}
    team_counts = {}
    for info in final_players.values():
        team_id = info.get("team")
        team_counts[team_id] = team_counts.get(team_id, 0) + 1

    # Don't assume team IDs are 0/1 — TeamAssigner labels teams 1 and 2, so
    # sort by count instead of indexing by a specific ID. 0 is
    # UNKNOWN_TEAM_ID (src/team_assigner.py) — a low-confidence-assignment
    # sentinel, not a real team — and a missing "team" key (info.get returns
    # None) is the same case in practice, so both are excluded here or they
    # can get counted as if they were one of the two teams.
    top_counts = sorted(
        (count for team_id, count in team_counts.items() if team_id not in (0, None)),
        reverse=True,
    )
    team_split = [top_counts[0] if len(top_counts) > 0 else 0,
                  top_counts[1] if len(top_counts) > 1 else 0]

    ball_observed = (sum(1 for f in ball_frames if f) if observed_ball_frames is None
                     else observed_ball_frames)
    return {
        "frames": num_frames,
        "avg_players_per_frame": (sum(len(f) for f in player_frames) / num_frames) if num_frames else 0.0,
        "avg_referees_per_frame": (sum(len(f) for f in referee_frames) / num_frames) if num_frames else 0.0,
        "ball_detection_rate": (ball_observed / num_frames) if num_frames else 0.0,
        "ball_observed_frames": ball_observed,
        "ball_available_frames": sum(1 for f in ball_frames if f),
        "player_track_id_count": len(player_track_ids),
        "team_split": team_split,
        "speed_kmh_max": max(speeds) if speeds else 0.0,
        "speed_kmh_min": min(speeds) if speeds else 0.0,
    }


def write_summary_json(stats: dict, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
