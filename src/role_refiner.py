"""Conservative, clip-level correction for a rare yellow/dark referee kit.

The detector's player/referee class can flicker on the same tracked person. A
yellow shirt is useful evidence only when yellow is rare in the current clip;
otherwise it may be a field team's kit and raw model classes are retained.
"""

from collections import defaultdict

import cv2
import numpy as np

from src.team_assigner import TeamAssigner


def _yellow_fraction(frame, bbox):
    crop = TeamAssigner._crop_upper_half(
        frame, dict(zip(("x1", "y1", "x2", "y2"), bbox, strict=True))
    )
    if crop.size == 0:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, int(x1 + 0.20 * (x2 - x1))))
    right = max(0, min(width, int(x1 + 0.80 * (x2 - x1))))
    top = max(0, min(height, int(y1 + 0.48 * (y2 - y1))))
    bottom = max(0, min(height, int(y1 + 0.75 * (y2 - y1))))
    shorts = frame[top:bottom, left:right]
    if shorts.size == 0:
        return None
    # A keeper may wear yellow from shirt to shorts. Referees in this clip
    # have dark shorts, whereas the yellow keeper's lower kit stays bright.
    shorts_hsv = cv2.cvtColor(shorts, cv2.COLOR_BGR2HSV)
    if float(np.mean(shorts_hsv[:, :, 2] < 70)) < 0.08:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # A narrow hue interval avoids counting orange/red kit pixels and grass
    # around very small boxes as part of a yellow shirt.
    yellow = ((hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 34)
              & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 70))
    return float(np.mean(yellow))


def refine_human_roles(frames, tracks, *, enabled=True, sample_stride=5,
                       yellow_threshold=0.20):
    """Correct player/referee labels when a rare yellow kit has model support.

    Mutates only the player/referee frame dictionaries. Goalkeepers and balls
    stay in their existing roles. The returned metadata records when the
    heuristic was inactive; it does not claim ground-truth role accuracy.
    """
    if sample_stride < 1 or not 0 < yellow_threshold < 1:
        raise ValueError("Invalid role-refinement settings")
    if len(frames) != len(tracks["players"]) or len(frames) != len(tracks["referees"]):
        raise ValueError("Role tracks must match video frames")

    sample_count = yellow_count = referee_count = yellow_referee_count = 0
    for frame_index in range(0, len(frames), sample_stride):
        for role in ("players", "referees"):
            for info in tracks[role][frame_index].values():
                fraction = _yellow_fraction(frames[frame_index], info["bbox"])
                if fraction is None:
                    continue
                sample_count += 1
                is_yellow = fraction >= yellow_threshold
                yellow_count += is_yellow
                referee_count += role == "referees"
                yellow_referee_count += is_yellow and role == "referees"

    yellow_rate = yellow_count / sample_count if sample_count else 0.0
    raw_referee_rate = referee_count / sample_count if sample_count else 0.0
    yellow_referee_rate = yellow_referee_count / yellow_count if yellow_count else 0.0
    raw_referee_observations = sum(len(row) for row in tracks["referees"])
    raw_referee_track_count = len({track_id for row in tracks["referees"] for track_id in row})
    active = (enabled and sample_count >= 50 and yellow_count >= 8 and yellow_rate <= 0.08
              and yellow_referee_count >= 3
              and yellow_referee_rate >= max(0.10, 2.0 * raw_referee_rate))
    metadata = {
        "method": "rare_yellow_kit_temporal_vote" if active else "raw_model_roles",
        "enabled": bool(enabled),
        "yellow_threshold": yellow_threshold,
        "sample_stride": sample_stride,
        "sampled_people": sample_count,
        "sampled_yellow": yellow_count,
        "yellow_fraction_of_people": round(yellow_rate, 4),
        "sampled_yellow_referee_labels": yellow_referee_count,
        "raw_referee_observations": raw_referee_observations,
        "raw_referee_track_count": raw_referee_track_count,
        "refined_referee_observations": raw_referee_observations,
        "refined_referee_track_count": raw_referee_track_count,
        "reclassified_observations": 0,
        "note": ("Rare yellow torso with dark shorts and referee-model support; five nearby "
                 "observations vote on player/referee role. This is a heuristic, "
                 "not a verified role label."
                 if active else "Correction disabled or yellow/dark kit was not distinct and rare enough; raw model roles retained."),
    }
    if not active:
        return metadata

    observations = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        for role in ("players", "referees"):
            for track_id, info in tracks[role][frame_index].items():
                fraction = _yellow_fraction(frame, info["bbox"])
                observations[track_id].append((frame_index, role, info, fraction))

    players = [{} for _ in frames]
    referees = [{} for _ in frames]
    changed = refined_referee_frames = 0
    for track_id, rows in observations.items():
        for index, (frame_index, raw_role, info, fraction) in enumerate(rows):
            nearby = [item[3] >= yellow_threshold for item in rows[max(0, index - 2):index + 3]
                      if item[3] is not None and abs(item[0] - frame_index) <= 3]
            is_referee = sum(nearby) > len(nearby) / 2 if nearby else raw_role == "referees"
            role = "referees" if is_referee else "players"
            (referees if is_referee else players)[frame_index][track_id] = info
            changed += role != raw_role
            refined_referee_frames += is_referee
    tracks["players"] = players
    tracks["referees"] = referees
    metadata.update({"reclassified_observations": changed,
                     "refined_referee_observations": refined_referee_frames,
                     "refined_referee_track_count": len({
                         track_id for row in referees for track_id in row
                     })})
    return metadata
