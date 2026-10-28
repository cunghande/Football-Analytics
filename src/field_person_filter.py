"""Exclude obvious touchline and technical-area people from match statistics."""

from collections import defaultdict

import cv2
import numpy as np

from src.team_assigner import TeamAssigner


def _pitch_distance(frame):
    """Distance in image pixels to the largest connected green pitch region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 35, 25), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green)
    if count < 2:
        return None
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[main, cv2.CC_STAT_AREA] < 0.25 * frame.shape[0] * frame.shape[1]:
        return None
    field = (labels == main).astype(np.uint8)
    return cv2.distanceTransform(1 - field, cv2.DIST_L2, 5)


def _outside_score(frame, distance, bbox, role):
    if distance is None:
        return False
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    foot_x = max(0, min(width - 1, int((x1 + x2) / 2)))
    foot_y = max(0, min(height - 1, int(y2 + 2)))
    if distance[foot_y, foot_x] > 8:
        return True
    # Coaches standing beside the near touchline can still have green pixels
    # under their box. A dark coat at the bottom edge is a conservative cue.
    if role == "players" and y2 / height > 0.80:
        crop = TeamAssigner._crop_upper_half(
            frame, dict(zip(("x1", "y1", "x2", "y2"), bbox, strict=True))
        )
        if crop.size:
            brightness = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
            if float(np.median(brightness)) < 70:
                return True
    return False


def filter_nonfield_people(frames, tracks, *, enabled=True):
    """Remove clear off-pitch person observations, keeping their raw stub intact.

    The result is a heuristic image-space eligibility filter, not a field
    homography or a verified in-play label. It mutates only player/referee/GK
    dictionaries in memory after tracking and before report aggregation.
    """
    roles = ("players", "referees", "goalkeepers")
    if any(len(tracks.get(role, [])) != len(frames) for role in roles):
        raise ValueError("Person tracks must match video frames")
    metadata = {
        "enabled": bool(enabled),
        "method": "green_pitch_and_near_touchline_dark_clothing" if enabled else "disabled",
        "pitch_min_fraction": 0.25,
        "max_foot_distance_px": 8,
        "near_touchline_bottom_fraction": 0.80,
        "excluded_observations": 0,
        "fully_excluded_track_ids": 0,
        "note": "Image-space heuristic; may omit a dark-kit player near the lower touchline or miss staff on grass.",
    }
    if not enabled:
        return metadata

    by_track = defaultdict(list)
    pitch_frames = 0
    for frame_index, frame in enumerate(frames):
        distance = _pitch_distance(frame)
        pitch_frames += distance is not None
        for role in roles:
            for track_id, info in tracks[role][frame_index].items():
                outside = _outside_score(frame, distance, info["bbox"], role)
                by_track[track_id].append((frame_index, role, info, outside))

    kept = {role: [{} for _ in frames] for role in roles}
    excluded = 0
    kept_ids = set()
    excluded_ids = set()
    for track_id, rows in by_track.items():
        outside_share = sum(row[3] for row in rows) / len(rows)
        whole_track_outside = (outside_share >= 0.80 or
                               (len(rows) <= 25 and outside_share >= 0.50))
        for index, (frame_index, role, info, outside) in enumerate(rows):
            nearby = [item[3] for item in rows[max(0, index - 2):index + 3]
                      if abs(item[0] - frame_index) <= 3]
            exclude = whole_track_outside or (sum(nearby) > len(nearby) / 2 if nearby else outside)
            if exclude:
                excluded += 1
                excluded_ids.add(track_id)
            else:
                kept[role][frame_index][track_id] = info
                kept_ids.add(track_id)
    for role in roles:
        tracks[role] = kept[role]
    metadata.update({
        "pitch_frames": pitch_frames,
        "excluded_observations": excluded,
        "fully_excluded_track_ids": len(excluded_ids - kept_ids),
    })
    return metadata
