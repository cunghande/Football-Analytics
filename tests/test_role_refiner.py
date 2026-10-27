import cv2
import numpy as np

from src.role_refiner import refine_human_roles


def _clip(yellow_people=1):
    frames = []
    tracks = {role: [] for role in ("players", "referees", "goalkeepers", "ball")}
    for frame_index in range(75):
        frame = np.zeros((80, 310, 3), dtype=np.uint8)
        per_frame = {role: {} for role in tracks}
        for track_id in range(13):
            x1 = 2 + track_id * 23
            bbox = [x1, 10, x1 + 18, 60]
            shirt = (0, 255, 255) if track_id < yellow_people or track_id == 4 else (
                (0, 0, 220) if track_id == 3 else
                (160, 70, 40) if track_id % 2 else (240, 240, 240))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), shirt, -1)
            if track_id < yellow_people:
                cv2.rectangle(frame, (bbox[0], 35), (bbox[2], 52), (25, 25, 25), -1)
            role = "referees" if (track_id in (1, 3) or
                                  (track_id == 0 and frame_index % 5 == 0)) else "players"
            per_frame[role][track_id] = {"bbox": bbox, "center": (x1 + 9, 35), "conf": 0.8}
        frames.append(frame)
        for role in tracks:
            tracks[role].append(per_frame[role])
    return frames, tracks


def test_rare_yellow_referee_corrects_both_class_directions():
    frames, tracks = _clip()
    metadata = refine_human_roles(frames, tracks)
    assert metadata["method"] == "rare_yellow_kit_temporal_vote"
    assert 0 in tracks["referees"][20]
    assert 1 in tracks["players"][20]
    assert 3 in tracks["players"][20]
    assert 4 in tracks["players"][20]
    assert metadata["reclassified_observations"] > 0
    assert metadata["refined_referee_track_count"] == 1


def test_yellow_team_does_not_trigger_referee_color_rule():
    frames, tracks = _clip(yellow_people=3)
    metadata = refine_human_roles(frames, tracks)
    assert metadata["method"] == "raw_model_roles"
    assert 1 in tracks["referees"][20]
    assert 0 in tracks["players"][21]


def test_user_can_leave_model_roles_unchanged():
    frames, tracks = _clip()
    metadata = refine_human_roles(frames, tracks, enabled=False)
    assert metadata["method"] == "raw_model_roles"
    assert 1 in tracks["referees"][20]
