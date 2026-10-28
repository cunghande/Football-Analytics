import cv2
import numpy as np

from src.field_person_filter import filter_nonfield_people


def _clip():
    frames = []
    tracks = {role: [] for role in ("players", "referees", "goalkeepers", "ball")}
    boxes = {
        1: ([20, 35, 40, 72], (240, 240, 240)),  # field player
        2: ([50, 48, 70, 84], (25, 25, 25)),     # dark coach near touchline
        3: ([125, 70, 145, 98], (0, 255, 255)),  # steward on walkway
        4: ([80, 35, 96, 83], (0, 255, 255)),    # assistant referee at field edge
    }
    for _ in range(5):
        frame = np.zeros((100, 180, 3), dtype=np.uint8)
        frame[:85] = (40, 150, 40)
        frame[85:] = (95, 95, 95)
        people = {}
        for track_id, (bbox, color) in boxes.items():
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)
            people[track_id] = {"bbox": bbox, "center": ((bbox[0] + bbox[2]) // 2,
                                                          (bbox[1] + bbox[3]) // 2), "conf": 0.9}
        frames.append(frame)
        tracks["players"].append(people)
        for role in ("referees", "goalkeepers", "ball"):
            tracks[role].append({})
    return frames, tracks


def test_excludes_staff_and_keeps_people_on_pitch():
    frames, tracks = _clip()
    metadata = filter_nonfield_people(frames, tracks)
    assert set(tracks["players"][0]) == {1, 4}
    assert metadata["excluded_observations"] == 10
    assert metadata["fully_excluded_track_ids"] == 2


def test_can_disable_filter():
    frames, tracks = _clip()
    metadata = filter_nonfield_people(frames, tracks, enabled=False)
    assert set(tracks["players"][0]) == {1, 2, 3, 4}
    assert metadata["excluded_observations"] == 0
