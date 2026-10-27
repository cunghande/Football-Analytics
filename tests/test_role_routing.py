"""Goalkeepers and referees must not enter the jersey team clusters."""

from types import SimpleNamespace

import numpy as np
import torch

from src.tracker import ByteTrackTracker
from src.yolo_detector import YOLODetector


def _box(class_id: int, confidence: float = 0.9):
    return SimpleNamespace(cls=torch.tensor([class_id]), conf=torch.tensor([confidence]),
                           xyxy=torch.tensor([[1.0, 2.0, 11.0, 32.0]]))


def test_detector_routes_four_class_model_roles_separately():
    result = SimpleNamespace(names={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
                             boxes=[_box(class_id) for class_id in range(4)])
    detector = YOLODetector.__new__(YOLODetector)
    detector.model = lambda *args, **kwargs: [result]
    detector.device, detector.conf, detector.ball_conf, detector.iou = "cpu", 0.25, 0.25, 0.45
    detections = detector.detect(np.zeros((40, 40, 3), dtype=np.uint8))
    assert {key: len(value) for key, value in detections.items()} == {
        "players": 1, "goalkeepers": 1, "referees": 1, "balls": 1}


def test_detector_uses_lower_threshold_only_for_ball():
    result = SimpleNamespace(names={0: "ball", 2: "player"},
                             boxes=[_box(0, 0.3), _box(2, 0.3)])
    detector = YOLODetector.__new__(YOLODetector)
    detector.model = lambda *args, **kwargs: [result]
    detector.device, detector.conf, detector.ball_conf, detector.iou = "cpu", 0.5, 0.25, 0.45
    detections = detector.detect(np.zeros((40, 40, 3), dtype=np.uint8))
    assert len(detections["balls"]) == 1
    assert detections["players"] == []


def test_tracker_split_preserves_goalkeeper_role():
    tracker = ByteTrackTracker.__new__(ByteTrackTracker)
    objects = {1: {"class_name": "goalkeeper", "bbox": [0, 0, 10, 20],
                   "center": (5, 10), "conf": 0.9},
               2: {"class_name": "referee", "bbox": [20, 0, 30, 20],
                   "center": (25, 10), "conf": 0.9}}
    split = tracker.split_tracks_by_class(objects)
    assert set(split["goalkeepers"]) == {1}
    assert set(split["referees"]) == {2}
    assert split["players"] == {}
