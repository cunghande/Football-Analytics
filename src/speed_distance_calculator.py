"""Estimate track movement from consecutive field or scaled image positions."""

from collections import defaultdict, deque
from math import hypot
from typing import Dict, List, Tuple

import cv2
import numpy as np


class SpeedDistanceCalculator:
    def __init__(self, fps: float = 30.0, meters_per_pixel: float = 0.1):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.meters_per_pixel = meters_per_pixel

    @staticmethod
    def _measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(hypot(p2[0] - p1[0], p2[1] - p1[1]))

    @staticmethod
    def _foot_position(bbox: List[float]) -> Tuple[int, int]:
        x1, _, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def add_speed_and_distance_to_tracks(self, tracks: Dict, frame_window: int = 5):
        """Accumulate consecutive-frame distance; smooth speed over recent steps.

        A missing frame breaks the path. No displacement is inferred across an
        occlusion, and each measured step is added exactly once.
        """
        if frame_window < 1:
            raise ValueError("frame_window must be at least one")
        for role in ("players", "goalkeepers"):
            previous = {}
            total_distance = defaultdict(float)
            recent_steps = defaultdict(lambda: deque(maxlen=frame_window))
            for frame_num, frame_tracks in enumerate(tracks.get(role, [])):
                current = {}
                for track_id, info in frame_tracks.items():
                    position = info.get("position_transformed")
                    if position is None:
                        recent_steps[track_id].clear()
                        continue
                    prior = previous.get(track_id)
                    if prior is not None and prior[0] == frame_num - 1:
                        step = self._measure_distance(prior[1], position)
                        total_distance[track_id] += step
                        recent_steps[track_id].append(step)
                    else:
                        recent_steps[track_id].clear()
                    info["distance"] = total_distance[track_id]
                    info["speed"] = (
                        sum(recent_steps[track_id]) / len(recent_steps[track_id])
                        * self.fps * 3.6 if recent_steps[track_id] else 0.0
                    )
                    current[track_id] = (frame_num, position)
                previous = current

    def draw_speed_and_distance(self, frames: List[np.ndarray], tracks: Dict) -> List[np.ndarray]:
        output_frames = []
        for frame_num, frame in enumerate(frames):
            output = frame.copy()
            for role in ("players", "goalkeepers"):
                role_frames = tracks.get(role, [])
                if frame_num >= len(role_frames):
                    continue
                for track_info in role_frames[frame_num].values():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance")
                    if speed is None or distance is None:
                        continue
                    x, y = self._foot_position(track_info["bbox"])
                    y += 35
                    cv2.putText(output, f"{speed:.1f} km/h", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(output, f"{distance:.1f} m", (x, y + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(output)
        return output_frames
