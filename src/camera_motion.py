import cv2
import numpy as np
from typing import Optional, Tuple
import pickle
import os

from config import CAMERA_CONFIG


class CameraMotionAnalyzer:
    """Analyze camera motion using optical flow"""

    def __init__(self):
        self.prev_frame = None
        self.prev_gray = None
        self.minimum_distance = CAMERA_CONFIG["minimum_distance"]

    def calculate_optical_flow(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculate optical flow between frames

        Returns:
            flow: Optical flow array, or None on the first frame
            magnitude: Flow magnitude, or None on the first frame
            angle: Flow angle, or None on the first frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None, None

        flow_buf = np.zeros((*gray.shape, 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, flow_buf,
            float(CAMERA_CONFIG["farneback_pyr_scale"]),
            int(CAMERA_CONFIG["farneback_levels"]),
            int(CAMERA_CONFIG["farneback_winsize"]),
            int(CAMERA_CONFIG["farneback_iterations"]),
            int(CAMERA_CONFIG["farneback_poly_n"]),
            float(CAMERA_CONFIG["farneback_poly_sigma"]),
            0,
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        self.prev_gray = gray

        return flow, magnitude, angle

    def get_camera_movement(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Estimate camera movement from optical flow

        Returns:
            dx, dy: Camera movement in x and y directions
        """
        flow, _, _ = self.calculate_optical_flow(frame)

        if flow is None:
            return 0.0, 0.0

        dx = float(np.mean(flow[:, :, 0]))
        dy = float(np.mean(flow[:, :, 1]))

        return dx, dy

    def get_camera_movement_batch(self, frames, read_from_stub: bool = False, stub_path: Optional[str] = None):
        """Estimate camera movement for all frames, optionally loading/saving stubs."""
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        if not frames:
            return []

        movements = [(0.0, 0.0)] * len(frames)
        self.prev_gray = None

        for frame_idx, frame in enumerate(frames):
            if frame_idx == 0:
                _ = self.get_camera_movement(frame)
                continue
            dx, dy = self.get_camera_movement(frame)
            if np.sqrt(dx * dx + dy * dy) < self.minimum_distance:
                dx, dy = 0.0, 0.0
            movements[frame_idx] = (float(dx), float(dy))

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(movements, f)

        return movements

    def add_adjusted_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """Remove accumulated frame-to-frame camera translation from positions."""
        offsets = []
        total_x = total_y = 0.0
        for dx, dy in camera_movement_per_frame:
            total_x += dx
            total_y += dy
            offsets.append((total_x, total_y))
        for _, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                dx, dy = offsets[frame_num]
                for _, track_info in frame_tracks.items():
                    x, y = track_info["position"]
                    track_info["position_adjusted"] = (x - dx, y - dy)
