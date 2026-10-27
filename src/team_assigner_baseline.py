import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Sequence

UNKNOWN_TEAM_ID = 0


class TeamAssigner:
    """Assign players to teams based on t-shirt color using KMeans"""

    def __init__(
        self,
        n_teams: int = 2,
        random_state: int = 42,
        color_sample_frames: int = 10,
        min_confidence_threshold: float = 0.15,
        team_color_overrides: Optional[Dict[int, tuple]] = None,
    ):
        self.n_teams = n_teams
        self.random_state = random_state
        self.color_sample_frames = color_sample_frames
        self.min_confidence_threshold = min_confidence_threshold
        self.team_color_overrides = team_color_overrides or {}
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_assignment: Dict[int, int] = {}
        self.kmeans: Optional[KMeans] = None

    def _crop_upper_half(self, frame: np.ndarray, bbox: Dict) -> np.ndarray:
        """Crop the upper half of the bounding box to avoid shorts and socks, which may have different colors than the jersey"""
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        return frame[y1:int((y1 + y2) * 0.5), x1:x2]

    def get_player_color(self, frame: np.ndarray, bbox: Dict) -> np.ndarray:
        """
        Extract player shirt color from the upper half of the bounding box in BGR format

        Args:
            frame: Input frame
            bbox: Bounding box dictionary with x1, y1, x2, y2

        Returns:
            Average color of player's t-shirt (BGR)
        """
        player_region = self._crop_upper_half(frame, bbox)
        if player_region.size == 0:
            return np.array([0, 0, 0])
        return np.mean(player_region.reshape(-1, 3), axis=0)

    def fit_team_colors(self, frames: list, all_player_tracks: list) -> None:
        """Fit team color clusters by aggregating jersey crops across the first N frames"""
        self.kmeans = None
        self.team_colors = {}
        self.player_team_assignment = {}

        n = min(self.color_sample_frames, len(frames), len(all_player_tracks))
        player_colors = []
        for frame_idx in range(n):
            frame = frames[frame_idx]
            for _, player_track in all_player_tracks[frame_idx].items():
                x1, y1, x2, y2 = player_track["bbox"]
                bbox = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                crop = self._crop_upper_half(frame, bbox)
                if crop.size == 0:
                    continue
                player_colors.append(np.mean(crop.reshape(-1, 3), axis=0))

        if len(player_colors) < self.n_teams:
            return

        # Skip fitting if we don't have at least n_teams visually distinct colors
        unique_colors = np.unique(np.round(np.array(player_colors), 0), axis=0)
        if len(unique_colors) < self.n_teams:
            return

        self.kmeans = KMeans(
            n_clusters=min(self.n_teams, len(player_colors)),
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans.fit(np.array(player_colors))

        for cluster_idx, center in enumerate(self.kmeans.cluster_centers_):
            self.team_colors[cluster_idx + 1] = center

        for team_id, bgr in self.team_color_overrides.items():
            if team_id in self.team_colors:
                self.team_colors[team_id] = np.array(bgr, dtype=float)

    def _compute_team_and_confidence(self, color: np.ndarray) -> tuple:
        """Compute the team assignment and confidence for a given color"""
        # Relative margin: 0 = equidistant from both clusters (ambiguous), 1 = unambiguous
        centers = self.kmeans.cluster_centers_  # type: ignore[union-attr]
        dists = np.linalg.norm(centers - color, axis=1)
        sorted_idx = np.argsort(dists)
        best_team_id = int(sorted_idx[0]) + 1
        if len(dists) == 1:
            return best_team_id, 1.0
        d_best = dists[sorted_idx[0]]
        d_second = dists[sorted_idx[1]]
        confidence = float((d_second - d_best) / (d_second + 1e-6))
        return best_team_id, confidence

    def get_player_team(self, frame: np.ndarray, player_bbox: Sequence[float], player_id: int) -> int:
        """Get the team assignment for a player based on their bounding box and ID"""
        if player_id in self.player_team_assignment:
            return self.player_team_assignment[player_id]

        if self.kmeans is None:
            return 1

        bbox = {
            "x1": int(player_bbox[0]),
            "y1": int(player_bbox[1]),
            "x2": int(player_bbox[2]),
            "y2": int(player_bbox[3]),
        }
        color = self.get_player_color(frame, bbox)
        team_id, confidence = self._compute_team_and_confidence(color)

        if confidence < self.min_confidence_threshold:
            # Don't cache: a wrong assignment sticks for the entire run via player_team_assignment
            return UNKNOWN_TEAM_ID

        self.player_team_assignment[player_id] = team_id
        return team_id
