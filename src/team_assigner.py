"""Temporal, field-player-only team assignment from jersey color."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import permutations
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans


UNKNOWN_TEAM_ID = 0


class TeamAssigner:
    """Cluster field-player shirt colors, periodically refreshing the clusters.

    Goalkeepers and referees must be routed separately by the detector/tracker.
    Team IDs stay aligned across refreshes, while short per-track voting reduces
    frame-to-frame flicker. A team ID is never inferred when two clusters cannot
    be fitted.
    """

    def __init__(
        self,
        n_teams: int = 2,
        random_state: int = 42,
        color_sample_frames: int = 10,
        min_confidence_threshold: float = 0.15,
        team_color_overrides: Optional[Dict[int, tuple]] = None,
        recluster_interval: int = 30,
        recluster_window: int = 90,
        team_vote_window: int = 7,
        max_samples_per_track: int = 8,
        min_center_separation: float = 12.0,
        suppress_pitch_green: bool = True,
    ):
        if n_teams < 2 or color_sample_frames < 1:
            raise ValueError("n_teams must be >= 2 and color_sample_frames >= 1")
        if recluster_interval < 0 or recluster_window < 1 or team_vote_window < 1:
            raise ValueError("Invalid temporal window settings")
        if max_samples_per_track < 1:
            raise ValueError("max_samples_per_track must be positive")
        self.n_teams = n_teams
        self.random_state = random_state
        self.color_sample_frames = color_sample_frames
        self.min_confidence_threshold = min_confidence_threshold
        self.team_color_overrides = team_color_overrides or {}
        self.recluster_interval = recluster_interval
        self.recluster_window = recluster_window
        self.team_vote_window = team_vote_window
        self.max_samples_per_track = max_samples_per_track
        self.min_center_separation = min_center_separation
        self.suppress_pitch_green = suppress_pitch_green
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_assignment: Dict[int, int] = {}
        self._votes: dict[int, deque[int]] = {}
        self.kmeans: Optional[KMeans] = None
        self.refresh_count = 0

    @staticmethod
    def _crop_upper_half(frame: np.ndarray, bbox: Dict) -> np.ndarray:
        """Select the central torso; clip coordinates to the image boundary."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]
        left = max(0, min(width, int(x1 + 0.20 * (x2 - x1))))
        right = max(0, min(width, int(x1 + 0.80 * (x2 - x1))))
        top = max(0, min(height, int(y1 + 0.15 * (y2 - y1))))
        bottom = max(0, min(height, int(y1 + 0.50 * (y2 - y1))))
        return frame[top:bottom, left:right]

    def get_player_color(self, frame: np.ndarray, bbox: Dict) -> np.ndarray:
        """Return the median BGR color of a field player's torso crop."""
        crop = self._crop_upper_half(frame, bbox)
        if crop.size == 0:
            return np.zeros(3, dtype=float)
        return self._color_from_crop(crop)

    def _color_from_crop(self, crop: np.ndarray) -> np.ndarray:
        pixels = crop.reshape(-1, 3)
        if self.suppress_pitch_green:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            is_pitch_green = ((hsv[:, 0] >= 25) & (hsv[:, 0] <= 95)
                              & (hsv[:, 1] >= 40) & (hsv[:, 2] >= 30))
            remaining = pixels[~is_pitch_green]
            if len(remaining) >= max(4, int(0.05 * len(pixels))):
                pixels = remaining
        return np.median(pixels, axis=0)

    def _sample_colors(self, frames: list, tracks: list, indices: range) -> np.ndarray:
        by_track: dict[int, list[np.ndarray]] = defaultdict(list)
        for frame_idx in indices:
            if frame_idx >= len(frames) or frame_idx >= len(tracks):
                continue
            for track_id, track in tracks[frame_idx].items():
                x1, y1, x2, y2 = track["bbox"]
                crop = self._crop_upper_half(frames[frame_idx],
                                             {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                if crop.size:
                    by_track[track_id].append(self._color_from_crop(crop))
        samples = []
        for colors in by_track.values():
            sample_count = min(len(colors), self.max_samples_per_track)
            if sample_count:
                sample_indices = np.linspace(0, len(colors) - 1, sample_count, dtype=int)
                samples.extend(colors[index] for index in sample_indices)
        return np.asarray(samples, dtype=float).reshape(-1, 3)

    @staticmethod
    def _color_features(bgr_colors: np.ndarray) -> np.ndarray:
        """Lab chroma with lightly weighted brightness, robust to stadium lighting."""
        values = np.clip(np.asarray(bgr_colors).reshape(-1, 1, 3), 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(values, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        lab[:, 0] *= 0.15
        return lab

    def _fit(self, colors: np.ndarray, preserve_team_ids: bool) -> bool:
        if len(colors) < self.n_teams or len(np.unique(np.round(colors), axis=0)) < self.n_teams:
            return False
        features = self._color_features(colors)
        candidate = KMeans(n_clusters=self.n_teams, random_state=self.random_state, n_init=10)
        candidate.fit(features)
        centers = candidate.cluster_centers_.copy()
        if any(np.linalg.norm(centers[i] - centers[j]) < self.min_center_separation
               for i in range(self.n_teams) for j in range(i + 1, self.n_teams)):
            return False
        if preserve_team_ids and self.kmeans is not None:
            previous = self.kmeans.cluster_centers_
            order = min(permutations(range(self.n_teams)),
                        key=lambda perm: sum(np.linalg.norm(previous[idx] - centers[perm[idx]])
                                             for idx in range(self.n_teams)))
            centers = centers[list(order)]
        candidate.cluster_centers_ = centers
        self.kmeans = candidate
        self.team_colors = {}
        sample_teams = np.argmin(np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2), axis=1)
        for idx in range(self.n_teams):
            cluster_bgr = colors[sample_teams == idx]
            self.team_colors[idx + 1] = np.median(cluster_bgr, axis=0) if len(cluster_bgr) else colors.mean(axis=0)
        for team_id, bgr in self.team_color_overrides.items():
            if team_id in self.team_colors:
                self.team_colors[team_id] = np.asarray(bgr, dtype=float)
        return True

    def fit_team_colors(self, frames: list, all_player_tracks: list) -> None:
        """Initial fit from early frames, excluding goalkeeper/referee tracks."""
        self.kmeans = None
        self.team_colors = {}
        self.player_team_assignment = {}
        self._votes = {}
        self.refresh_count = 0
        n = min(self.color_sample_frames, len(frames), len(all_player_tracks))
        colors = self._sample_colors(frames, all_player_tracks, range(n))
        self._fit(colors, preserve_team_ids=False)

    def update_team_colors(self, frames: list, all_player_tracks: list, frame_index: int) -> bool:
        """Refit on the recent window at the configured interval.

        Returns True only when a valid new fit replaces the previous centers.
        """
        if self.recluster_interval == 0 or frame_index <= 0 or frame_index % self.recluster_interval:
            return False
        end = min(frame_index + 1, len(frames), len(all_player_tracks))
        start = max(0, end - self.recluster_window)
        colors = self._sample_colors(frames, all_player_tracks, range(start, end))
        changed = self._fit(colors, preserve_team_ids=True)
        if changed:
            self.refresh_count += 1
        return changed

    def _compute_team_and_confidence(self, color: np.ndarray) -> tuple[int, float]:
        centers = self.kmeans.cluster_centers_  # type: ignore[union-attr]
        features = self._color_features(color)[0]
        dists = np.linalg.norm(centers - features, axis=1)
        order = np.argsort(dists)
        best = int(order[0]) + 1
        confidence = float((dists[order[1]] - dists[order[0]]) / (dists[order[1]] + 1e-6))
        return best, confidence

    def get_player_team(self, frame: np.ndarray, player_bbox: Sequence[float], player_id: int) -> int:
        """Classify a field player and smooth the track's recent confident votes."""
        if self.kmeans is None:
            return UNKNOWN_TEAM_ID
        bbox = dict(zip(("x1", "y1", "x2", "y2"), player_bbox, strict=True))
        crop = self._crop_upper_half(frame, bbox)
        if crop.size == 0:
            return self.player_team_assignment.get(player_id, UNKNOWN_TEAM_ID)
        color = self._color_from_crop(crop)
        team_id, confidence = self._compute_team_and_confidence(color)
        if confidence < self.min_confidence_threshold:
            return self.player_team_assignment.get(player_id, UNKNOWN_TEAM_ID)
        votes = self._votes.setdefault(player_id, deque(maxlen=self.team_vote_window))
        votes.append(team_id)
        counts = Counter(votes)
        current = self.player_team_assignment.get(player_id)
        chosen = max(counts, key=lambda candidate: (counts[candidate], candidate == current))
        self.player_team_assignment[player_id] = chosen
        return chosen
