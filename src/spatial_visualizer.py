"""2D spatial views with explicit calibrated and image-space modes."""

import json
import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SpatialVisualizer:
    def __init__(self, frame_size, field_size=(120.0, 80.0), calibrated=False,
                 grid_shape=(48, 72)):
        self.frame_width, self.frame_height = frame_size
        self.field_width, self.field_height = field_size
        self.calibrated = calibrated
        self.grid_shape = grid_shape
        if min(self.frame_width, self.frame_height, self.field_width,
               self.field_height, *grid_shape) <= 0:
            raise ValueError("Spatial dimensions must be positive")

    @property
    def coordinate_mode(self):
        return "calibrated_field" if self.calibrated else "normalized_image"

    def normalized_position(self, info):
        """Return (x,y) in [0,1] or None for missing/out-of-bounds points."""
        point = info.get("position_transformed" if self.calibrated else "position")
        if point is None or len(point) != 2:
            return None
        x, y = point
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        width = self.field_width if self.calibrated else self.frame_width
        height = self.field_height if self.calibrated else self.frame_height
        nx, ny = x / width, y / height
        if not (0 <= nx <= 1 and 0 <= ny <= 1):
            return None
        return nx, ny

    def density(self, tracks):
        """Count player observations per grid cell, grouped by team at that frame."""
        rows, cols = self.grid_shape
        grids = {team: np.zeros((rows, cols), dtype=np.float32) for team in (1, 2, 0)}
        counts = {team: 0 for team in (1, 2, 0)}
        discarded = 0
        for frame_tracks in tracks.get("players", []):
            for info in frame_tracks.values():
                point = self.normalized_position(info)
                if point is None:
                    discarded += 1
                    continue
                team = info.get("team", 0)
                team = team if team in (1, 2) else 0
                x = min(int(point[0] * cols), cols - 1)
                y = min(int(point[1] * rows), rows - 1)
                grids[team][y, x] += 1
                counts[team] += 1
        return grids, counts, discarded

    @staticmethod
    def _draw_pitch_lines(canvas, left, top, right, bottom, color, thick=1):
        mid = (left + right) // 2
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thick)
        cv2.line(canvas, (mid, top), (mid, bottom), color, thick)
        radius = max(3, int((bottom - top) * 0.12))
        cv2.circle(canvas, (mid, (top + bottom) // 2), radius, color, thick)
        box_width = max(4, int((right - left) * 0.13))
        box_margin = max(4, int((bottom - top) * 0.22))
        cv2.rectangle(canvas, (left, top + box_margin),
                      (left + box_width, bottom - box_margin), color, thick)
        cv2.rectangle(canvas, (right - box_width, top + box_margin),
                      (right, bottom - box_margin), color, thick)

    def minimap_canvas(self, frame_tracks, width=320, height=210):
        """Return a standalone BGR schematic; no implied field geometry uncalibrated."""
        canvas = np.full((height, width, 3), (64, 62, 58), dtype=np.uint8)
        cv2.rectangle(canvas, (3, 3), (width - 4, height - 4), (230, 230, 230), 1)
        title = "FIELD (CALIBRATED)" if self.calibrated else "IMAGE SPACE (UNCALIBRATED)"
        cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        left, top, right, bottom = 12, 32, width - 13, height - 13
        if self.calibrated:
            canvas[top:bottom + 1, left:right + 1] = (48, 100, 47)
            self._draw_pitch_lines(canvas, left, top, right, bottom, (230, 230, 230))
        else:
            cv2.rectangle(canvas, (left, top), (right, bottom), (185, 185, 185), 1)
        styles = (
            ("players", lambda info: {1: (230, 110, 45), 2: (60, 70, 220)}.get(info.get("team"), (170, 170, 170)), 5),
            ("goalkeepers", lambda info: (0, 165, 255), 5),
            ("referees", lambda info: (0, 230, 230), 4),
            ("ball", lambda info: (255, 255, 255), 3),
        )
        for role, color_fn, radius in styles:
            for info in frame_tracks.get(role, {}).values():
                point = self.normalized_position(info)
                if point is None:
                    continue
                x = left + round(point[0] * (right - left))
                y = top + round(point[1] * (bottom - top))
                cv2.circle(canvas, (x, y), radius + 1, (20, 20, 20), -1)
                cv2.circle(canvas, (x, y), radius, color_fn(info), -1)
        return canvas

    def overlay_minimap(self, frame, frame_tracks):
        """Place minimap in lower-right without changing video frame dimensions."""
        height, width = frame.shape[:2]
        map_width = min(320, max(100, width // 3))
        map_height = min(210, max(75, round(map_width * 210 / 320)))
        if map_width + 12 > width or map_height + 12 > height:
            return frame.copy()
        canvas = self.minimap_canvas(frame_tracks, map_width, map_height)
        result = frame.copy()
        x, y = width - map_width - 12, height - map_height - 12
        result[y:y + map_height, x:x + map_width] = canvas
        return result

    def save_heatmaps(self, tracks, output_stem):
        """Save comparable three-panel occupancy heatmaps and method metadata."""
        grids, counts, discarded = self.density(tracks)
        smoothed = {team: cv2.GaussianBlur(grid, (0, 0), 1.2)
                    for team, grid in grids.items()}
        vmax = max(float(grid.max()) for grid in smoothed.values())
        vmax = max(vmax, 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        image = None
        for ax, team, title in zip(axes, (1, 2, 0), ("Team 1", "Team 2", "Unknown")):
            image = ax.imshow(smoothed[team], cmap="magma", vmin=0, vmax=vmax,
                              origin="upper", extent=(0, 100, 100, 0),
                              interpolation="nearest")
            ax.set_title(f"{title}: {counts[team]} observations")
            ax.set_xlim(0, 100)
            ax.set_ylim(100, 0)
            ax.set_xlabel("Field width (%)" if self.calibrated else "Image width (%)")
            if team == 1:
                ax.set_ylabel("Field height (%)" if self.calibrated else "Image height (%)")
            if self.calibrated:
                ax.axvline(50, color="white", linewidth=0.6, alpha=0.5)
        fig.colorbar(image, ax=axes, label="Smoothed observations per grid cell", shrink=0.78)
        mode = "calibrated field coordinates" if self.calibrated else "normalized image coordinates"
        fig.suptitle(f"Player observation density — {mode}", fontsize=11)
        paths = {
            "png": output_stem + "_heatmaps.png",
            "pdf": output_stem + "_heatmaps.pdf",
            "metadata": output_stem + "_heatmaps_metadata.json",
        }
        parent = os.path.dirname(output_stem)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(paths["png"], dpi=200, bbox_inches="tight")
        fig.savefig(paths["pdf"], bbox_inches="tight")
        plt.close(fig)
        metadata = {
            "coordinate_mode": self.coordinate_mode,
            "frame_size": [self.frame_width, self.frame_height],
            "field_size_m": [self.field_width, self.field_height] if self.calibrated else None,
            "frames": len(tracks.get("players", [])),
            "grid_shape": list(self.grid_shape),
            "gaussian_sigma_cells": 1.2,
            "shared_color_scale_max": vmax,
            "observations_by_team": {str(k): v for k, v in counts.items()},
            "discarded_outside_view": discarded,
            "unit_of_count": "one tracked player detection per frame",
            "note": "Descriptive position density; not a trajectory or a measured distance.",
        }
        with open(paths["metadata"], "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return paths
