import json

import cv2
import numpy as np

from src.spatial_visualizer import SpatialVisualizer


def test_image_space_coordinates_and_density(tmp_path):
    visualizer = SpatialVisualizer((200, 100), grid_shape=(10, 20))
    tracks = {"players": [
        {1: {"position": (100, 50), "team": 1},
         2: {"position": (200, 100), "team": 2},
         3: {"position": (201, 50), "team": 2}},
        {1: {"position": (100, 50), "team": 1},
         4: {"position": (0, 0), "team": 0}},
    ]}
    grids, counts, discarded = visualizer.density(tracks)
    assert counts == {1: 2, 2: 1, 0: 1}
    assert discarded == 1
    assert grids[1][5, 10] == 2
    assert grids[2][9, 19] == 1
    paths = visualizer.save_heatmaps(tracks, str(tmp_path / "analysis"))
    assert all(cv2.imread(paths[key]) is not None for key in ("png",))
    metadata = json.loads((tmp_path / "analysis_heatmaps_metadata.json").read_text(encoding="utf-8"))
    assert metadata["coordinate_mode"] == "normalized_image"
    assert metadata["discarded_outside_view"] == 1


def test_calibrated_minimap_uses_field_position():
    visualizer = SpatialVisualizer((200, 100), field_size=(120, 80), calibrated=True)
    info = {"position": (0, 0), "position_transformed": (60, 40)}
    assert visualizer.normalized_position(info) == (0.5, 0.5)
    assert visualizer.normalized_position({"position_transformed": (121, 40)}) is None
    canvas = visualizer.minimap_canvas({"players": {1: {**info, "team": 1}}})
    assert canvas.shape == (210, 320, 3)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    result = visualizer.overlay_minimap(frame, {"players": {1: {**info, "team": 1}}})
    assert result.shape == frame.shape
    assert np.any(result != frame)
    assert not np.any(frame)
