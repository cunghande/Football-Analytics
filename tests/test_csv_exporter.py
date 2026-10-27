import csv
import json

import pytest

from src.csv_exporter import export_analysis_csv
from src.speed_distance_calculator import SpeedDistanceCalculator


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_csv_totals_unknown_frames_and_fps(tmp_path):
    tracks = {
        "players": [
            {7: {"position_transformed": (0, 0), "team": 1, "has_ball": True}},
            {7: {"position_transformed": (3, 4), "team": 1, "has_ball": True},
             8: {"position_transformed": (0, 0), "team": 2}},
            {7: {"position_transformed": (6, 8), "team": 1},
             8: {"position_transformed": (0, 2), "team": 2, "has_ball": True}},
            {},
        ],
        "goalkeepers": [
            {9: {"position_transformed": (0, 0)}},
            {9: {"position_transformed": (0, 1)}},
            {}, {},
        ],
    }
    SpeedDistanceCalculator(fps=2).add_speed_and_distance_to_tracks(tracks, frame_window=2)
    paths = export_analysis_csv(tracks, [7, 7, 8, None], [1, 1, 2, 0], 2,
                                str(tmp_path / "analysis"), meters_per_pixel=0.1)
    players = _read(paths["players"])
    teams = _read(paths["teams"])
    timeline = _read(paths["timeline"])
    p7 = next(row for row in players if row["track_id"] == "7")
    assert float(p7["distance_m_estimate"]) == pytest.approx(10.0)
    assert float(p7["possession_seconds_estimate"]) == pytest.approx(1.0)
    assert float(p7["possession_pct_video"]) == pytest.approx(50.0)
    assert next(row for row in players if row["track_id"] == "9")["role"] == "goalkeeper"
    assert float(teams[0]["possession_pct_video"]) == pytest.approx(50.0)
    assert float(teams[0]["possession_pct_assigned_frames"]) == pytest.approx(66.6667)
    assert float(teams[1]["possession_pct_assigned_frames"]) == pytest.approx(33.3333)
    assert float(teams[2]["possession_pct_video"]) == pytest.approx(25.0)
    assert timeline[-1]["owner_track_id"] == ""
    metadata = json.loads((tmp_path / "analysis_csv_metadata.json").read_text(encoding="utf-8"))
    assert metadata["fps"] == 2
    assert metadata["coordinate_method"] == "constant_meter_per_pixel_assumption"


def test_csv_requires_aligned_frame_counts(tmp_path):
    with pytest.raises(ValueError, match="possession arrays"):
        export_analysis_csv({"players": [{}]}, [], [0], 30, str(tmp_path / "x"))
