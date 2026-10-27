import csv
import json

import pytest

from src.pass_map import detect_pass_candidates, save_pass_map
from src.spatial_visualizer import SpatialVisualizer


def _fixture(receiver_team=1, ball_in_gap=True):
    players = []
    balls = []
    for frame in range(6):
        players.append({
            11: {"team": 1, "position": (100, 250), "position_transformed": (12, 40)},
            22: {"team": receiver_team, "position": (600, 250),
                 "position_transformed": (72, 40)},
        })
        conf = 0.0 if frame in (2, 3) and not ball_in_gap else 0.9
        balls.append({1: {"conf": conf}})
    return {"players": players, "ball": balls}


def test_candidate_requires_stable_same_team_and_observed_free_ball(tmp_path):
    owners = [11, 11, None, None, 22, 22]
    visualizer = SpatialVisualizer((1000, 500))
    events = detect_pass_candidates(_fixture(), owners, 10, visualizer)
    assert len(events) == 1
    event = events[0]
    assert (event["start_frame"], event["end_frame"], event["gap_frames"]) == (1, 4, 2)
    assert (event["from_track_id"], event["to_track_id"], event["team_id"]) == (11, 22, 1)
    assert event["distance_norm"] == pytest.approx(0.5)
    assert event["distance_m_if_calibrated"] == ""
    paths = save_pass_map(events, visualizer, str(tmp_path / "analysis"), 6, 10)
    with open(paths["csv"], newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    assert json.loads((tmp_path / "analysis_pass_map_metadata.json").read_text(encoding="utf-8"))["candidate_count"] == 1
    assert (tmp_path / "analysis_pass_map.png").stat().st_size > 0
    assert (tmp_path / "analysis_pass_map.pdf").stat().st_size > 0


def test_rejects_opponent_direct_switch_and_interpolated_ball():
    visualizer = SpatialVisualizer((1000, 500))
    assert detect_pass_candidates(_fixture(receiver_team=2),
                                  [11, 11, None, None, 22, 22], 10, visualizer) == []
    assert detect_pass_candidates(_fixture(),
                                  [11, 11, 22, 22, 22, 22], 10, visualizer) == []
    assert detect_pass_candidates(_fixture(ball_in_gap=False),
                                  [11, 11, None, None, 22, 22], 10, visualizer) == []


def test_empty_pass_map_is_a_valid_output(tmp_path):
    paths = save_pass_map([], SpatialVisualizer((1000, 500)),
                          str(tmp_path / "empty"), 3, 10)
    with open(paths["csv"], newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    assert (tmp_path / "empty_pass_map.png").stat().st_size > 0


def test_calibrated_candidate_distance_and_gap_gate():
    visualizer = SpatialVisualizer((1000, 500), calibrated=True)
    events = detect_pass_candidates(_fixture(),
                                    [11, 11, None, None, 22, 22], 10, visualizer)
    assert events[0]["distance_m_if_calibrated"] == pytest.approx(60.0)
    assert detect_pass_candidates(_fixture(),
                                  [11, 11, None, None, 22, 22], 10, visualizer,
                                  max_gap_seconds=0.1) == []
