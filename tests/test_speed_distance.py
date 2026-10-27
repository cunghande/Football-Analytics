import pytest
from src.speed_distance_calculator import SpeedDistanceCalculator


def _tracks(positions_by_frame):
    """Build tracks dict for player id=1 with the given position_transformed per frame."""
    return {"players": [
        {1: {"bbox": [0, 0, 10, 10], "position_transformed": pos}}
        if pos is not None else {}
        for pos in positions_by_frame
    ]}


def test_speed_for_known_displacement():
    # 10 m in 5 frames at 5 fps = 1 second → 10 m/s = 36 km/h
    calc = SpeedDistanceCalculator(fps=5.0)
    tracks = _tracks([(0.0, 0.0)] * 5 + [(10.0, 0.0)])
    calc.add_speed_and_distance_to_tracks(tracks, frame_window=5)
    assert tracks["players"][0][1]["speed"] == 0.0
    assert tracks["players"][5][1]["speed"] == pytest.approx(36.0)


def test_distance_accumulates_across_windows():
    # Window 1 (frames 0→5): move 3 m east. Window 2 (frames 5→10): move 4 m north.
    # Total: 3 + 4 = 7 m.
    calc = SpeedDistanceCalculator(fps=5.0)
    positions = [(0.0, 0.0)] * 5 + [(3.0, 0.0)] * 5 + [(3.0, 4.0)]
    tracks = _tracks(positions)
    calc.add_speed_and_distance_to_tracks(tracks, frame_window=5)
    assert abs(tracks["players"][10][1]["distance"] - 7.0) < 0.01


def test_missing_position_transformed_does_not_crash():
    calc = SpeedDistanceCalculator(fps=5.0)
    tracks = _tracks([None] * 6)
    calc.add_speed_and_distance_to_tracks(tracks, frame_window=5)


def test_measure_distance_pythagorean():
    assert SpeedDistanceCalculator._measure_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert SpeedDistanceCalculator._measure_distance((1.0, 1.0), (1.0, 1.0)) == pytest.approx(0.0)


def test_no_distance_added_across_missing_frame_and_goalkeeper_measured():
    tracks = {"players": [
        {1: {"position_transformed": (0.0, 0.0)}},
        {},
        {1: {"position_transformed": (100.0, 0.0)}},
        {1: {"position_transformed": (103.0, 4.0)}},
    ], "goalkeepers": [
        {2: {"position_transformed": (0.0, 0.0)}},
        {2: {"position_transformed": (0.0, 1.0)}},
        {}, {},
    ]}
    SpeedDistanceCalculator(fps=10).add_speed_and_distance_to_tracks(tracks)
    assert tracks["players"][3][1]["distance"] == pytest.approx(5.0)
    assert tracks["goalkeepers"][1][2]["distance"] == pytest.approx(1.0)
