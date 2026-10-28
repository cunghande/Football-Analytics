from src.camera_motion import CameraMotionAnalyzer


def test_adjusted_position_subtracts_cumulative_camera_translation():
    tracks = {"players": [
        {1: {"position": (10, 20)}},
        {1: {"position": (12, 20)}},
        {1: {"position": (14, 20)}},
    ]}
    CameraMotionAnalyzer().add_adjusted_positions_to_tracks(
        tracks, [(0, 0), (2, 0), (2, 0)])
    assert [row[1]["position_adjusted"] for row in tracks["players"]] == [
        (10, 20), (10, 20), (10, 20)]
