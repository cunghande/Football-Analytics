"""
Integration test for everything downstream of detection.
Runs without YOLO or the input video by using synthetic track data.
Verifies that all modules compose correctly and write the expected keys
into the tracks dict.
"""
import pytest
from src.camera_motion import CameraMotionAnalyzer
from src.player_ball_assigner import PlayerBallAssigner
from src.speed_distance_calculator import SpeedDistanceCalculator
from src.team_assigner import TeamAssigner
from tests.conftest import N_FRAMES, PLAYER_ID, BALL_ID

SCALE = 0.1  # mirrors SPEED_CONFIG["meters_per_pixel"]


def _add_positions(tracks):
    for obj_name, obj_tracks in tracks.items():
        for frame_num, frame_tracks in enumerate(obj_tracks):
            for track_id, info in frame_tracks.items():
                x1, y1, x2, y2 = info["bbox"]
                if obj_name == "ball":
                    info["position"] = ((x1 + x2) // 2, (y1 + y2) // 2)
                else:
                    info["position"] = ((x1 + x2) // 2, int(y2))


def _add_fallback_transformed(tracks, scale):
    for obj_name, obj_tracks in tracks.items():
        for frame_num, frame_tracks in enumerate(obj_tracks):
            for track_id, info in frame_tracks.items():
                adj = info.get("position_adjusted")
                if adj is None:
                    info["position_transformed"] = None
                else:
                    info["position_transformed"] = (adj[0] * scale, adj[1] * scale)


def test_full_downstream_pipeline(synthetic_tracks, synthetic_frames):
    # Step 1 — position enrichment
    _add_positions(synthetic_tracks)

    # Step 2 — camera motion (zero movement for static synthetic data)
    camera_movement = [(0.0, 0.0)] * N_FRAMES
    CameraMotionAnalyzer().add_adjusted_positions_to_tracks(synthetic_tracks, camera_movement)

    # Step 3 — fallback perspective transform
    _add_fallback_transformed(synthetic_tracks, SCALE)

    # Step 4 — team assignment (only one player → not enough for KMeans → team defaults to 1)
    team_assigner = TeamAssigner(n_teams=2)
    team_assigner.fit_team_colors(synthetic_frames, synthetic_tracks["players"])
    for frame_num, player_frame in enumerate(synthetic_tracks["players"]):
        for pid, pinfo in player_frame.items():
            pinfo["team"] = team_assigner.get_player_team(synthetic_frames[frame_num], pinfo["bbox"], pid)

    # Step 5 — ball possession
    ball_assigner = PlayerBallAssigner(max_player_ball_distance=70)
    possession_found = False
    for frame_num, player_frame in enumerate(synthetic_tracks["players"]):
        ball_frame = synthetic_tracks["ball"][frame_num]
        if ball_frame:
            ball_bbox = next(iter(ball_frame.values()))["bbox"]
            assigned = ball_assigner.assign_ball_to_player(player_frame, ball_bbox)
            if assigned != -1:
                player_frame[assigned]["has_ball"] = True
                possession_found = True

    # Step 6 — speed and distance
    SpeedDistanceCalculator(fps=30.0).add_speed_and_distance_to_tracks(synthetic_tracks)

    # Assert all expected keys are present in the final player frame
    last_player = synthetic_tracks["players"][-1][PLAYER_ID]
    for key in ("position", "position_adjusted", "position_transformed", "team", "speed", "distance"):
        assert key in last_player, f"missing key: {key}"

    assert possession_found, "ball was never assigned to a player"
    assert synthetic_tracks["players"][0][PLAYER_ID]["team"] == 0
