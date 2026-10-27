"""
Unit tests for TeamAssigner multi-frame fitting and confidence threshold
"""
import numpy as np
from src.team_assigner import TeamAssigner, UNKNOWN_TEAM_ID

RED = (0, 0, 255)     # BGR
BLUE = (255, 0, 0)    # BGR
GREY = (127, 0, 127)  # equidistant from RED and BLUE in BGR space

RED_BBOX = [0, 0, 20, 40]
BLUE_BBOX = [50, 0, 70, 40]
GREY_BBOX = [0, 50, 20, 90]


def _frame_with_color(height, width, bbox, color):
    """Return a black frame with the upper-half of bbox filled with color"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    upper_y2 = int((y1 + y2) * 0.5)
    frame[y1:upper_y2, x1:x2] = color
    return frame


def _two_player_frame():
    """Frame with a red player and a blue player"""
    frame = _frame_with_color(100, 100, RED_BBOX, RED)
    frame[0:20, 50:70] = BLUE  # blue player upper-half crop region
    return frame


def test_fit_separates_two_distinct_teams():
    frames = [_two_player_frame()]
    tracks = [{1: {"bbox": RED_BBOX}, 2: {"bbox": BLUE_BBOX}}]

    ta = TeamAssigner(n_teams=2, min_confidence_threshold=0.1)
    ta.fit_team_colors(frames, tracks)

    assert ta.kmeans is not None
    assert len(ta.team_colors) == 2

    red_frame = _frame_with_color(100, 100, RED_BBOX, RED)
    blue_frame = _frame_with_color(100, 100, BLUE_BBOX, BLUE)

    team_red = ta.get_player_team(red_frame, RED_BBOX, player_id=1)
    team_blue = ta.get_player_team(blue_frame, BLUE_BBOX, player_id=2)

    assert team_red in (1, 2)
    assert team_blue in (1, 2)
    assert team_red != team_blue


def test_fit_aggregates_across_frames():
    """Blue player only visible in frame 1 — still forms a correct second cluster"""
    frame0 = _frame_with_color(100, 100, RED_BBOX, RED)  # red only
    frame1 = _two_player_frame()                         # red + blue

    frames = [frame0, frame1]
    tracks = [
        {1: {"bbox": RED_BBOX}},
        {1: {"bbox": RED_BBOX}, 2: {"bbox": BLUE_BBOX}},
    ]

    ta = TeamAssigner(n_teams=2, color_sample_frames=2, min_confidence_threshold=0.1)
    ta.fit_team_colors(frames, tracks)

    assert ta.kmeans is not None
    assert len(ta.team_colors) == 2

    red_frame = _frame_with_color(100, 100, RED_BBOX, RED)
    blue_frame = _frame_with_color(100, 100, BLUE_BBOX, BLUE)

    team_red = ta.get_player_team(red_frame, RED_BBOX, player_id=10)
    team_blue = ta.get_player_team(blue_frame, BLUE_BBOX, player_id=20)
    assert team_red != team_blue


def test_low_confidence_returns_unknown_and_is_not_cached():
    """Equidistant color → UNKNOWN_TEAM_ID returned and assignment not cached"""
    frames = [_two_player_frame()]
    tracks = [{1: {"bbox": RED_BBOX}, 2: {"bbox": BLUE_BBOX}}]

    ta = TeamAssigner(n_teams=2, min_confidence_threshold=0.9)
    ta.fit_team_colors(frames, tracks)

    grey_frame = _frame_with_color(100, 100, GREY_BBOX, GREY)
    result = ta.get_player_team(grey_frame, GREY_BBOX, player_id=99)

    assert result == UNKNOWN_TEAM_ID
    assert 99 not in ta.player_team_assignment


def test_high_confidence_assignment_is_cached():
    """Clearly colored player gets assigned and the result is cached"""
    frames = [_two_player_frame()]
    tracks = [{1: {"bbox": RED_BBOX}, 2: {"bbox": BLUE_BBOX}}]

    ta = TeamAssigner(n_teams=2, min_confidence_threshold=0.1)
    ta.fit_team_colors(frames, tracks)

    red_frame = _frame_with_color(100, 100, RED_BBOX, RED)
    team = ta.get_player_team(red_frame, RED_BBOX, player_id=10)

    assert team in (1, 2)
    assert ta.player_team_assignment.get(10) == team


def test_team_color_overrides_change_rendering_color():
    """team_color_overrides updates team_colors without affecting cluster geometry"""
    frames = [_two_player_frame()]
    tracks = [{1: {"bbox": RED_BBOX}, 2: {"bbox": BLUE_BBOX}}]

    green = (0, 255, 0)
    ta = TeamAssigner(n_teams=2, team_color_overrides={1: green})
    ta.fit_team_colors(frames, tracks)

    assert np.allclose(ta.team_colors[1], np.array(green, dtype=float))


def test_not_enough_players_leaves_kmeans_none():
    """Fewer than two colors leaves the team unknown rather than guessing team 1."""
    frame = _frame_with_color(100, 100, RED_BBOX, RED)
    frames = [frame]
    tracks = [{1: {"bbox": RED_BBOX}}]  # only one player, n_teams=2

    ta = TeamAssigner(n_teams=2)
    ta.fit_team_colors(frames, tracks)

    assert ta.kmeans is None
    result = ta.get_player_team(frame, RED_BBOX, player_id=1)
    assert result == UNKNOWN_TEAM_ID


def test_non_distinct_colors_leave_kmeans_none():
    """Identical colors leave the team unknown."""
    black_frame = np.zeros((200, 640, 3), dtype=np.uint8)
    frames = [black_frame] * 20
    tracks = [{1: {"bbox": [100, 100, 120, 200]}}] * 20

    ta = TeamAssigner(n_teams=2)
    ta.fit_team_colors(frames, tracks)

    assert ta.kmeans is None
    result = ta.get_player_team(black_frame, [100, 100, 120, 200], player_id=1)
    assert result == UNKNOWN_TEAM_ID


def test_recluster_preserves_team_ids_under_color_shift():
    frames = []
    tracks = []
    for idx in range(6):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[0:20, 0:20] = (0, 0, 240 - idx * 10)
        frame[0:20, 50:70] = (240 - idx * 10, 0, 0)
        frames.append(frame)
        tracks.append({1: {"bbox": RED_BBOX}, 2: {"bbox": BLUE_BBOX}})
    ta = TeamAssigner(n_teams=2, color_sample_frames=2, recluster_interval=3,
                      recluster_window=3, min_confidence_threshold=0.1)
    ta.fit_team_colors(frames, tracks)
    initial_red = ta.get_player_team(frames[0], RED_BBOX, 1)
    initial_blue = ta.get_player_team(frames[0], BLUE_BBOX, 2)
    assert ta.update_team_colors(frames, tracks, 3)
    assert ta.refresh_count == 1
    assert ta.get_player_team(frames[4], RED_BBOX, 1) == initial_red
    assert ta.get_player_team(frames[4], BLUE_BBOX, 2) == initial_blue
