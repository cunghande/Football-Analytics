"""Unit tests for src/pipeline_summary.py — pure computation, no ML deps."""
from src.pipeline_summary import compute_summary_stats, write_summary_json
from tests.conftest import N_FRAMES, PLAYER_ID


def test_compute_summary_stats_basic(synthetic_tracks):
    for frame_tracks in synthetic_tracks["players"]:
        frame_tracks[PLAYER_ID]["team"] = 1
        frame_tracks[PLAYER_ID]["speed"] = 12.5

    stats = compute_summary_stats(synthetic_tracks, num_frames=N_FRAMES)

    assert stats["frames"] == N_FRAMES
    assert stats["avg_players_per_frame"] == 1.0
    assert stats["avg_referees_per_frame"] == 0.0
    assert stats["ball_detection_rate"] == 1.0
    assert stats["player_track_id_count"] == 1
    assert stats["team_split"] == [1, 0]
    assert stats["speed_kmh_max"] == 12.5
    assert stats["speed_kmh_min"] == 12.5


def test_compute_summary_stats_team_split_uses_real_team_ids():
    # TeamAssigner labels teams 1 and 2 (src/team_assigner.py), never 0 —
    # team_split must not assume a fixed 0/1 key convention.
    tracks = {
        "players": [{1: {"team": 1}, 2: {"team": 1}, 3: {"team": 2}}],
        "referees": [{}],
        "ball": [{}],
    }
    stats = compute_summary_stats(tracks, num_frames=1)
    assert stats["team_split"] == [2, 1]


def test_compute_summary_stats_team_split_excludes_unknown_team():
    # team == 0 is UNKNOWN_TEAM_ID (src/team_assigner.py), returned for
    # low-confidence assignments — it must not be counted as if it were a
    # real team, even when it's the largest bucket (strictly larger than
    # either real team here, so a version that doesn't exclude it would
    # rank it into team_split instead of [1, 1]).
    tracks = {
        "players": [
            {1: {"team": 1}, 2: {"team": 2}, 3: {"team": 0}, 4: {"team": 0}, 5: {"team": 0}}
        ],
        "referees": [{}],
        "ball": [{}],
    }
    stats = compute_summary_stats(tracks, num_frames=1)
    assert stats["team_split"] == [1, 1]


def test_compute_summary_stats_team_split_excludes_missing_team_key():
    # A player with no "team" key at all (info.get("team") -> None) is the
    # same case as UNKNOWN_TEAM_ID in practice and must be excluded the same
    # way. Unreachable today since main.py always sets "team" on every
    # player, but defensive against a caller that doesn't.
    tracks = {
        "players": [
            {1: {"team": 1}, 2: {"team": 2}, 3: {}, 4: {}, 5: {}}
        ],
        "referees": [{}],
        "ball": [{}],
    }
    stats = compute_summary_stats(tracks, num_frames=1)
    assert stats["team_split"] == [1, 1]


def test_compute_summary_stats_empty_tracks():
    tracks = {"players": [{}], "referees": [{}], "ball": [{}]}
    stats = compute_summary_stats(tracks, num_frames=1)
    assert stats["ball_detection_rate"] == 0.0
    assert stats["player_track_id_count"] == 0
    assert stats["speed_kmh_max"] == 0.0
    assert stats["speed_kmh_min"] == 0.0
    assert stats["team_split"] == [0, 0]


def test_ball_detection_rate_excludes_interpolated_frames():
    tracks = {"players": [{}, {}, {}], "referees": [{}, {}, {}],
              "ball": [{0: {}}, {1: {}}, {0: {}}]}
    stats = compute_summary_stats(tracks, num_frames=3, observed_ball_frames=2)
    assert stats["ball_detection_rate"] == 2 / 3
    assert stats["ball_observed_frames"] == 2
    assert stats["ball_available_frames"] == 3


def test_write_summary_json(tmp_path):
    import json
    stats = {"frames": 10, "ball_detection_rate": 0.5}
    path = tmp_path / "summary.json"
    write_summary_json(stats, str(path))
    assert json.loads(path.read_text()) == stats


def test_write_summary_json_creates_missing_parent_dir(tmp_path):
    import json
    stats = {"frames": 10}
    path = tmp_path / "nested" / "does" / "not" / "exist" / "summary.json"
    write_summary_json(stats, str(path))
    assert json.loads(path.read_text()) == stats
