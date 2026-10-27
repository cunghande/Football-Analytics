import csv
import json

import pytest

from src.session_report import generate_session_report


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path, *, calibrated=False):
    stem = tmp_path / "analysis"
    _csv(tmp_path / "analysis_players.csv", [{
        "role": "player", "track_id": "7", "team_id": "1", "observed_frames": "3",
        "possession_frames": "2", "distance_m_estimate": "99999",
        "mean_speed_kmh_estimate": "5.5", "max_speed_kmh_estimate": "8.0",
    }])
    _csv(tmp_path / "analysis_teams.csv", [
        {"team_id": "1", "tracked_people": "1", "possession_frames": "2", "possession_pct_video": "50"},
        {"team_id": "2", "tracked_people": "0", "possession_frames": "1", "possession_pct_video": "25"},
        {"team_id": "0", "tracked_people": "0", "possession_frames": "1", "possession_pct_video": "25"},
    ])
    (tmp_path / "analysis_csv_metadata.json").write_text(json.dumps({
        "frames": 4, "fps": 2,
        "coordinate_method": ("four_point_field_homography" if calibrated
                              else "constant_meter_per_pixel_assumption"),
    }), encoding="utf-8")
    (tmp_path / "analysis_summary.json").write_text(json.dumps({
        "frames": 4, "ball_detection_rate": 0.75,
    }), encoding="utf-8")
    (tmp_path / "dashboard_run.json").write_text(json.dumps({
        "source_name": "practice.mp4", "model_path": "/runs/freeze/best.pt",
    }), encoding="utf-8")
    return stem


def _text(path):
    pypdf = pytest.importorskip("pypdf")
    return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(path)).pages)


def test_uncalibrated_report_hides_false_precision_and_has_track_data(tmp_path):
    stem = _fixture(tmp_path)
    output = generate_session_report(stem)
    assert output.read_bytes().startswith(b"%PDF")
    content = _text(output)
    assert "practice.mp4" in content
    assert "P7" in content
    assert "99999" not in content
    assert "chấm điểm" in content


def test_calibrated_report_includes_estimated_movement(tmp_path):
    stem = _fixture(tmp_path, calibrated=True)
    content = " ".join(_text(generate_session_report(stem)).split())
    assert "99 999.0" in content
    assert "Tốc độ" in content


def test_report_rejects_inconsistent_team_totals(tmp_path):
    stem = _fixture(tmp_path)
    path = tmp_path / "analysis_teams.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows[0]["possession_frames"] = "4"
    _csv(path, rows)
    with pytest.raises(ValueError, match="frame totals"):
        generate_session_report(stem)


def test_no_ball_report_marks_possession_unavailable(tmp_path):
    stem = _fixture(tmp_path)
    _csv(tmp_path / "analysis_teams.csv", [
        {"team_id": "1", "tracked_people": "1", "possession_frames": "0", "possession_pct_video": "0"},
        {"team_id": "2", "tracked_people": "0", "possession_frames": "0", "possession_pct_video": "0"},
        {"team_id": "0", "tracked_people": "0", "possession_frames": "4", "possession_pct_video": "100"},
    ])
    (tmp_path / "analysis_summary.json").write_text(json.dumps({
        "frames": 4, "ball_detection_rate": 0,
    }), encoding="utf-8")
    content = " ".join(_text(generate_session_report(stem)).split())
    assert "Không có số liệu kiểm soát bóng hợp lệ" in content
    assert "Không đo được" in content
    assert "100.0%" not in content
