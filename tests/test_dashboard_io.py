import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from src.dashboard_io import (
    analysis_files, create_job, create_job_from_local_file, create_report_zip,
    prepare_analysis_clip, probe_video,
    transcode_for_browser,
)


class FakeUpload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def test_uploaded_name_never_controls_destination(tmp_path):
    uploaded = FakeUpload(b"test", "../../somewhere/evil.mp4")
    job, path = create_job(uploaded, tmp_path / "jobs")
    assert path == job / "input.mp4"
    assert path.read_bytes() == b"test"
    with pytest.raises(ValueError, match="MP4"):
        create_job(FakeUpload(b"test", "evil.exe"), tmp_path / "jobs")


def test_local_video_avoids_upload_copy(tmp_path, monkeypatch):
    import src.dashboard_io as dashboard_io

    monkeypatch.setattr(dashboard_io, "MAX_UPLOAD_BYTES", 2)
    source = tmp_path / "local.mp4"
    source.write_bytes(b"1234")
    with pytest.raises(ValueError, match="dung lượng"):
        create_job(FakeUpload(b"1234", "local.mp4"), tmp_path / "jobs")
    job, selected = create_job_from_local_file(source, tmp_path / "jobs")
    assert selected == source.resolve()
    assert not list(job.iterdir())


def test_probe_and_browser_transcode(tmp_path):
    source = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (64, 48))
    assert writer.isOpened()
    for index in range(5):
        writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    writer.release()
    details = probe_video(source)
    assert details["frames"] == 5
    assert details["fps"] == pytest.approx(10)
    assert details["duration_seconds"] == pytest.approx(0.5)
    browser = transcode_for_browser(source, tmp_path / "clip_browser.mp4")
    assert browser.stat().st_size > 0
    assert cv2.VideoCapture(str(browser)).isOpened()


def test_long_video_selects_real_segment(tmp_path, monkeypatch):
    import src.dashboard_io as dashboard_io

    monkeypatch.setattr(dashboard_io, "MAX_FRAMES", 10)
    source = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (64, 48))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    original = probe_video(source, enforce_limits=False)
    selected, clip = prepare_analysis_clip(source, tmp_path, original, start_seconds=0.5)
    analyzed = probe_video(selected)
    assert analyzed["frames"] == 10
    assert analyzed["fps"] == pytest.approx(10)
    assert clip["start_seconds"] == 0.5
    assert clip["end_seconds"] == pytest.approx(1.5)
    capture = cv2.VideoCapture(str(selected))
    ok, first = capture.read()
    capture.release()
    assert ok and 35 < float(first.mean()) < 65


def test_extended_segment_reduces_pixels_per_frame(tmp_path, monkeypatch):
    import src.dashboard_io as dashboard_io

    monkeypatch.setattr(dashboard_io, "MAX_FRAMES", 10)
    monkeypatch.setattr(dashboard_io, "MAX_FRAME_PIXELS", 160 * 90)
    source = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (160, 90))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((90, 160, 3), index * 10, dtype=np.uint8))
    writer.release()
    original = probe_video(source, enforce_limits=False)
    selected, clip = prepare_analysis_clip(source, tmp_path, original, max_frames=20)
    analyzed = probe_video(selected, max_frames=20, max_frame_pixels=160 * 90 // 2)
    assert analyzed["frames"] == 20
    assert analyzed["width"] * analyzed["height"] <= 160 * 90 // 2
    assert clip["analyzed_frames"] == 20


def test_report_zip_excludes_videos(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    files = analysis_files(job / "analysis")
    for key, path in files.items():
        path.write_bytes(b"demo")
    (job / "dashboard_run.json").write_text(json.dumps({"device": "cpu"}), encoding="utf-8")
    report = create_report_zip(job)
    with zipfile.ZipFile(report) as archive:
        names = set(archive.namelist())
    assert "analysis_players.csv" in names
    assert "analysis_heatmaps.pdf" in names
    assert "analysis_pass_candidates.csv" in names
    assert "analysis_pass_map.png" in names
    assert "analysis_session_report.pdf" in names
    assert "dashboard_run.json" in names
    assert "analysis.mp4" not in names
    assert "analysis_browser.mp4" not in names


def test_report_zip_keeps_technical_demo_protocol(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    for path in analysis_files(job / "analysis").values():
        path.write_bytes(b"demo")
    (job / "dashboard_run.json").write_text("{}", encoding="utf-8")
    (job / "dataset_demo_protocol.json").write_text('{"purpose":"technical_demo_only"}', encoding="utf-8")
    with zipfile.ZipFile(create_report_zip(job)) as archive:
        assert "dataset_demo_protocol.json" in archive.namelist()
