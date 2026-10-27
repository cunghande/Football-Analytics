import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from src.dashboard_io import (
    analysis_files, create_job, create_report_zip, probe_video,
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
    assert "dashboard_run.json" in names
    assert "analysis.mp4" not in names
    assert "analysis_browser.mp4" not in names
