"""Exercise upload-equivalent analysis and PDF/ZIP generation on 3 stills.

The resulting clip uses an assumed FPS and is only a technical smoke test.
"""

import json
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dashboard_io import create_report_zip, run_job


def main():
    images = ROOT / "datasets/football_4class_clip_split/valid/images"
    checkpoint = ROOT / "runs/modal/detection_player_freeze10_v1/weights/best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    selected = [next(images.glob(f"2_mp4-{frame}_*.jpg")) for frame in (148, 149, 150)]
    job = ROOT / "output/dashboard_jobs/session_report_smoke"
    job.mkdir(parents=True, exist_ok=True)
    source = job / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Could not write the technical test clip")
    for path in selected:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read {path}")
        writer.write(cv2.resize(frame, (640, 360)))
    writer.release()
    (job / "dataset_demo_protocol.json").write_text(json.dumps({
        "purpose": "technical_demo_only",
        "source": "FSDM detection-player v4 validation, clip 2, frames 148-150",
        "assumed_fps": 10,
        "warning": "Still images; no source timing or verified track IDs.",
    }, indent=2), encoding="utf-8")
    files = run_job(source, job, checkpoint, "cpu")
    bundle = create_report_zip(job)
    for key in ("video", "session_report", "players", "teams", "heatmaps"):
        assert files[key].exists() and files[key].stat().st_size > 0, key
    assert bundle.stat().st_size > 0
    print(f"PDF: {files['session_report']}")
    print(f"ZIP: {bundle}")


if __name__ == "__main__":
    main()
