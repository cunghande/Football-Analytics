"""Build a 50-frame technical demo from validation stills, with assumed FPS.

This is not a real-time source video and must not be used for speed, distance,
ball-control duration, pass precision, or match-level generalization claims.
"""

import json
from pathlib import Path
import re
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dashboard_io import create_report_zip, run_job


def main():
    source = ROOT / "datasets/football_4class_clip_split/valid/images"
    by_frame = {}
    for path in sorted(source.glob("2_mp4-*.jpg")):
        match = re.match(r"2_mp4-(\d+)_", path.name)
        if match:
            by_frame.setdefault(int(match.group(1)), path)
    frames = list(range(148, 198))
    missing = set(frames) - set(by_frame)
    if missing:
        raise FileNotFoundError(f"Missing source frames: {sorted(missing)}")
    job = ROOT / "output/dashboard_jobs/dataset_demo50"
    job.mkdir(parents=True, exist_ok=True)
    input_path = job / "input.mp4"
    assumed_fps = 25.0
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             assumed_fps, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Cannot write demo input")
    for frame_id in frames:
        image = cv2.imread(str(by_frame[frame_id]))
        if image is None:
            raise ValueError(f"Cannot read frame {frame_id}")
        writer.write(cv2.resize(image, (640, 360)))
    writer.release()
    protocol = {
        "purpose": "technical_demo_only",
        "dataset": "FSDM detection-player v4, CC BY 4.0",
        "source": "https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4",
        "split": "derived validation, clip 2",
        "source_frame_range": [148, 197],
        "frame_count": len(frames),
        "selection": "first sorted Roboflow export per source frame number",
        "assumed_fps": assumed_fps,
        "warning": "Source images have no verified FPS or real track IDs. Video-derived speed, distance, possession duration and pass counts are not scientific measurements.",
    }
    (job / "dataset_demo_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = ROOT / "runs/modal/detection_player_freeze10_v1/weights/best.pt"
    files = run_job(input_path, job, checkpoint, "cpu")
    bundle = create_report_zip(job)
    print(f"Browser video: {files['video']}")
    print(f"Technical report ZIP: {bundle}")


if __name__ == "__main__":
    main()
