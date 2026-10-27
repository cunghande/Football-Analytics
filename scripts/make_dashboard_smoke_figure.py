"""Verify the video-to-report path on three labeled images; not a model evaluation."""

from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dashboard_io import analysis_files, run_job


def main():
    images = ROOT / "datasets/football_4class_clip_split/valid/images"
    checkpoint = ROOT / "runs/modal/detection_player_freeze10_v1/weights/best.pt"
    job = ROOT / "output/dashboard_jobs/academic_smoke_v2"
    job.mkdir(parents=True, exist_ok=True)
    source = job / "input.mp4"
    files = analysis_files(job / "analysis")
    if not files["video"].exists():
        selected = [next(images.glob(f"2_mp4-{frame}_*.jpg")) for frame in (148, 149, 150)]
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10.0, (640, 360))
        if not writer.isOpened():
            raise RuntimeError("Cannot create technical smoke-test clip")
        for path in selected:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Cannot read {path}")
            writer.write(cv2.resize(frame, (640, 360)))
        writer.release()
        run_job(source, job, checkpoint, "cpu")

    capture = cv2.VideoCapture(str(files["video"]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Cannot read browser-compatible output video")
    figure, axis = plt.subplots(figsize=(11, 6.2))
    figure.subplots_adjust(top=0.88, bottom=0.12, left=0.02, right=0.98)
    axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axis.axis("off")
    figure.suptitle("Technical smoke test: 3 validation images at assumed 10 FPS",
                     fontsize=12, y=0.98)
    figure.text(0.5, 0.04,
                "Pipeline/video rendering check only — no real match timing or performance score",
                ha="center", fontsize=10)
    out = ROOT / "report/figures/dashboard_pipeline_smoke"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
    figure.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(figure)
    return out


if __name__ == "__main__":
    print(main())
