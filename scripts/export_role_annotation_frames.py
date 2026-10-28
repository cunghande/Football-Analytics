"""Export spaced video frames for manual four-class role annotation.

This script writes images and provenance only. It deliberately creates no
labels: YOLO predictions on this clip include goalkeeper/player mistakes.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def export_frames(video, destination, count=60):
    video = Path(video)
    destination = Path(destination)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total < 1 or fps <= 0 or count < 1:
            raise ValueError("Video frame count, FPS, and requested count must be positive")
        indices = sorted(set(np.linspace(0, total - 1, min(count, total), dtype=int).tolist()))
        images = destination / "images"
        images.mkdir(parents=True, exist_ok=True)
        rows = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {index} from {video}")
            filename = f"frame_{index:06d}.jpg"
            if not cv2.imwrite(str(images / filename), frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise IOError(f"Could not write {filename}")
            rows.append({"image": filename, "source_video": video.name,
                         "frame": index, "time_seconds": round(index / fps, 4),
                         "width": frame.shape[1], "height": frame.shape[0],
                         "match_group": video.stem})
    finally:
        capture.release()
    with (destination / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (destination / "LABELING_INSTRUCTIONS.txt").write_text(
        "Ảnh CHƯA CÓ NHÃN. Hãy vẽ box cho mọi ball, goalkeeper, player và referee "
        "nhìn thấy được. Thủ môn vẫn là goalkeeper dù màu áo thay đổi. Không coi ảnh "
        "chưa gán nhãn là ảnh rỗng. Giữ toàn bộ frame cùng trận trong một nhóm chia "
        "train/validation/test để tránh rò rỉ. Kiểm tra riêng cảnh cận và cảnh cắt.\n",
        encoding="utf-8",
    )
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    print(f"Exported {export_frames(args.video, args.destination, args.count)} frames to {args.destination}")
