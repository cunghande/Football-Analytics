import csv

import cv2
import numpy as np

from scripts.export_role_annotation_frames import export_frames


def test_exports_spaced_unlabelled_frames_with_source_group(tmp_path):
    video = tmp_path / "match_a.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for frame in range(10):
        writer.write(np.full((48, 64, 3), frame * 20, dtype=np.uint8))
    writer.release()

    output = tmp_path / "review"
    assert export_frames(video, output, count=4) == 4
    rows = list(csv.DictReader((output / "manifest.csv").open(encoding="utf-8")))
    assert [int(row["frame"]) for row in rows] == [0, 3, 6, 9]
    assert {row["match_group"] for row in rows} == {"match_a"}
    assert len(list((output / "images").glob("*.jpg"))) == 4
    assert not (output / "labels").exists()
