"""Prepare 50 frame/player crop pairs for human team-label review.

This only proposes candidates. A person must verify every crop before the
labels may be used as reference data. Run: python scripts/prepare_team_review.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.team_assigner import TeamAssigner


DATASET = ROOT / "datasets" / "football_4class_clip_split" / "valid"
OUTPUT = ROOT / "runs" / "team_review"
CALIBRATION_FRAME = 147
EVALUATION_FRAMES = range(148, 198)
FRAME_IDS = range(CALIBRATION_FRAME, 198)
# These proposed white samples were visually identified as red/black players.
# Keys are (source frame number, YOLO label-line index).
MANUAL_CORRECTIONS = {
    (156, 14): "red_black",
    (161, 15): "red_black",
    (162, 15): "red_black",
    (178, 15): "red_black",
    (184, 14): "red_black",
}


def chosen_images() -> dict[int, Path]:
    by_frame = {}
    for path in sorted((DATASET / "images").glob("2_mp4-*.jpg")):
        match = re.match(r"2_mp4-(\d+)_", path.name)
        if match:
            by_frame.setdefault(int(match.group(1)), path)
    missing = set(FRAME_IDS) - set(by_frame)
    if missing:
        raise ValueError(f"Missing selected frames: {sorted(missing)}")
    return by_frame


def player_boxes(image_path: Path, width: int, height: int) -> list[tuple[int, list[int]]]:
    labels = DATASET / "labels" / f"{image_path.stem}.txt"
    boxes = []
    for line_index, line in enumerate(labels.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if not parts or int(parts[0]) != 2:
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        box = [int((cx - bw / 2) * width), int((cy - bh / 2) * height),
               int((cx + bw / 2) * width), int((cy + bh / 2) * height)]
        if box[3] - box[1] >= 25 and box[2] - box[0] >= 8:
            boxes.append((line_index, box))
    return boxes


def expanded_crop(image: Image.Image, box: list[int]) -> Image.Image:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * 0.55, (y2 - y1) * 0.25
    coords = (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
              min(image.width, int(x2 + dx)), min(image.height, int(y2 + dy)))
    return image.crop(coords)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = chosen_images()
    rows = []
    tiles = []
    assigner = TeamAssigner()
    for frame_id in FRAME_IDS:
        path = selected[frame_id]
        image = Image.open(path).convert("RGB")
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        candidates = []
        for line_index, box in player_boxes(path, image.width, image.height):
            x1, y1, x2, y2 = box
            color = assigner.get_player_color(frame, dict(zip(("x1", "y1", "x2", "y2"), box)))
            b, g, r = color
            white_score = float(color.mean() - 0.65 * (color.max() - color.min()) + 0.08 * (y2 - y1))
            red_score = float(r - (b + g) / 2 + 0.02 * (y2 - y1))
            candidates.append({"line_index": line_index, "bbox": box,
                               "white_score": white_score, "red_score": red_score,
                               "bgr": [int(v) for v in color]})
        if len(candidates) < 2:
            raise ValueError(f"Need two field-player candidates in frame {frame_id}")
        white = max(candidates, key=lambda item: item["white_score"])
        red = max((item for item in candidates if item is not white),
                  key=lambda item: item["red_score"])
        chosen = [("white", "clear", white), ("red_black", "clear", red)]
        if frame_id != CALIBRATION_FRAME:
            remaining = [item for item in candidates if item is not white and item is not red]
            white_pool = sorted(remaining, key=lambda item: item["white_score"], reverse=True)[:5]
            white_far = min(white_pool, key=lambda item: item["bbox"][3] - item["bbox"][1])
            remaining = [item for item in remaining if item is not white_far]
            red_pool = sorted(remaining, key=lambda item: item["red_score"], reverse=True)[:5]
            red_far = min(red_pool, key=lambda item: item["bbox"][3] - item["bbox"][1])
            chosen.extend((("white", "additional", white_far),
                           ("red_black", "additional", red_far)))
        for label, sample_type, candidate in chosen:
            correction = MANUAL_CORRECTIONS.get((frame_id, candidate["line_index"]))
            if correction:
                if label != "white":
                    raise ValueError(f"Expected a proposed white label before correcting frame {frame_id}")
                label = correction
            rows.append({"frame_id": frame_id, "image": path.name,
                         "line_index": candidate["line_index"], "manual_team": label,
                         "purpose": "calibration" if frame_id == CALIBRATION_FRAME else "evaluation",
                         "sample_type": sample_type,
                         "bbox_xyxy": " ".join(map(str, candidate["bbox"])),
                         "bgr_median": " ".join(map(str, candidate["bgr"]))})
            crop = expanded_crop(image, candidate["bbox"])
            crop.thumbnail((170, 140), Image.Resampling.NEAREST)
            tile = Image.new("RGB", (220, 177), "white")
            tile.paste(crop, ((220 - crop.width) // 2, 24 + (140 - crop.height) // 2))
            draw = ImageDraw.Draw(tile)
            draw.text((5, 4), f"{frame_id} {label} {sample_type} #{candidate['line_index']}", fill="black")
            draw.text((5, 158), f"BGR {candidate['bgr']}", fill="black")
            tiles.append(tile)
    with (OUTPUT / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for sheet_index in range((len(tiles) + 19) // 20):
        sheet = Image.new("RGB", (5 * 220, 4 * 177), "white")
        for offset, tile in enumerate(tiles[sheet_index * 20:(sheet_index + 1) * 20]):
            sheet.paste(tile, ((offset % 5) * 220, (offset // 5) * 177))
        sheet.save(OUTPUT / f"candidate_sheet_{sheet_index + 1}.jpg", quality=94)
    print(f"Proposed {len(rows)} player labels across {len(FRAME_IDS)} frames in {OUTPUT}")


if __name__ == "__main__":
    main()
