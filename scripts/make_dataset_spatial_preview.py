"""Create an honest image-space preview from 50 validation frames and GT boxes."""

import json
from pathlib import Path
import re
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spatial_visualizer import SpatialVisualizer
VALID = ROOT / "datasets" / "football_4class_clip_split" / "valid"
OUT = ROOT / "report" / "figures"
FRAME_IDS = range(148, 198)
ROLES = {0: "ball", 1: "goalkeepers", 2: "players", 3: "referees"}
COLORS = {0: (255, 255, 255), 1: (0, 165, 255), 2: (255, 100, 50), 3: (0, 230, 230)}


def _image_for_frame():
    found = {}
    for path in sorted((VALID / "images").glob("2_mp4-*.jpg")):
        match = re.match(r"2_mp4-(\d+)_", path.name)
        if match:
            found.setdefault(int(match.group(1)), path)
    missing = set(FRAME_IDS) - set(found)
    if missing:
        raise FileNotFoundError(f"Missing validation frames: {sorted(missing)}")
    return found


def make_preview():
    image_paths = _image_for_frame()
    sample = cv2.imread(str(image_paths[148]))
    if sample is None:
        raise ValueError("Cannot read sample frame")
    reference_height, reference_width = sample.shape[:2]
    visualizer = SpatialVisualizer((reference_width, reference_height))
    tracks = {"players": []}
    sample_tracks = None
    marked = sample.copy()
    for frame_id in FRAME_IDS:
        image_path = image_paths[frame_id]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read {image_path}")
        height, width = image.shape[:2]
        roles = {role: {} for role in ROLES.values()}
        label_path = VALID / "labels" / f"{image_path.stem}.txt"
        for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Malformed annotation in {label_path}")
            class_id = int(values[0])
            if class_id not in ROLES:
                raise ValueError(f"Unexpected class {class_id} in {label_path}")
            cx, cy, bw, bh = map(float, values[1:])
            foot_y = cy if class_id == 0 else cy + bh / 2
            info = {"position": (cx * reference_width, foot_y * reference_height)}
            if class_id == 2:
                info["team"] = 0  # Dataset has no team labels.
            roles[ROLES[class_id]][line_index] = info
            if frame_id == 148:
                x1, y1 = round((cx - bw / 2) * width), round((cy - bh / 2) * height)
                x2, y2 = round((cx + bw / 2) * width), round((cy + bh / 2) * height)
                cv2.rectangle(marked, (x1, y1), (x2, y2), COLORS[class_id], 2)
        tracks["players"].append(roles["players"])
        if frame_id == 148:
            sample_tracks = roles

    grids, counts, discarded = visualizer.density(tracks)
    density = cv2.GaussianBlur(grids[0], (0, 0), 1.2)
    minimap = visualizer.minimap_canvas(sample_tracks, width=480, height=300)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Frame 148: ground-truth boxes")
    axes[1].imshow(cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Frame 148: normalized image positions")
    image = axes[2].imshow(density, cmap="magma", origin="upper",
                           extent=(0, 100, 100, 0), interpolation="nearest")
    axes[2].set_title(f"Player density: {counts[0]} boxes / 50 frames")
    axes[2].set_xlabel("Image width (%)")
    axes[2].set_ylabel("Image height (%)")
    axes[2].set_ylim(100, 0)
    fig.colorbar(image, ax=axes[2], shrink=0.74, label="Smoothed box count per cell")
    for ax in axes[:2]:
        ax.axis("off")
    fig.suptitle("detection-player v4 validation — image-space illustration, not player trajectories")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "dataset_spatial_preview_50frames"
    fig.savefig(str(stem) + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "dataset": "FSDM detection-player v4, CC BY 4.0",
        "source": "https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4",
        "split": "derived validation, clip 2, one export per source frame",
        "frame_range": [148, 197],
        "frame_count": 50,
        "sample_image": image_paths[148].name,
        "player_boxes_count": counts[0],
        "discarded_player_boxes": discarded,
        "coordinate_mode": "normalized image; no field calibration",
        "qualification": "Boxes are GT detections, not tracked identities; no speed or distance inferred.",
    }
    (OUT / "dataset_spatial_preview_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return stem


if __name__ == "__main__":
    print(make_preview())
