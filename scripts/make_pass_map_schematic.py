"""Render a visibly synthetic example of the pass-candidate rule for the report."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pass_map import detect_pass_candidates, save_pass_map
from src.spatial_visualizer import SpatialVisualizer


def main():
    locations = {
        11: ((200, 420), 1), 22: ((700, 240), 1),
        33: ((750, 420), 2), 44: ((300, 180), 2),
    }
    owners = [11, 11, None, None, 22, 22, 33, 33, None, None, 44, 44]
    players = [
        {track_id: {"position": point, "team": team}
         for track_id, (point, team) in locations.items()}
        for _ in owners
    ]
    ball = [{1: {"conf": 0.9}} for _ in owners]
    tracks = {"players": players, "ball": ball}
    visualizer = SpatialVisualizer((1000, 600))
    events = detect_pass_candidates(tracks, owners, 10, visualizer)
    if len(events) != 2:
        raise AssertionError(f"Expected two synthetic candidates, got {len(events)}")
    stem = ROOT / "report/figures/pass_map_schematic"
    paths = save_pass_map(events, visualizer, str(stem), len(owners), 10,
                          example_label="SYNTHETIC EXAMPLE")
    return paths


if __name__ == "__main__":
    print(main())
