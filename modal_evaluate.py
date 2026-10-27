"""Evaluate baseline, freeze=10 and full YOLO detectors on Modal.

Run after the freeze=10 job has saved its best.pt checkpoint:
    python -X utf8 -m modal run modal_evaluate.py
"""

from pathlib import Path
import subprocess
import sys

import modal


app = modal.App("football-detector-evaluation")
project_root = Path(__file__).resolve().parent
dataset_volume = modal.Volume.from_name("football-dataset", create_if_missing=False)
results_volume = modal.Volume.from_name("football-training-runs", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(project_root / "requirements.txt"))
    .pip_install("pycocotools==2.0.11", "Pillow")
    .add_local_file(str(project_root / "scripts" / "evaluate_detectors.py"),
                    remote_path="/app/evaluate_detectors.py")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=12 * 60 * 60,
    volumes={"/data": dataset_volume, "/runs": results_volume},
)
def evaluate_on_h100(dataset: str, full: str, freeze: str, output: str) -> str:
    dataset_root = Path("/data") / dataset
    if not (dataset_root / "data.yaml").is_file():
        raise FileNotFoundError(dataset_root / "data.yaml")
    full_path = Path("/runs") / full / "weights" / "best.pt"
    freeze_path = Path("/runs") / freeze / "weights" / "best.pt"
    for checkpoint in (full_path, freeze_path):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    output_path = Path("/runs") / output
    if output_path.exists():
        raise FileExistsError(output_path)
    subprocess.run([
        sys.executable, "/app/evaluate_detectors.py",
        "--data-root", str(dataset_root),
        "--baseline", "yolo26x.pt",
        "--full", str(full_path),
        "--freeze", str(freeze_path),
        "--output-dir", str(output_path),
        "--device", "0",
        "--split", "both",
    ], cwd="/app", check=True)
    results_volume.commit()
    return str(output_path)


@app.local_entrypoint()
def main(
    dataset: str = "football_4class_clip_split",
    full: str = "detection_player_full_v1",
    freeze: str = "detection_player_freeze10_v1",
    output: str = "evaluation_b_fixed640_v2",
) -> None:
    for value in (dataset, full, freeze, output):
        if value in ("", ".", "..") or "/" in value or "\\" in value:
            raise ValueError("All names must be one directory component")
    path = evaluate_on_h100.remote(dataset, full, freeze, output)
    print(f"Evaluation artifacts are saved in football-training-runs:{path}")
