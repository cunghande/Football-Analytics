"""Fine-tune the football detector on a Modal H100.

Upload a YOLO dataset first, for example:
    modal volume create football-dataset
    modal volume put football-dataset datasets/football_4class_clip_split /football_4class_clip_split
    modal run modal_train.py --dataset football_4class_clip_split --model yolo26x.pt --epochs 3
"""

from pathlib import Path
import subprocess
import sys

import modal


app = modal.App("football-detector-training")
project_root = Path(__file__).resolve().parent
dataset_volume = modal.Volume.from_name("football-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("football-training-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(project_root / "requirements.txt"))
    .add_local_file(str(project_root / "train.py"), remote_path="/app/train.py")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=12 * 60 * 60,
    volumes={"/data": dataset_volume, "/runs": results_volume},
)
def train_on_h100(dataset: str, model: str, epochs: int, imgsz: int, batch: int, name: str, freeze: int) -> str:
    import yaml

    dataset_root = Path("/data") / dataset
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. Upload the complete YOLO export to "
            f"the football-dataset Volume at /{dataset}."
        )

    with data_yaml.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not data.get("names"):
        raise ValueError(f"Invalid YOLO data.yaml: {data_yaml} must define class names")
    raw_names = data["names"]
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        names = {int(class_id): name for class_id, name in raw_names.items()}
    else:
        raise ValueError("data.yaml names must be a list or class-ID mapping")
    expected_names = {"ball", "goalkeeper", "player", "referee"}
    if set(names) != set(range(4)) or set(names.values()) != expected_names:
        raise ValueError(f"Expected exactly four classes {sorted(expected_names)}; found {names}")
    if "nc" in data and data["nc"] != 4:
        raise ValueError(f"Expected nc: 4; found {data['nc']}")
    data["path"] = str(dataset_root)

    # Roboflow exports commonly use ../train/images relative to data.yaml.
    # Normalize each split against the actual directory mounted on Modal.
    for split in ("train", "val", "test"):
        if split not in data:
            if split in ("train", "val"):
                raise ValueError(f"Missing '{split}' split in {data_yaml}")
            continue
        relative = data[split]
        if not isinstance(relative, str):
            raise ValueError(f"'{split}' must be a single directory path in {data_yaml}")
        split_path = dataset_root / relative.removeprefix("../")
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"'{split}' images not found for {relative!r} under {dataset_root}. "
                "Check the dataset upload and data.yaml paths."
            )
        data[split] = str(split_path)

    normalized_yaml = Path("/tmp/football_dataset.yaml")
    with normalized_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)

    run_dir = Path("/runs") / name
    if run_dir.exists():
        raise FileExistsError(f"Training run already exists: {run_dir}. Choose a new --name.")

    command = [
        sys.executable,
        "/app/train.py",
        "--data", str(normalized_yaml),
        "--model", model,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--device", "0",
        "--project", "/runs",
        "--name", name,
    ]
    if freeze > 0:
        command.extend(("--freeze", str(freeze)))
    subprocess.run(command, check=True)
    results_volume.commit()
    return f"/runs/{name}"


@app.local_entrypoint()
def main(
    dataset: str = "football_4class_clip_split",
    model: str = "yolo26x.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    name: str = "football_detector",
    freeze: int = 0,
) -> None:
    if dataset in {"", ".", ".."} or "/" in dataset or "\\" in dataset:
        raise ValueError("--dataset must be one directory name in the dataset Volume")
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError("--name must be one directory name for this training run")
    if min(epochs, imgsz, batch) <= 0:
        raise ValueError("--epochs, --imgsz, and --batch must be positive")
    if freeze < 0:
        raise ValueError("--freeze must be nonnegative")
    try:
        data_yaml_entries = dataset_volume.listdir(f"/{dataset}/data.yaml")
    except modal.exception.NotFoundError:
        data_yaml_entries = []
    if not data_yaml_entries:
        raise FileNotFoundError(
            f"Missing /{dataset}/data.yaml in football-dataset. "
            "Upload a YOLO dataset before starting the H100 job."
        )
    run_path = train_on_h100.remote(dataset, model, epochs, imgsz, batch, name, freeze)
    print(f"Training artifacts are saved in football-training-runs:{run_path}")
