import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on an exported Roboflow dataset (YOLO format)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/football-players-detection.v2i.yolo26/data.yaml",
        help="Path to data.yaml exported from Roboflow.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26x.pt",
        help="Base model checkpoint for transfer learning.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze the first N model layers; 0 keeps all layers trainable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help='Training device: e.g. "cpu", "mps", "0".',
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Output directory for training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="football_detector",
        help="Run name inside project directory.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    return parser.parse_args()


def validate_dataset_yaml(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found: {path}\n"
            "Export your Roboflow dataset in YOLO format first and pass --data <path/to/data.yaml>."
        )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    validate_dataset_yaml(data_path)

    print("Starting training with:")
    print(f"  data:    {data_path}")
    print(f"  model:   {args.model}")
    print(f"  device:  {args.device}")
    print(f"  epochs:  {args.epochs}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  batch:   {args.batch}")
    print(f"  freeze:  {args.freeze}")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
    )
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    results = model.train(**train_kwargs)

    run_dir = Path(args.project) / args.name
    best_weights = run_dir / "weights" / "best.pt"

    print("\nTraining finished.")
    print(f"Run directory: {run_dir.resolve()}")
    if best_weights.exists():
        print(f"Best model: {best_weights.resolve()}")
    else:
        print("Best weights not found at expected location.")

    # Keep variable used and visible for notebooks/scripts.
    _ = results


if __name__ == "__main__":
    # Avoid OpenMP thread over-subscription in some environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
