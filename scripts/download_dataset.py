"""Download the selected Roboflow football dataset without storing the API key.

Usage: python scripts/download_dataset.py --version VERSION
Requires: python -m pip install roboflow
"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True, help="Roboflow dataset version number")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/football_4class"),
        help="New local directory for the YOLO export",
    )
    args = parser.parse_args()
    if args.version <= 0:
        parser.error("--version must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    from roboflow import Roboflow

    api_key = getpass.getpass("Roboflow API key: ")
    dataset = (
        Roboflow(api_key=api_key)
        .workspace("fsdm-olss1")
        .project("detection-player")
        .version(args.version)
        .download("yolov8", location=str(output))
    )
    source = {
        "source": "https://universe.roboflow.com/fsdm-olss1/detection-player",
        "version": args.version,
        "export_format": "yolov8",
        "license": "VERIFY_ON_VERSION_PAGE",
    }
    (output / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded: {Path(dataset.location).resolve()}")
    print("Verify the license and run: python scripts/audit_dataset.py", output)


if __name__ == "__main__":
    main()
