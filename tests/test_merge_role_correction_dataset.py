from pathlib import Path

import yaml

from scripts.merge_role_correction_dataset import main


NAMES = ["ball", "goalkeeper", "player", "referee"]


def make_dataset(root: Path, splits: tuple[str, ...]) -> None:
    root.mkdir()
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": NAMES}), encoding="utf-8"
    )
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / f"{split}.jpg").write_bytes(b"image")
        (root / split / "labels" / f"{split}.txt").write_text(
            "1 0.5 0.5 0.2 0.3\n", encoding="utf-8"
        )


def test_correction_is_added_only_to_train(tmp_path, monkeypatch):
    base = tmp_path / "base"
    correction = tmp_path / "correction"
    output = tmp_path / "merged"
    make_dataset(base, ("train", "valid", "test"))
    make_dataset(correction, ("train",))
    (base / "train" / "labels.cache").write_bytes(b"stale")
    monkeypatch.setattr(
        "sys.argv", ["merge", str(base), str(correction), str(output)]
    )

    main()

    assert sorted(path.name for path in (output / "train" / "images").iterdir()) == [
        "role_v1__train.jpg", "train.jpg",
    ]
    assert [path.name for path in (output / "valid" / "images").iterdir()] == ["valid.jpg"]
    assert [path.name for path in (output / "test" / "images").iterdir()] == ["test.jpg"]
    assert not list(output.rglob("*.cache"))
    merged_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert merged_yaml["names"] == NAMES
