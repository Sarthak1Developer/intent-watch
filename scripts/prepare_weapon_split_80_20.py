from __future__ import annotations

import random
from pathlib import Path

import yaml


def _iter_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not images_dir.exists():
        return []
    return [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    dataset_root = repo / "weapon detection.v1i.yolov8"
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found at: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    images = []
    for split in ("train", "valid", "test"):
        images += _iter_images(dataset_root / split / "images")

    images = sorted(set(images))
    if not images:
        raise SystemExit("No images found under train/valid/test images folders")

    rng = random.Random(42)
    rng.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * 0.80)
    train_images = images[:n_train]
    val_images = images[n_train:]

    out_dir = dataset_root / "split_80_20"
    out_dir.mkdir(parents=True, exist_ok=True)

    def norm(p: Path) -> str:
        # Ultralytics handles forward slashes fine on Windows.
        return str(p.resolve()).replace("\\", "/")

    (out_dir / "train.txt").write_text("\n".join(norm(p) for p in train_images) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(norm(p) for p in val_images) + "\n", encoding="utf-8")

    out_yaml = {
        "train": str((out_dir / "train.txt").resolve()).replace("\\", "/"),
        "val": str((out_dir / "val.txt").resolve()).replace("\\", "/"),
        # Keep the original classes.
        "nc": int(data.get("nc", 0)),
        "names": data.get("names", []),
    }

    with (out_dir / "data_80_20.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)

    print("Prepared 80/20 split")
    print("Total images:", n_total)
    print("Train images:", len(train_images))
    print("Val images:", len(val_images))
    print("Wrote:", out_dir / "data_80_20.yaml")


if __name__ == "__main__":
    main()
