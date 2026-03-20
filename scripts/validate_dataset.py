from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_simple_yolo_yaml(yaml_path: Path) -> dict[str, str]:
    """Parse a minimal subset of Ultralytics/YOLO dataset YAML.

    We intentionally avoid adding a YAML dependency; this handles the common keys:
    `path`, `train`, `val`, `test`.
    """
    out: dict[str, str] = {}
    for raw_line in yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in {"path", "train", "val", "test"}:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            out[key] = value
    return out


def _resolve_dataset_root(dataset_arg: Path) -> Path:
    """Allow passing either a dataset root folder or a dataset YAML file."""
    if dataset_arg.is_file() and dataset_arg.suffix.lower() in {".yaml", ".yml"}:
        cfg = _parse_simple_yolo_yaml(dataset_arg)
        base = dataset_arg.parent
        if "path" in cfg:
            base = (base / cfg["path"]).resolve()
        return base
    return dataset_arg


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _read_label_lines(label_path: Path) -> list[str]:
    try:
        txt = label_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = [ln.strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _is_valid_yolo_line(parts: list[str]) -> bool:
    if len(parts) != 5:
        return False
    try:
        int(parts[0])
        nums = [float(x) for x in parts[1:]]
    except Exception:
        return False
    if any(math.isnan(x) or math.isinf(x) for x in nums):
        return False
    x, y, w, h = nums
    # YOLO normalized format expects 0..1, but some tools allow slight epsilon.
    if not (-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6):
        return False
    if not (0 <= w <= 1 + 1e-6 and 0 <= h <= 1 + 1e-6):
        return False
    # A width/height of 0 is not useful.
    if w <= 0 or h <= 0:
        return False
    return True


def validate_split(split_dir: Path) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images = list(_iter_images(images_dir)) if images_dir.exists() else []
    labels = list(labels_dir.glob("**/*.txt")) if labels_dir.exists() else []

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    missing_labels = sorted(img_stems - lbl_stems)
    missing_images = sorted(lbl_stems - img_stems)

    class_counts: Counter[int] = Counter()
    invalid_lines = 0
    out_of_range_ids: Counter[int] = Counter()

    for lp in labels:
        for line in _read_label_lines(lp):
            parts = line.split()
            if not _is_valid_yolo_line(parts):
                invalid_lines += 1
                continue
            class_id = int(parts[0])
            class_counts[class_id] += 1
            if class_id < 0:
                out_of_range_ids[class_id] += 1

    return {
        "split": split_dir.name,
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "invalid_label_lines": invalid_lines,
        "class_counts": dict(class_counts),
        "negative_class_ids": dict(out_of_range_ids),
        "sample_missing_labels": missing_labels[:10],
        "sample_missing_images": missing_images[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a YOLO-format dataset folder")
    ap.add_argument(
        "dataset_root",
        type=Path,
        help="Root containing train/val/test splits OR a YOLO data.yaml file",
    )
    args = ap.parse_args()

    root: Path = _resolve_dataset_root(args.dataset_root)
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    # Many exported datasets use `valid/` instead of `val/`.
    split_candidates = [
        ("train", ["train"]),
        ("val", ["val", "valid", "validation"]),
        ("test", ["test"]),
    ]

    splits: list[Path] = []
    for _, names in split_candidates:
        for name in names:
            p = root / name
            if p.exists():
                splits.append(p)
                break
    if not splits:
        raise SystemExit("No train/val/test split folders found under dataset_root")

    report = {
        "dataset_root": str(root),
        "splits": [validate_split(p) for p in splits],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
