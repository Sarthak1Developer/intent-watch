from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_simple_yolo_yaml(yaml_path: Path) -> dict[str, str]:
    """Parse a minimal subset of Ultralytics/YOLO dataset YAML.

    We intentionally avoid adding a YAML dependency; this handles common keys:
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


def _to_long_path(p: Path) -> str:
    """Convert to a Windows long-path string when needed."""

    s = str(p.resolve())
    if os.name != "nt":
        return s
    if s.startswith("\\\\?\\"):
        return s
    # Keep a little buffer under MAX_PATH.
    if len(s) >= 240:
        return "\\\\?\\" + s
    return s


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _read_label_lines(label_path: Path) -> list[str]:
    try:
        with open(_to_long_path(label_path), "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
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
    if not (-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6):
        return False
    if not (0 <= w <= 1 + 1e-6 and 0 <= h <= 1 + 1e-6):
        return False
    if w <= 0 or h <= 0:
        return False
    return True


def _write_text(path: Path, text: str) -> None:
    lp = _to_long_path(path)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    with open(lp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _unlink(path: Path) -> None:
    os.unlink(_to_long_path(path))


def clean_split(
    split_dir: Path,
    *,
    dry_run: bool,
    delete_orphan_labels: bool,
    fix_invalid_lines: bool,
) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images = list(_iter_images(images_dir)) if images_dir.exists() else []
    labels = list(labels_dir.glob("**/*.txt")) if labels_dir.exists() else []

    img_stems = {p.stem for p in images}
    orphan_labels = [p for p in labels if p.stem not in img_stems]

    deleted_orphans = 0
    if delete_orphan_labels:
        for lp in orphan_labels:
            if dry_run:
                deleted_orphans += 1
                continue
            try:
                _unlink(lp)
                deleted_orphans += 1
            except Exception:
                # Best-effort: keep going.
                pass

    total_invalid_lines = 0
    total_lines_removed = 0
    files_touched = 0

    if fix_invalid_lines:
        for lp in labels:
            if delete_orphan_labels and lp.stem not in img_stems:
                # Orphan was already removed (or scheduled).
                continue
            raw_lines = _read_label_lines(lp)
            kept: list[str] = []
            removed = 0
            invalid = 0
            for line in raw_lines:
                parts = line.split()
                if not _is_valid_yolo_line(parts):
                    removed += 1
                    invalid += 1
                    continue
                kept.append(" ".join(parts))

            if invalid:
                total_invalid_lines += invalid
            if removed:
                total_lines_removed += removed
                files_touched += 1
                if not dry_run:
                    _write_text(lp, "\n".join(kept) + ("\n" if kept else ""))

    return {
        "split": split_dir.name,
        "images": len(images),
        "labels": len(labels),
        "orphan_labels": len(orphan_labels),
        "deleted_orphan_labels": deleted_orphans,
        "invalid_label_lines_removed": total_lines_removed,
        "label_files_touched": files_touched,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean a YOLO dataset (remove orphan labels, strip invalid lines)")
    ap.add_argument(
        "dataset_root",
        type=Path,
        help="Root containing train/val/test splits OR a YOLO data.yaml file",
    )
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")
    ap.add_argument(
        "--keep-orphan-labels",
        action="store_true",
        help="Do not delete label files that have no matching image",
    )
    ap.add_argument(
        "--keep-invalid-lines",
        action="store_true",
        help="Do not remove invalid label lines",
    )
    args = ap.parse_args()

    root = _resolve_dataset_root(args.dataset_root)
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

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
        "dry_run": bool(args.dry_run),
        "splits": [
            clean_split(
                split,
                dry_run=args.dry_run,
                delete_orphan_labels=not args.keep_orphan_labels,
                fix_invalid_lines=not args.keep_invalid_lines,
            )
            for split in splits
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
