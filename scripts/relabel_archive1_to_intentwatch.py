from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# IntentWatch weapon model schema (must match archive/weapon-detection/data.yaml)
INTENTWATCH_NAMES = [
    "pistol",
    "smartphone",
    "knife",
    "monedero",
    "billete",
    "tarjeta",
]

# Map archive1 filename prefix -> target class id in IntentWatch schema.
# We treat all firearms as 'pistol' (class 0) and blade weapons as 'knife' (class 2).
FIREARM_PREFIXES = {
    "Automatic Rifle",
    "Bazooka",
    "Grenade Launcher",
    "Handgun",
    "Shotgun",
    "SMG",
    "Sniper",
}
BLADE_PREFIXES = {"Knife", "Sword"}


def _prefix_from_stem(stem: str) -> str:
    # Files are like 'Automatic Rifle_10' or 'Grenade Launcher_100'
    return stem.split("_", 1)[0]


def _target_class_id(prefix: str) -> int | None:
    if prefix in FIREARM_PREFIXES:
        return 0  # pistol
    if prefix in BLADE_PREFIXES:
        return 2  # knife
    return None


def _rewrite_label_file(src_label: Path, dst_label: Path, target_id: int) -> None:
    try:
        text = src_label.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""

    out_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            # skip malformed
            continue
        # Keep xywh as-is, only change class id
        out_lines.append(" ".join([str(target_id)] + parts[1:5]))

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def relabel_split(split_dir: Path) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return {"split": split_dir.name, "status": "skipped (missing images/labels)"}

    raw_dir = split_dir / "labels_raw"
    if raw_dir.exists():
        raise SystemExit(f"Backup already exists: {raw_dir} (refusing to overwrite)")

    # Backup existing labels
    shutil.copytree(labels_dir, raw_dir)

    # Rewrite into labels_dir
    changed = 0
    skipped = 0
    for src_label in raw_dir.rglob("*.txt"):
        stem = src_label.stem
        prefix = _prefix_from_stem(stem)
        tid = _target_class_id(prefix)
        if tid is None:
            skipped += 1
            continue
        dst_label = labels_dir / src_label.relative_to(raw_dir)
        _rewrite_label_file(src_label, dst_label, tid)
        changed += 1

    return {
        "split": split_dir.name,
        "backed_up_to": str(raw_dir),
        "labels_rewritten": changed,
        "labels_skipped": skipped,
    }


def write_data_yaml(dataset_root: Path, out_path: Path) -> None:
    # 'dataset_root' is currently unused but kept for API compatibility.
    _ = dataset_root
    # Ultralytics expects paths relative to this yaml, so keep it simple.
    # This yaml will live under archive1/weapon_detection/
    out_path.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: val/images",
                "",
                "names:",
                *[f"  - {n}" for n in INTENTWATCH_NAMES],
                "",
                f"nc: {len(INTENTWATCH_NAMES)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Relabel archive1 weapon_detection to IntentWatch classes")
    ap.add_argument("dataset_root", type=Path, help="Path like archive1/weapon_detection")
    args = ap.parse_args()

    root: Path = args.dataset_root
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    results: list[dict] = []
    for split in (root / "train", root / "val"):
        if split.exists():
            results.append(relabel_split(split))

    data_yaml = root / "data_intentwatch.yaml"
    write_data_yaml(root, data_yaml)

    print("Relabel complete")
    for r in results:
        print(r)
    print(f"Wrote: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
