import argparse
import glob
import os
import statistics
from collections import defaultdict


def _win_long_path(path: str) -> str:
    """Return a path that can be opened on Windows even if it's >260 chars."""
    if os.name != "nt":
        return path
    # Already a long-path or UNC-prefixed path
    if path.startswith("\\\\?\\"):
        return path
    abs_path = os.path.abspath(path)
    # Prefix only when near legacy MAX_PATH to avoid confusing other tools.
    if len(abs_path) < 240:
        return path
    return "\\\\?\\" + abs_path


def iter_boxes(labels_dir: str):
    for path in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(_win_long_path(path), "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    yield (None, None, None, None, "invalid")
                    continue
                try:
                    cls = int(float(parts[0]))
                    _, _, w, h = map(float, parts[1:])
                except Exception:
                    yield (None, None, None, None, "invalid")
                    continue
                yield (cls, w, h, w * h, None)


def summarize(labels_dir: str):
    if not os.path.isdir(labels_dir):
        raise SystemExit(f"labels dir not found: {labels_dir}")

    invalid = 0
    areas = []
    wh_gt_095 = 0
    area_gt_090 = 0
    by_cls = defaultdict(list)

    for cls, w, h, area, err in iter_boxes(labels_dir):
        if err:
            invalid += 1
            continue
        areas.append(area)
        by_cls[cls].append(area)
        if w > 0.95 or h > 0.95:
            wh_gt_095 += 1
        if area > 0.90:
            area_gt_090 += 1

    if not areas:
        raise SystemExit(f"no valid boxes found in {labels_dir} (invalid_lines={invalid})")

    areas_sorted = sorted(areas)
    p90 = areas_sorted[int(0.9 * len(areas_sorted)) - 1]
    result = {
        "labels_dir": labels_dir,
        "label_files": len(glob.glob(os.path.join(labels_dir, "*.txt"))),
        "boxes": len(areas),
        "invalid_lines": invalid,
        "area_mean": statistics.mean(areas),
        "area_median": statistics.median(areas),
        "area_p90": p90,
        "boxes_w_or_h_gt_0.95": wh_gt_095,
        "boxes_area_gt_0.90": area_gt_090,
        "per_class": {},
    }

    for cls in sorted(by_cls):
        arr = by_cls[cls]
        arr_sorted = sorted(arr)
        p90_cls = arr_sorted[int(0.9 * len(arr_sorted)) - 1]
        result["per_class"][str(cls)] = {
            "n": len(arr),
            "area_mean": statistics.mean(arr),
            "area_median": statistics.median(arr),
            "area_p90": p90_cls,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize YOLO label box-size quality stats")
    parser.add_argument("labels_dir", help="Path to YOLO labels directory (contains .txt files)")
    args = parser.parse_args()

    r = summarize(args.labels_dir)
    # Pretty print without adding dependencies
    import json

    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
