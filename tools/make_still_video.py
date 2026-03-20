from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument("--seconds", type=float, default=3.0, help="Video duration")
    p.add_argument("--fps", type=float, default=5.0, help="Frames per second")
    args = p.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    h, w = frame.shape[:2]
    fps = float(args.fps)
    total = max(1, int(round(float(args.seconds) * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for: {out_path}")

    try:
        for _ in range(total):
            writer.write(frame)
    finally:
        writer.release()

    print(f"Wrote {total} frames to {out_path} ({w}x{h} @ {fps}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
