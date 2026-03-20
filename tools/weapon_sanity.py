from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image",
        required=True,
        help="Path to an image to test (should contain a person holding a weapon).",
    )
    p.add_argument(
        "--base-model",
        default="d:/intent-watch/backend/yolov8n.pt",
        help="Path to the base COCO model used for people/bags.",
    )
    p.add_argument(
        "--weapon-model",
        default="d:/intent-watch/runs_weapon/weapon80_20/weights/best.pt",
        help="Path to the trained weapon model checkpoint.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    img = Path(args.image)
    if not img.exists():
        raise SystemExit(f"Image not found: {img}")

    base = YOLO(args.base_model)
    weapon = YOLO(args.weapon_model)

    r_base = base(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)
    r_weapon = weapon(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)

    print("base_model:", args.base_model)
    print("weapon_model:", args.weapon_model)
    print("image:", str(img))
    print("base names:", base.names)
    print("weapon names:", weapon.names)

    for r in r_base:
        boxes = r.boxes
        print("base detections:", len(boxes))
        for b in boxes:
            cls = int(b.cls[0])
            name = str(base.names.get(cls, cls))
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(x) for x in b.xyxy[0])
            if name in {"person", "backpack", "handbag", "suitcase", "bag"}:
                print(f"  {name:10s} conf={conf:.3f} xyxy={[x1, y1, x2, y2]}")

    for r in r_weapon:
        boxes = r.boxes
        print("weapon detections:", len(boxes))
        for b in boxes:
            cls = int(b.cls[0])
            name = str(weapon.names.get(cls, cls))
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(x) for x in b.xyxy[0])
            print(f"  {name:30s} conf={conf:.3f} xyxy={[x1, y1, x2, y2]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
