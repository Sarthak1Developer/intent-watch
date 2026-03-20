from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Train YOLOv8s binary verify model (person/weapon) for IntentWatch")
    ap.add_argument(
        "--data",
        default=str(Path("datasets") / "data_cctv_v3_person_weapon.yaml"),
        help="Dataset YAML path (default: datasets/data_cctv_v3_person_weapon.yaml)",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=2, help="Use small batch for 4GB VRAM")
    ap.add_argument("--device", default="0", help="GPU id, e.g. 0; use 'cpu' to force CPU")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--project", default=str(Path("runs_weapon")))
    ap.add_argument("--name", default="weapon_verify_v8s")
    ap.add_argument(
        "--model",
        default=str(Path("yolov8s.pt")),
        help="Base model checkpoint (default: yolov8s.pt in repo root)",
    )

    args = ap.parse_args()

    try:
        from ultralytics import YOLO
        import ultralytics

        print("ultralytics", getattr(ultralytics, "__version__", "?"))
    except Exception as e:
        raise SystemExit(f"Ultralytics not importable in this environment: {e}")

    repo_root = Path(__file__).resolve().parents[1]

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (repo_root / model_path).resolve()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (repo_root / data_path).resolve()

    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (repo_root / project_path).resolve()

    if not model_path.exists():
        raise SystemExit(f"Model checkpoint not found: {model_path}")
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    print("Training YOLOv8s verify model")
    print("- model  :", model_path)
    print("- data   :", data_path)
    print("- epochs :", args.epochs)
    print("- imgsz  :", args.imgsz)
    print("- batch  :", args.batch)
    print("- device :", args.device)
    print("- out    :", project_path / args.name)

    try:
        y = YOLO(str(model_path))
        y.train(
            data=str(data_path),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            device=str(args.device),
            workers=int(args.workers),
            patience=int(args.patience),
            close_mosaic=int(args.close_mosaic),
            project=str(project_path),
            name=str(args.name),
            exist_ok=True,
        )
    except Exception as e:
        print("\nTraining failed.")
        raise

    best = project_path / args.name / "weights" / "best.pt"
    if best.exists():
        print("\nDone.")
        print("Verify checkpoint:", best)
        return 0

    raise SystemExit(f"Training finished but best checkpoint not found at: {best}")


if __name__ == "__main__":
    raise SystemExit(main())
