from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import threading
import time

import numpy as np

from api.torch_compat import apply_torch_load_weights_only_default_false


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _workspace_dir() -> Path:
    # backend/api/demo_inference.py -> backend/api -> backend -> workspace
    return Path(__file__).resolve().parents[2]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _default_device() -> int | str:
    raw = (os.getenv("INTENTWATCH_DEMO_DEVICE") or "").strip()
    if raw:
        if raw.isdigit():
            return int(raw)
        return raw
    return 0 if _cuda_available() else "cpu"


def _sorted_checkpoints(root: Path, *, filename: str = "best.pt", max_items: int = 20) -> list[Path]:
    if not root.exists():
        return []
    try:
        found = [p for p in root.rglob(filename) if p.is_file()]
    except Exception:
        return []
    found.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return found[: max(0, int(max_items))]


def pick_demo_model_path() -> Path:
    """Pick a YOLO checkpoint for demo inference.

    Priority:
    1) INTENTWATCH_DEMO_MODEL_PATH (explicit)
    2) INTENTWATCH_WEAPON_MODEL_PATH (existing runtime weapon model)
    3) newest best.pt under runs_weapon/ or runs/detect/
    4) fallback backend/yolov8n.pt
    """

    explicit = (os.getenv("INTENTWATCH_DEMO_MODEL_PATH") or "").strip()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p

    weapon_env = (os.getenv("INTENTWATCH_WEAPON_MODEL_PATH") or "").strip()
    if weapon_env:
        p = Path(weapon_env)
        if p.exists():
            return p

    ws = _workspace_dir()

    # Prefer repo's common training outputs.
    candidates: list[Path] = []
    candidates.extend(_sorted_checkpoints(ws / "runs_weapon", filename="best.pt", max_items=10))
    candidates.extend(_sorted_checkpoints(ws / "runs" / "detect", filename="best.pt", max_items=10))

    # If best.pt isn't found (e.g., only last.pt exists), fallback to last.pt.
    if not candidates:
        candidates.extend(_sorted_checkpoints(ws / "runs_weapon", filename="last.pt", max_items=5))
        candidates.extend(_sorted_checkpoints(ws / "runs" / "detect", filename="last.pt", max_items=5))

    for p in candidates:
        if p.exists():
            return p

    # Last resort: the bundled coco model.
    fallback = ws / "backend" / "yolov8n.pt"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "No YOLO checkpoint found for demo inference. Set INTENTWATCH_DEMO_MODEL_PATH to a .pt file."
    )


@dataclass(frozen=True)
class DemoModelInfo:
    model_path: str
    device: int | str


_model_lock = threading.Lock()
_demo_model: object | None = None
_demo_info: DemoModelInfo | None = None


def get_demo_model() -> tuple[object, DemoModelInfo]:
    """Load and cache the demo YOLO model once per process."""
    global _demo_model, _demo_info
    with _model_lock:
        if _demo_model is not None and _demo_info is not None:
            return _demo_model, _demo_info

        # Ensure torch.load can load Ultralytics checkpoints in environments
        # where weights_only=True is the default (e.g., some Colab runtimes).
        apply_torch_load_weights_only_default_false()

        # Lazy import so torch patch is applied before Ultralytics loads weights.
        from ultralytics import YOLO

        model_path = pick_demo_model_path()
        device = _default_device()

        model = YOLO(str(model_path))

        _demo_model = model
        _demo_info = DemoModelInfo(model_path=str(model_path), device=device)
        return model, _demo_info


def warmup_demo_model(*, imgsz: int = 640, conf: float = 0.25) -> dict:
    model, info = get_demo_model()

    torch_version = None
    cuda_available = None
    cuda_device_name = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            try:
                cuda_device_name = torch.cuda.get_device_name(0)
            except Exception:
                cuda_device_name = None
    except Exception:
        pass

    ultralytics_version = None
    try:
        import ultralytics

        ultralytics_version = getattr(ultralytics, "__version__", None)
    except Exception:
        pass

    # Small dummy image warmup (helps first-run latency on GPU).
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    _ = model.predict(dummy, imgsz=imgsz, conf=conf, device=info.device, verbose=False)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "ok": True,
        "model_path": info.model_path,
        "device": info.device,
        "warmup_ms": round(dt_ms, 2),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "ultralytics_version": ultralytics_version,
    }
