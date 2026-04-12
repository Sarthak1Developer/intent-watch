from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.demo_inference import get_demo_model, warmup_demo_model
from api.routes.alerts import add_alert

router = APIRouter()

BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
SNAP_DIR = BACKEND_DIR / "data" / "snaps"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = BACKEND_DIR / "data" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stream_id(stream_id: str) -> str:
    s = str(stream_id or "").strip()
    if not s or "/" in s or "\\" in s or ".." in s:
        raise HTTPException(status_code=400, detail="Invalid stream_id")
    return s


@router.post("/warmup")
def warmup():
    """Warm up model load + first inference.

    Recommended to call once after starting Colab backend.
    """

    try:
        return warmup_demo_model(imgsz=int(os.getenv("INTENTWATCH_DEMO_IMG_SIZE") or 640))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")


@router.post("/detect-image")
def detect_image(
    file: UploadFile = File(...),
    stream_id: str = "demo",
    emit_alert: bool = True,
):
    """Run one-shot inference on an uploaded image.

    This is the recommended demo path (no real-time streaming loop).
    """

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Basic allowlist (content-type can be unreliable; keep it permissive).
    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    sid = _safe_stream_id(stream_id)

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    model, info = get_demo_model()

    conf = float(os.getenv("INTENTWATCH_DEMO_CONF") or 0.25)
    imgsz = int(os.getenv("INTENTWATCH_DEMO_IMG_SIZE") or 800)

    results = model.predict(img, imgsz=imgsz, conf=conf, device=info.device, verbose=False)
    if not results:
        detections: list[dict] = []
        annotated = img
    else:
        r0 = results[0]
        names = getattr(r0, "names", None) or getattr(model, "names", {})

        detections = []
        try:
            boxes = r0.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

                for i in range(len(xyxy)):
                    cls_id = int(clss[i])
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    detections.append(
                        {
                            "label": str(label),
                            "confidence": float(confs[i]),
                            "bbox": [float(x) for x in xyxy[i].tolist()],
                        }
                    )
        except Exception:
            detections = []

        try:
            annotated = r0.plot()
        except Exception:
            annotated = img

    # Persist annotated image under the existing snapshot path so the frontend can display it.
    now = datetime.now()
    date_dir = now.date().isoformat()
    out_dir = SNAP_DIR / sid / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = f"demo_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = out_dir / fn

    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        out_path.write_bytes(buf.tobytes())

    snapshot_url = f"/alerts/snapshot/{sid}/{date_dir}/{fn}" if ok else None

    # Emit an alert so the Dashboard updates without needing the streaming pipeline.
    if emit_alert:
        if detections:
            top = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)[:3]
            desc = ", ".join(f"{d['label']} ({d['confidence']:.2f})" for d in top)
            msg = f"Demo image detection: {desc}"
            severity = "Critical"
        else:
            msg = "Demo image detection: no objects detected"
            severity = "Low"

        add_alert("Weapon", msg, severity=severity, camera="demo-image", snapshot_url=snapshot_url)

    return {
        "ok": True,
        "model_path": info.model_path,
        "device": info.device,
        "detections": detections,
        "snapshot_url": snapshot_url,
    }


@router.post("/detect-video")
def detect_video(
    file: UploadFile = File(...),
    stream_id: str = "demo",
    emit_alert: bool = True,
):
    """Run one-shot inference over a short uploaded video.

    This processes only a limited number of frames (sampling) to keep demo runs fast and reliable.
    """

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    sid = _safe_stream_id(stream_id)

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    tmp_name = f"demo_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}{ext or '.mp4'}"
    tmp_path = (VIDEO_DIR / tmp_name).resolve()
    tmp_path.write_bytes(raw)

    model, info = get_demo_model()
    conf = float(os.getenv("INTENTWATCH_DEMO_CONF") or 0.25)
    imgsz = int(os.getenv("INTENTWATCH_DEMO_IMG_SIZE") or 800)

    max_frames = int(os.getenv("INTENTWATCH_DEMO_VIDEO_MAX_FRAMES") or 90)
    every_n = int(os.getenv("INTENTWATCH_DEMO_VIDEO_EVERY_N") or 5)
    if max_frames < 1:
        max_frames = 1
    if every_n < 1:
        every_n = 1

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video")

    best_dets: list[dict] = []
    best_conf = -1.0
    best_annotated: np.ndarray | None = None
    frames_seen = 0
    frames_used = 0

    try:
        while frames_used < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frames_seen += 1
            if (frames_seen - 1) % every_n != 0:
                continue

            frames_used += 1
            results = model.predict(frame, imgsz=imgsz, conf=conf, device=info.device, verbose=False)
            if not results:
                continue

            r0 = results[0]
            names = getattr(r0, "names", None) or getattr(model, "names", {})

            dets: list[dict] = []
            max_det_conf = -1.0
            try:
                boxes = r0.boxes
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
                    for i in range(len(xyxy)):
                        c = float(confs[i])
                        cls_id = int(clss[i])
                        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                        dets.append(
                            {
                                "label": str(label),
                                "confidence": c,
                                "bbox": [float(x) for x in xyxy[i].tolist()],
                            }
                        )
                        if c > max_det_conf:
                            max_det_conf = c
            except Exception:
                dets = []

            if max_det_conf > best_conf:
                best_conf = max_det_conf
                best_dets = dets
                try:
                    best_annotated = r0.plot()
                except Exception:
                    best_annotated = frame
    finally:
        try:
            cap.release()
        except Exception:
            pass

    now = datetime.now()
    date_dir = now.date().isoformat()
    out_dir = SNAP_DIR / sid / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_url = None
    if best_annotated is not None:
        fn = f"demo_video_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        out_path = out_dir / fn
        ok, buf = cv2.imencode(".jpg", best_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            out_path.write_bytes(buf.tobytes())
            snapshot_url = f"/alerts/snapshot/{sid}/{date_dir}/{fn}"

    if emit_alert:
        if best_dets:
            top = sorted(best_dets, key=lambda d: d.get("confidence", 0.0), reverse=True)[:3]
            desc = ", ".join(f"{d['label']} ({d['confidence']:.2f})" for d in top)
            msg = f"Demo video detection: {desc}"
            severity = "Critical"
        else:
            msg = "Demo video detection: no objects detected"
            severity = "Low"

        add_alert("Weapon", msg, severity=severity, camera="demo-video", snapshot_url=snapshot_url)

    return {
        "ok": True,
        "model_path": info.model_path,
        "device": info.device,
        "frames_seen": frames_seen,
        "frames_used": frames_used,
        "detections": best_dets,
        "snapshot_url": snapshot_url,
    }
