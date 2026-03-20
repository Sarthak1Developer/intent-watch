from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import os
import re
import shutil
import time
import uuid
import cv2
from api.stream_manager import NormalizedZone, StreamManager

router = APIRouter()

# ---------------- PATHS / CONFIG ----------------
BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
WORKSPACE_DIR = BACKEND_DIR.parent

VIDEO_DIR = BACKEND_DIR / "data" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CANDIDATES = [BACKEND_DIR / "yolov8n.pt"]


def _sorted_checkpoints(root: Path, *, max_items: int = 20) -> list[Path]:
    """Return best.pt checkpoints under root sorted newest-first."""
    if not root.exists():
        return []
    found: list[Path] = []
    try:
        found = [p for p in root.rglob("best.pt") if p.is_file()]
    except Exception:
        return []
    found.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return found[: max(0, int(max_items))]


def _weapon_model_candidates() -> list[Path]:
    # Optional second model dedicated to weapons.
    # If provided, any detection from this model will be treated as a Weapon alert.
    env_path = os.getenv("INTENTWATCH_WEAPON_MODEL_PATH")

    def _env_bool(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    allow_legacy = _env_bool("INTENTWATCH_WEAPON_ALLOW_LEGACY_MODEL", False)

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    # Default to known training output locations used by this repo.
    # Ultralytics CLI often writes under: runs/detect/<project>/<name>/weights/best.pt
    candidates.extend(
        [
            # Existing trained models used in this workspace
            WORKSPACE_DIR
            / "runs"
            / "detect"
            / "runs_weapon"
            / "combined_yolov8s_gpu"
            / "weights"
            / "best.pt",
            WORKSPACE_DIR
            / "runs"
            / "detect"
            / "runs_weapon"
            / "combined_ft_from_archive1_best"
            / "weights"
            / "best.pt",
            # New multi-class weapon model (preferred)
            WORKSPACE_DIR
            / "runs"
            / "detect"
            / "runs_weapon"
            / "weapon_types_v1"
            / "weights"
            / "best.pt",
            # Legacy location (if user trained with project=runs_weapon but without the extra runs/detect prefix)
            WORKSPACE_DIR / "runs_weapon" / "weapon_types_v1" / "weights" / "best.pt",
        ]
    )

    # Legacy weapon80_20 checkpoint has historically produced many false positives
    # (and often contains non-weapon/placeholder labels). Only include it when
    # explicitly allowed.
    if allow_legacy:
        candidates.extend(
            [
                WORKSPACE_DIR
                / "runs"
                / "detect"
                / "runs_weapon"
                / "weapon80_20"
                / "weights"
                / "best.pt",
                WORKSPACE_DIR / "runs_weapon" / "weapon80_20" / "weights" / "best.pt",
            ]
        )

    # Also consider any other trained checkpoints under runs/detect.
    # This helps when model run names change.
    auto_found = _sorted_checkpoints(WORKSPACE_DIR / "runs" / "detect", max_items=10)
    if not allow_legacy:
        auto_found = [p for p in auto_found if "weapon80_20" not in str(p).lower()]
    candidates.extend(auto_found)

    # De-dupe while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _weapon_fallback_model_candidates() -> list[Path]:
    """Candidates for a fallback 'person-with-weapon' model.

    This model is NOT expected to draw a tight weapon box; it usually predicts a person-sized box.
    We use it only when the weapon-type model fails to produce a weapon box for the frame.
    """
    env_path = os.getenv("INTENTWATCH_WEAPON_FALLBACK_MODEL_PATH")

    def _env_bool(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Fallback model is noisy; keep it OFF by default.
    enable_fallback = _env_bool("INTENTWATCH_WEAPON_ENABLE_FALLBACK", False)
    if not env_path and not enable_fallback:
        return []

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            # Old training output (most common in this repo)
            WORKSPACE_DIR / "runs_weapon" / "weapon80_20" / "weights" / "best.pt",
            # If trained via Ultralytics default directory layout
            WORKSPACE_DIR
            / "runs"
            / "detect"
            / "runs_weapon"
            / "weapon80_20"
            / "weights"
            / "best.pt",
        ]
    )

    # De-dupe while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _weapon_verify_model_candidates() -> list[Path]:
    """Candidates for a secondary 'verify' weapon model.

    This model is executed only at Weapon event start, right before alert emission,
    to suppress false positives from the primary (fast) model.
    """
    env_path = os.getenv("INTENTWATCH_WEAPON_VERIFY_MODEL_PATH")

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    # Default locations used by this repo for a verify model run.
    candidates.extend(
        [
            WORKSPACE_DIR
            / "runs"
            / "detect"
            / "runs_weapon"
            / "weapon_verify_v8s"
            / "weights"
            / "best.pt",
            WORKSPACE_DIR / "runs_weapon" / "weapon_verify_v8s" / "weights" / "best.pt",
        ]
    )

    auto_found = _sorted_checkpoints(WORKSPACE_DIR / "runs" / "detect", max_items=10)
    # Heuristic: prefer run names that contain "verify".
    auto_found = [p for p in auto_found if "verify" in str(p).lower()]
    candidates.extend(auto_found)

    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq

# ---------------- STREAM MANAGER ----------------
PRIMARY_STREAM_ID = "primary"
manager = StreamManager(
    MODEL_CANDIDATES,
    _weapon_model_candidates(),
    _weapon_verify_model_candidates(),
    _weapon_fallback_model_candidates(),
)


_IP_WEBCAM_HOSTPORT_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}$")


def _normalize_ip_webcam_source(source: str) -> str:
    """Normalize common Android IP Webcam inputs.

    Supported user inputs:
    - '10.12.26.111:8080' -> 'http://10.12.26.111:8080/video'
    - 'http://10.12.26.111:8080' -> 'http://10.12.26.111:8080/video'
    - 'http://10.12.26.111:8080/video' -> unchanged
    """
    s = str(source or "").strip()
    if not s:
        return s

    # If user provided just IP:PORT (no scheme)
    if _IP_WEBCAM_HOSTPORT_RE.match(s):
        return f"http://{s}/video"

    # If user provided http(s)://host:port with no path
    s_lower = s.lower()
    if s_lower.startswith("http://") or s_lower.startswith("https://"):
        # If it already includes a path beyond the host:port, keep it.
        # Otherwise append /video (IP Webcam MJPEG endpoint).
        after_scheme = s.split("://", 1)[-1]
        if "/" not in after_scheme:
            return s.rstrip("/") + "/video"
        # If it ends with just a trailing slash, assume root.
        if after_scheme.endswith("/"):
            return s.rstrip("/") + "/video"

    return s


class StartCameraRequest(BaseModel):
    # Can be a numeric device id (0,1,2,...) or an IP Webcam URL/host:port.
    device_id: int | str = 0


class StartVideoRequest(BaseModel):
    source: str


class StartStreamRequest(BaseModel):
    stream_id: str
    source: str


class StopStreamRequest(BaseModel):
    stream_id: str


class ZonesRequest(BaseModel):
    zones: list[dict]


def _safe_filename(name: str) -> str:
    # Keep it simple: remove path separators and trim.
    name = (name or "video").strip().replace("\\", "_").replace("/", "_")
    return name if name else "video"


def _unique_upload_path(original_filename: str) -> Path:
    base = _safe_filename(original_filename)
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    # Always create a unique filename to prevent overwriting a file that is currently being streamed.
    suffix = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    return VIDEO_DIR / f"{stem}_{suffix}{ext}"


def _open_capture_for_validation(source: int):
    if os.name == "nt":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)

# ---------------- API ENDPOINTS ----------------

@router.post("/upload")
def upload_video(file: UploadFile = File(...)):
    """Upload and validate video file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate file extension
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Allowed: {', '.join(allowed_extensions)}",
            )
        
        file_path = _unique_upload_path(file.filename)
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        
        # Save file (write to temp then rename for best-effort atomicity)
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        os.replace(tmp_path, file_path)
        
        # Validate that file was written
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="File upload failed or file is empty")

        # NOTE: Do not attempt to open the video with OpenCV here.
        # Some codecs/containers can cause VideoCapture to hang on Windows.
        # Streaming will surface an error if the codec is unsupported.
        
        abs_path = str(file_path.resolve())

        # Selecting/uploading a file becomes the primary stream source.
        manager.start(PRIMARY_STREAM_ID, abs_path, mode="file")
        
        print(f"✓ Video uploaded successfully: {abs_path}")
        return {"message": "Video uploaded successfully", "path": abs_path, "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Upload error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {error_msg}")

@router.post("/start-camera")
def start_camera(body: StartCameraRequest | None = None):
    """Start live camera feed"""
    try:
        # If user has configured an IP Webcam URL, prefer it.
        webcam_url = str(os.getenv("INTENTWATCH_WEBCAM_URL") or "").strip()
        if webcam_url:
            src = _normalize_ip_webcam_source(webcam_url)
            manager.start(PRIMARY_STREAM_ID, src, mode="camera")
            print("✓ IP webcam selected successfully")
            return {"message": "IP webcam selected", "source": src}

        raw = 0 if body is None else body.device_id
        # Numeric => local webcam device id
        if isinstance(raw, int) or (isinstance(raw, str) and raw.strip().isdigit()):
            device_id = int(raw)
            # Test camera access first
            test_cap = _open_capture_for_validation(device_id)
            if not test_cap.isOpened():
                test_cap.release()
                raise HTTPException(status_code=500, detail="Camera not available or already in use")
            test_cap.release()

            manager.start(PRIMARY_STREAM_ID, device_id, mode="camera")
            print("✓ Camera selected successfully")
            return {"message": "Live camera selected", "device_id": device_id}

        # Otherwise treat as IP Webcam url/host:port
        src = _normalize_ip_webcam_source(str(raw))
        manager.start(PRIMARY_STREAM_ID, src, mode="camera")
        print("✓ IP webcam selected successfully")
        return {"message": "IP webcam selected", "source": src}
    except Exception as e:
        print(f"✗ Camera error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Camera error: {str(e)}")


@router.post("/start")
def start_video(body: StartVideoRequest):
    """Start stream from a source.

    This endpoint exists to match the frontend API client.
    Accepted sources:
    - 'webcam'/'camera' -> device 0
    - numeric string -> that camera device id
    - any other string -> treated as a path/URL (validated by trying to open)
    """
    if body is None or not body.source:
        raise HTTPException(status_code=400, detail="Missing source")

    source_raw = str(body.source).strip()
    try:
        if source_raw.lower() in {"webcam", "camera"}:
            source: int | str = 0
        else:
            # Allow Android IP Webcam shorthand like '10.12.26.111:8080'
            source_raw = _normalize_ip_webcam_source(source_raw)
            # If numeric, treat as camera device id.
            try:
                source = int(source_raw)
            except Exception:
                source = source_raw

        # Validation strategy:
        # - Camera devices: validate by attempting to open (fast failure if device busy).
        # - File paths / URLs: do NOT open with OpenCV here (can hang on some codecs on Windows).
        if isinstance(source, int):
            test_cap = _open_capture_for_validation(source)
            if not test_cap.isOpened():
                try:
                    test_cap.release()
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail="Camera not available")
            test_cap.release()
        else:
            # If it's a local file path, ensure it exists. URLs/RTSP/IPcam may not exist as files.
            if os.path.exists(str(source)) is False and ("://" not in str(source)):
                raise HTTPException(status_code=404, detail="Video file not found")

        manager.start(
            PRIMARY_STREAM_ID,
            source,
            mode=(
                "camera"
                if (isinstance(source, int) or (isinstance(source, str) and "://" in str(source)))
                else "file"
            ),
        )
        return {"message": "Video source selected", "source": source_raw}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Start error: {str(e)}")

@router.post("/stop")
def stop_video():
    """Stop video stream"""
    try:
        manager.stop(PRIMARY_STREAM_ID)
        print("✓ Video stream stopped")
        return {"message": "Video stream stopped"}
    except Exception as e:
        print(f"✗ Stop error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
def video_status():
    # Backward-compatible shape: include mode/path, plus a running bit.
    return manager.get_status(PRIMARY_STREAM_ID)


@router.get("/streams")
def list_streams():
    """List active streams (including primary if running)."""
    ids = manager.list_streams()
    return {"streams": [{"id": sid, **manager.get_status(sid)} for sid in ids]}


@router.post("/streams/start")
def start_stream(body: StartStreamRequest):
    if body is None or not body.stream_id or not body.source:
        raise HTTPException(status_code=400, detail="Missing stream_id or source")

    stream_id = str(body.stream_id).strip()
    if not stream_id:
        raise HTTPException(status_code=400, detail="Invalid stream_id")

    source_raw = str(body.source).strip()
    try:
        # Allow Android IP Webcam shorthand like '10.12.26.111:8080'
        source_raw = _normalize_ip_webcam_source(source_raw)
        # camera device ids can be numeric
        try:
            source: int | str = int(source_raw)
        except Exception:
            source = source_raw

        if isinstance(source, int):
            test_cap = _open_capture_for_validation(source)
            if not test_cap.isOpened():
                try:
                    test_cap.release()
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail="Camera not available")
            test_cap.release()
        else:
            if os.path.exists(str(source)) is False and ("://" not in str(source)):
                raise HTTPException(status_code=404, detail="Video file not found")

        manager.start(
            stream_id,
            source,
            mode=(
                "camera"
                if (isinstance(source, int) or (isinstance(source, str) and "://" in str(source)))
                else "file"
            ),
        )
        return {"message": "Stream started", "stream_id": stream_id, "source": source_raw}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Start error: {str(e)}")


@router.post("/streams/stop")
def stop_stream(body: StopStreamRequest):
    if body is None or not body.stream_id:
        raise HTTPException(status_code=400, detail="Missing stream_id")
    stream_id = str(body.stream_id).strip()
    manager.stop(stream_id)
    return {"message": "Stream stopped", "stream_id": stream_id}


@router.get("/status/{stream_id}")
def stream_status(stream_id: str):
    return manager.get_status(str(stream_id))


@router.post("/zones/{stream_id}")
def set_zones_for_stream(stream_id: str, body: ZonesRequest):
    raw = body.zones if body and body.zones else []

    def to_float(v, default: float | None = None) -> float | None:
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    zones: list[NormalizedZone] = []
    for z in raw:
        try:
            x = to_float(z.get("x"))
            y = to_float(z.get("y"))
            w = to_float(z.get("width"))
            h = to_float(z.get("height"))
            if x is None or y is None or w is None or h is None:
                continue
            zones.append(
                NormalizedZone(
                    id=str(z.get("id")),
                    name=str(z.get("name")),
                    severity=str(z.get("severity")) if z.get("severity") is not None else "medium",
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                )
            )
        except Exception:
            continue

    manager.set_zones(str(stream_id), zones)
    return {"message": "Zones updated", "count": len(zones), "stream_id": str(stream_id)}


@router.post("/zones")
def set_zones(body: ZonesRequest):
    """Set normalized zones for the primary stream.

    Zones are expected to be normalized floats in [0,1] relative to the frame.
    """
    def to_float(v, default: float | None = None) -> float | None:
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    raw = body.zones if body and body.zones else []
    zones: list[NormalizedZone] = []
    for z in raw:
        try:
            x = to_float(z.get("x"))
            y = to_float(z.get("y"))
            w = to_float(z.get("width"))
            h = to_float(z.get("height"))
            if x is None or y is None or w is None or h is None:
                continue
            zones.append(
                NormalizedZone(
                    id=str(z.get("id")),
                    name=str(z.get("name")),
                    severity=str(z.get("severity")) if z.get("severity") is not None else "medium",
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                )
            )
        except Exception:
            continue

    manager.set_zones(PRIMARY_STREAM_ID, zones)
    return {"message": "Zones updated", "count": len(zones)}

# ---------------- STREAM ----------------


def _mjpeg_generator(stream_id: str):
    """Yield MJPEG frames from a running StreamWorker."""
    last_ts: float | None = None
    stall_started = time.time()
    try:
        while True:
            worker = manager.get_worker(stream_id)
            if worker is None or not worker.is_running():
                time.sleep(0.05)
                # If the worker is gone, stop streaming.
                if manager.get_status(stream_id)["mode"] is None:
                    break
                # If frames have stalled for too long, end the response so clients reconnect.
                if (time.time() - stall_started) > 5.0:
                    break
                continue

            jpg, ts = worker.get_latest_jpeg()
            if jpg is None or ts is None or ts == last_ts:
                time.sleep(0.02)
                if (time.time() - stall_started) > 5.0:
                    break
                continue

            last_ts = ts
            stall_started = time.time()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
    except GeneratorExit:
        # Client disconnected.
        return
@router.get("/stream")
def stream_video():
    status = manager.get_status(PRIMARY_STREAM_ID)
    if status["mode"] is None:
        raise HTTPException(status_code=400, detail="No video source selected")

    # For local files, validate existence before streaming.
    source = status["path"]
    if isinstance(source, str) and ("://" not in source) and not os.path.exists(source):
        raise HTTPException(status_code=404, detail=f"Video file not found: {source}")

    return StreamingResponse(
        _mjpeg_generator(PRIMARY_STREAM_ID),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/stream/{stream_id}")
def stream_video_by_id(stream_id: str):
    status = manager.get_status(str(stream_id))
    if status["mode"] is None:
        raise HTTPException(status_code=400, detail="No video source selected")

    source = status["path"]
    if isinstance(source, str) and ("://" not in source) and not os.path.exists(source):
        raise HTTPException(status_code=404, detail=f"Video file not found: {source}")

    return StreamingResponse(
        _mjpeg_generator(str(stream_id)),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )
