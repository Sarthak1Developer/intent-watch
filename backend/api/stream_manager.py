from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict
import os
import threading
import time
import json
import uuid

import cv2
from ultralytics import YOLO


# Disable FFmpeg threading to avoid codec errors
os.environ.setdefault("FFREPORT", "file=/dev/null")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

# Limit OpenCV internal threading to reduce race conditions
cv2.setNumThreads(1)


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _pick_history_fourcc_candidates() -> list[str]:
    """Return preferred FourCC codes for browser-friendly MP4 recording.

    Many browsers cannot play `mp4v` (MPEG-4 Part 2) reliably. Prefer H.264 when
    available (avc1/H264). Actual support depends on the OpenCV build and OS codecs.
    """
    override = (os.getenv("INTENTWATCH_HISTORY_FOURCC") or "").strip()
    if override:
        # Allow comma-separated overrides like: "avc1,H264,mp4v"
        parts = [p.strip() for p in override.split(",") if p.strip()]
        return parts or [override]

    # Default preference order
    return ["avc1", "H264", "X264", "mp4v"]


@dataclass(frozen=True)
class NormalizedZone:
    id: str
    name: str
    severity: str
    x: float
    y: float
    width: float
    height: float


class StreamStatus(TypedDict):
    mode: str | None
    path: str | int | None
    running: bool


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _open_capture(source: str | int):
    if isinstance(source, int) and os.name == "nt":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)

    # On Windows, OpenCV+FFmpeg can crash on some videos (libavcodec threading assertions).
    # Prefer Media Foundation for file/URL sources.
    if os.name == "nt":
        cap_try = cv2.VideoCapture(source, cv2.CAP_MSMF)
        if cap_try.isOpened():
            return cap_try
        try:
            cap_try.release()
        except Exception:
            pass
        return cv2.VideoCapture(source)

    # Non-Windows: try FFmpeg first, then default.
    cap_try = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if cap_try.isOpened():
        return cap_try

    try:
        cap_try.release()
    except Exception:
        pass

    return cv2.VideoCapture(source)


class _ModelHolder:
    def __init__(self, model_candidates: Iterable[Path]):
        self._candidates = list(model_candidates)
        self._model: YOLO | None = None
        self._lock = threading.Lock()

    def get_optional(self) -> YOLO | None:
        """Return a loaded YOLO model if any candidate exists, else None."""
        with self._lock:
            if self._model is not None:
                return self._model

            model_path = next((p for p in self._candidates if p.exists()), None)
            if model_path is None:
                return None

            self._model = YOLO(str(model_path))
            return self._model

    def get(self) -> YOLO:
        model = self.get_optional()
        if model is None:
            raise FileNotFoundError(
                "YOLO model file not found. Expected one of: "
                + ", ".join(str(p) for p in self._candidates)
            )
        return model


class StreamWorker:
    def __init__(
        self,
        stream_id: str,
        source: str | int,
        mode: str,
        model_holder: _ModelHolder,
        weapon_model_holder: _ModelHolder | None = None,
        weapon_verify_model_holder: _ModelHolder | None = None,
        weapon_fallback_model_holder: _ModelHolder | None = None,
        *,
        conf: float = 0.4,
        weapon_conf: float = 0.85,
        weapon_knife_conf: float = 0.8,
        weapon_persist_frames: int = 3,
        weapon_max_area_ratio: float = 0.3,
        weapon_verify_conf: float = 0.85,
        weapon_verify_imgsz: int = 800,
        weapon_verify_cooldown_s: float = 2.0,
        weapon_verify_retry_s: float = 0.4,
        weapon_verify_window_s: float = 1.5,
        weapon_fallback_conf: float = 0.8,
        weapon_fallback_persist_frames: int = 3,
        weapon_allow_person_labels: bool = False,
        weapon_labels_allowlist: set[str] | None = None,
        fps_limit: int = 30,
        infer_imgsz: int = 640,
        infer_half: bool = False,
        max_frame_height: int = 720,
        jpeg_quality: int = 80,
        weapon_infer_every_n_frames: int = 1,
        weapon_rearm_seconds: float = 20.0,
        weapon_clear_seconds: float = 2.0,
        running_persist_frames: int = 2,
        zone_dwell_seconds: float = 2.0,
        zone_cooldown_s: float = 10.0,
        history_enabled: bool = False,
        history_root_dir: Path | None = None,
        history_clip_seconds: int = 60,
        history_upload_supabase: bool = False,
        history_bucket: str = "footages",
        history_table: str = "footage_clips",
        snapshots_enabled: bool = True,
        snapshots_root_dir: Path | None = None,
        snapshots_upload_supabase: bool = False,
        snapshots_bucket: str = "Snapshots",
    ):
        self.stream_id = stream_id
        self.source = source
        self.mode = str(mode or "").strip().lower() or "file"
        self._model_holder = model_holder
        self._weapon_model_holder = weapon_model_holder
        self._weapon_verify_model_holder = weapon_verify_model_holder
        self._weapon_fallback_model_holder = weapon_fallback_model_holder
        self._conf = conf
        self._weapon_conf = weapon_conf
        self._weapon_knife_conf = float(weapon_knife_conf)
        self._weapon_persist_frames = max(1, int(weapon_persist_frames))
        self._weapon_max_area_ratio = float(weapon_max_area_ratio)
        self._weapon_verify_conf = float(weapon_verify_conf)
        self._weapon_verify_imgsz = max(128, int(weapon_verify_imgsz))
        self._weapon_verify_cooldown_s = max(0.0, float(weapon_verify_cooldown_s))
        self._weapon_verify_retry_s = max(0.0, float(weapon_verify_retry_s))
        self._weapon_verify_window_s = max(0.1, float(weapon_verify_window_s))
        self._weapon_fallback_conf = float(weapon_fallback_conf)
        self._weapon_fallback_persist_frames = max(1, int(weapon_fallback_persist_frames))
        self._weapon_allow_person_labels = bool(weapon_allow_person_labels)
        if weapon_labels_allowlist is None:
            self._weapon_labels_allowlist: set[str] | None = None
        else:
            self._weapon_labels_allowlist = {
                s.strip().lower() for s in (weapon_labels_allowlist or set()) if str(s).strip()
            }
        self._fps_limit = fps_limit

        self._infer_imgsz = max(128, int(infer_imgsz))
        self._infer_half = bool(infer_half)
        self._max_frame_height = max(0, int(max_frame_height))
        self._jpeg_quality = int(jpeg_quality)
        if self._jpeg_quality < 30:
            self._jpeg_quality = 30
        if self._jpeg_quality > 95:
            self._jpeg_quality = 95

        self._weapon_infer_every_n_frames = max(1, int(weapon_infer_every_n_frames))
        self._weapon_rearm_seconds = max(0.0, float(weapon_rearm_seconds))
        self._weapon_clear_seconds = max(0.0, float(weapon_clear_seconds))
        self._running_persist_frames = max(1, int(running_persist_frames))
        self._zone_dwell_seconds = max(0.0, float(zone_dwell_seconds))
        self._zone_cooldown_s = max(0.0, float(zone_cooldown_s))

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None

        self._latest_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_frame_ts: float | None = None

        self._zones_lock = threading.Lock()
        self._zones: List[NormalizedZone] = []

        # Detection state
        self._person_start_time: float | None = None
        self._bag_start_time: float | None = None
        self._person_positions: Dict[int, Tuple[int, int, float]] = {}
        self._person_speed_history: Dict[int, List[float]] = {}

        # Simple cooldown to avoid alert spam
        self._alert_lock = threading.Lock()
        self._last_alert_ts: Dict[str, float] = {}

        self.last_people_count: int = 0
        self.started_at: float = time.time()

        # Weapon de-noising: require persistence across frames
        self._weapon_streak: int = 0
        self._weapon_fallback_streak: int = 0

        # Weapon event gating: emit only once until the weapon clears.
        self._weapon_event_active: bool = False
        self._weapon_last_seen_ts: float | None = None
        self._weapon_last_emit_ts: float | None = None

        # Weapon verification (secondary model): cache recent decision to avoid re-running
        # verification on every frame when a false positive persists.
        self._weapon_last_verify_ts: float | None = None
        self._weapon_last_verify_ok: bool | None = None
        self._weapon_verify_pending_since: float | None = None

        # Debounce running alerts to reduce one-frame ID-swap spikes
        self._running_streak: int = 0

        # Zone dwell tracking (zone id -> first time person observed inside)
        self._zone_presence_start: Dict[str, float] = {}

        # History recording
        self._history_enabled = bool(history_enabled) and (self.mode == "camera")
        self._history_root_dir = history_root_dir
        self._history_clip_seconds = max(5, int(history_clip_seconds))
        self._history_upload_supabase = bool(history_upload_supabase)
        self._history_bucket = str(history_bucket or "footages").strip() or "footages"
        self._history_table = str(history_table or "footage_clips").strip() or "footage_clips"

        # Alert snapshots
        self._snapshots_enabled = bool(snapshots_enabled)
        self._snapshots_root_dir = snapshots_root_dir
        self._snapshots_upload_supabase = bool(snapshots_upload_supabase)
        self._snapshots_bucket = str(snapshots_bucket or "Snapshots").strip() or "Snapshots"

        if self._snapshots_root_dir is not None:
            try:
                self._snapshots_root_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # If the folder can't be created, disable snapshots (best-effort).
                self._snapshots_enabled = False

        self._clip_writer: cv2.VideoWriter | None = None
        self._clip_started_at: float | None = None
        self._clip_path: Path | None = None
        self._clip_fps: float = 15.0
        self._clip_fourcc: str | None = None

        # thresholds (kept same as previous behavior)
        self.LOITER_THRESHOLD = 5
        self.BAG_THRESHOLD = 5
        self.PERSON_BAG_DISTANCE = 150
        self.RUNNING_SPEED_THRESHOLD = 120

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"stream-{self.stream_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Clear buffered frame so new clients don't see stale last-frame forever.
        with self._latest_lock:
            self._latest_jpeg = None
            self._latest_frame_ts = None
        # Best-effort: releasing the capture from the caller thread can help unblock a stuck read().
        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        self._finalize_clip(reason="stop")

    def is_alive(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def join(self, timeout: float = 1.0) -> None:
        t = self._thread
        if t is None:
            return
        try:
            t.join(timeout=timeout)
        except Exception:
            pass

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive() and not self._stop_event.is_set())

    def set_zones(self, zones: List[NormalizedZone]) -> None:
        with self._zones_lock:
            self._zones = zones

    def get_zones(self) -> List[NormalizedZone]:
        with self._zones_lock:
            return list(self._zones)

    def get_latest_jpeg(self) -> Tuple[bytes | None, float | None]:
        with self._latest_lock:
            return self._latest_jpeg, self._latest_frame_ts

    def _emit_alert(
        self,
        add_alert_fn,
        alert_type: str,
        message: str,
        *,
        cooldown_s: float = 5.0,
        severity: str | None = None,
        snapshot_provider: Callable[[], str | None] | None = None,
        snapshot_url: str | None = None,
    ) -> None:
        key = f"{alert_type}:{message}" if message else alert_type
        now = time.time()
        with self._alert_lock:
            last = self._last_alert_ts.get(key)
            if last is not None and (now - last) < cooldown_s:
                return
            self._last_alert_ts[key] = now

        if snapshot_provider is not None:
            try:
                snapshot_url = snapshot_provider()
            except Exception:
                snapshot_url = snapshot_url

        # Defer to shared alert store
        add_alert_fn(alert_type, message, severity=severity, camera=self.stream_id, snapshot_url=snapshot_url)

    def _save_snapshot_jpeg(
        self,
        frame_bgr,
        bbox: Tuple[int, int, int, int],
        *,
        alert_type: str,
        now: float,
    ) -> str | None:
        if not self._snapshots_enabled:
            return None
        root = self._snapshots_root_dir
        if root is None:
            return None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        # Add a small padding so the crop includes context.
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)
        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(w - 1, x2 + pad_x)
        y2p = min(h - 1, y2 + pad_y)

        crop = frame_bgr[y1p : y2p, x1p : x2p]
        if crop.size == 0:
            return None

        date_s = time.strftime("%Y-%m-%d", time.localtime(now))
        time_s = time.strftime("%H%M%S", time.localtime(now))
        out_dir = root / self.stream_id / date_s
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_type = "".join(ch for ch in str(alert_type or "snap") if ch.isalnum() or ch in {"-", "_"}).lower() or "snap"
        filename = f"{time_s}-{safe_type}-{uuid.uuid4().hex[:8]}.jpg"
        out_path = out_dir / filename

        ok = False
        try:
            ok = bool(cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90]))
        except Exception:
            ok = False
        if not ok:
            return None

        local_url = f"/alerts/snapshot/{self.stream_id}/{date_s}/{filename}"

        # Optional Supabase upload (small file; ok to do synchronously when alert triggers)
        if self._snapshots_upload_supabase:
            try:
                from api.supabase_client import is_configured, upload_file

                if is_configured():
                    storage_key = f"{self.stream_id}/{date_s}/{filename}"
                    public_url = upload_file(
                        self._snapshots_bucket,
                        storage_key,
                        str(out_path),
                        content_type="image/jpeg",
                    )
                    if public_url:
                        # Sidecar metadata for local debugging
                        try:
                            meta_path = out_path.with_suffix(out_path.suffix + ".json")
                            meta_path.write_text(
                                json.dumps({"public_url": public_url, "storage_key": storage_key}, ensure_ascii=False),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                        return str(public_url)
            except Exception:
                pass

        return local_url

    def _finalize_clip(self, *, reason: str) -> None:
        writer = self._clip_writer
        clip_path = self._clip_path
        started_at = self._clip_started_at
        fourcc = self._clip_fourcc

        self._clip_writer = None
        self._clip_path = None
        self._clip_started_at = None
        self._clip_fourcc = None

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        if not clip_path or not clip_path.exists():
            return

        # Optionally upload to Supabase (best-effort, non-blocking)
        if not self._history_upload_supabase:
            return

        try:
            from api.supabase_client import is_configured, upload_file, insert_row
        except Exception:
            return

        if not is_configured():
            return

        def _upload() -> None:
            try:
                # storage key: <stream_id>/<YYYY-MM-DD>/<filename>
                parts = clip_path.parts
                # Find the last 3 path components: stream_id/date/filename
                storage_key = "/".join(parts[-3:])
                public_url = upload_file(
                    self._history_bucket,
                    storage_key,
                    str(clip_path),
                    content_type="video/mp4",
                )

                # Sidecar metadata for the local history browser
                meta_path = clip_path.with_suffix(clip_path.suffix + ".json")
                meta = {
                    "public_url": public_url,
                    "uploaded": bool(public_url),
                    "reason": reason,
                    "started_at": float(started_at) if started_at else None,
                    "stream_id": self.stream_id,
                    "fourcc": fourcc,
                }
                try:
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

                if public_url:
                    insert_row(
                        self._history_table,
                        {
                            "stream_id": self.stream_id,
                            "storage_key": storage_key,
                            "public_url": public_url,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at or time.time())),
                        },
                    )
            except Exception:
                return

        threading.Thread(target=_upload, name="history-upload", daemon=True).start()

    def _ensure_clip_writer(self, frame_w: int, frame_h: int, *, now: float) -> None:
        if not self._history_enabled:
            return

        root = self._history_root_dir
        if root is None:
            return

        if self._clip_writer is not None and self._clip_started_at is not None:
            # Rotate if clip duration exceeded
            if (now - self._clip_started_at) >= float(self._history_clip_seconds):
                self._finalize_clip(reason="rotate")
            else:
                return

        # Start new clip
        date_s = time.strftime("%Y-%m-%d", time.localtime(now))
        time_s = time.strftime("%H%M%S", time.localtime(now))
        out_dir = root / self.stream_id / date_s
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{time_s}.mp4"
        out_path = out_dir / filename
        # If the name already exists, add a suffix
        if out_path.exists():
            out_path = out_dir / f"{time_s}-{int(now)}.mp4"

        # Use a conservative FPS; if capture reports a reasonable FPS, use it.
        fps = float(self._clip_fps) if self._clip_fps > 1 else 15.0
        size = (int(frame_w), int(frame_h))

        # Prefer browser-friendly H.264 MP4 when available.
        # Fallback to mp4v if the environment lacks H.264 support.
        force_mp4v = _bool_env("INTENTWATCH_HISTORY_FORCE_MP4V", False)
        candidates = ["mp4v"] if force_mp4v else _pick_history_fourcc_candidates()

        writer = None
        chosen = None
        for code in candidates:
            try:
                cc = cv2.VideoWriter_fourcc(*str(code)[:4])
                w = cv2.VideoWriter(str(out_path), cc, fps, size)
                if w.isOpened():
                    writer = w
                    chosen = str(code)[:4]
                    break
                try:
                    w.release()
                except Exception:
                    pass
            except Exception:
                continue

        if writer is None:
            return

        self._clip_writer = writer
        self._clip_started_at = now
        self._clip_path = out_path
        self._clip_fourcc = chosen

    def _run(self) -> None:
        cap_local = None
        try:
            cap_local = _open_capture(self.source)
            self._cap = cap_local

            # Best-effort: keep capture buffering low to reduce latency.
            # (Some backends ignore this; we also implement a latest-frame reader for camera mode below.)
            if cap_local is not None:
                try:
                    cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            if cap_local is None or not cap_local.isOpened():
                return

            try:
                fps_cap = float(cap_local.get(cv2.CAP_PROP_FPS) or 0.0)
                if fps_cap and 1.0 < fps_cap < 121.0:
                    self._clip_fps = fps_cap
            except Exception:
                pass

            # When streaming a local video file, heavy inference can make playback run
            # slower than real-time, which users perceive as "late" labels/alerts.
            # To keep the stream close to real-time, optionally skip frames when behind.
            file_realtime = bool(self.mode == "file") and _bool_env("INTENTWATCH_FILE_REALTIME", True)
            file_base_ts = time.time()
            file_base_frame = 0
            file_fps = 0.0
            cap_frame_idx = 0
            if file_realtime:
                try:
                    file_fps = float(cap_local.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    file_fps = 0.0
                if not (1.0 < file_fps < 121.0):
                    # Fall back to an assumed FPS; this only affects skip behavior.
                    file_fps = 30.0
                try:
                    cap_frame_idx = int(cap_local.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                except Exception:
                    cap_frame_idx = 0
                file_base_frame = cap_frame_idx

            model = self._model_holder.get()

            last_yield = 0.0
            frame_index = 0

            # For live camera streams, we prefer low latency over processing every frame.
            # If inference is slower than the camera FPS, processing frames sequentially causes
            # the pipeline to fall behind, making labels/alerts appear "late". To prevent this,
            # keep only the latest captured frame and drop stale ones.
            drop_stale_frames = bool(self.mode == "camera") and _bool_env(
                "INTENTWATCH_CAMERA_DROP_STALE_FRAMES",
                True,
            )

            capture_lock = threading.Lock()
            latest_frame = None
            latest_frame_ts: float | None = None
            latest_frame_id = 0
            capture_done = threading.Event()

            def _capture_reader() -> None:
                nonlocal latest_frame, latest_frame_ts, latest_frame_id
                try:
                    while not self._stop_event.is_set():
                        ret, frm = cap_local.read()
                        if not ret or frm is None:
                            break
                        ts_now = time.time()
                        with capture_lock:
                            latest_frame = frm
                            latest_frame_ts = ts_now
                            latest_frame_id += 1
                finally:
                    capture_done.set()

            capture_thread: threading.Thread | None = None
            if drop_stale_frames:
                capture_thread = threading.Thread(
                    target=_capture_reader,
                    name=f"capture-{self.stream_id}",
                    daemon=True,
                )
                capture_thread.start()

            # Import here to avoid circular imports
            from api.routes.alerts import add_alert

            last_processed_frame_id = -1
            while not self._stop_event.is_set():
                now: float = time.time()
                if drop_stale_frames:
                    with capture_lock:
                        fid = latest_frame_id
                        frame = latest_frame
                        captured_ts = latest_frame_ts

                    # Prefer a capture timestamp if available; otherwise fall back.
                    if captured_ts is not None:
                        now = float(captured_ts)

                    if frame is None or fid == last_processed_frame_id:
                        if capture_done.is_set():
                            break
                        time.sleep(0.005)
                        continue

                    last_processed_frame_id = fid

                else:
                    # If we're playing a file in real-time mode, fast-forward by grabbing
                    # frames until we're close to the expected frame index for wall time.
                    if file_realtime and file_fps > 1.0:
                        desired = file_base_frame + int((time.time() - file_base_ts) * file_fps)
                        behind = desired - cap_frame_idx
                        if behind > 0:
                            # Don't skip unboundedly; cap at ~2 seconds worth each loop.
                            max_skip = int(max(1.0, file_fps * 2.0))
                            to_skip = min(int(behind), int(max_skip))
                            for _ in range(to_skip):
                                ok = False
                                try:
                                    ok = bool(cap_local.grab())
                                except Exception:
                                    ok = False
                                if not ok:
                                    break
                                cap_frame_idx += 1

                    ret, frame = cap_local.read()
                    if not ret or frame is None:
                        if isinstance(self.source, str):
                            cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap_local.read()
                            if not ret or frame is None:
                                break
                        else:
                            break

                    now = time.time()
                    cap_frame_idx += 1

                # Basic FPS limiting (camera streams only).
                if (not file_realtime) and self._fps_limit > 0:
                    min_dt = 1.0 / float(self._fps_limit)
                    if (now - last_yield) < min_dt:
                        time.sleep(max(min_dt - (now - last_yield), 0.0))
                        now = time.time()
                    last_yield = now

                # Resize large frames
                h, w = frame.shape[:2]
                if self._max_frame_height > 0 and h > self._max_frame_height:
                    scale = float(self._max_frame_height) / float(h)
                    w = int(w * scale)
                    frame = cv2.resize(frame, (w, self._max_frame_height), interpolation=cv2.INTER_LINEAR)
                    h = int(self._max_frame_height)

                # IMPORTANT: run inference on a clean frame.
                # Drawing overlays before inference can create false positives.
                frame_clean = frame.copy()
                frame_infer = frame_clean

                # Run YOLO
                results = model(
                    frame_infer,
                    conf=self._conf,
                    imgsz=self._infer_imgsz,
                    half=self._infer_half,
                    verbose=False,
                )

                persons: List[Tuple[int, int, int, int, int, int]] = []
                bags: List[Tuple[int, int, int, int]] = []
                weapons: List[Tuple[int, int, int, int, str]] = []
                weapon_candidates: List[Tuple[int, int, int, int, str, float]] = []
                fallback_candidates: List[Tuple[int, int, int, int, float]] = []

                weapon_model = (
                    self._weapon_model_holder.get_optional() if self._weapon_model_holder is not None else None
                )

                weapon_verify_model = (
                    self._weapon_verify_model_holder.get_optional()
                    if self._weapon_verify_model_holder is not None
                    else None
                )

                weapon_fallback_model = (
                    self._weapon_fallback_model_holder.get_optional()
                    if self._weapon_fallback_model_holder is not None
                    else None
                )

                def _weapon_alias(label_norm: str) -> str:
                    # Normalize similar labels to a single user-facing weapon type.
                    # NOTE: This does not create new detection capability; it only renames detected classes.
                    if label_norm in {"pistol", "handgun", "gun", "rifle", "firearm"}:
                        return "gun"
                    # Many custom datasets use a single generic class name like "weapon".
                    if ("weapon" in label_norm) or ("firearm" in label_norm):
                        return "gun"
                    return label_norm

                def _intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
                    ax1, ay1, ax2, ay2 = a
                    bx1, by1, bx2, by2 = b
                    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

                def _near_person(wb: Tuple[int, int, int, int], *, base_px: int = 220) -> bool:
                    """Return True if weapon box is close enough to any detected person.

                    We use this instead of requiring strict intersection because many real weapon
                    boxes (rifles/knives) sit just outside the person bbox, especially with tight
                    person boxes or partial occlusion.
                    """
                    if not persons:
                        return False
                    x1, y1, x2, y2 = wb
                    wcx = (x1 + x2) // 2
                    wcy = (y1 + y2) // 2
                    for (px1, py1, px2, py2, pcx, pcy) in persons:
                        pw = max(1, px2 - px1)
                        ph = max(1, py2 - py1)
                        # Dynamic threshold scales with person size.
                        thresh = max(int(base_px), int(0.55 * float(max(pw, ph))))
                        dx = float(wcx - pcx)
                        dy = float(wcy - pcy)
                        if (dx * dx + dy * dy) <= float(thresh * thresh):
                            return True
                    return False

                def _expanded_person_boxes(pad_ratio: float = 0.2) -> List[Tuple[int, int, int, int]]:
                    # Expand person bboxes slightly so a weapon box near a hand
                    # still counts as "associated".
                    expanded: List[Tuple[int, int, int, int]] = []
                    for (px1, py1, px2, py2, _pcx, _pcy) in persons:
                        pw = max(1, px2 - px1)
                        ph = max(1, py2 - py1)
                        pad_x = int(pw * pad_ratio)
                        pad_y = int(ph * pad_ratio)
                        ex1 = max(0, px1 - pad_x)
                        ey1 = max(0, py1 - pad_y)
                        ex2 = min(w - 1, px2 + pad_x)
                        ey2 = min(h - 1, py2 + pad_y)
                        expanded.append((ex1, ey1, ex2, ey2))
                    return expanded

                # Gather detections
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = str(model.names.get(cls, cls))
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        if label == "person":
                            persons.append((x1, y1, x2, y2, cx, cy))
                        elif label in {"backpack", "handbag", "suitcase", "bag"}:
                            bags.append((x1, y1, x2, y2))
                        elif weapon_model is None and label.lower() in {"knife", "gun", "pistol", "rifle", "weapon"}:
                            weapons.append((x1, y1, x2, y2, label))

                        # Draw common objects
                        if label in {"backpack", "handbag", "suitcase", "bag"}:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self.last_people_count = len(persons)

                # Optional dedicated weapon model: treat any detection as a weapon.
                if weapon_model is not None:
                    run_weapon = (frame_index % self._weapon_infer_every_n_frames) == 0
                    weapon_results = (
                        weapon_model(
                            frame_infer,
                            conf=self._weapon_conf,
                            imgsz=self._infer_imgsz,
                            half=self._infer_half,
                            verbose=False,
                        )
                        if run_weapon
                        else []
                    )
                    expanded_people = _expanded_person_boxes(pad_ratio=0.25)
                    for wr in weapon_results:
                        for box in wr.boxes:
                            cls = int(box.cls[0])
                            label = str(weapon_model.names.get(cls, cls))
                            label_norm = str(label).strip().lower()
                            # Roboflow exports sometimes contain placeholder labels like '-' or 'undefined'.
                            # If we have a verification model configured, we can still use these boxes
                            # as *candidates* (verification will decide whether it's truly a weapon).
                            # Without verification, ignore them to avoid false positives.
                            if label_norm in {"-", "undefined", "background"}:
                                if weapon_verify_model is None:
                                    continue
                                # Treat as a generic weapon candidate; keep overlays/alerts gated by verification.
                                label = "weapon"
                                label_norm = "weapon"

                            label_alias = _weapon_alias(label_norm)

                            # If an allowlist is explicitly configured, enforce it.
                            # Otherwise, treat all non-placeholder labels as "weapon" candidates and
                            # rely on verification (if configured) to suppress false positives.
                            if self._weapon_labels_allowlist is not None:
                                if (label_norm not in self._weapon_labels_allowlist) and (
                                    label_alias not in self._weapon_labels_allowlist
                                ):
                                    continue

                            # Some weapon datasets are mislabeled as "person-with-weapon" which can
                            # false-trigger on any person in frame.
                            # If a verification model is configured, we can allow these labels and let
                            # verification suppress false positives.
                            if (weapon_verify_model is None) and (not self._weapon_allow_person_labels) and (
                                "person" in label_norm
                            ):
                                continue
                            try:
                                score = float(box.conf[0])
                            except Exception:
                                score = 0.0
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Per-class tightening: knife false-positives are common.
                            # Keep gun more permissive (it is often intermittent), but require
                            # a higher confidence for knives.
                            if label_alias == "knife" and score < max(self._weapon_conf, self._weapon_knife_conf):
                                continue

                            # Anti-false-positive guard:
                            # If we have people in the scene, only accept weapon boxes that
                            # intersect an expanded person box (weapon in/near hands).
                            # If there are *no* people detected, require a higher confidence.
                            wb = (x1, y1, x2, y2)
                            if expanded_people:
                                if (not any(_intersects(wb, pb) for pb in expanded_people)) and (not _near_person(wb)):
                                    continue
                            else:
                                # "weapon" with no person nearby is much more likely to be a false positive.
                                if score < max(self._weapon_conf, 0.75):
                                    continue

                            # Guard against huge "whole frame" boxes which are a common false-positive pattern.
                            box_area = max(0, x2 - x1) * max(0, y2 - y1)
                            frame_area = max(1, int(w) * int(h))
                            area_ratio = float(box_area) / float(frame_area)
                            if area_ratio > self._weapon_max_area_ratio:
                                continue

                            # Extra guard: labels that look like "person with weapon" often trigger on any person.
                            # Only apply this guard when there's no verification model.
                            if (weapon_verify_model is None) and ("person" in label_norm) and (
                                score < max(self._weapon_conf, 0.9)
                            ):
                                continue

                            weapon_candidates.append((x1, y1, x2, y2, label, score))

                # Require persistence across frames before declaring a weapon candidate.
                if weapon_candidates:
                    self._weapon_streak += 1
                else:
                    self._weapon_streak = 0

                weapon_primary_ready = bool(weapon_candidates) and (self._weapon_streak >= self._weapon_persist_frames)

                # Fallback: if weapon-type model found nothing, try a "person-with-weapon" model.
                # This draws a person-sized box labeled "PERSON WITH GUN".
                # It is intentionally gated heavily to reduce false positives.
                if (not weapon_primary_ready) and (weapon_fallback_model is not None):
                    fallback_candidates.clear()
                    expanded_people = _expanded_person_boxes(pad_ratio=0.15)
                    run_fallback = (frame_index % self._weapon_infer_every_n_frames) == 0
                    fb_results = (
                        weapon_fallback_model(
                            frame_infer,
                            conf=self._weapon_fallback_conf,
                            imgsz=self._infer_imgsz,
                            half=self._infer_half,
                            verbose=False,
                        )
                        if run_fallback
                        else []
                    )
                    for fr in fb_results:
                        for box in fr.boxes:
                            cls = int(box.cls[0])
                            raw_label = str(weapon_fallback_model.names.get(cls, cls))
                            label_norm = str(raw_label).strip().lower()
                            if label_norm in {"-", "undefined", "background"}:
                                continue
                            try:
                                score = float(box.conf[0])
                            except Exception:
                                score = 0.0
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Only allow fallback boxes that intersect a person.
                            # This prevents the fallback model from labeling random objects as "person with gun".
                            if expanded_people:
                                if (not any(_intersects((x1, y1, x2, y2), pb) for pb in expanded_people)) and (not _near_person((x1, y1, x2, y2), base_px=260)):
                                    continue
                            else:
                                # If we can't detect a person at all, avoid triggering fallback.
                                continue
                            fallback_candidates.append((x1, y1, x2, y2, score))

                    if fallback_candidates:
                        self._weapon_fallback_streak += 1
                    else:
                        self._weapon_fallback_streak = 0

                    if self._weapon_fallback_streak >= self._weapon_fallback_persist_frames:
                        # Use the highest-confidence box.
                        x1, y1, x2, y2, _score = max(fallback_candidates, key=lambda t: t[4])
                        # Treat as a weapon candidate (fallback).
                        weapon_primary_ready = False

                # Pick a single "event candidate" box/label for verification + alerting.
                candidate_bbox: Tuple[int, int, int, int] | None = None
                candidate_label: str | None = None
                if weapon_primary_ready and weapon_candidates:
                    x1, y1, x2, y2, label, _score = max(weapon_candidates, key=lambda t: float(t[5]))
                    candidate_bbox = (int(x1), int(y1), int(x2), int(y2))
                    candidate_label = str(label)
                elif (weapon_fallback_model is not None) and (self._weapon_fallback_streak >= self._weapon_fallback_persist_frames) and fallback_candidates:
                    x1, y1, x2, y2, _score = max(fallback_candidates, key=lambda t: float(t[4]))
                    candidate_bbox = (int(x1), int(y1), int(x2), int(y2))
                    candidate_label = "person with gun"

                def _weapon_label_for_overlay(raw_label: str) -> str:
                    # Keep overlay readable even if model has long class names.
                    s = str(raw_label).strip()
                    s_norm = s.lower()
                    s_alias = _weapon_alias(s_norm)
                    if s_alias != s_norm:
                        s = s_alias
                    if len(s) > 32:
                        s = s[:29] + "..."
                    return s

                # Emit at most one weapon alert per cycle (cooldown handles further spam)
                weapon_signal = bool(candidate_bbox is not None) or bool(weapon_candidates) or bool(fallback_candidates)
                if weapon_signal:
                    self._weapon_last_seen_ts = now
                else:
                    # Reset verify pending state when the weapon is no longer present.
                    self._weapon_verify_pending_since = None
                    if (
                        self._weapon_event_active
                        and self._weapon_last_seen_ts is not None
                        and (now - self._weapon_last_seen_ts) >= self._weapon_clear_seconds
                    ):
                        self._weapon_event_active = False

                # If a verify model is configured, suppress weapon overlays unless verification passes.
                # This prevents "false gun label" in clips where the fast model misfires.
                if candidate_bbox is not None and candidate_label is not None:
                    cb = candidate_bbox
                    label_text = _weapon_label_for_overlay(candidate_label)

                    def _verify_weapon_event_start() -> bool:
                        # Only verify at event-start (when event isn't active yet).
                        # Uses a cooldown cache so persistent FPs don't run verify every frame.
                        if weapon_verify_model is None:
                            return True
                        if self._weapon_event_active:
                            return True

                        # Start/maintain a short verification window so a single bad frame
                        # doesn't permanently suppress a real weapon event.
                        if self._weapon_verify_pending_since is None:
                            self._weapon_verify_pending_since = now
                        elif (now - self._weapon_verify_pending_since) > self._weapon_verify_window_s:
                            # Give up for now; will reset once weapon clears.
                            self._weapon_last_verify_ts = now
                            self._weapon_last_verify_ok = False
                            return False

                        # Cooldown handling:
                        # - If last verify was OK, honor the configured cooldown.
                        # - If last verify was NOT OK, retry more frequently (retry_s).
                        if self._weapon_last_verify_ts is not None and self._weapon_last_verify_ok is not None:
                            dt = now - self._weapon_last_verify_ts
                            if self._weapon_last_verify_ok:
                                if dt < self._weapon_verify_cooldown_s:
                                    return True
                            else:
                                if dt < self._weapon_verify_retry_s:
                                    return False

                        ok = False
                        try:
                            vres = weapon_verify_model(
                                frame_infer,
                                conf=self._weapon_verify_conf,
                                imgsz=self._weapon_verify_imgsz,
                                half=self._infer_half,
                                verbose=False,
                            )
                        except Exception:
                            vres = []

                        expanded_people = _expanded_person_boxes(pad_ratio=0.25)
                        for vr in vres:
                            for box in vr.boxes:
                                cls = int(box.cls[0])
                                raw_label = str(weapon_verify_model.names.get(cls, cls))
                                label_norm = str(raw_label).strip().lower()

                                # Ignore placeholder labels and person-only outputs.
                                if label_norm in {"-", "undefined", "background"}:
                                    continue
                                if "person" in label_norm:
                                    continue
                                try:
                                    score = float(box.conf[0])
                                except Exception:
                                    score = 0.0
                                x1, y1, x2, y2 = map(int, box.xyxy[0])

                                # If people exist, require an intersection so we don't verify on random objects.
                                if expanded_people:
                                    if (not any(_intersects((x1, y1, x2, y2), pb) for pb in expanded_people)) and (not _near_person((x1, y1, x2, y2))):
                                        continue
                                else:
                                    # If we can't see a person, still allow verification based on
                                    # confidence alone (respect the configured threshold).
                                    if score < float(self._weapon_verify_conf):
                                        continue

                                # Avoid whole-frame boxes.
                                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                                frame_area = max(1, int(w) * int(h))
                                if (float(box_area) / float(frame_area)) > max(self._weapon_max_area_ratio, 0.6):
                                    continue

                                ok = True
                                break
                            if ok:
                                break

                        self._weapon_last_verify_ts = now
                        self._weapon_last_verify_ok = bool(ok)
                        return bool(ok)

                    event_start_confirmed = _verify_weapon_event_start()

                    can_emit = not self._weapon_event_active
                    if self._weapon_last_emit_ts is not None and (now - self._weapon_last_emit_ts) < self._weapon_rearm_seconds:
                        can_emit = False

                    def _weapon_snapshot() -> str | None:
                        if not self._snapshots_enabled:
                            return None
                        wx1, wy1, wx2, wy2 = cb
                        wcx = (wx1 + wx2) // 2
                        wcy = (wy1 + wy2) // 2
                        best_bbox: Tuple[int, int, int, int] = (wx1, wy1, wx2, wy2)
                        best_score: float | None = None
                        for (px1, py1, px2, py2, pcx, pcy) in persons:
                            ix1 = max(wx1, px1)
                            iy1 = max(wy1, py1)
                            ix2 = min(wx2, px2)
                            iy2 = min(wy2, py2)
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            if inter > 0:
                                score = 1_000_000_000 + float(inter)
                            else:
                                dx = float(pcx - wcx)
                                dy = float(pcy - wcy)
                                score = -((dx * dx) + (dy * dy))
                            if best_score is None or score > best_score:
                                best_score = score
                                best_bbox = (px1, py1, px2, py2)
                        return self._save_snapshot_jpeg(frame_clean, best_bbox, alert_type="weapon", now=now)

                    # Only consider the event "active" after verification passes (if configured).
                    if event_start_confirmed:
                        self._weapon_event_active = True
                        self._weapon_verify_pending_since = None

                    # Draw weapon overlay only when verified (or when no verify model is configured).
                    if event_start_confirmed or (weapon_verify_model is None) or self._weapon_event_active:
                        x1, y1, x2, y2 = cb
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        text_y = max(y1 - 8, 15)
                        cv2.putText(
                            frame,
                            f"{label_text.upper()}!",
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                    if can_emit and event_start_confirmed:
                        self._weapon_last_emit_ts = now
                        self._emit_alert(
                            add_alert,
                            "Weapon",
                            f"{label_text} detected",
                            cooldown_s=12.0,
                            severity="Critical",
                            snapshot_provider=_weapon_snapshot,
                        )

                # Running detection (same tracking approach as before, simplified)
                running_persons: set[int] = set()

                prev_positions = dict(self._person_positions)
                used_pids: set[int] = set()
                person_ids_by_index: Dict[int, int] = {}

                for i, (x1, y1, x2, y2, cx, cy) in enumerate(persons):
                    pid = None
                    min_dist = float("inf")
                    for tracked_pid, (prev_x, prev_y, prev_t) in prev_positions.items():
                        if tracked_pid in used_pids:
                            continue
                        dx = cx - prev_x
                        dy = cy - prev_y
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < 90 and dist < min_dist:
                            min_dist = dist
                            pid = tracked_pid

                    if pid is None:
                        pid = (max(prev_positions.keys()) + 1) if prev_positions else 0
                        while pid in used_pids:
                            pid += 1

                    used_pids.add(pid)
                    person_ids_by_index[i] = pid

                    if pid in prev_positions:
                        prev_x, prev_y, prev_t = prev_positions[pid]
                        dist = ((cx - prev_x) ** 2 + (cy - prev_y) ** 2) ** 0.5
                        dt = max(now - prev_t, 0.016)
                        speed = dist / dt

                        hist = self._person_speed_history.setdefault(pid, [])
                        hist.append(speed)
                        if len(hist) > 5:
                            hist.pop(0)

                        if len(hist) >= 3:
                            recent = hist[-3:]
                            avg_speed = sum(recent) / len(recent)
                            if avg_speed > self.RUNNING_SPEED_THRESHOLD:
                                running_persons.add(pid)

                    self._person_positions[pid] = (cx, cy, now)

                # Draw people
                for i, (x1, y1, x2, y2, cx, cy) in enumerate(persons):
                    pid = person_ids_by_index.get(i)
                    label = f"person {pid}" if pid is not None else "person"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for pid in running_persons:
                    # Debounce running alerts to avoid single-frame spikes
                    self._running_streak += 1
                    if self._running_streak >= self._running_persist_frames:
                        self._emit_alert(
                            add_alert,
                            "Running",
                            "Person running detected",
                            cooldown_s=8.0,
                            severity="Low",
                        )
                        self._running_streak = 0

                if not running_persons:
                    self._running_streak = 0

                # Cleanup old tracking
                self._person_positions = {k: v for k, v in self._person_positions.items() if now - v[2] < 1.5}
                self._person_speed_history = {k: v for k, v in self._person_speed_history.items() if k in self._person_positions}

                # Loitering detection
                if persons:
                    if self._person_start_time is None:
                        self._person_start_time = now
                    elif now - self._person_start_time > self.LOITER_THRESHOLD:
                        self._emit_alert(add_alert, "Loitering", "Person loitering detected", cooldown_s=5.0, severity="Medium")
                        self._person_start_time = now
                else:
                    self._person_start_time = None

                # Unattended bag detection
                unattended_bag = False
                unattended_bbox: Tuple[int, int, int, int] | None = None
                for bx1, by1, bx2, by2 in bags:
                    bcx = (bx1 + bx2) // 2
                    bcy = (by1 + by2) // 2
                    near_person = any(
                        (((bcx - px[4]) ** 2 + (bcy - px[5]) ** 2) ** 0.5) < self.PERSON_BAG_DISTANCE
                        for px in persons
                    )
                    if not near_person:
                        unattended_bag = True
                        unattended_bbox = (bx1, by1, bx2, by2)
                        break

                if unattended_bbox is not None:
                    bx1, by1, bx2, by2 = unattended_bbox
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(frame, "UNATTENDED BAG", (bx1, max(by1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if unattended_bag:
                    if self._bag_start_time is None:
                        self._bag_start_time = now
                    elif now - self._bag_start_time > self.BAG_THRESHOLD:

                        def _bag_snapshot() -> str | None:
                            if unattended_bbox is None:
                                return None
                            return self._save_snapshot_jpeg(frame_clean, unattended_bbox, alert_type="bag", now=now)

                        self._emit_alert(
                            add_alert,
                            "Unattended Bag",
                            "Suspicious item detected",
                            cooldown_s=10.0,
                            severity="High",
                            snapshot_provider=_bag_snapshot,
                        )
                        self._bag_start_time = now
                else:
                    self._bag_start_time = None

                # Zones overlay + zone alerts
                zones = self.get_zones()
                if zones:
                    observed_zone_ids: set[str] = set()
                    for z in zones:
                        zid = str(z.id)
                        observed_zone_ids.add(zid)
                        zx1 = int(_clamp01(z.x) * w)
                        zy1 = int(_clamp01(z.y) * h)
                        zx2 = int(_clamp01(z.x + z.width) * w)
                        zy2 = int(_clamp01(z.y + z.height) * h)
                        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 255), 2)
                        cv2.putText(frame, z.name, (zx1, max(zy1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        in_zone = any(zx1 <= pcx <= zx2 and zy1 <= pcy <= zy2 for (_, _, _, _, pcx, pcy) in persons)
                        if in_zone:
                            start = self._zone_presence_start.get(zid)
                            if start is None:
                                self._zone_presence_start[zid] = now
                            elif (now - start) >= self._zone_dwell_seconds:
                                sev = (z.severity or "medium").capitalize()

                                def _zone_snapshot() -> str | None:
                                    # Find a person currently in the zone and crop them.
                                    for (px1, py1, px2, py2, pcx, pcy) in persons:
                                        if zx1 <= pcx <= zx2 and zy1 <= pcy <= zy2:
                                            return self._save_snapshot_jpeg(
                                                frame_clean,
                                                (px1, py1, px2, py2),
                                                alert_type="zone",
                                                now=now,
                                            )
                                    return None

                                self._emit_alert(
                                    add_alert,
                                    "Zone",  # keep type stable
                                    f"{z.name}: person detected in zone",
                                    cooldown_s=self._zone_cooldown_s,
                                    severity=sev,
                                    snapshot_provider=_zone_snapshot,
                                )
                                # reset dwell timer so it requires re-dwell after an alert
                                self._zone_presence_start[zid] = now
                        else:
                            self._zone_presence_start.pop(zid, None)

                    # Prune state for zones no longer configured
                    for zid in list(self._zone_presence_start.keys()):
                        if zid not in observed_zone_ids:
                            self._zone_presence_start.pop(zid, None)

                # Encode JPEG and publish
                ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)],
                )
                if ok:
                    data = buf.tobytes()
                    with self._latest_lock:
                        self._latest_jpeg = data
                        self._latest_frame_ts = now

                # Record history clip after overlays
                if self._history_enabled:
                    self._ensure_clip_writer(frame.shape[1], frame.shape[0], now=now)
                    if self._clip_writer is not None:
                        try:
                            self._clip_writer.write(frame)
                        except Exception:
                            # If recording fails, stop recording for this session.
                            self._finalize_clip(reason="write_error")
                            self._history_enabled = False

                frame_index += 1

        except Exception:
            # swallow; status endpoints can show it as stopped
            pass
        finally:
            # Ensure any open clip is finalized on exit
            self._finalize_clip(reason="exit")
            try:
                if cap_local is not None:
                    cap_local.release()
            except Exception:
                pass
            self._cap = None


class StreamManager:
    def __init__(
        self,
        model_candidates: Iterable[Path],
        weapon_model_candidates: Iterable[Path] | None = None,
        weapon_verify_model_candidates: Iterable[Path] | None = None,
        weapon_fallback_model_candidates: Iterable[Path] | None = None,
    ):
        self._model_holder = _ModelHolder(model_candidates)
        self._weapon_model_holder = _ModelHolder(weapon_model_candidates) if weapon_model_candidates else None
        self._weapon_verify_model_holder = (
            _ModelHolder(weapon_verify_model_candidates) if weapon_verify_model_candidates else None
        )
        self._weapon_fallback_model_holder = (
            _ModelHolder(weapon_fallback_model_candidates) if weapon_fallback_model_candidates else None
        )
        self._lock = threading.Lock()
        self._workers: Dict[str, StreamWorker] = {}
        self._sources: Dict[str, Tuple[str | int, str]] = {}  # id -> (source, mode)
        self._last_restart_ts: Dict[str, float] = {}

        # Weapon model de-noising controls (configurable via env).
        def _env_float(name: str, default: float) -> float:
            try:
                v = os.getenv(name)
                return float(v) if v is not None and str(v).strip() != "" else float(default)
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            try:
                v = os.getenv(name)
                return int(v) if v is not None and str(v).strip() != "" else int(default)
            except Exception:
                return int(default)

        def _env_bool(name: str, default: bool) -> bool:
            try:
                v = os.getenv(name)
                if v is None:
                    return bool(default)
                s = str(v).strip().lower()
                if s in {"1", "true", "yes", "y", "on"}:
                    return True
                if s in {"0", "false", "no", "n", "off"}:
                    return False
                return bool(default)
            except Exception:
                return bool(default)

        # Defaults are tuned for real-world videos:
        # - Lower conf than 0.85 (too strict, misses guns)
        # - Persistence>1 to suppress single-frame false positives
        # - Slightly larger max box (some models output loose boxes)
        self._weapon_conf = _env_float("INTENTWATCH_WEAPON_CONF", 0.75)
        self._weapon_knife_conf = _env_float("INTENTWATCH_WEAPON_KNIFE_CONF", 0.8)
        self._weapon_persist_frames = _env_int("INTENTWATCH_WEAPON_PERSIST_FRAMES", 2)
        self._weapon_max_area_ratio = _env_float("INTENTWATCH_WEAPON_MAX_AREA_RATIO", 0.6)

        # Fallback model thresholds (person-with-weapon model).
        # Keep these stricter because this model has historically produced false positives.
        self._weapon_fallback_conf = _env_float("INTENTWATCH_WEAPON_FALLBACK_CONF", 0.9)
        self._weapon_fallback_persist_frames = _env_int("INTENTWATCH_WEAPON_FALLBACK_PERSIST_FRAMES", 3)

        # Weapon alert spam control
        self._weapon_rearm_seconds = _env_float("INTENTWATCH_WEAPON_REARM_SECONDS", 20.0)
        self._weapon_clear_seconds = _env_float("INTENTWATCH_WEAPON_CLEAR_SECONDS", 2.0)

        # Weapon verification model (secondary model run only at event start).
        self._weapon_verify_conf = _env_float("INTENTWATCH_WEAPON_VERIFY_CONF", 0.85)
        self._weapon_verify_imgsz = _env_int("INTENTWATCH_WEAPON_VERIFY_IMGSZ", 800)
        self._weapon_verify_cooldown_s = _env_float("INTENTWATCH_WEAPON_VERIFY_COOLDOWN_SECONDS", 2.0)
        self._weapon_verify_retry_s = _env_float("INTENTWATCH_WEAPON_VERIFY_RETRY_SECONDS", 0.4)
        self._weapon_verify_window_s = _env_float("INTENTWATCH_WEAPON_VERIFY_WINDOW_SECONDS", 1.5)

        allow_person_labels_raw = str(os.getenv("INTENTWATCH_WEAPON_ALLOW_PERSON_LABELS", "0")).strip().lower()
        self._weapon_allow_person_labels = allow_person_labels_raw in {"1", "true", "yes", "y", "on"}

        labels_env = os.getenv("INTENTWATCH_WEAPON_LABELS")
        # If INTENTWATCH_WEAPON_LABELS is NOT set, do not enforce an allowlist.
        # This improves recall for custom single-class weapon models.
        # If users want strict filtering (e.g., dataset includes non-weapon labels), set it explicitly.
        if labels_env is None:
            self._weapon_labels_allowlist: set[str] | None = None
        else:
            # IMPORTANT: If the env var is present but empty, do NOT treat it as "allow everything".
            # That would re-enable false positives for datasets/models that include non-weapon labels.
            labels_raw = str(labels_env).strip()
            if not labels_raw:
                labels_raw = "pistol,knife,gun,rifle,weapon,firearm"
            self._weapon_labels_allowlist = {s.strip().lower() for s in labels_raw.split(",") if s.strip()}

        # Performance and alert tuning
        # Keep inference image size fixed at 640 for consistent behavior/accuracy.
        # (Do not allow overriding via env var.)
        self._infer_imgsz = 640
        self._infer_half = _env_bool("INTENTWATCH_INFER_HALF", False)
        self._max_frame_height = _env_int("INTENTWATCH_MAX_FRAME_HEIGHT", 720)
        self._jpeg_quality = _env_int("INTENTWATCH_JPEG_QUALITY", 80)
        self._weapon_infer_every_n_frames = _env_int("INTENTWATCH_WEAPON_INFER_EVERY_N_FRAMES", 1)

        self._running_persist_frames = _env_int("INTENTWATCH_RUNNING_PERSIST_FRAMES", 2)
        # Default to a shorter dwell so zone alerts feel real-time.
        # Users can override via env if they want less sensitive zones.
        self._zone_dwell_seconds = _env_float("INTENTWATCH_ZONE_DWELL_SECONDS", 0.5)
        self._zone_cooldown_s = _env_float("INTENTWATCH_ZONE_COOLDOWN_SECONDS", 10.0)

        # History recording (webcam streams only)
        backend_dir = Path(__file__).resolve().parents[1]  # .../backend
        self._history_root_dir = backend_dir / "data" / "history"
        self._history_root_dir.mkdir(parents=True, exist_ok=True)
        self._history_enabled = _env_bool("INTENTWATCH_HISTORY_ENABLED", True)
        self._history_clip_seconds = _env_int("INTENTWATCH_HISTORY_CLIP_SECONDS", 60)
        self._history_upload_supabase = _env_bool("INTENTWATCH_HISTORY_UPLOAD_SUPABASE", False)
        self._history_bucket = (os.getenv("INTENTWATCH_HISTORY_BUCKET") or "footages").strip() or "footages"
        self._history_table = (os.getenv("INTENTWATCH_HISTORY_TABLE") or "footage_clips").strip() or "footage_clips"

        # Alert snapshots
        self._snapshots_root_dir = backend_dir / "data" / "snaps"
        self._snapshots_root_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_enabled = _env_bool("INTENTWATCH_SNAPSHOTS_ENABLED", True)
        self._snapshots_upload_supabase = _env_bool(
            "INTENTWATCH_SNAPSHOT_UPLOAD_SUPABASE",
            bool(self._history_upload_supabase),
        )
        self._snapshots_bucket = (os.getenv("INTENTWATCH_SNAPSHOT_BUCKET") or "Snapshots").strip() or "Snapshots"

    def _ensure_worker_running(self, stream_id: str) -> StreamWorker | None:
        """Ensure a worker exists and is running if we have a remembered source.

        This prevents a "stuck" MJPEG connection when the background thread dies
        (e.g., due to a transient decoder/camera error).
        """
        now = time.time()
        worker = self._workers.get(stream_id)
        source_mode = self._sources.get(stream_id)
        if source_mode is None:
            return worker

        # If there's no worker or it has died, restart with a small backoff.
        if worker is None or (worker is not None and not worker.is_alive()):
            last = self._last_restart_ts.get(stream_id, 0.0)
            if (now - last) < 5.0:
                return worker
            self._last_restart_ts[stream_id] = now
            source, mode = source_mode
            new_worker = StreamWorker(
                stream_id,
                source,
                mode,
                self._model_holder,
                self._weapon_model_holder,
                self._weapon_verify_model_holder,
                self._weapon_fallback_model_holder,
                weapon_conf=self._weapon_conf,
                weapon_knife_conf=self._weapon_knife_conf,
                weapon_persist_frames=self._weapon_persist_frames,
                weapon_max_area_ratio=self._weapon_max_area_ratio,
                weapon_verify_conf=self._weapon_verify_conf,
                weapon_verify_imgsz=self._weapon_verify_imgsz,
                weapon_verify_cooldown_s=self._weapon_verify_cooldown_s,
                weapon_verify_retry_s=self._weapon_verify_retry_s,
                weapon_verify_window_s=self._weapon_verify_window_s,
                weapon_fallback_conf=self._weapon_fallback_conf,
                weapon_fallback_persist_frames=self._weapon_fallback_persist_frames,
                weapon_allow_person_labels=self._weapon_allow_person_labels,
                weapon_labels_allowlist=self._weapon_labels_allowlist,
                infer_imgsz=self._infer_imgsz,
                infer_half=self._infer_half,
                max_frame_height=self._max_frame_height,
                jpeg_quality=self._jpeg_quality,
                weapon_infer_every_n_frames=self._weapon_infer_every_n_frames,
                weapon_rearm_seconds=self._weapon_rearm_seconds,
                weapon_clear_seconds=self._weapon_clear_seconds,
                running_persist_frames=self._running_persist_frames,
                zone_dwell_seconds=self._zone_dwell_seconds,
                zone_cooldown_s=self._zone_cooldown_s,
                history_enabled=self._history_enabled,
                history_root_dir=self._history_root_dir,
                history_clip_seconds=self._history_clip_seconds,
                history_upload_supabase=self._history_upload_supabase,
                history_bucket=self._history_bucket,
                history_table=self._history_table,
                snapshots_enabled=self._snapshots_enabled,
                snapshots_root_dir=self._snapshots_root_dir,
                snapshots_upload_supabase=self._snapshots_upload_supabase,
                snapshots_bucket=self._snapshots_bucket,
            )
            self._workers[stream_id] = new_worker
            new_worker.start()
            return new_worker

        return worker

    def list_streams(self) -> List[str]:
        with self._lock:
            return sorted(self._workers.keys())

    def start(self, stream_id: str, source: str | int, mode: str) -> None:
        with self._lock:
            # Stop any existing worker first
            existing = self._workers.get(stream_id)
            if existing is not None:
                existing.stop()
                existing.join(timeout=2.0)

            worker = StreamWorker(
                stream_id,
                source,
                mode,
                self._model_holder,
                self._weapon_model_holder,
                self._weapon_verify_model_holder,
                self._weapon_fallback_model_holder,
                weapon_conf=self._weapon_conf,
                weapon_knife_conf=self._weapon_knife_conf,
                weapon_persist_frames=self._weapon_persist_frames,
                weapon_max_area_ratio=self._weapon_max_area_ratio,
                weapon_verify_conf=self._weapon_verify_conf,
                weapon_verify_imgsz=self._weapon_verify_imgsz,
                weapon_verify_cooldown_s=self._weapon_verify_cooldown_s,
                weapon_fallback_conf=self._weapon_fallback_conf,
                weapon_fallback_persist_frames=self._weapon_fallback_persist_frames,
                weapon_allow_person_labels=self._weapon_allow_person_labels,
                weapon_labels_allowlist=self._weapon_labels_allowlist,
                infer_imgsz=self._infer_imgsz,
                infer_half=self._infer_half,
                max_frame_height=self._max_frame_height,
                jpeg_quality=self._jpeg_quality,
                weapon_infer_every_n_frames=self._weapon_infer_every_n_frames,
                weapon_rearm_seconds=self._weapon_rearm_seconds,
                weapon_clear_seconds=self._weapon_clear_seconds,
                running_persist_frames=self._running_persist_frames,
                zone_dwell_seconds=self._zone_dwell_seconds,
                zone_cooldown_s=self._zone_cooldown_s,
                history_enabled=self._history_enabled,
                history_root_dir=self._history_root_dir,
                history_clip_seconds=self._history_clip_seconds,
                history_upload_supabase=self._history_upload_supabase,
                history_bucket=self._history_bucket,
                history_table=self._history_table,
                snapshots_enabled=self._snapshots_enabled,
                snapshots_root_dir=self._snapshots_root_dir,
                snapshots_upload_supabase=self._snapshots_upload_supabase,
                snapshots_bucket=self._snapshots_bucket,
            )
            self._workers[stream_id] = worker
            self._sources[stream_id] = (source, mode)
            self._last_restart_ts[stream_id] = time.time()
            worker.start()

    def stop(self, stream_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(stream_id, None)
            self._sources.pop(stream_id, None)
            self._last_restart_ts.pop(stream_id, None)

        if worker is not None:
            worker.stop()
            worker.join(timeout=1.0)

    def get_worker(self, stream_id: str) -> StreamWorker | None:
        with self._lock:
            return self._ensure_worker_running(stream_id)

    def get_status(self, stream_id: str) -> StreamStatus:
        with self._lock:
            worker = self._ensure_worker_running(stream_id)
            src = self._sources.get(stream_id)

        if src is None:
            return {"mode": None, "path": None, "running": False}

        source, mode = src
        return {"mode": mode, "path": source, "running": bool(worker and worker.is_running())}

    def set_zones(self, stream_id: str, zones: List[NormalizedZone]) -> None:
        worker = self.get_worker(stream_id)
        if worker is None:
            return
        worker.set_zones(zones)

    def get_people_count(self, stream_id: str) -> int:
        worker = self.get_worker(stream_id)
        if worker is None:
            return 0
        return int(worker.last_people_count)
