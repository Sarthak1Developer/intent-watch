from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict
import os
import threading
import time

import cv2
from ultralytics import YOLO


# Disable FFmpeg threading to avoid codec errors
os.environ.setdefault("FFREPORT", "file=/dev/null")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

# Limit OpenCV internal threading to reduce race conditions
cv2.setNumThreads(1)


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
        model_holder: _ModelHolder,
        weapon_model_holder: _ModelHolder | None = None,
        *,
        conf: float = 0.4,
        fps_limit: int = 30,
    ):
        self.stream_id = stream_id
        self.source = source
        self._model_holder = model_holder
        self._weapon_model_holder = weapon_model_holder
        self._conf = conf
        self._fps_limit = fps_limit

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

    def _emit_alert(self, add_alert_fn, alert_type: str, message: str, *, cooldown_s: float = 5.0, severity: str | None = None) -> None:
        key = f"{alert_type}:{message}" if message else alert_type
        now = time.time()
        with self._alert_lock:
            last = self._last_alert_ts.get(key)
            if last is not None and (now - last) < cooldown_s:
                return
            self._last_alert_ts[key] = now

        # Defer to shared alert store
        add_alert_fn(alert_type, message, severity=severity, camera=self.stream_id)

    def _run(self) -> None:
        cap_local = None
        try:
            cap_local = _open_capture(self.source)
            self._cap = cap_local

            if isinstance(self.source, int) and cap_local is not None:
                cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap_local is None or not cap_local.isOpened():
                return

            model = self._model_holder.get()

            last_yield = 0.0

            # Import here to avoid circular imports
            from api.routes.alerts import add_alert

            while not self._stop_event.is_set():
                ret, frame = cap_local.read()
                if not ret or frame is None:
                    if isinstance(self.source, str):
                        cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap_local.read()
                        if not ret or frame is None:
                            break
                    else:
                        break

                # Basic FPS limiting
                now = time.time()
                if self._fps_limit > 0:
                    min_dt = 1.0 / float(self._fps_limit)
                    if (now - last_yield) < min_dt:
                        time.sleep(max(min_dt - (now - last_yield), 0.0))
                        now = time.time()
                    last_yield = now

                # Resize large frames
                h, w = frame.shape[:2]
                if h > 720:
                    scale = 720 / h
                    w = int(w * scale)
                    frame = cv2.resize(frame, (w, 720), interpolation=cv2.INTER_LINEAR)
                    h = 720

                # Run YOLO
                results = model(frame, conf=self._conf, verbose=False)

                persons: List[Tuple[int, int, int, int, int, int]] = []
                bags: List[Tuple[int, int, int, int]] = []
                weapons: List[Tuple[int, int, int, int, str]] = []

                weapon_model = (
                    self._weapon_model_holder.get_optional() if self._weapon_model_holder is not None else None
                )

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
                    weapon_results = weapon_model(frame, conf=self._conf, verbose=False)
                    for wr in weapon_results:
                        for box in wr.boxes:
                            cls = int(box.cls[0])
                            label = str(weapon_model.names.get(cls, cls))
                            label_norm = str(label).strip().lower()
                            # Roboflow exports sometimes contain placeholder labels like '-' or 'undefined'.
                            # Ignore these so we don't spam Weapon alerts for non-weapon classes.
                            if label_norm in {"-", "undefined", "background"}:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            weapons.append((x1, y1, x2, y2, label))

                # Weapon detection (only works if model has these classes)
                for (x1, y1, x2, y2, label) in weapons:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label.upper()}!", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self._emit_alert(add_alert, "Weapon", f"{label} detected", cooldown_s=10.0, severity="Critical")

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
                        x_dist = abs(cx - prev_x)
                        if x_dist < 80 and x_dist < min_dist:
                            min_dist = x_dist
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
                    self._emit_alert(add_alert, "Running", "Person running detected", cooldown_s=5.0, severity="Low")

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
                        self._emit_alert(add_alert, "Unattended Bag", "Suspicious item detected", cooldown_s=10.0, severity="High")
                        self._bag_start_time = now
                else:
                    self._bag_start_time = None

                # Zones overlay + zone alerts
                zones = self.get_zones()
                if zones:
                    for z in zones:
                        zx1 = int(_clamp01(z.x) * w)
                        zy1 = int(_clamp01(z.y) * h)
                        zx2 = int(_clamp01(z.x + z.width) * w)
                        zy2 = int(_clamp01(z.y + z.height) * h)
                        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 255), 2)
                        cv2.putText(frame, z.name, (zx1, max(zy1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        for (_, _, _, _, pcx, pcy) in persons:
                            if zx1 <= pcx <= zx2 and zy1 <= pcy <= zy2:
                                sev = (z.severity or "medium").capitalize()
                                self._emit_alert(
                                    add_alert,
                                    "Zone",  # keep type stable
                                    f"{z.name}: person detected in zone",
                                    cooldown_s=5.0,
                                    severity=sev,
                                )
                                break

                # Encode JPEG and publish
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    data = buf.tobytes()
                    with self._latest_lock:
                        self._latest_jpeg = data
                        self._latest_frame_ts = now

        except Exception:
            # swallow; status endpoints can show it as stopped
            pass
        finally:
            try:
                if cap_local is not None:
                    cap_local.release()
            except Exception:
                pass
            self._cap = None


class StreamManager:
    def __init__(self, model_candidates: Iterable[Path], weapon_model_candidates: Iterable[Path] | None = None):
        self._model_holder = _ModelHolder(model_candidates)
        self._weapon_model_holder = _ModelHolder(weapon_model_candidates) if weapon_model_candidates else None
        self._lock = threading.Lock()
        self._workers: Dict[str, StreamWorker] = {}
        self._sources: Dict[str, Tuple[str | int, str]] = {}  # id -> (source, mode)
        self._last_restart_ts: Dict[str, float] = {}

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
            new_worker = StreamWorker(stream_id, source, self._model_holder, self._weapon_model_holder)
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

            worker = StreamWorker(stream_id, source, self._model_holder, self._weapon_model_holder)
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
