from __future__ import annotations

from datetime import datetime, time
import os
import threading
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from api.routes.alerts import add_alert

router = APIRouter()

_override_lock = threading.Lock()
_override_active_start: time | None = None
_override_active_end: time | None = None


def _parse_hhmm(value: str | None) -> time | None:
    v = (value or "").strip()
    if not v:
        return None
    try:
        parts = v.split(":")
        if len(parts) != 2:
            return None
        hh = int(parts[0])
        mm = int(parts[1])
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return None
        return time(hour=hh, minute=mm)
    except Exception:
        return None


def _within_active_window(now: datetime) -> bool:
    """Return True if IoT alerting is active at `now`.

    Controlled by:
    - INTENTWATCH_IOT_ACTIVE_START=HH:MM
    - INTENTWATCH_IOT_ACTIVE_END=HH:MM

    If either is missing/invalid, defaults to "always active".
    """

    with _override_lock:
        start = _override_active_start
        end = _override_active_end

    if start is None or end is None:
        start = _parse_hhmm(os.getenv("INTENTWATCH_IOT_ACTIVE_START"))
        end = _parse_hhmm(os.getenv("INTENTWATCH_IOT_ACTIVE_END"))

    if start is None or end is None:
        return True

    n = now.time()

    # Same-day window (e.g., 09:00-17:00)
    if start < end:
        return start <= n < end

    # Overnight window (e.g., 22:00-06:00)
    if start > end:
        return n >= start or n < end

    # start == end -> interpret as always active
    return True


def _require_shared_secret(provided: str | None) -> None:
    secret = (os.getenv("INTENTWATCH_IOT_SHARED_SECRET") or "").strip()
    if not secret:
        # Not configured: allow all requests (useful for local prototyping).
        return

    if not provided or str(provided).strip() != secret:
        raise HTTPException(status_code=401, detail="Invalid IoT shared secret")


class DoorEventIn(BaseModel):
    device_id: str = Field(default="door-1", max_length=64)
    state: Literal["open", "closed", "tamper"]

    # Optional telemetry (helps debugging)
    battery_v: float | None = None
    rssi: int | None = None
    ts: str | None = None  # device timestamp (best-effort)


class IoTActiveWindowIn(BaseModel):
    active_start: str | None = None  # HH:MM
    active_end: str | None = None  # HH:MM


@router.get("/ping")
def ping():
    return {"ok": True, "service": "iot"}


@router.post("/door")
def door_event(
    body: DoorEventIn,
    x_intentwatch_key: str | None = Header(default=None),
    x_iot_key: str | None = Header(default=None),
):
    # Accept either header name to make firmware simpler.
    _require_shared_secret(x_intentwatch_key or x_iot_key)

    now = datetime.now()
    if not _within_active_window(now):
        return {"ok": True, "status": "ignored", "reason": "outside_active_window"}

    device_id = (body.device_id or "door-1").strip() or "door-1"

    # Map states to alert metadata.
    if body.state == "open":
        severity = "High"
        message = f"Door opened (device_id={device_id})"
    elif body.state == "tamper":
        severity = "Critical"
        message = f"Door tamper detected (device_id={device_id})"
    else:
        # closed
        severity = "Low"
        message = f"Door closed (device_id={device_id})"

    # Treat IoT as its own alert type.
    # NOTE: If you want phone alerts, include "door" in INTENTWATCH_PHONE_ALERT_TYPES.
    add_alert("door", message, severity=severity, camera=device_id)

    return {
        "ok": True,
        "status": "alerted",
        "received_at": now.isoformat(timespec="seconds"),
        "device_id": device_id,
        "state": body.state,
    }


@router.get("/config")
def get_config():
    with _override_lock:
        start = _override_active_start
        end = _override_active_end

    return {
        "active_start": start.strftime("%H:%M") if start else None,
        "active_end": end.strftime("%H:%M") if end else None,
    }


@router.post("/config")
def update_config(
    body: IoTActiveWindowIn,
    x_intentwatch_key: str | None = Header(default=None),
    x_iot_key: str | None = Header(default=None),
):
    _require_shared_secret(x_intentwatch_key or x_iot_key)

    start = _parse_hhmm(body.active_start)
    end = _parse_hhmm(body.active_end)

    # If either value is missing/invalid, clear overrides (fallback to env vars).
    if (body.active_start and start is None) or (body.active_end and end is None):
        raise HTTPException(status_code=400, detail="Invalid active window (expected HH:MM)")

    with _override_lock:
        _override_active_start = start
        _override_active_end = end

    return {
        "ok": True,
        "active_start": start.strftime("%H:%M") if start else None,
        "active_end": end.strftime("%H:%M") if end else None,
    }
