from __future__ import annotations

from fastapi import APIRouter
import time

from api.routes import video
from api.routes.alerts import alerts, alerts_lock

router = APIRouter()

_started_at = time.time()


def _format_uptime(seconds: int) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


@router.get("/metrics")
def get_metrics():
    now = time.time()
    uptime_seconds = int(now - _started_at)

    stream_ids = video.manager.list_streams()
    statuses = [video.manager.get_status(sid) for sid in stream_ids]

    running = [s for s in statuses if s.get("running")]
    cameras_online = sum(1 for s in running if s.get("mode") == "camera")
    streams_running = len(running)

    people_detected = 0
    for sid in stream_ids:
        st = video.manager.get_status(sid)
        if st.get("running"):
            people_detected += video.manager.get_people_count(sid)

    with alerts_lock:
        active_alerts = len(alerts)

    return {
        "uptime_seconds": uptime_seconds,
        "uptime": _format_uptime(uptime_seconds),
        "streams_running": streams_running,
        "cameras_online": cameras_online,
        "people_detected": people_detected,
        "active_alerts": active_alerts,
    }
