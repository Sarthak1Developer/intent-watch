from fastapi import APIRouter
from datetime import datetime, timedelta
import threading
import uuid

from api.phone_notify import notify_async

router = APIRouter()

alerts = []
alerts_lock = threading.Lock()

# Keep the alert store bounded so /alerts/live stays fast.
MAX_ALERTS_STORED = 2000
MAX_ALERTS_LIVE_RESPONSE = 200

def add_alert(alert_type, message, *, severity: str | None = None, camera: str | None = None):
    """Add alert to the list."""
    now = datetime.now()
    alert = {
        "id": uuid.uuid4().hex,
        "type": alert_type,
        "message": message,
        "severity": severity,
        "camera": camera,
        # Human-friendly time for the UI
        "time": now.strftime("%H:%M:%S"),
        # Machine-friendly timestamp for analytics bucketing
        "timestamp": now.isoformat(),
    }
    with alerts_lock:
        alerts.append(alert)
        if len(alerts) > MAX_ALERTS_STORED:
            # Drop oldest alerts to keep memory bounded.
            del alerts[: len(alerts) - MAX_ALERTS_STORED]
    print(f"[ALERT] {alert_type}: {message}")

    # Best-effort, non-blocking phone notifications.
    notify_async(alert)

def clear_all_alerts():
    """Clear all alerts"""
    global alerts
    with alerts_lock:
        alerts = []

@router.get("/live")
def get_alerts():
    """Get all alerts"""
    with alerts_lock:
        if not alerts:
            return []
        return list(alerts[-MAX_ALERTS_LIVE_RESPONSE:])

@router.get("/analytics")
def get_analytics():
    """Get aggregated analytics data"""
    def severity_for_type(alert_type: str) -> str:
        t = (alert_type or "").lower()
        if "bag" in t:
            return "High"
        if "loiter" in t:
            return "Medium"
        if "running" in t:
            return "Low"
        if "weapon" in t:
            return "Critical"
        if "zone" in t:
            return "Medium"
        return "Low"

    now = datetime.now()
    with alerts_lock:
        snapshot = list(alerts)

    alert_counts = {}
    severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}

    # Build totals by type + severity.
    for alert in snapshot:
        alert_type = alert.get("type", "Unknown")
        alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        sev = (alert.get("severity") or "").strip() or severity_for_type(alert_type)
        sev = sev[:1].upper() + sev[1:].lower() if sev else "Low"
        if sev not in severity_counts:
            sev = severity_for_type(alert_type)
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Time-bucketed analytics (from ISO timestamps).
    def parse_ts(a: dict) -> datetime | None:
        ts = a.get("timestamp")
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    # Last 7 days (including today)
    day_keys = []
    by_day_map = {}
    for i in range(6, -1, -1):
        d = (now - timedelta(days=i)).date()
        key = d.isoformat()
        day_keys.append(key)
        by_day_map[key] = {"date": key, "day": d.strftime("%a"), "alerts": 0}

    # Last 24 hours hourly buckets
    by_hour_map = {}
    hour_keys = []
    start_hour = (now - timedelta(hours=23)).replace(minute=0, second=0, microsecond=0)
    for i in range(24):
        h = start_hour + timedelta(hours=i)
        key = h.isoformat(timespec="minutes")
        hour_keys.append(key)
        by_hour_map[key] = {"hour": h.strftime("%H:00"), "alerts": 0}

    # Threat trends for last 7 days
    threat_types = ["Running", "Loitering", "Unattended Bag"]
    threat_by_day = {k: {"date": k, "day": datetime.fromisoformat(k).strftime("%a"), **{t: 0 for t in threat_types}} for k in day_keys}

    for alert in snapshot:
        dt = parse_ts(alert)
        if dt is None:
            continue

        # Daily bucket
        dkey = dt.date().isoformat()
        if dkey in by_day_map:
            by_day_map[dkey]["alerts"] += 1

        # Hourly bucket (round down to hour)
        hour_dt = dt.replace(minute=0, second=0, microsecond=0)
        hkey = hour_dt.isoformat(timespec="minutes")
        if hkey in by_hour_map:
            by_hour_map[hkey]["alerts"] += 1

        # Threat type per day
        at = alert.get("type")
        if at in threat_types and dkey in threat_by_day:
            threat_by_day[dkey][at] += 1

    return {
        "total": len(snapshot),
        "counts": alert_counts,
        "severity": severity_counts,
        "by_day": [by_day_map[k] for k in day_keys],
        "by_hour": [by_hour_map[k] for k in hour_keys],
        "threat_trends": [threat_by_day[k] for k in day_keys],
        "recent": snapshot[-20:] if snapshot else [],
    }

@router.post("/clear")
def clear_alerts():
    """Clear all alerts"""
    global alerts
    with alerts_lock:
        alerts = []
    print("✓ Alerts cleared")
    return {"message": "Alerts cleared"}
