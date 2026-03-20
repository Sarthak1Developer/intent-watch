from __future__ import annotations

import os
import threading
import urllib.parse
import urllib.request


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _telegram_config() -> tuple[str, str] | None:
    token = (os.getenv("INTENTWATCH_TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("INTENTWATCH_TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return None
    return token, chat_id


def should_notify(alert: dict) -> bool:
    """Return True if this alert should be forwarded to phone notification channels."""
    if not _env_bool("INTENTWATCH_PHONE_ALERTS_ENABLED", default=False):
        return False

    alert_type = str(alert.get("type", "") or "").strip()

    # Default: only high-value alerts
    configured = os.getenv("INTENTWATCH_PHONE_ALERT_TYPES")
    if configured:
        allowed = {t.strip().lower() for t in configured.split(",") if t.strip()}
    else:
        allowed = {"weapon", "unattended bag"}

    return alert_type.lower() in allowed


def _telegram_send_message(text: str) -> None:
    # Explicit opt-in only. User asked for notifications from the website (browser), not Telegram.
    if not _env_bool("INTENTWATCH_TELEGRAM_ENABLED", default=False):
        return

    cfg = _telegram_config()
    if cfg is None:
        return

    token, chat_id = cfg
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    # Keep timeout short so we never block the video loop.
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


def _telegram_send_photo(photo_url: str, caption: str) -> None:
    if not _env_bool("INTENTWATCH_TELEGRAM_ENABLED", default=False):
        return

    cfg = _telegram_config()
    if cfg is None:
        return

    token, chat_id = cfg
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "photo": photo_url,
            "caption": caption,
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


def notify_async(alert: dict) -> None:
    """Send alert to phone notification channels (best-effort, non-blocking)."""
    if not should_notify(alert):
        return

    # Keep the message compact and phone-friendly.
    alert_type = str(alert.get("type", "") or "Alert").strip() or "Alert"
    message = str(alert.get("message", "") or "").strip()
    severity = str(alert.get("severity", "") or "").strip()
    camera = str(alert.get("camera", "") or "").strip()
    time_s = str(alert.get("time", "") or "").strip()
    snapshot_url = str(alert.get("snapshot_url", "") or "").strip()

    parts: list[str] = [f"{alert_type}"]
    if severity:
        parts.append(f"({severity})")
    if camera:
        parts.append(f"camera={camera}")
    if time_s:
        parts.append(f"at {time_s}")

    header = " ".join(parts)
    text = header + (f"\n{message}" if message else "")

    # Telegram can only fetch publicly reachable URLs.
    can_send_photo = snapshot_url.lower().startswith("http://") or snapshot_url.lower().startswith("https://")

    def _run() -> None:
        try:
            if can_send_photo:
                _telegram_send_photo(snapshot_url, text)
            else:
                _telegram_send_message(text)
        except Exception:
            # Best-effort: never let notification errors break alerting.
            return

    threading.Thread(target=_run, name="phone-notify", daemon=True).start()
