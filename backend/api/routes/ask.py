from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
import json

from api import rag


router = APIRouter()

_HISTORY_EXTS = {".mp4", ".webm"}


def _load_recent_history_clips(max_items: int = 20) -> list[dict]:
    """Return lightweight history clip metadata for RAG context."""
    backend_dir = Path(__file__).resolve().parents[2]
    history_dir = backend_dir / "data" / "history"
    if not history_dir.exists():
        return []

    clips: list[tuple[float, Path]] = []
    try:
        for p in history_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in _HISTORY_EXTS:
                continue
            try:
                mtime = float(p.stat().st_mtime)
            except Exception:
                mtime = 0.0
            clips.append((mtime, p))
    except Exception:
        return []

    clips.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for mtime, p in clips[: max(1, int(max_items))]:
        parts = p.parts
        stream_id = parts[-3] if len(parts) >= 3 else "unknown"
        date = parts[-2] if len(parts) >= 2 else "unknown"
        filename = parts[-1]

        meta = {}
        meta_path = p.with_suffix(p.suffix + ".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        started_at = meta.get("started_at")
        if isinstance(started_at, (int, float)):
            ts = datetime.fromtimestamp(float(started_at)).isoformat()
        else:
            ts = datetime.fromtimestamp(mtime).isoformat()

        reason = str(meta.get("reason") or "recorded").strip() or "recorded"
        public_url = str(meta.get("public_url") or "").strip() or None

        out.append(
            {
                "id": f"history-{stream_id}-{filename}",
                "type": "HistoryClip",
                "message": f"History clip recorded ({reason}) file={filename} date={date}",
                "timestamp": ts,
                "severity": None,
                "camera": stream_id,
                "snapshot_url": public_url,
            }
        )

    return out


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    k: int | None = Field(default=5, ge=1, le=12)
    max_alerts: int | None = Field(default=1000, ge=50, le=2000)


@router.post("/ask")
def ask(req: AskRequest):
    # Pull alerts from the in-memory store.
    try:
        from api.routes.alerts import alerts, alerts_lock
    except Exception:
        alerts = []
        alerts_lock = None

    max_alerts = int(req.max_alerts or 1000)
    if alerts_lock is None:
        snapshot = []
    else:
        with alerts_lock:
            snapshot = list(alerts[-max_alerts:]) if alerts else []

    # If user asks about history/video/clips, add recent clip metadata to context.
    q_low = str(req.question or "").lower()
    if any(token in q_low for token in ("history", "clip", "video", "footage")):
        snapshot.extend(_load_recent_history_clips(max_items=20))

    answer, sources = rag.answer_question(snapshot, req.question, k=int(req.k or 5))

    return {
        "answer": answer,
        "sources": [
            {
                "id": s.id,
                "type": s.type,
                "message": s.message,
                "timestamp": s.timestamp,
                "severity": s.severity,
                "camera": s.camera,
                "snapshot_url": s.snapshot_url,
            }
            for s in sources
        ],
    }
