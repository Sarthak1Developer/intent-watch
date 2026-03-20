from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import os
import re
import json

router = APIRouter()

BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
HISTORY_DIR = BACKEND_DIR / "data" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_stream_id(stream_id: str) -> str:
    s = str(stream_id or "").strip()
    if not s or "/" in s or "\\" in s or ".." in s:
        raise HTTPException(status_code=400, detail="Invalid stream_id")
    return s


def _safe_date(date: str) -> str:
    d = str(date or "").strip()
    if not _DATE_RE.match(d):
        raise HTTPException(status_code=400, detail="Invalid date (expected YYYY-MM-DD)")
    return d


def _safe_filename(name: str) -> str:
    n = os.path.basename(str(name or "").strip())
    if not n or n in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if any(sep in n for sep in ["/", "\\"]):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return n


@router.get("/dates")
def list_dates(stream_id: str = "primary"):
    sid = _safe_stream_id(stream_id)
    base = HISTORY_DIR / sid
    if not base.exists():
        return {"stream_id": sid, "dates": []}

    dates: list[str] = []
    for p in base.iterdir():
        if p.is_dir() and _DATE_RE.match(p.name):
            dates.append(p.name)
    dates.sort(reverse=True)
    return {"stream_id": sid, "dates": dates}


@router.get("/supabase/status")
def supabase_status():
    """Report whether Supabase is configured for this backend process.

    This endpoint never returns secrets; it only reports boolean presence.
    """
    bucket = (os.getenv("INTENTWATCH_HISTORY_BUCKET") or "footages").strip() or "footages"
    upload_enabled = _env_bool("INTENTWATCH_HISTORY_UPLOAD_SUPABASE", False)
    has_url = bool((os.getenv("SUPABASE_URL") or "").strip())
    has_key = bool(
        (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        or (os.getenv("SUPABASE_KEY") or "").strip()
        or (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    )

    configured = False
    try:
        from api.supabase_client import is_configured

        configured = bool(is_configured())
    except Exception:
        configured = False

    return {
        "upload_enabled": upload_enabled,
        "bucket": bucket,
        "has_url": has_url,
        "has_key": has_key,
        "configured": configured,
    }


@router.get("/clips")
def list_clips(stream_id: str = "primary", date: str | None = None):
    sid = _safe_stream_id(stream_id)
    if not date:
        raise HTTPException(status_code=400, detail="Missing date")
    d = _safe_date(date)

    day_dir = HISTORY_DIR / sid / d
    if not day_dir.exists():
        return {"stream_id": sid, "date": d, "clips": []}

    clips = []
    for p in sorted(day_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            size_bytes = int(p.stat().st_size)
        except Exception:
            continue

        # Skip tiny/corrupt stub files (e.g., if a writer was created but never wrote frames).
        if size_bytes < 1024:
            continue

        meta_path = p.with_suffix(p.suffix + ".json")
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        clips.append(
            {
                "filename": p.name,
                "size_bytes": size_bytes,
                "mtime": int(p.stat().st_mtime),
                "public_url": (meta or {}).get("public_url"),
                "url": f"/history/clip/{sid}/{d}/{p.name}",
            }
        )

    return {"stream_id": sid, "date": d, "clips": clips}


@router.get("/clip/{stream_id}/{date}/{filename}")
def get_clip(stream_id: str, date: str, filename: str, request: Request):
    sid = _safe_stream_id(stream_id)
    d = _safe_date(date)
    fn = _safe_filename(filename)

    p = (HISTORY_DIR / sid / d / fn).resolve()
    # Ensure resolved path is still under HISTORY_DIR
    if HISTORY_DIR.resolve() not in p.parents:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Clip not found")

    file_size = int(p.stat().st_size)
    range_header = request.headers.get("range") or request.headers.get("Range")

    # Support byte-range requests for HTML5 video playback.
    # Example: Range: bytes=0-1023
    if range_header:
        m = re.match(r"^bytes=(\d+)-(\d*)$", range_header.strip())
        if not m:
            raise HTTPException(status_code=416, detail="Invalid Range header")

        start = int(m.group(1))
        end_s = m.group(2)
        end = int(end_s) if end_s else (file_size - 1)

        if start >= file_size:
            raise HTTPException(status_code=416, detail="Range start out of bounds")
        end = min(end, file_size - 1)
        if end < start:
            raise HTTPException(status_code=416, detail="Invalid Range")

        chunk_size = 1024 * 1024
        length = (end - start) + 1

        def _iterfile():
            with open(p, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    data = f.read(min(chunk_size, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        }
        return StreamingResponse(_iterfile(), status_code=206, media_type="video/mp4", headers=headers)

    # No range: serve full file. Still advertise Accept-Ranges.
    return FileResponse(
        str(p),
        media_type="video/mp4",
        filename=fn,
        headers={"Accept-Ranges": "bytes"},
    )
