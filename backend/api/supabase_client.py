from __future__ import annotations

import os
from typing import Any, Optional

try:
    # supabase-py
    from supabase import Client, create_client
except Exception:  # pragma: no cover
    Client = Any  # type: ignore
    create_client = None  # type: ignore


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_url_and_key() -> tuple[str, str] | None:
    url = (os.getenv("SUPABASE_URL") or "").strip()

    # Prefer server-side keys if present.
    key = (
        (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        or (os.getenv("SUPABASE_KEY") or "").strip()
        or (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    )

    if not url or not key:
        return None
    return url, key


def is_configured() -> bool:
    if not _env_bool("INTENTWATCH_SUPABASE_ENABLED", default=True):
        return False
    return _get_url_and_key() is not None and create_client is not None


_client: Client | None = None


def get_client() -> Client | None:
    global _client
    if _client is not None:
        return _client

    if not is_configured():
        return None

    cfg = _get_url_and_key()
    if cfg is None:
        return None

    url, key = cfg
    try:
        _client = create_client(url, key)
    except Exception:
        _client = None
    return _client


def upload_file(bucket: str, storage_key: str, file_path: str, *, content_type: str | None = None) -> str | None:
    """Upload a local file to Supabase Storage and return a public URL (if available).

    This is best-effort: returns None on any error.
    """
    client = get_client()
    if client is None:
        return None

    bucket = str(bucket or "").strip() or "footages"
    storage_key = str(storage_key or "").lstrip("/")
    if not storage_key:
        return None

    try:
        # Open the file in binary mode and pass the file object directly to Supabase.
        with open(file_path, "rb") as f:
            # supabase-py accepts file-like objects or bytes for upload.
            options: dict[str, Any] = {}
            if content_type:
                options["content-type"] = content_type

            client.storage.from_(bucket).upload(storage_key, f, file_options=options or None)
        public = client.storage.from_(bucket).get_public_url(storage_key)
        return str(public) if public else None
    except Exception:
        return None


def insert_row(table: str, data: dict[str, Any]) -> bool:
    """Insert a row into Supabase Postgres via the REST API."""
    client = get_client()
    if client is None:
        return False

    try:
        client.table(str(table)).insert(data).execute()
        return True
    except Exception:
        return False
