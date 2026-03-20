"""Supabase connectivity smoke test for IntentWatch.

This script:
- Reads Supabase credentials from environment variables
- Attempts a small upload to the configured Storage bucket
- Prints only success/failure (no secrets)

Required env vars:
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY (recommended) or SUPABASE_KEY / SUPABASE_ANON_KEY

Optional env vars:
- INTENTWATCH_HISTORY_BUCKET (defaults to 'footages')
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    # Ensure backend package imports work when running from repo root.
    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    if backend_dir.exists():
        sys.path.insert(0, str(backend_dir))

    # Load .env files if present (so this script matches backend behavior)
    try:
        from dotenv import load_dotenv

        workspace_dir = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=workspace_dir / ".env", override=False)
        load_dotenv(dotenv_path=backend_dir / ".env", override=False)
    except Exception:
        pass

    # Import from backend package
    try:
        from api.supabase_client import get_client, insert_row, is_configured, upload_file  # type: ignore[import-not-found]
    except Exception as e:
        print("import_failed= True")
        print(f"reason= {type(e).__name__}")
        return 2

    has_url = bool(os.getenv("SUPABASE_URL"))
    has_key = bool(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    bucket = (os.getenv("INTENTWATCH_HISTORY_BUCKET") or "footages").strip() or "footages"
    table = (os.getenv("INTENTWATCH_HISTORY_TABLE") or "footage_clips").strip() or "footage_clips"

    print(f"has_SUPABASE_URL= {has_url}")
    print(f"has_SUPABASE_KEY= {has_key}")
    print(f"bucket= {bucket}")
    print(f"table= {table}")
    print(f"supabase_configured= {is_configured()}")

    if not is_configured():
        print("upload_ok= False")
        print("note= Not configured (missing env vars or supabase not installed)")
        return 1

    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"intentwatch supabase smoke test")
        tmp_path = f.name

    storage_key = f"_intentwatch_test/{int(time.time())}.txt"
    url = upload_file(bucket, storage_key, tmp_path, content_type="text/plain")

    print(f"upload_ok= {bool(url)}")
    print(f"storage_key= {storage_key}")
    print(f"public_url_prefix= {(url[:50] + '...') if url else None}")

    # DB check (optional): verify table exists and attempt an insert.
    client = get_client()
    if client is None:
        print("db_table_ok= False")
        print("db_insert_ok= False")
        return 0 if url else 1

    try:
        client.table(table).select("id").limit(1).execute()
        print("db_table_ok= True")
    except Exception as e:
        print("db_table_ok= False")
        print(f"db_error_prefix= {str(e)[:120]}")
        print("db_insert_ok= False")
        return 0 if url else 1

    payload = {
        "stream_id": "primary",
        "storage_key": f"_intentwatch_db_test/{int(time.time())}.txt",
        "public_url": url,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ok = insert_row(table, payload)
    print(f"db_insert_ok= {bool(ok)}")

    return 0 if url else 1


if __name__ == "__main__":
    raise SystemExit(main())
