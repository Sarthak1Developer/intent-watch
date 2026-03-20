from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class PurgePlan:
    local_files: list[Path]
    supabase_storage_keys: list[str]
    supabase_row_ids: list[Any]


def _load_env(workspace_root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    # Load workspace .env then backend/.env (best-effort)
    load_dotenv(workspace_root / ".env", override=False)
    load_dotenv(workspace_root / "backend" / ".env", override=False)


def _history_dir(workspace_root: Path) -> Path:
    return workspace_root / "backend" / "data" / "history"


def _iter_local_clip_files(history_dir: Path) -> list[Path]:
    if not history_dir.exists():
        return []

    files: list[Path] = []
    # Delete both the video and its sidecar json.
    files.extend(history_dir.rglob("*.mp4"))
    files.extend(history_dir.rglob("*.mp4.json"))
    return sorted({p for p in files if p.is_file()})


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0:
            return f"{size:.1f}{u}"
        size /= 1024.0
    return f"{size:.1f}PB"


def _chunked(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _get_supabase_client(workspace_root: Path):
    # Import backend api client helper.
    backend_dir = workspace_root / "backend"
    sys.path.insert(0, str(backend_dir))
    from api.supabase_client import get_client  # type: ignore

    return get_client()


def _plan_supabase_deletes(
    workspace_root: Path, *, table: str, want_db: bool, want_storage: bool
) -> tuple[list[str], list[Any]]:
    if not (want_db or want_storage):
        return [], []

    client = _get_supabase_client(workspace_root)
    if client is None:
        return [], []

    storage_keys: list[str] = []
    row_ids: list[Any] = []

    # Page through rows to avoid huge payloads.
    page_size = 1000
    offset = 0
    while True:
        try:
            res = client.table(table).select("id,storage_key").range(offset, offset + page_size - 1).execute()
            data = getattr(res, "data", None)
            if data is None:
                # Some clients return dict-like
                data = res.get("data") if isinstance(res, dict) else None
            if not data:
                break

            for row in data:
                if isinstance(row, dict):
                    if want_storage and row.get("storage_key"):
                        storage_keys.append(str(row["storage_key"]).lstrip("/"))
                    if want_db and ("id" in row):
                        row_ids.append(row["id"])

            if len(data) < page_size:
                break
            offset += page_size
        except Exception:
            break

    # De-dupe
    storage_keys = sorted(set(storage_keys))
    # Keep row_ids order (not required), but de-dupe
    seen: set[str] = set()
    dedup_ids: list[Any] = []
    for rid in row_ids:
        key = str(rid)
        if key in seen:
            continue
        seen.add(key)
        dedup_ids.append(rid)

    return storage_keys, dedup_ids


def build_plan(
    workspace_root: Path,
    *,
    local: bool,
    supabase_db: bool,
    supabase_storage: bool,
) -> PurgePlan:
    hist = _history_dir(workspace_root)
    local_files = _iter_local_clip_files(hist) if local else []

    bucket = (os.getenv("INTENTWATCH_HISTORY_BUCKET") or "footages").strip() or "footages"
    table = (os.getenv("INTENTWATCH_HISTORY_TABLE") or "footage_clips").strip() or "footage_clips"

    # Plan from DB rows (storage_key + ids)
    storage_keys, row_ids = _plan_supabase_deletes(
        workspace_root,
        table=table,
        want_db=supabase_db,
        want_storage=supabase_storage,
    )

    # Note: bucket is used at execution time.
    _ = bucket

    return PurgePlan(local_files=local_files, supabase_storage_keys=storage_keys, supabase_row_ids=row_ids)


def execute_plan(
    workspace_root: Path,
    plan: PurgePlan,
    *,
    local: bool,
    supabase_db: bool,
    supabase_storage: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "local_deleted": 0,
        "local_failed": 0,
        "supabase_storage_deleted": 0,
        "supabase_storage_failed": 0,
        "supabase_db_deleted": 0,
        "supabase_db_failed": 0,
    }

    if local:
        for p in plan.local_files:
            try:
                p.unlink(missing_ok=True)
                results["local_deleted"] += 1
            except Exception:
                results["local_failed"] += 1

    if supabase_storage or supabase_db:
        client = _get_supabase_client(workspace_root)
        if client is None:
            if supabase_storage:
                results["supabase_storage_failed"] = len(plan.supabase_storage_keys)
            if supabase_db:
                results["supabase_db_failed"] = len(plan.supabase_row_ids)
            return results

        bucket = (os.getenv("INTENTWATCH_HISTORY_BUCKET") or "footages").strip() or "footages"
        table = (os.getenv("INTENTWATCH_HISTORY_TABLE") or "footage_clips").strip() or "footage_clips"

        if supabase_storage and plan.supabase_storage_keys:
            # Supabase Storage remove() expects list of paths.
            for chunk in _chunked(plan.supabase_storage_keys, 100):
                try:
                    client.storage.from_(bucket).remove(chunk)
                    results["supabase_storage_deleted"] += len(chunk)
                except Exception:
                    results["supabase_storage_failed"] += len(chunk)

        if supabase_db and plan.supabase_row_ids:
            for chunk in _chunked(plan.supabase_row_ids, 500):
                try:
                    client.table(table).delete().in_("id", chunk).execute()
                    results["supabase_db_deleted"] += len(chunk)
                except Exception:
                    results["supabase_db_failed"] += len(chunk)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge IntentWatch history clips (local + Supabase).")
    parser.add_argument("--local", action="store_true", help="Delete local clip files under backend/data/history")
    parser.add_argument("--supabase-storage", action="store_true", help="Delete Supabase Storage objects for clips")
    parser.add_argument("--supabase-db", action="store_true", help="Delete Supabase DB rows for clips")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable local + Supabase storage + Supabase DB deletions",
    )
    parser.add_argument("--yes", action="store_true", help="Actually perform deletions (otherwise dry-run)")

    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    _load_env(workspace_root)

    local = bool(args.local or args.all)
    supabase_storage = bool(args.supabase_storage or args.all)
    supabase_db = bool(args.supabase_db or args.all)

    hist = _history_dir(workspace_root)
    bucket = (os.getenv("INTENTWATCH_HISTORY_BUCKET") or "footages").strip() or "footages"
    table = (os.getenv("INTENTWATCH_HISTORY_TABLE") or "footage_clips").strip() or "footage_clips"

    plan = build_plan(
        workspace_root,
        local=local,
        supabase_db=supabase_db,
        supabase_storage=supabase_storage,
    )

    # Summarize plan
    local_bytes = 0
    for p in plan.local_files:
        try:
            local_bytes += int(p.stat().st_size)
        except Exception:
            pass

    print("=== IntentWatch clip purge ===")
    print(f"Workspace: {workspace_root}")
    print(f"Local history dir: {hist}")
    print(f"Supabase bucket: {bucket}")
    print(f"Supabase table: {table}")
    print()
    print("Planned deletions:")
    print(f"- Local files: {len(plan.local_files)} ({_fmt_bytes(local_bytes)})")
    print(f"- Supabase Storage keys (from DB): {len(plan.supabase_storage_keys)}")
    print(f"- Supabase DB rows: {len(plan.supabase_row_ids)}")

    if not args.yes:
        print()
        print("Dry-run only. Re-run with --yes to actually delete.")
        return 0

    print()
    print("Deleting...")
    started = time.time()

    results = execute_plan(
        workspace_root,
        plan,
        local=local,
        supabase_db=supabase_db,
        supabase_storage=supabase_storage,
    )

    elapsed = time.time() - started
    print("Done.")
    print(f"Elapsed: {elapsed:.2f}s")
    for k, v in results.items():
        print(f"- {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
