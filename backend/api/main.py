from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Load environment variables from local .env files (if present) BEFORE importing routes.
# This ensures modules that read env vars at import/initialization time pick them up.
try:  # pragma: no cover
    from pathlib import Path

    from dotenv import load_dotenv

    backend_dir = Path(__file__).resolve().parents[1]  # .../backend
    workspace_dir = backend_dir.parent

    # Prefer workspace root .env, but also allow backend/.env.
    # Do not override variables already set in the process environment.
    load_dotenv(dotenv_path=workspace_dir / ".env", override=False)
    load_dotenv(dotenv_path=backend_dir / ".env", override=False)
except Exception:
    pass

# Torch compatibility: ensure Ultralytics checkpoints load in environments
# where torch defaults to weights-only loading.
try:  # pragma: no cover
    from api.torch_compat import apply_torch_load_weights_only_default_false

    apply_torch_load_weights_only_default_false()
except Exception:
    pass

from api.routes import video, alerts, metrics, history, iot, ask, demo


def _parse_csv_env(name: str) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]

# ✅ FIRST create the app
app = FastAPI(
    title="IntentWatch Backend",
    description="AI-powered CCTV Intent Detection System",
    version="1.0.0"
)

# ✅ THEN add middleware
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

extra_origins = _parse_csv_env("INTENTWATCH_CORS_ORIGINS")
allow_origins = default_origins + [o for o in extra_origins if o not in default_origins]

allow_origin_regex = (os.getenv("INTENTWATCH_CORS_ORIGIN_REGEX") or "").strip() or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ THEN register routes
app.include_router(video.router, prefix="/video", tags=["Video"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(history.router, prefix="/history", tags=["History"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(iot.router, prefix="/iot", tags=["IoT"])
app.include_router(ask.router, tags=["AI"])
app.include_router(demo.router, prefix="/demo", tags=["Demo"])


@app.on_event("startup")
def _startup_tasks():
    try:
        history.start_history_retention_worker()
    except Exception:
        pass

@app.get("/")
def root():
    return {"status": "IntentWatch backend running"}
