from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

from api.routes import video, alerts, metrics, history

# ✅ FIRST create the app
app = FastAPI(
    title="IntentWatch Backend",
    description="AI-powered CCTV Intent Detection System",
    version="1.0.0"
)

# ✅ THEN add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ THEN register routes
app.include_router(video.router, prefix="/video", tags=["Video"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(history.router, prefix="/history", tags=["History"])
app.include_router(metrics.router, tags=["Metrics"])

@app.get("/")
def root():
    return {"status": "IntentWatch backend running"}
