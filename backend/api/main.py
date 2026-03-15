from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import video, alerts, metrics

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
app.include_router(metrics.router, tags=["Metrics"])

@app.get("/")
def root():
    return {"status": "IntentWatch backend running"}
