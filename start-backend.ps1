# IntentWatch - Start Backend Server
# This script activates the virtual environment and starts the FastAPI backend

Write-Host "Starting IntentWatch Backend..." -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create a virtual environment first with: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Check if backend directory exists
if (-Not (Test-Path ".\backend\api\main.py")) {
    Write-Host "Error: Backend files not found!" -ForegroundColor Red
    exit 1
}

# Start the FastAPI server
$BackendPort = if ($env:INTENTWATCH_BACKEND_PORT) { [int]$env:INTENTWATCH_BACKEND_PORT } else { 8000 }
Write-Host "Starting FastAPI server on http://localhost:$BackendPort..." -ForegroundColor Green
Write-Host ""

# Optional: auto-wire trained weapon model if present.
if (-Not $env:INTENTWATCH_WEAPON_MODEL_PATH) {
    $WeaponModelPath = Join-Path $PSScriptRoot "runs_weapon\weapon80_20\weights\best.pt"
    if (Test-Path $WeaponModelPath) {
        $env:INTENTWATCH_WEAPON_MODEL_PATH = (Resolve-Path $WeaponModelPath).Path
        Write-Host "Using trained weapon model: $env:INTENTWATCH_WEAPON_MODEL_PATH" -ForegroundColor Yellow
        Write-Host ""
    }
}

Push-Location backend
try {
    # Avoid --reload on Windows here; it can spawn reloader processes that keep ports open and lead to 'buffering' / hung servers.
    & ..\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port $BackendPort
} finally {
    Pop-Location
}
