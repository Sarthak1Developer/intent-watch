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
if ($env:INTENTWATCH_WEAPON_MODEL_PATH) {
    # If user set a path but it doesn't exist, treat it as unset.
    if (-Not (Test-Path $env:INTENTWATCH_WEAPON_MODEL_PATH)) {
        Write-Host "INTENTWATCH_WEAPON_MODEL_PATH was set but not found; auto-detecting model..." -ForegroundColor Yellow
        Remove-Item Env:INTENTWATCH_WEAPON_MODEL_PATH -ErrorAction SilentlyContinue
    }
}

if (-Not $env:INTENTWATCH_WEAPON_MODEL_PATH) {
    $WeaponCandidates = @(
        # Existing trained model in this workspace
        (Join-Path $PSScriptRoot "runs\detect\runs_weapon\combined_yolov8s_gpu\weights\best.pt"),
        (Join-Path $PSScriptRoot "runs\detect\runs_weapon\combined_ft_from_archive1_best\weights\best.pt"),
        # Preferred new multi-class model
        (Join-Path $PSScriptRoot "runs\detect\runs_weapon\weapon_types_v1\weights\best.pt"),
        (Join-Path $PSScriptRoot "runs_weapon\weapon_types_v1\weights\best.pt")
    )

    # Legacy weapon80_20 is noisy; only include it when explicitly allowed.
    $AllowLegacy = $false
    if ($env:INTENTWATCH_WEAPON_ALLOW_LEGACY_MODEL) {
        $v = $env:INTENTWATCH_WEAPON_ALLOW_LEGACY_MODEL.ToString().Trim().ToLower()
        $AllowLegacy = @('1','true','yes','y','on') -contains $v
    }
    if ($AllowLegacy) {
        $WeaponCandidates += @(
            (Join-Path $PSScriptRoot "runs\detect\runs_weapon\weapon80_20\weights\best.pt"),
            (Join-Path $PSScriptRoot "runs_weapon\weapon80_20\weights\best.pt")
        )
    }

    # If none of the known candidates exist, pick the newest best.pt from likely Ultralytics output roots.
    # Note: Ultralytics can sometimes nest output folders (e.g., runs\detect\runs\detect\...).
    $AutoDetectRoots = @(
        (Join-Path $PSScriptRoot "runs\detect\runs_weapon"),
        (Join-Path $PSScriptRoot "runs\detect\runs\detect\runs_weapon"),
        (Join-Path $PSScriptRoot "runs\detect")
    ) | Select-Object -Unique

    foreach ($root in $AutoDetectRoots) {
        if (-Not (Test-Path $root)) { continue }

        $Newest = Get-ChildItem -Path $root -Recurse -Filter best.pt -ErrorAction SilentlyContinue |
            Where-Object {
                $_.FullName -like "*runs_weapon*" -and (
                    $AllowLegacy -or ($_.FullName -notlike "*weapon80_20*")
                )
            } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($Newest) {
            $WeaponCandidates = @($Newest.FullName) + $WeaponCandidates
            break
        }
    }

    foreach ($cand in $WeaponCandidates) {
        if (Test-Path $cand) {
            $env:INTENTWATCH_WEAPON_MODEL_PATH = (Resolve-Path $cand).Path
            Write-Host "Using trained weapon model: $env:INTENTWATCH_WEAPON_MODEL_PATH" -ForegroundColor Yellow
            Write-Host ""
            break
        }
    }
}

Push-Location backend
try {
    # Avoid --reload on Windows here; it can spawn reloader processes that keep ports open and lead to 'buffering' / hung servers.
    & ..\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port $BackendPort
} finally {
    Pop-Location
}
