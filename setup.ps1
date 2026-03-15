# IntentWatch - Setup Script
# This script sets up the complete development environment.
# Note: Keep console output ASCII-only to avoid parsing/encoding issues.

$ErrorActionPreference = "Stop"

function Write-Section([string]$text) {
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ("  " + $text) -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
}

Write-Section "IntentWatch Setup"

# Step 1: Check Python installation
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host ("  [OK] Found: " + $pythonVersion) -ForegroundColor Green

    # Check Python version (3.8+ recommended)
    if ($pythonVersion -notmatch "Python 3\.[89]|Python 3\.1[0-9]") {
        Write-Host "  [WARN] Python 3.8 or higher is recommended" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] Python not found. Install Python 3.8+" -ForegroundColor Red
    Write-Host "  Download: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Step 2: Create virtual environment
Write-Host ""
Write-Host "[2/5] Creating Python virtual environment..." -ForegroundColor Yellow
if (Test-Path ".\venv") {
    Write-Host "  [OK] Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "  [OK] Virtual environment created" -ForegroundColor Green
}

# Step 3: Install Python dependencies
Write-Host ""
Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Yellow
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "  [ERROR] Virtual environment activation script not found." -ForegroundColor Red
    exit 1
}

& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "  [OK] Python dependencies installed" -ForegroundColor Green

# Step 4: Check NodeJS installation
Write-Host ""
Write-Host "[4/5] Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host ("  [OK] Found: " + $nodeVersion) -ForegroundColor Green

    # Check Node version
    $nodeVersionNum = [int]($nodeVersion -replace 'v([0-9]+)\..*', '$1')
    if ($nodeVersionNum -lt 16) {
        Write-Host "  [WARN] Node.js 16 or higher is recommended" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] Node.js not found. Install Node.js 16+" -ForegroundColor Red
    Write-Host "  Download: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Step 5: Install frontend dependencies
Write-Host ""
Write-Host "[5/5] Installing frontend dependencies..." -ForegroundColor Yellow
Push-Location "Build AI Surveillance System"
try {
    if (Test-Path ".\node_modules") {
        Write-Host "  [OK] Frontend dependencies already installed" -ForegroundColor Green
    } else {
        npm install
        Write-Host "  [OK] Frontend dependencies installed" -ForegroundColor Green
    }
} catch {
    Write-Host "  [ERROR] Failed to install frontend dependencies" -ForegroundColor Red
    throw
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1) Run: .\start-intentwatch.ps1" -ForegroundColor Yellow
Write-Host "  2) Or:  .\start-backend.ps1  and  .\start-frontend.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "URLs:" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Green
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
