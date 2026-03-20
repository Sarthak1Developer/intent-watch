param(
  [switch]$RemoveRuns,
  [switch]$RemovePythonCaches,
  [switch]$RemoveYoloCaches
)

$ErrorActionPreference = 'Stop'

function Remove-IfExists($path) {
  if (Test-Path $path) {
    Write-Host "Removing $path" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $path
  }
}

if ($RemoveRuns) {
  Remove-IfExists "runs"
  Remove-IfExists "runs_weapon"
}

if ($RemovePythonCaches) {
  Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-IfExists $_.FullName }

  Get-ChildItem -Recurse -File -Include "*.pyc","*.pyo" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Force $_.FullName }
}

if ($RemoveYoloCaches) {
  # Ultralytics/YOLO label cache files (safe to delete; will regenerate)
  Get-ChildItem -Recurse -File -Filter "*.cache" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "\\labels\\.cache$" -or $_.Name -eq "labels.cache" } |
    ForEach-Object { Remove-Item -Force $_.FullName }
}

Write-Host "Done." -ForegroundColor Green
