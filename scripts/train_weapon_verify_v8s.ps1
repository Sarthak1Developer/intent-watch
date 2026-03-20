Param(
  [string]$Data = "d:\intent-watch\datasets\data_cctv_v3_person_weapon.yaml",
  [int]$Epochs = 60,
  [int]$ImgSz = 800,
  [string]$Device = "0",
  [int]$Batch = 2,
  [string]$Name = "weapon_verify_v8s",
  [string]$Project = "d:\intent-watch\runs_weapon"
)

$ErrorActionPreference = 'Stop'

Write-Host "Training YOLOv8s verify model" -ForegroundColor Cyan
Write-Host "- data   : $Data" -ForegroundColor DarkGray
Write-Host "- epochs : $Epochs" -ForegroundColor DarkGray
Write-Host "- imgsz  : $ImgSz" -ForegroundColor DarkGray
Write-Host "- device : $Device" -ForegroundColor DarkGray
Write-Host "- batch  : $Batch" -ForegroundColor DarkGray
Write-Host "- out    : $Project/$Name" -ForegroundColor DarkGray
Write-Host "";

if (-Not (Test-Path "$PSScriptRoot\..\venv\Scripts\python.exe")) {
  throw "Python venv not found at ..\\venv\\Scripts\\python.exe"
}

# Ensure ultralytics CLI is available in this venv
& "$PSScriptRoot\..\venv\Scripts\python.exe" -c "import ultralytics; print('ultralytics', ultralytics.__version__)" | Out-Host

# NOTE:
# - Model is YOLOv8s (more accurate) for verification.
# - Dataset is CCTV-focused binary person/weapon.
# - imgsz=800 improves small weapon details but uses more VRAM.

$ModelPath = "d:\intent-watch\yolov8s.pt"
if (-Not (Test-Path $ModelPath)) {
  # Fall back to Ultralytics resolution (may download if not present).
  $ModelPath = "yolov8s.pt"
}

& "$PSScriptRoot\..\venv\Scripts\python.exe" "$PSScriptRoot\train_weapon_verify_v8s.py" `
  --model "$ModelPath" `
  --data "$Data" `
  --epochs $Epochs `
  --imgsz $ImgSz `
  --batch $Batch `
  --device "$Device" `
  --workers 8 `
  --patience 10 `
  --close-mosaic 10 `
  --project "$Project" `
  --name "$Name"

if ($LASTEXITCODE -ne 0) {
  throw "Training failed (exit code $LASTEXITCODE). See output above."
}

$BestPath = "$Project\$Name\weights\best.pt"
if (-Not (Test-Path $BestPath)) {
  throw "Training finished but best checkpoint not found at: $BestPath"
}

Write-Host "";
Write-Host "Done. Verify checkpoint:" -ForegroundColor Green
Write-Host "- $BestPath" -ForegroundColor Green
Write-Host "";
Write-Host "To enable verification in backend, set:" -ForegroundColor Yellow
Write-Host "- INTENTWATCH_WEAPON_VERIFY_MODEL_PATH=$BestPath" -ForegroundColor Yellow
