Param(
  [string]$RunDir = "d:\intent-watch\runs_weapon\weapon80_20",
  [int]$RefreshSeconds = 10
)

$results = Join-Path $RunDir "results.csv"
$weightsDir = Join-Path $RunDir "weights"

Write-Host "Watching YOLO training progress in: $RunDir" -ForegroundColor Cyan
Write-Host "- Results file: $results" -ForegroundColor DarkGray
Write-Host "- Weights dir : $weightsDir" -ForegroundColor DarkGray
Write-Host "(Press Ctrl+C to stop watching)" -ForegroundColor DarkGray
Write-Host "";

while ($true) {
  $now = Get-Date

  # Basic signal: GPU activity
  try {
    $smi = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $smi) {
      $parts = $smi -split ',' | ForEach-Object { $_.Trim() }
      Write-Host ("[{0}] GPU util={1}% mem={2}/{3}MiB temp={4}C" -f $now.ToString("HH:mm:ss"), $parts[0], $parts[1], $parts[2], $parts[3])
    } else {
      Write-Host ("[{0}] nvidia-smi not available" -f $now.ToString("HH:mm:ss"))
    }
  } catch {
    Write-Host ("[{0}] nvidia-smi error: {1}" -f $now.ToString("HH:mm:ss"), $_.Exception.Message)
  }

  # Training metrics (after first epoch, results.csv usually appears)
  if (Test-Path $results) {
    try {
      $last = Get-Content $results -Tail 1
      Write-Host ("[{0}] results.csv last row: {1}" -f $now.ToString("HH:mm:ss"), $last) -ForegroundColor Green
    } catch {
      Write-Host ("[{0}] Could not read results.csv: {1}" -f $now.ToString("HH:mm:ss"), $_.Exception.Message) -ForegroundColor Yellow
    }
  } else {
    Write-Host ("[{0}] Waiting for results.csv (usually after epoch 1 completes)..." -f $now.ToString("HH:mm:ss")) -ForegroundColor Yellow
  }

  # Checkpoints
  $best = Join-Path $weightsDir "best.pt"
  $lastPt = Join-Path $weightsDir "last.pt"

  if (Test-Path $best) {
    $sz = (Get-Item $best).Length
    Write-Host ("[{0}] best.pt exists ({1:N0} bytes)" -f $now.ToString("HH:mm:ss"), $sz) -ForegroundColor Green
  }

  if (Test-Path $lastPt) {
    $sz = (Get-Item $lastPt).Length
    Write-Host ("[{0}] last.pt exists ({1:N0} bytes)" -f $now.ToString("HH:mm:ss"), $sz) -ForegroundColor Green
  }

  Write-Host "";
  Start-Sleep -Seconds $RefreshSeconds
}
