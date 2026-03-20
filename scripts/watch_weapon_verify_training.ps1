Param(
  [string]$RunDir = "d:\intent-watch\runs_weapon\weapon_verify_v8s",
  [int]$TotalEpochs = 60,
  [int]$IntervalSeconds = 30
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $RunDir)) {
  throw "Run directory not found: $RunDir"
}

$resultsCsv = Join-Path $RunDir 'results.csv'
$weightsDir = Join-Path $RunDir 'weights'
$lastPt = Join-Path $weightsDir 'last.pt'
$bestPt = Join-Path $weightsDir 'best.pt'

$start = Get-Date

Write-Host "Watching training progress..." -ForegroundColor Cyan
Write-Host "- run dir : $RunDir" -ForegroundColor DarkGray
Write-Host "- epochs  : $TotalEpochs" -ForegroundColor DarkGray
Write-Host "- refresh : every $IntervalSeconds sec" -ForegroundColor DarkGray
Write-Host "";

function Format-Duration([TimeSpan]$ts) {
  if ($ts.TotalDays -ge 1) { return "{0:0}d {1:00}h {2:00}m" -f [int]$ts.TotalDays, $ts.Hours, $ts.Minutes }
  if ($ts.TotalHours -ge 1) { return "{0:0}h {1:00}m {2:00}s" -f [int]$ts.TotalHours, $ts.Minutes, $ts.Seconds }
  if ($ts.TotalMinutes -ge 1) { return "{0:0}m {1:00}s" -f [int]$ts.TotalMinutes, $ts.Seconds }
  return "{0:0}s" -f [int]$ts.TotalSeconds
}

while ($true) {
  $now = Get-Date
  $elapsed = $now - $start

  $epochDone = $null
  if (Test-Path $resultsCsv) {
    try {
      # results.csv header + epoch rows; take last non-empty row
      $lines = Get-Content $resultsCsv -ErrorAction Stop
      $last = ($lines | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -Last 1)
      if ($last -and ($last -ne $lines[0])) {
        $firstCol = ($last -split ',')[0]
        $parsed = 0
        if ([int]::TryParse($firstCol.Trim(), [ref]$parsed)) {
          # Ultralytics uses 0-based epoch index in results.csv
          $epochDone = $parsed + 1
        }
      }
    } catch {
      $epochDone = $null
    }
  }

  $hasLast = Test-Path $lastPt
  $hasBest = Test-Path $bestPt

  if ($null -eq $epochDone) {
    $msg = "[{0}] elapsed={1} | epochs=?/{2} | last.pt={3} best.pt={4}" -f $now.ToString('HH:mm:ss'), (Format-Duration $elapsed), $TotalEpochs, $hasLast, $hasBest
    Write-Host $msg
  } else {
    $rate = $elapsed.TotalSeconds / [double]$epochDone
    $remainingEpochs = [Math]::Max(0, $TotalEpochs - $epochDone)
    $etaSeconds = $rate * $remainingEpochs
    $eta = [TimeSpan]::FromSeconds($etaSeconds)
    $finishAt = $now + $eta

    $msg = "[{0}] elapsed={1} | epochs={2}/{3} | ETA={4} (finish ~ {5}) | last.pt={6} best.pt={7}" -f `
      $now.ToString('HH:mm:ss'), (Format-Duration $elapsed), $epochDone, $TotalEpochs, (Format-Duration $eta), $finishAt.ToString('HH:mm'), $hasLast, $hasBest
    Write-Host $msg

    if ($epochDone -ge $TotalEpochs) {
      Write-Host "Training reached target epochs." -ForegroundColor Green
      break
    }
  }

  Start-Sleep -Seconds $IntervalSeconds
}
