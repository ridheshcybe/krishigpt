<#
.SYNOPSIS
    KisanGPT - Stage 3A Trajectory Model Training & Evaluation Pipeline
.DESCRIPTION
    Runs the leak-free, causal trajectory training and evaluation pipeline locally on the GPU.
#>

param (
    [int]$Epochs = 60,
    [int]$BatchSize = 32,
    [double]$Lr = 1e-4,
    [int]$Patience = 15,
    [int]$AlertHorizon = 3,
    [string]$AlertLoss = "focal",
    [int]$Seed = 42,
    [switch]$RebuildData,
    [switch]$MultiSeed
)

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " KisanGPT - Stage 3A Trajectory Model Training Pipeline" -ForegroundColor Cyan
Write-Host " (Local GPU Execution - NVIDIA GeForce RTX 5060)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path "artifacts")) { New-Item -ItemType Directory -Path "artifacts" | Out-Null }
if (-not (Test-Path "plots")) { New-Item -ItemType Directory -Path "plots" | Out-Null }

$argsList = @(
    "run_training.py",
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--lr", $Lr,
    "--patience", $Patience,
    "--alert_horizon", $AlertHorizon,
    "--alert_loss", $AlertLoss,
    "--seed", $Seed
)

if ($RebuildData) {
    $argsList += "--rebuild_data"
}

if ($MultiSeed) {
    $argsList += "--multi_seed"
}

& python $argsList

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Pipeline failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host " Pipeline Completed Successfully!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
