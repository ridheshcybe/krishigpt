@echo off
setlocal

echo =====================================================================
echo  KisanGPT - Stage 3A Trajectory Model Training Pipeline
echo  (Local GPU Execution - NVIDIA GeForce RTX 5060)
echo =====================================================================
echo.
echo Started at: %DATE% %TIME%
echo.

if not exist artifacts mkdir artifacts
if not exist plots mkdir plots

REM Run Python pipeline script
python run_training.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline failed with exit code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo.
echo =====================================================================
echo  Pipeline Completed Successfully at: %DATE% %TIME%
echo =====================================================================
