@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  KisanGPT Autonomous Long-Run Optimization and Training Pipeline
echo  (Zero LLM token consumption - Local GPU Execution)
echo =====================================================================
echo.
echo Started at: %DATE% %TIME%
echo Output log: artifacts\autonomous_run.log
echo.

if not exist artifacts mkdir artifacts

(
echo =====================================================================
echo Autonomous Run Started: %DATE% %TIME%
echo =====================================================================
) > artifacts\autonomous_run.log

echo [STEP 1/4] Fetching Real Weather Data from Open-Meteo and Building Trajectories...
echo [STEP 1/4] Fetching Real Weather Data... >> artifacts\autonomous_run.log
python trajectory_data.py --weather-source open-meteo --n-fields 1200 --force >> artifacts\autonomous_run.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Open-Meteo fetch encountered an issue, regenerating with proxy fallback...
    python trajectory_data.py --weather-source proxy --n-fields 1200 --force >> artifacts\autonomous_run.log 2>&1
)

echo [STEP 2/4] Training Trajectory Model on Expanded Field Dataset...
echo [STEP 2/4] Training Trajectory Model... >> artifacts\autonomous_run.log
python train_trajectory.py --mode train --n-fields 1200 --epochs 80 --weather-source open-meteo >> artifacts\autonomous_run.log 2>&1

echo [STEP 3/4] Deep Training Attention Fusion Model (RTX 5060)...
echo [STEP 3/4] Deep Training Attention Fusion Model... >> artifacts\autonomous_run.log
python attention_fusion.py --epochs 50 --stability_patience 8 >> artifacts\autonomous_run.log 2>&1

echo [STEP 4/4] Validating Pipeline, Benchmarking and Exporting ONNX / INT8...
echo [STEP 4/4] Validating Pipeline and Exporting... >> artifacts\autonomous_run.log
python edge_deploy.py >> artifacts\autonomous_run.log 2>&1
python significance_tests.py >> artifacts\autonomous_run.log 2>&1
python validate_models.py >> artifacts\autonomous_run.log 2>&1

echo.
echo =====================================================================
echo Autonomous Run Completed at: %DATE% %TIME%
echo =====================================================================
(
echo =====================================================================
echo Autonomous Run Completed at: %DATE% %TIME%
echo =====================================================================
) >> artifacts\autonomous_run.log

echo Full log written to artifacts\autonomous_run.log
echo To inspect results, run: python -m streamlit run app.py
pause
