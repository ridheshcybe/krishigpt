@echo off
REM Usage: run.bat          -> evaluate existing checkpoints + launch dashboard
REM        run.bat train    -> retrain models first
echo Installing dependencies...
python -m pip install -r requirements.txt
python -m pip install streamlit
if /I "%1"=="train" (
    echo Training unified model...
    python train_unified.py --mode train
    echo Training trajectory model...
    python train_trajectory.py --mode train
)
echo Running full validation...
python main.py --mode validate
python evaluate_unified.py --mode full
python evaluate_trajectory.py --mode full
echo Running edge benchmark...
python edge_deploy.py --benchmark
python significance_tests.py
echo Launching KisanGPT dashboard...
python -m streamlit run app.py
