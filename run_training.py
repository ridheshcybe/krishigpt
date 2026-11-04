#!/usr/bin/env python3
"""Stage 3A Trajectory Model Training and Evaluation Pipeline.

Executes the complete, verified, leak-free pipeline:
1. Builds/validates the causal trajectory dataset (seed-aware, strictly causal).
2. Trains TrajectoryModel (Transformer with residual NDVI head, decoupled alert head,
   and lead-time-aware EarlyWarningLoss).
3. Evaluates test split, tunes 5-level alert threshold on healthy validation fields,
   and plots lead-time histogram, threshold curve, and confusion matrix.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_cmd(cmd_args: list[str], desc: str) -> int:
    cmd_str = " ".join(cmd_args)
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {desc}")
    print(f"Command: {cmd_str}")
    print("=" * 70, flush=True)
    
    t0 = time.time()
    res = subprocess.run(cmd_args)
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {res.returncode} after {elapsed:.1f}s")
        return res.returncode
    
    print(f"[{time.strftime('%H:%M:%S')}] {desc} finished successfully ({elapsed:.1f}s)")
    return 0

def main():
    parser = argparse.ArgumentParser(description="KisanGPT Trajectory Model Training Runner")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--alert_horizon", type=int, default=3, help="Early warning horizon in days (default: 3)")
    parser.add_argument("--alert_loss", type=str, default="focal", choices=["bce", "focal", "ranking"], help="Alert loss kind")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--rebuild_data", action="store_true", help="Force rebuild of trajectory data cache")
    parser.add_argument("--multi_seed", action="store_true", help="Run 5-seed benchmark after training")
    args = parser.parse_args()

    python_exe = sys.executable

    # STEP 1: Rebuild or validate trajectory dataset cache
    step1_args = [python_exe, "trajectory_data.py", "--seed", str(args.seed)]
    if args.rebuild_data:
        step1_args.append("--force")
    ret = run_cmd(step1_args, "STEP 1: Trajectory Dataset Cache Setup & Validation")
    if ret != 0:
        return ret

    # STEP 2: Train trajectory model
    step2_args = [
        python_exe, "train_trajectory.py",
        "--mode", "train",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--alert_horizon", str(args.alert_horizon),
        "--alert_loss", args.alert_loss,
        "--alert_focal_gamma", "2.0",
        "--lambda_alert", "0.5",
        "--lead_weight_3d", "1.0",
        "--lead_weight_2d", "0.8",
        "--lead_weight_1d", "0.6",
        "--exclude_pest",
        "--merge_generic_into_nearest",
        "--seed", str(args.seed),
    ]
    ret = run_cmd(step2_args, "STEP 2: Training Trajectory Model on GPU")
    if ret != 0:
        return ret

    # STEP 3: Comprehensive evaluation & plotting
    step3_args = [
        python_exe, "evaluate_trajectory.py",
        "--split", "test",
        "--alert_horizon", str(args.alert_horizon),
        "--seed", str(args.seed),
        "--checkpoint", "artifacts/trajectory_model.pth",
        "--data", "artifacts/trajectory_data.npz",
    ]
    ret = run_cmd(step3_args, "STEP 3: Test Split Evaluation, Threshold Tuning & Plotting")
    if ret != 0:
        return ret

    # STEP 4: Optional 5-seed benchmark
    if args.multi_seed:
        step4_args = [
            python_exe, "evaluate_multi_seed.py",
            "--seeds", "42", "43", "44", "45", "46",
            "--alert_horizon", str(args.alert_horizon),
            "--split", "test",
        ]
        ret = run_cmd(step4_args, "STEP 4: 5-Seed Reproducibility & Robustness Benchmark")
        if ret != 0:
            return ret

    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("Metrics written to: artifacts/trajectory_metrics.json")
    print("Report written to:  report_trajectory.txt")
    print("Plots generated in: plots/")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
