"""Evaluate Trajectory Model across multiple random seeds.

Runs evaluation on held-out field splits generated with different random seeds
(default: 42, 43, 44, 45, 46) and reports mean +/- standard deviation for:
- NDVI MAE, RMSE, R^2
- Classification accuracy and macro F1
- Alert precision, recall, F1, mean lead time, and percentage before onset

Saves results to ``artifacts/multi_seed_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import config as cfg
from stage1_config import UNIFIED_CLASSES
from trajectory_data import build_dataset, decode_labels_from_ndvi, field_split, log
from trajectory_model import detect_onset_alerts, load_trajectory_model
from evaluate_trajectory import (
    binary_onset_scores,
    build_class_mapping,
    classification_report,
    first_onset,
    predict_split,
    regression_metrics,
    tune_alert_threshold,
)


def run_seed_eval(
    model: Any,
    blob: dict[str, Any],
    seed: int,
    active_classes: list[str],
    horizon: int,
    merge_generic: bool,
) -> dict[str, float]:
    """Evaluate model on a field split generated with the given seed."""
    ckpt_cfg = blob.get("config", {})
    class_mapping = build_class_mapping(active_classes, merge_generic=merge_generic)

    data = build_dataset(seed=seed, cache=False)
    split = field_split(data["n_fields"], data["labels"], seed=seed, save=False)

    val_indices = np.asarray(split["val"], dtype=np.int64)
    test_indices = np.asarray(split["test"], dtype=np.int64)

    # 1. Validation threshold tuning on healthy-at-endpoint fields
    val_preds = predict_split(
        model, data, val_indices,
        channel_mean=ckpt_cfg.get("input_mean"),
        channel_std=ckpt_cfg.get("input_std"),
        class_mapping=class_mapping,
        alert_horizon=horizon,
    )
    best_threshold, _, _, _ = tune_alert_threshold(
        val_preds["probs"],
        val_preds["y_cls_raw"],
        val_preds["early_probs"],
        val_preds["endpoint_stressed"],
        horizon=horizon,
        active_classes=active_classes,
    )

    # 2. Test evaluation
    test_preds = predict_split(
        model, data, test_indices,
        channel_mean=ckpt_cfg.get("input_mean"),
        channel_std=ckpt_cfg.get("input_std"),
        class_mapping=class_mapping,
        alert_horizon=horizon,
    )

    y_ndvi = test_preds["y_ndvi"]
    ndvi_pred = test_preds["ndvi_pred"]
    y_cls = test_preds["y_cls"]
    y_cls_raw = test_preds["y_cls_raw"]
    probs = test_preds["probs"]
    early_probs = test_preds["early_probs"]
    endpoint_stressed = test_preds["endpoint_stressed"]

    # NDVI metrics
    ndvi_m = regression_metrics(y_ndvi, ndvi_pred)

    # Classification metrics
    cls_pred = probs.argmax(axis=-1)
    clf_m = classification_report(y_cls, cls_pred, class_names=active_classes)

    # Alert quality evaluated strictly on fields that were healthy at history endpoint
    healthy_eval_mask = ~np.asarray(endpoint_stressed, dtype=bool)
    h_probs = probs[healthy_eval_mask]
    h_early = early_probs[healthy_eval_mask] if early_probs is not None else None
    h_raw = y_cls_raw[healthy_eval_mask]

    actual_onsets = []
    for row in h_raw:
        sub = row[:horizon]
        nz = np.flatnonzero(sub != 0)
        actual_onsets.append(int(nz[0]) + 1 if nz.size > 0 else None)
    actual_pos = np.array([d is not None for d in actual_onsets], dtype=bool)

    alerts = detect_onset_alerts(
        torch.from_numpy(h_probs),
        horizon=horizon,
        early_probs=torch.from_numpy(h_early) if h_early is not None else None,
        threshold=best_threshold,
        classes=active_classes,
    )
    alert_flags = np.array([a["alerted"] for a in alerts], dtype=bool)
    alert_days = np.array([a["day"] if a["day"] is not None else 0 for a in alerts], dtype=float)

    tp = int(np.sum(alert_flags & actual_pos))
    fp = int(np.sum(alert_flags & ~actual_pos))
    fn = int(np.sum(~alert_flags & actual_pos))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    alert_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    lead_times = [
        float(actual_onsets[i]) - float(alert_days[i])
        for i in range(len(alerts))
        if alert_flags[i] and actual_pos[i]
    ]
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    pct_before = (sum(1 for lt in lead_times if lt > 0) / len(lead_times) * 100.0) if lead_times else 0.0

    return {
        "ndvi_mae": float(ndvi_m["mae"]),
        "ndvi_rmse": float(ndvi_m["rmse"]),
        "ndvi_r2": float(ndvi_m["r2"]),
        "accuracy": float(clf_m["accuracy"]),
        "macro_f1": float(clf_m["macro_f1"]),
        "alert_precision": float(prec),
        "alert_recall": float(rec),
        "alert_f1": float(alert_f1),
        "mean_lead_time": float(mean_lead),
        "pct_before_onset": float(pct_before),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed evaluation of Stage 3A Trajectory Model.")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated random seeds")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class names")
    parser.add_argument("--alert_horizon", type=int, default=None, help="Early warning horizon (e.g. 3 or 5)")
    parser.add_argument("--merge_generic_into_nearest", action="store_true", default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    try:
        checkpoint = Path(cfg.TRAJ_CKPT)
        model, blob = load_trajectory_model(checkpoint)
    except FileNotFoundError as err:
        log(f"ERROR: {err}")
        return 1

    ckpt_cfg = blob.get("config", {})
    if args.classes is not None:
        active_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    elif "classes" in ckpt_cfg and ckpt_cfg["classes"]:
        active_classes = list(ckpt_cfg["classes"])
    else:
        active_classes = list(UNIFIED_CLASSES)

    horizon = args.alert_horizon if args.alert_horizon is not None else ckpt_cfg.get("alert_horizon", cfg.TRAJ_ALERT_HORIZON)
    merge_generic = args.merge_generic_into_nearest if args.merge_generic_into_nearest is not None else ckpt_cfg.get("merge_generic_into_nearest", False)

    log(f"Starting multi-seed evaluation across {len(seeds)} seeds: {seeds}")
    per_seed_results: dict[int, dict[str, float]] = {}

    for s in seeds:
        log(f"--> Evaluating seed {s} ...")
        t0 = time.time()
        res = run_seed_eval(model, blob, s, active_classes, horizon, merge_generic)
        per_seed_results[s] = res
        log(f"    Seed {s} done in {time.time()-t0:.1f}s: NDVI MAE={res['ndvi_mae']:.4f}, Macro F1={res['macro_f1']:.4f}, Alert F1={res['alert_f1']:.4f}, Mean Lead={res['mean_lead_time']:.2f}d")

    metrics_list = list(next(iter(per_seed_results.values())).keys())
    summary: dict[str, dict[str, float]] = {}

    for m in metrics_list:
        values = [per_seed_results[s][m] for s in seeds]
        summary[m] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "values": [round(float(v), 4) for v in values],
        }

    output_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "classes": active_classes,
        "alert_horizon": horizon,
        "summary": summary,
        "per_seed": {str(k): v for k, v in per_seed_results.items()},
    }

    out_path = Path("artifacts/multi_seed_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    log(f"Multi-seed evaluation saved to {out_path}")

    # Print summary table
    print()
    print("=" * 68)
    print(f" KisanGPT Stage 3A Multi-Seed Evaluation ({len(seeds)} seeds: {seeds})")
    print("=" * 68)
    print(f"{'Metric':<26}{'Mean':>12}{'Std':>12}{'Min - Max':>18}")
    print("-" * 68)
    for m in metrics_list:
        stat = summary[m]
        min_max_str = f"{stat['min']:.4f} - {stat['max']:.4f}"
        print(f"{m:<26}{stat['mean']:>12.4f}{stat['std']:>12.4f}{min_max_str:>18}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
