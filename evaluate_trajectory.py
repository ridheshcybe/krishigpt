"""Stage 3A - evaluate the trajectory model and its baselines.

    python evaluate_trajectory.py --mode full

What gets measured (spec order)
-------------------------------
1. NDVI regression       MAE / RMSE / R^2 + predicted-vs-actual curves for
                         5 sample fields   -> plots/trajectory_curves.png
2. Stress classification per-time-step accuracy, macro F1, aggregated 6x6
                         confusion matrix  -> plots/trajectory_confusion_matrix.png
3. Stress onset          precision / recall / F1 for "stress onset within the
   detection             next 7 days?"
4. Alert quality         alert recall, alert precision, average lead time in
                         days between the predicted onset day and the real one
5. Baselines             persistence (last value repeated) vs linear
                         extrapolation vs the model -> comparison table +
                         plots/trajectory_baselines.png

Outputs: ``artifacts/trajectory_metrics.json`` and ``report_trajectory.txt``.
Everything is computed on the held-out **field-level** test split (no field is
ever seen during training), and the PROXY flags are echoed into every output
file so no downstream consumer can mistake proxy NDVI/weather for satellite
readings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from stage1_config import N_CLASSES, SEED, UNIFIED_CLASSES, UNIFIED_INDEX
from trajectory_data import (
    TrajectoryDataset,
    build_dataset,
    decode_labels_from_ndvi,
    load_split,
    log,
)
from trajectory_model import TrajectoryModel, detect_onset_alerts, load_trajectory_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """MAE / RMSE / R^2 (uniform average over the horizon) plus per-day MAE and Day-1 MAE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    per_day_mae = np.mean(np.abs(err), axis=0)
    day1_mae = float(per_day_mae[0]) if len(per_day_mae) > 0 else 0.0
    return {
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(np.sqrt(mse)), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
        "day1_mae": round(day1_mae, 6),
        "per_day_mae": [round(float(v), 6) for v in per_day_mae],
        "n_samples": int(y_true.shape[0]),
    }


def persistence_forecast(history: np.ndarray, t_future: int) -> np.ndarray:
    """Baseline 1 - the last observed NDVI value, held flat for the horizon."""
    return np.full(t_future, float(history[-1]), dtype=np.float64)


def linear_forecast(history: np.ndarray, t_future: int, window: int = 5) -> np.ndarray:
    """Baseline 2 - least-squares linear extrapolation of the recent NDVI trend."""
    w = int(min(window, len(history)))
    ys = history[-w:].astype(np.float64)
    xs = np.arange(w, dtype=np.float64)
    xs = xs - xs.mean()
    denom = float((xs ** 2).sum())
    slope = float((xs * (ys - ys.mean())).sum() / denom) if denom > 0 else 0.0
    steps = np.arange(1, t_future + 1, dtype=np.float64)
    return np.clip(float(history[-1]) + slope * steps, 0.05, 1.0)


def first_onset(classes: np.ndarray) -> int | None:
    """First day index of a non-healthy class, or ``None`` if it never happens."""
    nonzero = np.flatnonzero(np.asarray(classes) != 0)
    return int(nonzero[0]) if nonzero.size else None


def binary_onset_scores(
    actual_pos: np.ndarray, predicted_pos: np.ndarray
) -> dict[str, float]:
    """Precision / recall / F1 for the binary "onset within the horizon" call."""
    tp = int(np.sum(actual_pos & predicted_pos))
    fp = int(np.sum(~actual_pos & predicted_pos))
    fn = int(np.sum(actual_pos & ~predicted_pos))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support_positive": int(actual_pos.sum()),
        "n": int(actual_pos.size),
    }


def compute_alert_metrics(
    alerts: list[dict[str, Any]],
    actual_onsets: list[int | None],
    actual_pos: np.ndarray,
    n_healthy_evaluated: int | None = None,
    best_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute alert quality, distinguishing operational issuance lead from fallback onset timing.

    - Binary head: alert is issued at t=0; lead_time = actual_onset - 0 = actual_onset days.
    - Fallback scan: alert records 0-based predicted onset step k; converted to 1-based day (k+1);
      timing_diff = actual_onset - (k+1). A predicted index 1 and actual onset day 2 are on the same day -> diff 0.
    """
    alert_flags = np.array([a["alerted"] for a in alerts], dtype=bool)
    tp = int(np.sum(alert_flags & actual_pos))
    fp = int(np.sum(alert_flags & ~actual_pos))
    fn = int(np.sum(~alert_flags & actual_pos))
    tn = int(np.sum(~alert_flags & ~actual_pos))

    has_alerts = bool(tp + fp > 0)
    has_positives = bool(tp + fn > 0)

    precision = round(float(tp / (tp + fp)), 4) if has_alerts else None
    recall = round(float(tp / (tp + fn)), 4) if has_positives else None
    # F1 from counts: 0.0 (defined) when TP=0 with FP+FN>0; undefined only when TP=FP=FN=0.
    f1 = round(float(2 * tp / (2 * tp + fp + fn)), 4) if (2 * tp + fp + fn) > 0 else None
    fpr = round(float(fp / (fp + tn)), 4) if (fp + tn > 0) else None

    # Effective origin per record: the explicit alert_origin field wins; a record
    # carrying an explicit predicted-onset field is fallback evidence. Anything else
    # is unknown and is NEVER scored with issuance math.
    def _record_origin(rec: dict[str, Any]) -> str | None:
        o = rec.get("alert_origin")
        if o in ("binary_head", "fallback_scan"):
            return o
        if "predicted_onset_day" in rec or "predicted_onset_step" in rec:
            return "fallback_scan"
        return None

    # Set-level origin from ALERTED records (inferred origins included); mixed
    # origins are rejected so the two timing conventions are never pooled.
    per_alert_origin = [_record_origin(a) for a in alerts]
    alerted_origins = {
        ro for a, ro in zip(alerts, per_alert_origin) if a["alerted"] and ro is not None
    }
    if len(alerted_origins) > 1:
        raise ValueError(f"mixed alert origins in one evaluation: {sorted(alerted_origins)}")
    origin = (
        next(iter(alerted_origins))
        if alerted_origins
        else (per_alert_origin[0] or "unknown" if alerts else "unknown")
    )

    lead_times: list[float] = []
    n_unknown_origin = 0
    for i in range(len(alerts)):
        if alert_flags[i] and actual_pos[i]:
            act_d = float(actual_onsets[i])
            rec_origin = per_alert_origin[i]
            if rec_origin == "fallback_scan":
                # Fallback scan: predicted_onset_day is 1-based (day + 1)
                pred_d = float(alerts[i].get("predicted_onset_day", alerts[i]["day"] + 1))
                lead_times.append(act_d - pred_d)
            elif rec_origin == "binary_head":
                # Dedicated binary head: issued at evaluation time (day 0)
                issuance_day = float(alerts[i].get("day", 0))
                lead_times.append(act_d - issuance_day)
            else:
                # Record without an identifiable origin: excluded, counted, never guessed.
                n_unknown_origin += 1

    if lead_times:
        mean_lead = round(float(np.mean(lead_times)), 3)
        median_lead = round(float(np.median(lead_times)), 3)
        std_lead = round(float(np.std(lead_times)), 3)
        n_matched = len(lead_times)
        pct_before = round(sum(1 for lt in lead_times if lt > 0) / n_matched * 100.0, 2)
        pct_on = round(sum(1 for lt in lead_times if lt == 0) / n_matched * 100.0, 2)
        pct_after = round(sum(1 for lt in lead_times if lt < 0) / n_matched * 100.0, 2)
        lead_hist = {
            f"{k:+d}d" if k != 0 else "0d": sum(1 for lt in lead_times if int(round(lt)) == k)
            for k in range(-3, 4)
        }
    else:
        mean_lead = None
        median_lead = None
        std_lead = None
        pct_before = None
        pct_on = None
        pct_after = None
        lead_hist = {f"{k:+d}d" if k != 0 else "0d": 0 for k in range(-3, 4)}

    warning_msg = ""
    if pct_before is not None and (pct_before < 20.0 or (mean_lead is not None and mean_lead < 0.5)):
        warning_msg = (
            f"Alert timing does not yet satisfy true early-warning criteria: "
            f"only {pct_before:.1f}% of alerts fired before onset, mean lead time is {mean_lead:.2f} days."
        )
    elif pct_before is None:
        warning_msg = "No matched true positive events; lead-time metrics undefined."

    if n_unknown_origin:
        note = (
            f"{n_unknown_origin} matched alert(s) had unknown origin and were "
            f"excluded from timing metrics."
        )
        warning_msg = f"{warning_msg} {note}".strip() if warning_msg else note

    example = next((a["message"] for a in alerts if a["alerted"]), "")
    always_alert = binary_onset_scores(actual_pos, np.ones_like(actual_pos))

    lead_def = (
        "actual_onset_day - alert_issuance_day (positive = alert fired ahead of real onset)"
        if origin == "binary_head"
        else "actual_onset_day - (predicted_onset_step + 1) (0 = predicted on exact onset day, positive = predicted early)"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "mean_lead_days": mean_lead,
        "median_lead_days": median_lead,
        "std_lead_days": std_lead,
        "pct_before_onset": pct_before,
        "pct_before_onset_among_matched_tp": pct_before,
        "pct_on_onset": pct_on,
        "pct_after_onset": pct_after,
        "lead_time_histogram": lead_hist,
        "lead_times": lead_times,
        "n_unknown_origin": n_unknown_origin,
        "n_alerts": int(alert_flags.sum()),
        "n_true": int(actual_pos.sum()),
        "n_matched_true_positives": tp,
        "n_false_alarms": fp,
        "n_missed_events": fn,
        "n_true_negatives": tn,
        "n_healthy_evaluated": n_healthy_evaluated if n_healthy_evaluated is not None else int(actual_pos.size),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "positive_rate": round(float(actual_pos.mean()), 4) if actual_pos.size else 0.0,
        "always_alert_reference": always_alert,
        "example": example,
        "best_threshold": round(best_threshold, 2),
        "warning": warning_msg,
        "lead_definition": lead_def,
        "alert_origin": origin,
        "lead_metric_note": (
            "When alert_origin is binary_head, alert is issued at t=0 and pct_before_onset is 100% "
            "by construction for future events; event detection skill is measured by precision and recall. "
            "When alert_origin is fallback_scan, lead measures onset day difference relative to predicted onset step + 1."
        ),
    }


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Per-time-step accuracy/F1, aggregate accuracy, macro F1, per-class F1."""
    names = class_names or UNIFIED_CLASSES
    n_c = len(names)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    valid_mask = (y_true != -100) & (y_pred != -100)

    step_acc = []
    step_f1 = []
    for d in range(y_true.shape[1]):
        v_d = valid_mask[:, d]
        if v_d.any():
            step_acc.append(float(accuracy_score(y_true[v_d, d], y_pred[v_d, d])))
            step_f1.append(float(f1_score(y_true[v_d, d], y_pred[v_d, d], average="macro", zero_division=0)))
        else:
            step_acc.append(1.0)
            step_f1.append(1.0)

    flat_true = y_true[valid_mask]
    flat_pred = y_pred[valid_mask]
    per_class = f1_score(
        flat_true, flat_pred, average=None, labels=list(range(n_c)), zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(flat_true, flat_pred)), 4),
        "macro_f1": round(
            float(f1_score(flat_true, flat_pred, average="macro", zero_division=0)),
            4,
        ),
        "mean_step_accuracy": round(float(np.mean(step_acc)), 4),
        "mean_step_macro_f1": round(float(np.mean(step_f1)), 4),
        "step_accuracy": [round(v, 4) for v in step_acc],
        "per_class_f1": {
            names[c]: round(float(per_class[c]), 4) for c in range(n_c)
        },
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plt():  # pragma: no cover - import guard
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_sample_curves(
    history: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    band: float = cfg.TRAJ_BAND,
) -> None:
    """Historical NDVI (solid) + model forecast (dashed) + truth for 5 fields."""
    try:
        plt = _plt()
        n = min(5, history.shape[0])
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
        days_hist = np.arange(history.shape[1])
        days_fut = np.arange(history.shape[1], history.shape[1] + y_true.shape[1])
        for i in range(n):
            ax = axes[0][i]
            ax.plot(days_hist, history[i], color="#2E7D32", lw=1.8, label="history")
            ax.plot(
                days_fut, y_true[i], color="#1565C0", lw=1.8, ls=":", label="actual"
            )
            ax.plot(
                days_fut, y_pred[i], color="#C62828", lw=1.8, ls="--", label="forecast"
            )
            ax.fill_between(
                days_fut, y_pred[i] - band, y_pred[i] + band,
                color="#C62828", alpha=0.15, label=f"+/-{band}",
            )
            ax.axvline(history.shape[1] - 0.5, color="#455A64", lw=1.2)
            ax.set_title(f"Field {i + 1}", fontsize=10)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        axes[0][0].set_ylabel("NDVI")
        for ax in axes[0]:
            ax.set_xlabel("day (dashed = today)")
        fig.suptitle(
            "NDVI trajectories: history, model forecast, ground truth "
            "[PROXY series - prototype data]",
            fontsize=11,
        )
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log(f"sample curves written to {path}")
    except Exception as err:
        log(f"WARNING: could not write sample curves ({err})")


def plot_confusion(matrix: np.ndarray, path: Path, class_names: list[str] | None = None) -> None:
    """Aggregated test confusion matrix heatmap for active classes."""
    try:
        plt = _plt()
        names_list = class_names or UNIFIED_CLASSES
        n_c = len(names_list)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        shown = matrix[:n_c, :n_c].astype(float)
        row_sums = shown.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm = shown / row_sums
        im = ax.imshow(norm, cmap="Greens", vmin=0.0, vmax=1.0)
        for i in range(n_c):
            for j in range(n_c):
                ax.text(
                    j, i, f"{int(matrix[i, j])}\n{norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black",
                )
        display_names = [c.replace("_", "\n") for c in names_list]
        ax.set_xticks(range(n_c), display_names, fontsize=8)
        ax.set_yticks(range(n_c), display_names, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title("Aggregated stress confusion (test fields x future days)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log(f"confusion matrix written to {path}")
    except Exception as err:
        log(f"WARNING: could not write confusion matrix ({err})")


def plot_alert_threshold_curve(
    records: list[dict[str, Any]],
    best_threshold: float,
    baseline_prec: float,
    path: Path,
) -> None:
    """Plot threshold tuning curves for precision, recall, F1, and mean lead time."""
    try:
        plt = _plt()
        fig, ax1 = plt.subplots(figsize=(8, 4.8))

        thresholds = [r["threshold"] for r in records]
        precs = [r["precision"] for r in records]
        recalls = [r["recall"] for r in records]
        f1s = [r["f1"] for r in records]
        leads = [r["mean_lead_days"] for r in records]

        ax1.plot(thresholds, precs, marker="o", label="Precision", color="#1565C0", lw=2)
        ax1.plot(thresholds, recalls, marker="s", label="Recall", color="#2E7D32", lw=2)
        ax1.plot(thresholds, f1s, marker="^", label="F1 Score", color="#6A1B9A", lw=2)
        ax1.axhline(baseline_prec, color="#546E7A", ls=":", label=f"Always-Alert Baseline ({baseline_prec:.2f})")
        ax1.set_xlabel("Confidence Threshold")
        ax1.set_ylabel("Score (0 - 1)")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(thresholds, leads, marker="d", label="Mean Lead Time (days)", color="#D84315", lw=2, ls="--")
        ax2.set_ylabel("Mean Lead Time (days)", color="#D84315")
        ax2.tick_params(axis="y", labelcolor="#D84315")

        ax1.axvline(best_threshold, color="#C62828", lw=2.5, ls="-", label=f"Selected Threshold ({best_threshold:.2f})")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

        fig.suptitle("Validation Early-Warning Threshold Tuning Curve", fontsize=11)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log(f"threshold curve written to {path}")
    except Exception as err:
        log(f"WARNING: could not write threshold curve ({err})")


def plot_lead_time_histogram(
    lead_times: list[float],
    pct_before: float,
    pct_on: float,
    pct_after: float,
    path: Path,
) -> None:
    """Plot distribution of alert lead times relative to actual stress onset."""
    try:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 4.2))

        if lead_times:
            arr = np.array(lead_times)
            bins = np.arange(-3.5, 4.5, 1.0)
            n, bins_out, patches = ax.hist(arr, bins=bins, rwidth=0.8, color="#1976D2", edgecolor="black")

            for patch, b_left in zip(patches, bins_out[:-1]):
                if b_left >= 0.0:
                    patch.set_facecolor("#2E7D32")
                elif b_left >= -1.0:
                    patch.set_facecolor("#FB8C00")
                else:
                    patch.set_facecolor("#C62828")

            ax.axvline(0, color="black", lw=1.5, ls="--", label="Onset Day (0d)")
            mean_val = float(np.mean(arr))
            ax.axvline(mean_val, color="#D81B60", lw=2, label=f"Mean Lead ({mean_val:.2f}d)")

        ax.set_xticks(range(-3, 4))
        ax.set_xticklabels(["-3d", "-2d", "-1d", "0d", "+1d", "+2d", "+3d"])
        ax.set_xlim(-3.7, 3.7)

        stats_text = (
            f"Alerts before onset: {pct_before:.1f}%\n"
            f"Alerts on onset:     {pct_on:.1f}%\n"
            f"Alerts after onset:  {pct_after:.1f}%"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
            family="monospace",
        )

        ax.set_xlabel("Lead Time in Days (actual_onset_day - alert_day)")
        ax.set_ylabel("Number of Alerts")
        ax.set_title("Early-Warning Alert Lead Time Distribution")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log(f"lead time histogram written to {path}")
    except Exception as err:
        log(f"WARNING: could not write lead time histogram ({err})")


def plot_baselines(ndvi_rows: list[tuple[str, dict[str, Any]]],
                   clf_rows: list[tuple[str, dict[str, Any]]],
                   path: Path) -> None:
    """Grouped bar chart: NDVI RMSE/MAE and classification accuracy/macro-F1."""
    try:
        plt = _plt()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        labels = [name for name, _ in ndvi_rows]
        x = np.arange(len(labels))
        width = 0.38
        axes[0].bar(x - width / 2, [m["mae"] for _, m in ndvi_rows], width, label="MAE")
        axes[0].bar(x + width / 2, [m["rmse"] for _, m in ndvi_rows], width, label="RMSE")
        axes[0].set_xticks(x, labels, rotation=12)
        axes[0].set_title("NDVI forecast error (lower is better)")
        axes[0].set_ylabel("error")
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis="y")

        labels2 = [name for name, _ in clf_rows]
        x2 = np.arange(len(labels2))
        axes[1].bar(x2 - width / 2, [m["accuracy"] for _, m in clf_rows], width,
                    label="accuracy")
        axes[1].bar(x2 + width / 2, [m["macro_f1"] for _, m in clf_rows], width,
                    label="macro F1")
        axes[1].set_xticks(x2, labels2, rotation=12)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Stress-class forecasting (higher is better)")
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis="y")

        fig.suptitle("Trajectory baselines vs model (held-out fields, PROXY series)",
                     fontsize=11)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log(f"baseline comparison written to {path}")
    except Exception as err:
        log(f"WARNING: could not write baseline chart ({err})")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_split(
    model: TrajectoryModel,
    data: dict[str, Any],
    field_indices: np.ndarray,
    batch_size: int = cfg.TRAJ_BATCH_SIZE,
    channel_mean: list[float] | None = None,
    channel_std: list[float] | None = None,
    alert_horizon: int = 3,
    class_mapping: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Run the model over a field subset; returns forecasts, probs, early warning, and truth."""
    device = next(model.parameters()).device
    dataset = TrajectoryDataset(
        ndvi=data["ndvi"],
        weather=data["weather"],
        labels=data["labels"],
        field_indices=field_indices,
        t_history=data["t_history"],
        t_future=data["t_future"],
        use_weather=cfg.TRAJ_USE_WEATHER,
        channel_mean=channel_mean,
        channel_std=channel_std,
        alert_horizon=alert_horizon,
        class_mapping=class_mapping,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ndvi_pred, probs, early_probs, y_ndvi, y_cls, y_cls_raw, fields = [], [], [], [], [], [], []
    for batch in tqdm(loader, desc="inference", leave=False):
        last_ndvi = batch["last_ndvi"].to(device) if "last_ndvi" in batch else None
        out = model.predict(batch["x"].to(device), last_ndvi=last_ndvi)
        ndvi_pred.append(out["ndvi"].cpu().numpy())
        probs.append(out["probs"].cpu().numpy())
        if "early_prob" in out:
            early_probs.append(out["early_prob"].cpu().numpy())
        y_ndvi.append(batch["y_ndvi"].numpy())
        y_cls.append(batch["y_cls"].numpy())
        y_cls_raw.append(batch.get("y_cls_raw", batch["y_cls"]).numpy())
        fields.append(batch["field_id"].numpy())

    return {
        "ndvi_pred": np.concatenate(ndvi_pred),
        "probs": np.concatenate(probs),
        "early_probs": np.concatenate(early_probs) if early_probs else None,
        "y_ndvi": np.concatenate(y_ndvi),
        "y_cls": np.concatenate(y_cls),
        "y_cls_raw": np.concatenate(y_cls_raw),
        "fields": np.concatenate(fields),
        "history": data["ndvi"][field_indices, : data["t_history"]],
        "endpoint_stressed": (data["labels"][field_indices, data["t_history"] - 1] != 0),
        "endpoint_label": data["labels"][field_indices, data["t_history"] - 1],
    }


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
def tune_alert_threshold(
    val_probs: np.ndarray,
    val_y_cls_raw: np.ndarray,
    val_early_probs: np.ndarray | None,
    val_endpoint_stressed: np.ndarray,
    horizon: int,
    active_classes: list[str],
) -> tuple[float, list[dict[str, Any]], str, float]:
    """Search thresholds in [0.20, ..., 0.90] on healthy-at-endpoint validation fields.

    Hierarchy:
    1. Precision >= 0.80 and pct_before_onset >= 20% (select max mean lead time)
    2. Fallback: F1 >= 0.75 (select max mean lead time)
    3. Fallback: max F1 with explicit warning
    """
    candidate_thresholds = [
        0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ]
    records: list[dict[str, Any]] = []

    # Filter strictly to fields that are healthy (not stressed) at the history endpoint
    val_healthy_mask = ~np.asarray(val_endpoint_stressed, dtype=bool)
    if not np.any(val_healthy_mask):
        val_healthy_mask = np.ones(len(val_y_cls_raw), dtype=bool)

    h_raw = val_y_cls_raw[val_healthy_mask]
    h_early = val_early_probs[val_healthy_mask] if val_early_probs is not None else None
    h_probs = val_probs[val_healthy_mask]

    actual_onsets = []
    for row in h_raw:
        sub = row[:horizon]
        nz = np.flatnonzero(sub != 0)
        actual_onsets.append(int(nz[0]) + 1 if nz.size > 0 else None)

    actual_pos = np.array([d is not None for d in actual_onsets], dtype=bool)
    baseline_precision = float(np.mean(actual_pos)) if actual_pos.size > 0 else 0.0

    for thresh in candidate_thresholds:
        alerts = detect_onset_alerts(
            torch.from_numpy(h_probs),
            horizon=horizon,
            early_probs=h_early,
            threshold=thresh,
            classes=active_classes,
        )
        m = compute_alert_metrics(
            alerts=alerts,
            actual_onsets=actual_onsets,
            actual_pos=actual_pos,
            best_threshold=thresh,
        )
        records.append({
            "threshold": round(thresh, 2),
            "precision": m["precision"] if m["precision"] is not None else 0.0,
            "recall": m["recall"] if m["recall"] is not None else 0.0,
            "f1": m["f1"] if m["f1"] is not None else 0.0,
            "false_positive_rate": m["false_positive_rate"] if m["false_positive_rate"] is not None else 0.0,
            "mean_lead_days": m["mean_lead_days"] if m["mean_lead_days"] is not None else 0.0,
            "median_lead_days": m["median_lead_days"] if m["median_lead_days"] is not None else 0.0,
            "std_lead_days": m["std_lead_days"] if m["std_lead_days"] is not None else 0.0,
            "pct_before_onset": m["pct_before_onset"] if m["pct_before_onset"] is not None else 0.0,
            "pct_on_onset": m["pct_on_onset"] if m["pct_on_onset"] is not None else 0.0,
            "pct_after_onset": m["pct_after_onset"] if m["pct_after_onset"] is not None else 0.0,
            "n_alerts": m["n_alerts"],
            "n_true": m["n_true"],
        })

    # Selection hierarchy
    c1 = [r for r in records if r["precision"] >= 0.80 and r["pct_before_onset"] >= 20.0 and r["n_alerts"] > 0]
    if c1:
        best_rec = max(c1, key=lambda r: (r["mean_lead_days"], r["f1"]))
        selection_rule = "precision >= 0.80 and pct_before_onset >= 20% (max mean lead time)"
    else:
        c2 = [r for r in records if r["f1"] >= 0.75 and r["n_alerts"] > 0]
        if c2:
            best_rec = max(c2, key=lambda r: (r["mean_lead_days"], r["f1"]))
            selection_rule = "fallback: F1 >= 0.75 (max mean lead time)"
        else:
            best_rec = max(records, key=lambda r: (r["f1"], r["mean_lead_days"]))
            selection_rule = "fallback: max F1 (warning: does not meet strict early-warning criteria)"

    best_threshold = float(best_rec["threshold"])
    return best_threshold, records, selection_rule, baseline_precision


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(payload: dict[str, Any], path: Path) -> None:
    """Human-readable report_trajectory.txt with 7 clearly separated sections."""
    ndvi = payload["ndvi_regression"]
    clf = payload["classification"]
    onset = payload["onset_detection"]
    alert = payload["alert_quality"]
    early = payload.get("early_warning", {})
    nutr = payload.get("nutrient_stress_deep_dive", {})
    flags = payload["proxy_flags"]
    classes = payload.get("classes", UNIFIED_CLASSES)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("KisanGPT - Stage 3A Trajectory & Early-Warning Evaluation")
    lines.append(f"generated: {payload['generated_at']}")
    lines.append("=" * 78)
    lines.append("")
    prov = payload.get("provenance", {})
    if prov:
        lines.append(f"checkpoint       : {prov.get('checkpoint_path', 'unknown')}")
        lines.append(f"parameters       : {prov.get('parameter_count', 'unknown'):,}" if isinstance(prov.get('parameter_count'), int) else f"parameters       : {prov.get('parameter_count', 'unknown')}")
        lines.append(f"dataset sha256   : {prov.get('dataset_sha256', 'unknown')}")
        lines.append(f"split sha256     : {prov.get('split_sha256', 'unknown')}")
        lines.append(f"selection rule   : {prov.get('selection_rule', 'unknown')}")
    lines.append(f"split            : {payload['split']} (by FIELD, leakage-checked disjoint)")
    lines.append(f"horizon          : {payload['t_history']}d history -> {payload['t_future']}d forecast")
    lines.append(f"alert horizon    : {payload.get('alert_horizon', 3)} days ahead")
    lines.append(f"encoder          : {payload['encoder']} (pooling={payload['pooling']}, checkpoint epoch {payload['best_epoch']})")
    lines.append(f"classes ({len(classes)}): {', '.join(classes)}")
    lines.append(f"NDVI PROXY       : {'YES - synthetic class-conditional curves' if flags['ndvi_is_proxy'] else 'no (real export)'}")
    lines.append(f"weather PROXY    : {'YES - declared synthetic' if flags['weather_is_proxy'] else 'no (real source: ' + flags['weather_source'] + ')'}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("1) NDVI FORECASTING PERFORMANCE (test fields)")
    lines.append("-" * 78)
    lines.append(f"{'method':<24}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'Day-1 MAE':>12}")
    for name, m in ndvi["methods"].items():
        d1 = m.get("day1_mae", m.get("per_day_mae", [0.0])[0])
        lines.append(f"{name:<24}{m['mae']:>10.4f}{m['rmse']:>10.4f}{m['r2']:>10.4f}{d1:>12.4f}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("2) STRESS-TRAJECTORY CLASSIFICATION PERFORMANCE")
    lines.append("-" * 78)
    for name, m in clf.items():
        lines.append(
            f"{name:<24} accuracy={m['accuracy']:.4f}  macro F1={m['macro_f1']:.4f}  "
            f"mean step acc={m['mean_step_accuracy']:.4f}"
        )
    pc = clf["trajectory_model"]["per_class_f1"]
    lines.append("model per-class F1 : " + ", ".join(f"{k}={v:.3f}" for k, v in pc.items()))
    lines.append("per-day accuracy (model): " + ", ".join(f"{v:.2f}" for v in clf["trajectory_model"]["step_accuracy"]))
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"3) STRESS ONSET DETECTION PERFORMANCE (within {payload.get('alert_horizon', 3)} days)")
    lines.append("-" * 78)
    lines.append(
        f"precision={onset['precision']:.4f}  recall={onset['recall']:.4f}  "
        f"F1={onset['f1']:.4f}  (tp={onset['tp']} fp={onset['fp']} fn={onset['fn']} of {onset['n']})"
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("4) K-DAY-AHEAD EARLY-WARNING PERFORMANCE & THRESHOLD TUNING")
    lines.append("-" * 78)
    lines.append(f"selected threshold       : {early.get('best_threshold', 0.50):.2f}")
    lines.append(f"tuning selection rule    : {early.get('selection_rule', 'precision >= 0.80 and pct_before_onset >= 20%')}")
    lines.append(f"always-alert base prec   : {early.get('baseline_precision', 0.0):.4f}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("5) ALERT LEAD-TIME ANALYSIS (HEALTHY-AT-ENDPOINT FIELDS)")
    lines.append("-" * 78)
    lines.append(f"evaluated healthy fields : {alert.get('n_healthy_evaluated', 0)}")
    lines.append(f"excluded ongoing stress  : {payload.get('n_already_stressed', 0)} fields (already stressed at history endpoint)")
    lines.append(f"matched alerts (TP)      : {alert.get('n_matched_true_positives', alert.get('tp', 0))}")
    lines.append(f"false alarms (FP)        : {alert.get('n_false_alarms', alert.get('fp', 0))}")
    lines.append(f"missed events (FN)       : {alert.get('n_missed_events', alert.get('fn', 0))}")
    def _fmt_or_na(val: float | None, prec_digits: int = 4, suffix: str = "") -> str:
        if val is None:
            return "N/A (undefined denominator)"
        return f"{val:.{prec_digits}f}{suffix}"

    lines.append(f"alert precision (TP/pred): {_fmt_or_na(alert.get('precision'), 4)}")
    lines.append(f"event recall (TP/actual) : {_fmt_or_na(alert.get('recall'), 4)}")
    lines.append(f"alert F1                 : {_fmt_or_na(alert.get('f1'), 4)}")
    lines.append(f"false positive rate (FPR): {_fmt_or_na(alert.get('false_positive_rate'), 4)}")
    lines.append(f"alerts fired             : {alert['n_alerts']}")
    lines.append(f"true onsets              : {alert['n_true']}")
    lines.append(f"mean lead time           : {_fmt_or_na(alert.get('mean_lead_days'), 2, ' days')}")
    lines.append(f"median lead time         : {_fmt_or_na(alert.get('median_lead_days'), 2, ' days')}")
    lines.append(f"lead time std dev        : {_fmt_or_na(alert.get('std_lead_days'), 2, ' days')}")
    lines.append(f"alerts fired BEFORE onset: {_fmt_or_na(alert.get('pct_before_onset'), 1, '%')} (among matched TPs)")
    lines.append(f"alerts fired ON onset day: {_fmt_or_na(alert.get('pct_on_onset'), 1, '%')} (among matched TPs)")
    lines.append(f"alerts fired AFTER onset : {_fmt_or_na(alert.get('pct_after_onset'), 1, '%')} (among matched TPs)")
    hist_str = ", ".join(f"{k}:{v}" for k, v in alert.get("lead_time_histogram", {}).items())
    lines.append(f"lead time histogram      : {hist_str}")
    lines.append("lead definition          : actual_onset_day - alert_day (positive = ahead of real onset)")
    if alert.get("warning"):
        lines.append(f">>> WARNING: {alert['warning']} <<<")
    if alert.get("example"):
        lines.append(f"example alert            : {alert['example']}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("6) PER-CLASS PERFORMANCE & NUTRIENT STRESS DEEP DIVE")
    lines.append("-" * 78)
    if nutr:
        lines.append(
            f"nutrient_stress: precision={nutr['precision']:.4f}  recall={nutr['recall']:.4f}  "
            f"F1={nutr['f1']:.4f}  support={nutr['support']} tokens"
        )
        lines.append("top confusions when ground truth is nutrient_stress:")
        for cls_k, cnt in nutr.get("confusions_when_actual", {}).items():
            lines.append(f"   -> predicted as {cls_k:<18}: {cnt:>5} ({cnt / max(nutr['support'], 1):.1%})")
        lines.append("\nRecommended telemetry features to separate nutrient stress from healthy/water stress:")
        for feat in nutr.get("recommended_features", []):
            lines.append(f"   * {feat}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("7) BASELINE COMPARISONS")
    lines.append("-" * 78)
    lines.append(f"{'method':<24}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for name, m in ndvi["methods"].items():
        lines.append(f"{name:<24}{m['mae']:>10.4f}{m['rmse']:>10.4f}{m['r2']:>10.4f}")
    lines.append("=" * 78)

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"report written to {path}")


def build_class_mapping(active_classes: list[str], merge_generic: bool = False) -> dict[int, int] | None:
    """Construct class mapping from UNIFIED_CLASSES to active_classes."""
    if active_classes == list(UNIFIED_CLASSES) and not merge_generic:
        return None
    mapping = {}
    for old_idx, old_name in enumerate(UNIFIED_CLASSES):
        if old_name in active_classes:
            mapping[old_idx] = active_classes.index(old_name)
        elif old_name == "stress_generic" and merge_generic:
            tgt_name = "water_stress" if "water_stress" in active_classes else active_classes[0]
            mapping[old_idx] = active_classes.index(tgt_name)
        else:
            mapping[old_idx] = -100
    return mapping


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for evaluate_trajectory.py."""
    parser = argparse.ArgumentParser(
        description="Stage 3A - evaluate the trajectory model against baselines."
    )
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alert_horizon", type=int, default=None, help="Early warning horizon in days (e.g. 3 or 5)")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class names")
    parser.add_argument("--merge_generic_into_nearest", action="store_true", default=None, help="Merge stress_generic")
    parser.add_argument("--checkpoint", type=str, default=cfg.TRAJ_CKPT, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data", type=str, default=cfg.TRAJ_DATA_NPZ, help="Path to trajectory dataset (.npz)")
    parser.add_argument("--split_file", type=str, default=cfg.TRAJ_SPLIT_JSON, help="Path to trajectory split (.json)")
    return parser


def main() -> int:
    """CLI entrypoint: ``python evaluate_trajectory.py --mode full``."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        checkpoint = Path(args.checkpoint)
        model, blob = load_trajectory_model(checkpoint)
    except FileNotFoundError as err:
        log(f"ERROR: {err}")
        return 1

    try:
        dataset_path = Path(args.data)
        split_path = Path(args.split_file)
        data = build_dataset(seed=args.seed, cache_path=dataset_path, read_only=True)
        split = load_split(seed=args.seed, split_path=split_path, dataset_path=dataset_path, read_only=True)
    except Exception as err:
        log(f"ERROR loading evaluation data (read-only): {type(err).__name__}: {err}")
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
    class_mapping = build_class_mapping(active_classes, merge_generic=merge_generic)

    field_indices = np.asarray(split[args.split], dtype=np.int64)
    log(
        f"evaluating {len(field_indices)} {args.split} fields | "
        f"classes={len(active_classes)} | alert_horizon={horizon}d | "
        f"encoder={ckpt_cfg.get('encoder')} | best epoch {blob.get('best_epoch')}"
    )

    t_history, t_future = data["t_history"], data["t_future"]

    # --- Step 1: Validation Threshold Tuning ---
    val_indices = np.asarray(split["val"], dtype=np.int64)
    try:
        val_preds = predict_split(
            model, data, val_indices,
            channel_mean=ckpt_cfg.get("input_mean"),
            channel_std=ckpt_cfg.get("input_std"),
            class_mapping=class_mapping,
            alert_horizon=horizon,
        )
        best_threshold, thresh_records, selection_rule, baseline_prec = tune_alert_threshold(
            val_preds["probs"],
            val_preds["y_cls_raw"],
            val_preds["early_probs"],
            val_preds["endpoint_stressed"],
            horizon=horizon,
            active_classes=active_classes,
        )
    except Exception as err:
        log(f"WARNING: validation threshold tuning encountered error: {err}. Using default 0.50")
        best_threshold = 0.50
        thresh_records = []
        selection_rule = "default 0.50"
        baseline_prec = 0.0

    threshold_payload = {
        "best_threshold": round(best_threshold, 2),
        "baseline_precision": round(baseline_prec, 4),
        "alert_horizon": horizon,
        "selection_rule": selection_rule,
        "records": thresh_records,
    }
    thresh_path = Path("artifacts/trajectory_alert_threshold.json")
    thresh_path.parent.mkdir(parents=True, exist_ok=True)
    thresh_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
    log(f"alert threshold tuning saved to {thresh_path} (selected threshold = {best_threshold:.2f}, rule='{selection_rule}')")

    if thresh_records:
        plot_alert_threshold_curve(
            thresh_records, best_threshold, baseline_prec,
            Path("plots/trajectory_alert_threshold_curve.png")
        )

    # --- Step 2: Prediction on target split ---
    try:
        preds = predict_split(
            model, data, field_indices,
            channel_mean=ckpt_cfg.get("input_mean"),
            channel_std=ckpt_cfg.get("input_std"),
            class_mapping=class_mapping,
            alert_horizon=horizon,
        )
    except Exception as err:
        log(f"ERROR during inference: {type(err).__name__}: {err}")
        return 1

    y_ndvi, y_cls, y_cls_raw, ndvi_pred, probs = (
        preds["y_ndvi"], preds["y_cls"], preds["y_cls_raw"], preds["ndvi_pred"], preds["probs"]
    )
    early_probs = preds["early_probs"]
    history = preds["history"]
    endpoint_stressed = preds["endpoint_stressed"]

    # --- 3. NDVI regression + baselines ---
    persist = np.stack(
        [persistence_forecast(h, t_future) for h in tqdm(history, desc="baselines", leave=False)]
    )
    linear = np.stack(
        [linear_forecast(h, t_future) for h in tqdm(history, desc="baselines", leave=False)]
    )
    ndvi_methods = {
        "persistence": regression_metrics(y_ndvi, persist),
        "linear_extrapolation": regression_metrics(y_ndvi, linear),
        "trajectory_model": regression_metrics(y_ndvi, ndvi_pred),
    }
    log("NDVI: " + " | ".join(
        f"{name} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f}"
        for name, m in ndvi_methods.items()
    ))

    # --- 4. Stress classification ---
    cls_pred = probs.argmax(axis=-1)
    clf_methods = {
        "trajectory_model": classification_report(y_cls, cls_pred, class_names=active_classes),
        "last_observed_class": classification_report(
            y_cls,
            np.repeat(decode_labels_from_ndvi(history)[:, -1][:, None], t_future, axis=1),
            class_names=active_classes,
        ),
        "always_healthy": classification_report(
            y_cls, np.zeros_like(y_cls, dtype=int),
            class_names=active_classes,
        ),
    }

    valid_mask = (y_cls.ravel() != -100)
    y_true_valid = y_cls.ravel()[valid_mask]
    y_pred_valid = cls_pred.ravel()[valid_mask]
    n_c = len(active_classes)
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(n_c)))
    log("classification: model accuracy="
        f"{clf_methods['trajectory_model']['accuracy']:.4f} macro F1="
        f"{clf_methods['trajectory_model']['macro_f1']:.4f}")

    # --- 5. Onset detection (within alert horizon) ---
    actual_onsets_all = [first_onset(row[:horizon]) for row in y_cls_raw]
    actual_pos_all = np.array([d is not None for d in actual_onsets_all], dtype=bool)

    pred_onsets_all = [first_onset(row[:horizon]) for row in cls_pred]
    predicted_pos_all = np.array([d is not None for d in pred_onsets_all], dtype=bool)

    onset_scores = binary_onset_scores(actual_pos_all, predicted_pos_all)
    log(f"onset({horizon}d): P={onset_scores['precision']:.4f} "
        f"R={onset_scores['recall']:.4f} F1={onset_scores['f1']:.4f}")

    # --- 6. Alert quality (Evaluated on healthy-at-endpoint fields) ---
    healthy_eval_mask = ~np.asarray(endpoint_stressed, dtype=bool)
    n_already_stressed = int(endpoint_stressed.sum())
    n_healthy_eval = int(healthy_eval_mask.sum())
    log(f"test split composition: {n_healthy_eval} healthy at endpoint, {n_already_stressed} already stressed")

    h_probs = probs[healthy_eval_mask]
    h_early = early_probs[healthy_eval_mask] if early_probs is not None else None
    h_cls_raw = y_cls_raw[healthy_eval_mask]

    actual_onsets = []
    for row in h_cls_raw:
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
    alert_quality = compute_alert_metrics(
        alerts=alerts,
        actual_onsets=actual_onsets,
        actual_pos=actual_pos,
        n_healthy_evaluated=n_healthy_eval,
        best_threshold=best_threshold,
    )
    lead_times = alert_quality["lead_times"]
    prec = alert_quality["precision"] if alert_quality["precision"] is not None else 0.0
    rec = alert_quality["recall"] if alert_quality["recall"] is not None else 0.0
    alert_f1 = alert_quality["f1"] if alert_quality["f1"] is not None else 0.0
    fpr = alert_quality["false_positive_rate"] if alert_quality["false_positive_rate"] is not None else 0.0
    mean_lead = alert_quality["mean_lead_days"] if alert_quality["mean_lead_days"] is not None else 0.0
    median_lead = alert_quality["median_lead_days"] if alert_quality["median_lead_days"] is not None else 0.0
    std_lead = alert_quality["std_lead_days"] if alert_quality["std_lead_days"] is not None else 0.0
    pct_before = alert_quality["pct_before_onset"] if alert_quality["pct_before_onset"] is not None else 0.0
    pct_on = alert_quality["pct_on_onset"] if alert_quality["pct_on_onset"] is not None else 0.0
    pct_after = alert_quality["pct_after_onset"] if alert_quality["pct_after_onset"] is not None else 0.0
    warning_msg = alert_quality["warning"]
    tp = alert_quality["tp"]
    fp = alert_quality["fp"]
    fn = alert_quality["fn"]
    tn = alert_quality["tn"]
    log(f"alerts (thresh={best_threshold:.2f}): P={prec:.4f} R={rec:.4f} F1={alert_f1:.4f} "
        f"mean lead={mean_lead:.2f}d (before: {pct_before:.1f}%, on: {pct_on:.1f}%)")

    # --- 7. Nutrient stress deep dive ---
    nutrient_deep_dive = {}
    if "nutrient_stress" in active_classes:
        n_idx = active_classes.index("nutrient_stress")
        actual_is_nutr = (y_true_valid == n_idx)
        pred_is_nutr = (y_pred_valid == n_idx)
        n_support = int(actual_is_nutr.sum())
        n_tp = int((actual_is_nutr & pred_is_nutr).sum())
        n_fp = int((~actual_is_nutr & pred_is_nutr).sum())
        n_fn = int((actual_is_nutr & ~pred_is_nutr).sum())
        n_prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
        n_rec = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
        n_f1 = 2 * n_prec * n_rec / (n_prec + n_rec) if (n_prec + n_rec) else 0.0

        conf_counts = {}
        for other_idx, other_name in enumerate(active_classes):
            if other_idx != n_idx:
                conf_counts[other_name] = int(((y_true_valid == n_idx) & (y_pred_valid == other_idx)).sum())

        nutrient_deep_dive = {
            "precision": round(n_prec, 4),
            "recall": round(n_rec, 4),
            "f1": round(n_f1, 4),
            "support": n_support,
            "confusions_when_actual": conf_counts,
            "recommended_features": [
                "NDVI 7-day velocity/slope (chlorosis is gradual, not abrupt drought drop)",
                "Precipitation rolling sum (if rainfall normal but NDVI declining -> nutrient deficiency)",
                "Soil NPK sensors or fertilizer application dates",
                "Soil moisture decoupling (dry soil = drought; wet soil + low NDVI = nutrient leaching/deficiency)",
            ],
        }

    # --- 8. Figures ---
    curves_path = Path("plots/trajectory_curves.png")
    cm_path = Path("plots/trajectory_confusion_matrix.png")
    baseline_path = Path("plots/trajectory_baselines.png")
    hist_path = Path("plots/trajectory_lead_time_histogram.png")

    plot_sample_curves(history, y_ndvi, ndvi_pred, curves_path)
    plot_confusion(cm, cm_path, class_names=active_classes)
    plot_baselines(
        [(k, ndvi_methods[k]) for k in ndvi_methods],
        [(k, clf_methods[k]) for k in ("trajectory_model", "last_observed_class", "always_healthy")],
        baseline_path,
    )
    plot_lead_time_histogram(lead_times, pct_before, pct_on, pct_after, hist_path)

    # Provenance tracking
    def _file_sha256(filepath: Path | str) -> str:
        p = Path(filepath)
        if not p.is_file():
            return "not_found"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    dataset_consumed_path = Path(data.get("loaded_path", args.data))
    split_consumed_path = Path(split.get("loaded_path", args.split_file))
    dataset_hash = _file_sha256(dataset_consumed_path)
    split_hash = _file_sha256(split_consumed_path)
    n_params = sum(p.numel() for p in model.parameters())

    # --- Persist ---
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": {
            "checkpoint_path": str(checkpoint),
            "data_path": str(dataset_consumed_path),
            "dataset_sha256": dataset_hash,
            "split_path": str(split_consumed_path),
            "split_sha256": split_hash,
            "parameter_count": n_params,
            "class_mapping": class_mapping,
            "selection_rule": selection_rule,
            "channel_mean": ckpt_cfg.get("input_mean"),
            "channel_std": ckpt_cfg.get("input_std"),
        },
        "split": args.split,
        "split_sizes": {k: len(split[k]) for k in ("train", "val", "test")},
        "split_by": split.get("split_by", "field"),
        "n_eval_fields": int(len(field_indices)),
        "n_already_stressed": int(n_already_stressed),
        "n_healthy_evaluated": int(n_healthy_eval),
        "t_history": int(t_history),
        "t_future": int(t_future),
        "alert_horizon": int(horizon),
        "classes": active_classes,
        "encoder": str(ckpt_cfg.get("encoder", cfg.TRAJ_ENCODER)),
        "pooling": str(ckpt_cfg.get("pooling", cfg.TRAJ_POOLING)),
        "standardized_inputs": bool(ckpt_cfg.get("input_mean")),
        "best_epoch": int(blob.get("best_epoch", 0)),
        "seed": int(args.seed),
        "proxy_flags": {
            "ndvi_is_proxy": bool(data["ndvi_is_proxy"]),
            "weather_is_proxy": bool(data["weather_is_proxy"]),
            "weather_source": data["weather_source"],
        },
        "ndvi_regression": {"methods": ndvi_methods},
        "classification": clf_methods,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": active_classes,
        "onset_detection": onset_scores,
        "early_warning": threshold_payload,
        "alert_quality": alert_quality,
        "nutrient_stress_deep_dive": nutrient_deep_dive,
        "baselines": ["persistence", "linear_extrapolation"],
        "figures": [
            str(curves_path),
            str(cm_path),
            str(baseline_path),
            str(thresh_path),
            str(hist_path),
        ],
    }

    metrics_path = Path(cfg.TRAJ_METRICS_JSON)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"metrics written to {metrics_path}")
    write_report(payload, Path("report_trajectory.txt"))

    # --- Console comparison table ---
    print()
    print("-" * 78)
    print(f"{'NDVI method':<24}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    print("-" * 78)
    for name, m in ndvi_methods.items():
        print(f"{name:<24}{m['mae']:>10.4f}{m['rmse']:>10.4f}{m['r2']:>10.4f}")
    print("-" * 78)
    print(f"{'class method':<24}{'accuracy':>10}{'macro F1':>10}")
    print("-" * 78)
    for name, m in clf_methods.items():
        print(f"{name:<24}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}")
    print("-" * 78)
    print(f"Early-Warning Alerts (Horizon {horizon}d, Threshold {best_threshold:.2f}, Rule: '{selection_rule}'):")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {alert_f1:.4f} | FPR: {fpr:.4f}")
    print(f"  Lead Time: {mean_lead:.2f} days ahead (Median: {median_lead:.2f}d, Std: {std_lead:.2f}d)")
    print(f"  Distribution: Before Onset: {pct_before:.1f}% | On Onset: {pct_on:.1f}% | After Onset: {pct_after:.1f}%")
    if warning_msg:
        print(f"  {warning_msg}")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
