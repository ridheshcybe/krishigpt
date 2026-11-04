"""Stage 4.2a - farmer feedback analytics + retraining trigger.

    python feedback_loop.py --analyze        # summary only (artifacts/feedback_summary.json)
    python feedback_loop.py --check-retrain  # summary + set artifacts/retrain_flag.json

Reads ``feedback_log.csv`` written by the Stage-2B dashboard (schema:
``timestamp, image_name, predicted_class, confidence, user_actual_class,
feedback``) and answers three questions:

1. Which classes attract the most "incorrect" feedback (and what the agronomist
   said the real class was - the predicted -> actual confusion pairs).
2. Is accuracy improving over time?  Weekly accuracy trend
   (correct / total per ISO week of the timestamp).
3. Do the modalities differ between correct and incorrect predictions?  Only
   answerable when attention weights were logged; the Stage-2B log schema does
   not contain them by default, so the analysis reports ``available: false``
   with the reason instead of inventing a correlation.

Retrain trigger (spec): **> 50 corrections for any class** sets
``retrain_flag = True``.  A "correction" is a row with ``feedback == incorrect``
AND a non-empty ``user_actual_class`` (only those carry a usable label for
retraining).  The per-class key is the *actual* class being asserted, because
that is the label the new training sample will receive.  Per-predicted-class
counts are still reported in the summary for diagnosis.

Reuses Stage 2B's ``load_feedback_log`` (which owns schema migration and
missing-file handling) so the log format has exactly one owner.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as cfg

FEEDBACK_LOG_PATH = Path("feedback_log.csv")
SUMMARY_JSON = Path("artifacts/feedback_summary.json")
FLAG_JSON = Path("artifacts/retrain_flag.json")

#: Retrain when any class accumulates more corrections than this (spec).
CORRECTION_THRESHOLD = 50
#: Feedback values written by components/feedback.py.
VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"
#: Optional attention columns - analysed only if the log actually carries them.
ATTENTION_COLUMNS = ("attention_image", "attention_tabular", "attention_ndvi")


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[stage4 {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Loading (reuse Stage 2B, single owner of the log format)
# ---------------------------------------------------------------------------
def load_log(path: Path) -> pd.DataFrame:
    """Load the feedback log, tolerating a missing/empty/corrupt file."""
    if path != FEEDBACK_LOG_PATH and path.exists():
        # Non-default log (e.g. a copied/exported one) - read directly.
        try:
            return pd.read_csv(path)
        except Exception as err:
            log(f"WARNING: cannot read {path} ({err}); treating as empty")
            return pd.DataFrame()
    try:
        from components.feedback import load_feedback_log

        return load_feedback_log()
    except ImportError as err:
        # Edge install without streamlit: fall back to a plain read.
        log(f"NOTE: streamlit unavailable ({err}); reading the CSV directly")
        if not FEEDBACK_LOG_PATH.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(FEEDBACK_LOG_PATH)
        except Exception as exc:
            log(f"WARNING: cannot read {FEEDBACK_LOG_PATH} ({exc})")
            return pd.DataFrame()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column types/labels without touching the source file."""
    if df.empty:
        return df
    out = df.copy()
    for column in ("timestamp", "predicted_class", "user_actual_class", "feedback"):
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str).str.strip()
    out["feedback"] = out["feedback"].str.lower()
    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    else:
        out["confidence"] = np.nan
    out["timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# 2. Statistics
# ---------------------------------------------------------------------------
def weekly_trend(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Accuracy (correct/total) per ISO week of the feedback timestamp."""
    valid = df.dropna(subset=["timestamp_dt"])
    if valid.empty:
        return []
    keyed = valid["timestamp_dt"].dt.isocalendar()
    grouped = valid.groupby([keyed["year"], keyed["week"]], sort=True)
    trend = []
    for (year, week), group in grouped:
        correct = int((group["feedback"] == VERDICT_CORRECT).sum())
        total = int(len(group))
        trend.append({
            "week": f"{int(year)}-W{int(week):02d}",
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else None,
        })
    return trend


def incorrect_breakdown(df: pd.DataFrame) -> dict[str, Any]:
    """Which predicted classes get flagged, and what the correct class was."""
    wrong = df[df["feedback"] == VERDICT_INCORRECT]
    by_predicted: dict[str, int] = {}
    for value in wrong["predicted_class"]:
        key = value or "(empty)"
        by_predicted[key] = by_predicted.get(key, 0) + 1
    pairs: dict[str, int] = {}
    for pred, actual in zip(wrong["predicted_class"], wrong["user_actual_class"]):
        key = f"{pred or '(empty)'} -> {actual or '(unspecified)'}"
        pairs[key] = pairs.get(key, 0) + 1
    ordered = dict(sorted(by_predicted.items(), key=lambda kv: -kv[1]))
    return {
        "n_incorrect": int(len(wrong)),
        "by_predicted_class": ordered,
        "hottest_class": next(iter(ordered), None),
        "predicted_to_actual": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
    }


def corrections_by_actual_class(df: pd.DataFrame) -> dict[str, int]:
    """Count usable corrections (incorrect + labelled) per *actual* class.

    These are the counts the retrain trigger uses: each one becomes a training
    sample with ``user_actual_class`` as its label.
    """
    wrong = df[
        (df["feedback"] == VERDICT_INCORRECT)
        & (df["user_actual_class"] != "")
        & (~df["user_actual_class"].str.lower().isin(["none", "nan", "n/a"]))
    ]
    counts: dict[str, int] = {}
    for value in wrong["user_actual_class"]:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def modality_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Mean attention per modality for correct vs incorrect rows, if logged."""
    present = [c for c in ATTENTION_COLUMNS if c in df.columns]
    if not present:
        return {
            "available": False,
            "reason": "feedback_log.csv has no attention columns "
                      f"({', '.join(ATTENTION_COLUMNS)}); the Stage-2B logger "
                      "records verdicts only. Add them to append_feedback_record "
                      "to enable modality-vs-error correlation.",
        }
    out: dict[str, Any] = {"available": True, "columns": present, "groups": {}}
    for verdict in (VERDICT_CORRECT, VERDICT_INCORRECT):
        group = df[df["feedback"] == verdict]
        if group.empty:
            continue
        out["groups"][verdict] = {
            column.replace("attention_", ""): round(float(group[column].mean()), 4)
            for column in present
            if pd.to_numeric(group[column], errors="coerce").notna().any()
        }
        out["groups"][verdict]["n"] = int(len(group))
    return out


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Full feedback summary payload (everything except the retrain flag)."""
    total = int(len(df))
    correct = int((df["feedback"] == VERDICT_CORRECT).sum()) if total else 0
    incorrect = int((df["feedback"] == VERDICT_INCORRECT).sum()) if total else 0
    scored = correct + incorrect
    corrections = corrections_by_actual_class(df) if total else {}

    mean_conf_correct = mean_conf_incorrect = None
    if total and "confidence" in df.columns:
        for verdict, holder in ((VERDICT_CORRECT, "correct"),
                                (VERDICT_INCORRECT, "incorrect")):
            values = df.loc[df["feedback"] == verdict, "confidence"].dropna()
            value = round(float(values.mean()), 4) if len(values) else None
            if holder == "correct":
                mean_conf_correct = value
            else:
                mean_conf_incorrect = value

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": str(FEEDBACK_LOG_PATH),
        "totals": {
            "rows": total,
            "correct": correct,
            "incorrect": incorrect,
            "unlabelled": int(total - scored),
            "observed_accuracy": round(correct / scored, 4) if scored else None,
            "mean_confidence_correct": mean_conf_correct,
            "mean_confidence_incorrect": mean_conf_incorrect,
        },
        "incorrect": incorrect_breakdown(df) if total else
                     {"n_incorrect": 0, "by_predicted_class": {},
                      "hottest_class": None, "predicted_to_actual": {}},
        "weekly_trend": weekly_trend(df) if total else [],
        "corrections_by_actual_class": corrections,
        "modality": modality_analysis(df) if total else {"available": False},
        "retrain": {
            "threshold_per_class": CORRECTION_THRESHOLD,
            "max_corrections": max(corrections.values()) if corrections else 0,
            "would_trigger": bool(
                corrections and max(corrections.values()) > CORRECTION_THRESHOLD
            ),
        },
    }
    return summary


def write_flag(summary: dict[str, Any]) -> dict[str, Any]:
    """Set/clear artifacts/retrain_flag.json from the correction counts."""
    counts = summary["corrections_by_actual_class"]
    per_class = {
        name: {"corrections": count,
               "exceeds_threshold": count > CORRECTION_THRESHOLD}
        for name, count in counts.items()
    }
    triggered = any(v["exceeds_threshold"] for v in per_class.values())
    payload = {
        "retrain_flag": triggered,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": f"> {CORRECTION_THRESHOLD} corrections (incorrect + labelled) "
                "for any actual class",
        "per_class": per_class,
        "total_corrections": sum(counts.values()),
        "next_step": "python retrain.py --run" if triggered
                     else "no action yet - keep collecting feedback",
    }
    FLAG_JSON.parent.mkdir(parents=True, exist_ok=True)
    FLAG_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def print_summary(summary: dict[str, Any]) -> None:
    """Human-readable console summary."""
    totals = summary["totals"]
    print()
    print("-" * 64)
    print(" KisanGPT feedback summary")
    print("-" * 64)
    print(f" rows .................. {totals['rows']}")
    print(f" correct / incorrect ... {totals['correct']} / {totals['incorrect']}"
          + (f"  (observed accuracy {totals['observed_accuracy']:.1%})"
             if totals["observed_accuracy"] is not None else ""))
    incorrect = summary["incorrect"]
    if incorrect["n_incorrect"]:
        print(f" most-flagged class .... {incorrect['hottest_class']} "
              f"({max(incorrect['by_predicted_class'].values())} flags)")
        for pair, count in list(incorrect["predicted_to_actual"].items())[:3]:
            print(f"   confusions: {pair} x{count}")
    corrections = summary["corrections_by_actual_class"]
    if corrections:
        print(" corrections per actual class:")
        for name, count in corrections.items():
            marker = " <-- exceeds threshold" if count > CORRECTION_THRESHOLD else ""
            print(f"   {name:<18} {count}{marker}")
    else:
        print(" corrections (incorrect + labelled) ... 0")
    if summary["weekly_trend"]:
        print(" weekly accuracy trend:")
        for point in summary["weekly_trend"]:
            bar = "#" * int(round((point["accuracy"] or 0) * 30))
            print(f"   {point['week']}  {point['accuracy']:.2%}  ({point['total']})  {bar}")
    modality = summary["modality"]
    if not modality.get("available"):
        print(f" modality analysis ...... unavailable ({modality.get('reason', '')[:70]}...)")
    print("-" * 64)


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage-4.2a CLI options."""
    parser = argparse.ArgumentParser(
        description="Analyse feedback_log.csv and decide whether to retrain."
    )
    parser.add_argument("--analyze", action="store_true",
                        help="write artifacts/feedback_summary.json (summary only)")
    parser.add_argument("--check-retrain", action="store_true",
                        help="summary + set artifacts/retrain_flag.json")
    parser.add_argument("--log", type=Path, default=FEEDBACK_LOG_PATH,
                        help="feedback CSV to analyse")
    parser.add_argument("--threshold", type=int, default=CORRECTION_THRESHOLD,
                        help="corrections per class that trigger retraining (50)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if not args.analyze and not args.check_retrain:
        args.analyze = True  # default action = summary only (spec: --analyze)

    global CORRECTION_THRESHOLD
    CORRECTION_THRESHOLD = max(1, int(args.threshold))

    if not args.log.exists():
        log(f"NOTE: {args.log} not found - no feedback yet. "
            "Diagnose samples in the dashboard (streamlit run app.py) and press "
            "Correct/Incorrect to start the loop.")
    df = _clean(load_log(args.log))
    if df.empty:
        log("feedback log is empty; writing an empty summary")

    summary = build_summary(df)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"summary -> {SUMMARY_JSON}")
    print_summary(summary)

    if args.check_retrain:
        flag = write_flag(summary)
        status = "SET" if flag["retrain_flag"] else "not set"
        log(f"retrain_flag {status} -> {FLAG_JSON} "
            f"(max corrections {flag['total_corrections']} "
            f"/ threshold {CORRECTION_THRESHOLD} per class)")
        if flag["retrain_flag"]:
            log("next: python retrain.py --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
