"""Stage 3B - alert panel.

Three alert families, all computed from the Stage-3A model's own outputs
(``trajectory_model.detect_onset_alerts``) - no alert text is hand-written:

1. **Onset alerts** - ``"Field 3: water_stress onset in 4 days (confidence 82%)"``
2. **Escalation alerts** - an already-stressed field whose predicted
   confidence keeps climbing: ``"Field 7: disease_stress escalating, 60%
   confidence by day 10"``
3. **Coverage alerts** - farm-level spread: ``"disease_stress reaches 45%
   coverage by day 8"`` where coverage = share of grid cells in that class.

Severity coding follows the spec's green / yellow / red scheme and reuses
Streamlit's ``success`` / ``warning`` / ``error`` so it stays theme-aware:

    confidence < 0.50  -> green   (watch)
    0.50 - 0.80        -> yellow  (act soon)
    > 0.80             -> red     (act now)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from components_twin.field_map import CLASS_LABELS
from stage1_config import UNIFIED_CLASSES

#: Severity thresholds (spec: green / yellow / red).
SEVERITY_GREEN = 0.50
SEVERITY_RED = 0.80
#: Escalation: confidence must climb by at least this many points.
ESCALATION_DELTA = 0.15
#: Coverage: share of cells in one class that triggers a farm-level alert.
COVERAGE_THRESHOLD = 0.30
#: How many alerts each family prints before summarising the rest.  A 100-cell
#: grid can raise ~65 onset alerts; printing them all buries the panel.
MAX_ALERTS_SHOWN = 6


def _severity(confidence: float) -> str:
    """Map a confidence in [0, 1] to ``green`` / ``yellow`` / ``red``."""
    if confidence >= SEVERITY_RED:
        return "red"
    if confidence >= SEVERITY_GREEN:
        return "yellow"
    return "green"


def collect_onset_alerts(
    pred_classes: np.ndarray,
    pred_probs: np.ndarray,
    field_ids: np.ndarray,
    horizon: int = 7,
    min_conf: float = 0.0,
) -> list[dict[str, Any]]:
    """Onset alerts for the visible cells, using Stage-3A's alert logic.

    Args:
        pred_classes: ``(n_fields, t_future)`` predicted class indices.
        pred_probs: ``(n_fields, t_future, 6)`` per-day probabilities.
        field_ids: field identifiers matching the first axis.
        horizon: look-ahead window (default 7 days, as in Stage 3A).
        min_conf: optional confidence floor.

    Returns:
        One dict per alerting field: ``field``, ``class``, ``class_idx``,
        ``day_offset``, ``confidence``, ``message``, ``severity``.
    """
    from trajectory_model import detect_onset_alerts  # Stage 3A reuse

    records = detect_onset_alerts(
        pred_probs, horizon=horizon, min_conf=min_conf
    )
    alerts: list[dict[str, Any]] = []
    for fid, record in zip(field_ids, records):
        if not record["alerted"]:
            continue
        confidence = float(record["confidence"])
        alerts.append(
            {
                "field": int(fid),
                "class": str(record["class"]),
                "class_idx": int(record["class_idx"]),
                "day_offset": int(record["day"]),
                "confidence": confidence,
                "message": f"Field {int(fid)}: {record['message']}",
                "severity": _severity(confidence),
                "kind": "onset",
            }
        )
    alerts.sort(key=lambda a: (-a["confidence"], a["day_offset"]))
    return alerts


def collect_escalation_alerts(
    pred_classes: np.ndarray,
    pred_probs: np.ndarray,
    field_ids: np.ndarray,
    origin_classes: np.ndarray,
    delta: float = ESCALATION_DELTA,
    cap: int = 5,
    t_history: int | None = None,
) -> list[dict[str, Any]]:
    """Fields already under stress whose confidence keeps climbing.

    An alert fires when the predicted probability of the field's *current*
    stress class grows by ``delta`` or more between today and the best future
    day.  At most ``cap`` alerts are returned, ordered by how much confidence
    grows, so the panel stays readable.  ``day`` is reported as an **absolute**
    day number (matching the chart x-axis), i.e. ``t_history + step``.
    """
    if t_history is None:
        import config as _cfg

        t_history = _cfg.TRAJ_T_HISTORY
    alerts: list[dict[str, Any]] = []
    for i, fid in enumerate(field_ids):
        current = int(origin_classes[i])
        if current == 0:
            continue  # only fields already stressed can "escalate"
        series = pred_probs[i, :, current]
        if series.size == 0:
            continue
        best = int(np.argmax(series))
        growth = float(series[best]) - float(series[0])
        if growth < delta:
            continue
        alerts.append(
            {
                "field": int(fid),
                "class": UNIFIED_CLASSES[current],
                "class_idx": current,
                "day_offset": t_history + best,
                "confidence": float(series[best]),
                "growth": growth,
                "message": (
                    f"Field {int(fid)}: {CLASS_LABELS[UNIFIED_CLASSES[current]]} "
                    f"escalating, {series[best]:.0%} confidence by day "
                    f"{t_history + best}"
                ),
                "severity": _severity(float(series[best])),
                "kind": "escalation",
            }
        )
    alerts.sort(key=lambda a: -a["growth"])
    return alerts[:cap]


def collect_coverage_alerts(
    pred_classes: np.ndarray,
    t_history: int,
    min_share: float = COVERAGE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Farm-level alerts: share of cells in one class, per future day.

    Coverage is computed over the visible grid only (that is the area the
    farmer is looking at), and reported once per class at its peak day.
    """
    alerts: list[dict[str, Any]] = []
    if pred_classes.size == 0:
        return alerts
    n_fields = pred_classes.shape[0]
    for cls_idx, cls_name in enumerate(UNIFIED_CLASSES):
        if cls_idx == 0:
            continue
        share = (pred_classes == cls_idx).mean(axis=0)
        peak = int(np.argmax(share))
        if float(share[peak]) < min_share:
            continue
        alerts.append(
            {
                "class": cls_name,
                "class_idx": cls_idx,
                "day_offset": t_history + peak,
                "confidence": float(share[peak]),
                "message": (
                    f"{CLASS_LABELS[cls_name]} reaches "
                    f"{share[peak]:.0%} coverage by day {t_history + peak} "
                    f"({n_fields} cells in view)"
                ),
                "severity": _severity(float(share[peak])),
                "kind": "coverage",
            }
        )
    alerts.sort(key=lambda a: -a["confidence"])
    return alerts


def _emit(message: str, severity: str) -> None:
    """Render one alert with the matching green / yellow / red treatment."""
    if severity == "red":
        st.error(message, icon="🚨")
    elif severity == "yellow":
        st.warning(message, icon="⚠️")
    else:
        st.success(message, icon="👁️")


def render_alert_panel(
    onset: list[dict[str, Any]],
    escalation: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    horizon: int = 7,
) -> None:
    """Render the three alert families with severity colouring.

    Empty families are reported positively (a quiet panel is information, not
    an error) rather than left blank.
    """
    st.markdown("### 🔔 Alerts")
    total = len(onset) + len(escalation) + len(coverage)
    st.caption(
        f"{len(onset)} onset · {len(escalation)} escalation · "
        f"{len(coverage)} coverage — next {horizon} days"
    )

    if total == 0:
        st.success(
            "No stress onset predicted in the next "
            f"{horizon} days for the cells in view.",
            icon="✅",
        )
        return

    with st.container(border=True):
        for title, family in (
            ("Onset", onset),
            ("Escalation", escalation),
            ("Coverage (whole grid)", coverage),
        ):
            if not family:
                continue
            st.markdown(f"**{title}**")
            for alert in family[:MAX_ALERTS_SHOWN]:
                _emit(alert["message"], alert["severity"])
            hidden = len(family) - MAX_ALERTS_SHOWN
            if hidden > 0:
                st.caption(
                    f"…and {hidden} more {title.lower()} alerts - sort by "
                    "confidence or select a narrower grid page."
                )
