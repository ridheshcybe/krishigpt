"""Stage 3B - KisanGPT Digital Twin dashboard (Streamlit).

Animated view of the Stage-3A temporal digital twin: a 10x10 farm grid whose
cells change colour across history and forecast, an NDVI trajectory chart with
an uncertainty band and onset markers, a severity-coded alert panel, and a
cell drill-down that reuses the Stage-2 diagnosis engine.

Run locally:

    streamlit run dashboard_twin.py
    streamlit run dashboard_twin.py --server.port 8502
    streamlit run dashboard_twin.py --server.headless true

Layout
------
    sidebar      model/data status, field selector, grid page, view mode,
                 history/forecast/both toggle, day scrubber + play, LLM
                 toggle, PROXY disclosure
    main top     animated field map (click a cell to select it)
    main middle  NDVI trajectory (left) | alert panel (right)
    main bottom  cell drill-down with the Stage-2 full diagnosis

Everything heavy (model + dataset + per-field predictions) is cached, so moving
the day slider never re-runs inference.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import streamlit as st

import config as cfg
from components_twin.alert_panel import (
    collect_coverage_alerts,
    collect_onset_alerts,
    render_alert_panel,
)
from components_twin.drilldown import render_drilldown
from components_twin.field_map import (
    CLASS_COLORS,
    CLASS_LABELS,
    render_day_scrubber,
    render_field_map,
)
from components_twin.trajectory_chart import render_trajectory_chart
from stage1_config import UNIFIED_CLASSES, UNIFIED_INDEX

st.set_page_config(
    page_title="KisanGPT — Digital Twin",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

T_HISTORY = cfg.TRAJ_T_HISTORY
T_FUTURE = cfg.TRAJ_T_FUTURE
T_TOTAL = T_HISTORY + T_FUTURE
TODAY = T_HISTORY - 1


# ---------------------------------------------------------------------------
# Cached loads (model + data once, predictions per visible cell set)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_twin_bundle() -> dict[str, Any]:
    """Load the Stage-3A checkpoint and trajectory dataset exactly once.

    Raises with an actionable message when an artifact is missing:
    ``FileNotFoundError`` carries the training command, data problems surface
    the dataset command from ``trajectory_data``.
    """
    from trajectory_data import build_dataset, load_split
    from trajectory_model import load_trajectory_model

    model, blob = load_trajectory_model()   # FileNotFoundError + command
    try:
        data = build_dataset()              # cached .npz, PROXY flags inside
    except Exception as err:
        raise RuntimeError(
            f"{err}\n\nBuild it first: python trajectory_data.py --validate-only"
        ) from err
    split = load_split()

    stats = blob.get("config", {})
    return {
        "model": model,
        "meta": stats,
        "checkpoint_meta": blob,
        "ndvi": data["ndvi"],
        "weather": data["weather"],
        "labels": data["labels"],
        "n_fields": int(data["n_fields"]),
        "ndvi_is_proxy": bool(data["ndvi_is_proxy"]),
        "weather_is_proxy": bool(data["weather_is_proxy"]),
        "weather_source": data["weather_source"],
        "split": split,
        "input_mean": stats.get("input_mean"),
        "input_std": stats.get("input_std"),
    }


@st.cache_data(show_spinner=False)
def predict_fields(field_ids: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Forecast the given cells once; cached until the cell set changes.

    Returns ``probs (F, T_future, 6)``, ``classes (F, T_future)``,
    ``ndvi_pred (F, T_future)`` and ``conf (F, T_future)``.  Standardisation
    uses the statistics stored **in the checkpoint**, so inference matches
    training bit for bit.
    """
    import torch

    bundle = load_twin_bundle()
    model = bundle["model"]
    ids = np.asarray(field_ids, dtype=np.int64)

    raw = np.concatenate(
        [bundle["ndvi"][ids, :T_HISTORY, None], bundle["weather"][ids, :T_HISTORY, :]],
        axis=2,
    ).astype(np.float32)

    mean, std = bundle["input_mean"], bundle["input_std"]
    if mean is not None and std is not None:
        raw = (raw - np.asarray(mean, dtype=np.float32)) / np.asarray(
            std, dtype=np.float32
        )

    with torch.no_grad():
        out = model.predict(torch.from_numpy(raw))
    probs = out["probs"].numpy()
    classes = out["classes"].numpy()
    conf = probs.max(axis=-1)
    return {
        "probs": probs,
        "classes": classes,
        "ndvi_pred": out["ndvi"].numpy(),
        "conf": conf,
    }


def load_knowledge() -> dict[str, Any]:
    """Best-effort knowledge-base load (the drill-down degrades without it)."""
    try:
        from diagnosis_engine import load_knowledge_base

        return load_knowledge_base()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(bundle: dict[str, Any]) -> dict[str, Any]:
    """Draw the sidebar and return the current view controls."""
    st.sidebar.markdown("## 🌾 KisanGPT Digital Twin")
    st.sidebar.caption("Stage 3B · temporal stress forecasting")

    # --- model / data status ----------------------------------------------
    st.sidebar.markdown("### ⚙️ Status")
    meta = bundle["meta"]
    st.sidebar.markdown(
        f"🟢 **Model:** `{meta.get('encoder', 'transformer')}` / "
        f"`{meta.get('pooling', 'last')}` pool · epoch "
        f"{bundle['checkpoint_meta'].get('best_epoch', '?')}"
    )
    st.sidebar.markdown(f"🟢 **Fields loaded:** `{bundle['n_fields']}`")
    if bundle["ndvi_is_proxy"]:
        st.sidebar.markdown("🟡 **NDVI:** `PROXY` (synthetic class-conditional)")
    else:
        st.sidebar.markdown("🟢 **NDVI:** real export")
    if bundle["weather_is_proxy"]:
        st.sidebar.markdown(
            f"🟡 **Weather:** `PROXY` ({bundle['weather_source']})"
        )
    else:
        st.sidebar.markdown("🟢 **Weather:** real source")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ View")

    # --- grid page ---------------------------------------------------------
    per_page = 100
    pages = max(1, bundle["n_fields"] // per_page)
    page = 0
    if pages > 1:
        page = st.sidebar.selectbox(
            "Grid page (100 cells each)",
            options=list(range(pages)),
            format_func=lambda p: f"fields {p * per_page}–{p * per_page + 99}",
            key="twin_page",
        )
    page_fields = np.arange(
        page * per_page, min((page + 1) * per_page, bundle["n_fields"])
    )

    # --- field selector ----------------------------------------------------
    page_set = {int(f) for f in page_fields}
    stored = st.session_state.get("twin_selected_field")
    if stored is not None and int(stored) in page_set:
        # Keep the dropdown in sync with the cell the user clicked.
        st.session_state["twin_selector"] = int(stored)
    selected = st.sidebar.selectbox(
        "Selected cell / field",
        options=sorted(page_set),
        index=0,
        key="twin_selector",
        help="Or click a cell in the map - both drive the chart and drill-down.",
    )
    st.session_state["twin_selected_field"] = int(selected)

    # --- view mode ---------------------------------------------------------
    mode = st.sidebar.radio(
        "Days to show",
        ["Historical", "Forecast", "Both"],
        index=2,
        key="twin_mode",
        help=f"history = days 0–{TODAY}, forecast = days {T_HISTORY}–{T_TOTAL - 1}",
    )
    view = st.sidebar.radio(
        "Map view",
        ["Grid heatmap", "Small multiples"],
        key="twin_view",
        help="Small multiples = one mini NDVI chart per cell (all fields).",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Time")
    day = render_day_scrubber(mode, T_HISTORY, T_TOTAL, key="twin_day")

    st.sidebar.markdown("---")
    use_llm = st.sidebar.toggle(
        "LLM polish (Layer 2)",
        value=False,
        key="twin_llm",
        help="Rewrites the Stage-2 diagnosis for the farmer when a provider "
             "is configured; silently falls back to Layer 1 otherwise.",
    )
    try:
        from llm_polish import describe_provider

        st.sidebar.caption(describe_provider())
    except Exception:
        st.sidebar.caption("LLM polish: module unavailable")

    # --- proxy disclosure --------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Data disclosure")
    st.sidebar.warning(
        "**PROXY data.** NDVI and weather here are declared synthetic/"
        "class-conditional series used because no paired Sentinel-2 export "
        "exists for these records. Cells are independent field records - the "
        "grid is a layout, not a contagion model. Do not read these colours "
        "as satellite observations.",
        icon="⚠️",
    )

    return {
        "page_fields": page_fields,
        "selected": int(selected),
        "mode": mode,
        "view": view,
        "day": int(day),
        "use_llm": bool(use_llm),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the full dashboard."""
    started = time.time()

    try:
        bundle = load_twin_bundle()
    except FileNotFoundError as err:
        st.error(f"**Trajectory model missing**\n\n{err}", icon="🚨")
        st.stop()
    except Exception as err:
        st.error(
            f"**Trajectory data unavailable**\n\n{err}\n\n"
            "Build it with: `python trajectory_data.py --validate-only`",
            icon="🚨",
        )
        st.stop()

    controls = render_sidebar(bundle)
    page_fields = controls["page_fields"]
    day = controls["day"]
    mode = controls["mode"]

    if page_fields.size == 0:
        st.error("No fields in this grid page.", icon="❌")
        st.stop()

    # --- predictions for the visible cells (cached) ------------------------
    with st.spinner("Forecasting the visible cells..."):
        preds = predict_fields(tuple(int(f) for f in page_fields))

    local_index = {int(f): i for i, f in enumerate(page_fields)}

    # --- field map ---------------------------------------------------------
    st.markdown(
        "<div style='background:#2E7D32;padding:10px 16px;border-radius:8px;"
        "color:white;font-size:18px;font-weight:600'>"
        "Field map — stress state per cell</div>",
        unsafe_allow_html=True,
    )
    clicked = render_field_map(
        page_fields=page_fields,
        day=day,
        labels=bundle["labels"],
        pred_classes=preds["classes"],
        ndvi=bundle["ndvi"],
        ndvi_pred=preds["ndvi_pred"],
        t_history=T_HISTORY,
        view=controls["view"],
        confidences=preds["conf"],
        key="twin_map",
    )
    if clicked is not None:
        st.session_state["twin_selected_field"] = int(clicked)

    selected = int(st.session_state.get("twin_selected_field", controls["selected"]))
    if selected not in local_index:
        selected = int(page_fields[0])
        st.session_state["twin_selected_field"] = selected
    sel = local_index[selected]

    st.markdown("---")

    # --- chart + alerts ----------------------------------------------------
    col_chart, col_alerts = st.columns([3, 2], gap="medium")

    with col_chart:
        is_forecast_day = day >= T_HISTORY
        step = min(max(day - T_HISTORY, 0), T_FUTURE - 1)
        if is_forecast_day:
            class_idx = int(preds["classes"][sel, step])
            confidence = float(preds["conf"][sel, step])
            source = "forecast"
        else:
            class_idx = int(bundle["labels"][selected, day])
            confidence = float(preds["probs"][sel, 0, class_idx])
            source = "observed"
        true_class = (
            UNIFIED_CLASSES[int(bundle["labels"][selected, day])]
            if day < T_HISTORY else None
        )

        fig = render_trajectory_chart(
            field_id=selected,
            ndvi_history=bundle["ndvi"][selected, :T_HISTORY],
            ndvi_forecast=preds["ndvi_pred"][sel],
            labels_history=bundle["labels"][selected, :T_HISTORY],
            pred_classes=preds["classes"][sel],
            pred_probs=preds["probs"][sel],
            ndvi_is_proxy=bundle["ndvi_is_proxy"],
        )
        st.plotly_chart(fig, width="stretch", key="twin_traj")

    with col_alerts:
        horizon = cfg.TRAJ_ALERT_HORIZON
        onset = collect_onset_alerts(
            preds["classes"], preds["probs"], page_fields, horizon=horizon
        )
        escalation = _collect_escalation(
            preds["classes"], preds["probs"], page_fields,
            bundle["labels"][page_fields, TODAY],
        )
        coverage = collect_coverage_alerts(
            preds["classes"], t_history=T_HISTORY
        )
        render_alert_panel(onset, escalation, coverage, horizon=horizon)

    st.markdown("---")

    # --- drill-down --------------------------------------------------------
    ndvi_at_day = (
        bundle["ndvi"][selected, day] if day < T_HISTORY
        else preds["ndvi_pred"][sel, step]
    )
    cell = {
        "field_id": selected,
        "day": day,
        "step": step,
        "source": source,
        "predicted_class": UNIFIED_CLASSES[class_idx],
        "class_idx": class_idx,
        "confidence": confidence,
        "ndvi_at_day": ndvi_at_day,
        "ndvi_history": bundle["ndvi"][selected, :T_HISTORY],
        "ndvi_forecast": preds["ndvi_pred"][sel],
        "labels_history": bundle["labels"][selected, :T_HISTORY],
        "pred_classes": preds["classes"][sel],
        "pred_probs": preds["probs"][sel],
        "weather_history": bundle["weather"][selected, :T_HISTORY],
        "true_class": true_class,
        "ndvi_is_proxy": bundle["ndvi_is_proxy"],
        "model": bundle["model"],
        "input": _standardised_input(bundle, selected),
        "knowledge": load_knowledge(),
    }
    render_drilldown(cell, use_llm=controls["use_llm"])

    # --- footer ------------------------------------------------------------
    st.sidebar.caption(f"rendered in {time.time() - started:.2f}s")
    with st.expander("ℹ️ About this dashboard"):
        st.markdown(
            "**Stage 3B — digital twin visualization.**\n\n"
            "- **Cell = field record**: every cell carries a real NDVI series, "
            "weather series, day-level labels and Stage-3A predictions; the "
            "10x10 grid is a layout, not a contagion model.\n"
            "- **History vs forecast**: colours on days "
            f"0–{TODAY} are stored labels; days {T_HISTORY}–{T_TOTAL - 1} are "
            "the trajectory model's argmax.\n"
            "- **Uncertainty band**: the model has no variance head, so a "
            f"declared fixed ±{cfg.TRAJ_BAND} band is drawn rather than an "
            "invented per-point interval.\n"
            "- **Attribution**: gradient x input on the trajectory model, the "
            "same method Stage 2A uses; the `image` share is 0 because this "
            "path has no leaf photo.\n"
            "- **PROXY**: NDVI and weather are declared synthetic; see the "
            "sidebar disclosure.\n\n"
            "Pipeline: `trajectory_data.py` -> `train_trajectory.py` -> "
            "`dashboard_twin.py`. Stage-2 diagnosis is reused from "
            "`diagnosis_engine.py` / `llm_polish.py`."
        )


def _collect_escalation(
    pred_classes: np.ndarray,
    pred_probs: np.ndarray,
    field_ids: np.ndarray,
    origin_classes: np.ndarray,
) -> list[dict[str, Any]]:
    """Thin wrapper so the escalation helper stays in the alert component."""
    from components_twin.alert_panel import collect_escalation_alerts

    return collect_escalation_alerts(
        pred_classes, pred_probs, field_ids, origin_classes, t_history=T_HISTORY
    )


def _standardised_input(bundle: dict[str, Any], field_id: int) -> np.ndarray:
    """Build the exact ``(1, T_history, D)`` tensor the model was given."""
    raw = np.concatenate(
        [
            bundle["ndvi"][[field_id], :T_HISTORY, None],
            bundle["weather"][[field_id], :T_HISTORY, :],
        ],
        axis=2,
    ).astype(np.float32)
    mean, std = bundle["input_mean"], bundle["input_std"]
    if mean is not None and std is not None:
        raw = (raw - np.asarray(mean, dtype=np.float32)) / np.asarray(
            std, dtype=np.float32
        )
    return raw


if __name__ == "__main__":
    main()
