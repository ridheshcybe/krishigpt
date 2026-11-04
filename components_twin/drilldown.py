"""Stage 3B - cell drill-down with Stage-2 diagnosis reuse.

Clicking a cell answers three questions:

1. *What is this cell?* - current stress class + confidence from the Stage-3A
   trajectory model (observed label for history days, model argmax for
   forecast days).
2. *Why did the model say that?* - per-channel attribution computed with
   gradient x input over the trajectory model's own input tensor, the same
   attribution method Stage 2A uses for the tabular branch.  Shares are
   reported honestly: ``image = 0.0`` because the digital-twin path has **no
   leaf image**, and nothing is invented to fill the slot.
3. *What should the farmer do?* - the Stage-2 diagnosis engine
   (``diagnosis_engine.diagnose``) plus its optional LLM polish
   (``llm_polish.polish``), imported, never copied.  If either import fails the
   panel degrades to a warning instead of crashing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from components_twin.field_map import CLASS_COLORS, CLASS_LABELS
from stage1_config import UNIFIED_CLASSES, UNIFIED_INDEX

# Optional Stage-2 imports: the twin must still render if they are missing.
try:  # pragma: no cover - exercised only when Stage 2 files are absent
    from diagnosis_engine import ModelOutput, TopFeature, diagnose
    from llm_polish import describe_provider, polish

    DIAGNOSIS_AVAILABLE = True
    _DIAGNOSIS_ERROR = ""
except Exception as _err:  # never crash the dashboard
    DIAGNOSIS_AVAILABLE = False
    _DIAGNOSIS_ERROR = str(_err)

#: Input channels produced by ``trajectory_data`` in order
#: ``[NDVI, temperature, humidity, rainfall, soil_moisture]``.
CHANNEL_NAMES: list[str] = ["ndvi", "temperature", "humidity", "rainfall",
                            "soil_moisture"]


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
def attribute_channels(
    model: Any,
    x: np.ndarray,
    step: int,
    class_idx: int,
) -> tuple[list[str], np.ndarray]:
    """Gradient x input attribution of one predicted logit over the input.

    Args:
        model: a Stage-3A ``TrajectoryModel`` (any module works).
        x: ``(1, T_history, D)`` float array - the *standardised* input the
            model actually consumed.
        step: future day index whose logit to explain (0 = tomorrow).
        class_idx: class whose logit is explained.

    Returns:
        ``(channel_names, signed_shares)`` where ``signed_shares`` sums to 0
        over |values| (positive supports the predicted class, negative argues
        against it) - the same sign convention Stage 2A prints.

    Notes:
        Never raises: any failure returns equal shares plus an empty name list
        so the caller can fall back to a neutral display.
    """
    import torch

    try:
        x_t = torch.tensor(np.asarray(x, dtype=np.float32), requires_grad=True)
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            _, logits = model(x_t)
            target = logits[0, int(step), int(class_idx)]
            model.zero_grad(set_to_none=True)
            target.backward()
            grad = x_t.grad.detach()[0]                     # (T, D)
            signed = (grad * x_t.detach()[0]).sum(dim=0)     # (D,)
        if was_training:
            model.train()
        model.zero_grad(set_to_none=True)

        values = signed.numpy()
        denom = float(np.abs(values).sum())
        if denom <= 1e-12:
            return CHANNEL_NAMES, np.zeros(len(values), dtype=float)
        return CHANNEL_NAMES, values / denom
    except Exception:
        return [], np.zeros(len(CHANNEL_NAMES), dtype=float)


# ---------------------------------------------------------------------------
# ModelOutput construction (Stage 2 contract)
# ---------------------------------------------------------------------------
def build_model_output(
    cell: dict[str, Any],
    channel_names: list[str],
    signed_shares: np.ndarray,
) -> Any | None:
    """Assemble a Stage-2 ``ModelOutput`` from one cell's twin data.

    Every field is either computed from real cell data or explicitly marked as
    unavailable - no field is filled with a plausible-looking guess:

    * ``attention_weights`` - ``tabular`` = weather-channel share, ``ndvi`` =
      NDVI-channel share, ``image`` = **0.0** (no image in this path)
    * ``tabular_top_features`` - today's raw weather readings with the signed
      attribution as the contribution
    * ``gradcam_summary`` - states that no image branch ran
    * ``ndvi_trend`` - measured from the cell's own history
    """
    if not DIAGNOSIS_AVAILABLE:
        return None

    step = int(cell.get("step", 0))
    conf = float(cell["confidence"])
    predicted = str(cell["predicted_class"])

    # --- attribution -> Stage-2 "attention" --------------------------------
    if channel_names and signed_shares.size == len(channel_names):
        shares = {name: float(val) for name, val in zip(channel_names, signed_shares)}
        magnitude = {k: abs(v) for k, v in shares.items()}
        total = sum(magnitude.values()) or 1.0
        ndvi_share = magnitude.get("ndvi", 0.0) / total
        tabular_share = 1.0 - ndvi_share
    else:
        shares = {}
        ndvi_share = tabular_share = 0.0
    attention = {
        "tabular": round(tabular_share, 4),
        "image": 0.0,
        "ndvi": round(ndvi_share, 4),
    }

    # --- today's raw readings as evidence ----------------------------------
    weather_today = np.asarray(cell["weather_history"][-1], dtype=float)  # 4 values
    feature_values = {
        "temperature": float(weather_today[0]),
        "humidity": float(weather_today[1]),
        "rainfall": float(weather_today[2]),
        "soil_moisture": float(weather_today[3]),
    }
    top_features: list[Any] = []
    for name, value in feature_values.items():
        contribution = float(shares.get(name, 0.0))
        top_features.append(
            TopFeature(feature=name, value=round(value, 3),
                       contribution=round(contribution, 4))
        )
    top_features.sort(key=lambda f: abs(f.contribution), reverse=True)

    # --- NDVI trend measured from history ----------------------------------
    hist = np.asarray(cell["ndvi_history"], dtype=float)
    direction = "declining" if hist[-1] < hist[0] - 0.02 else (
        "improving" if hist[-1] > hist[0] + 0.02 else "stable"
    )
    ndvi_trend = (
        f"{direction} from {hist[0]:.2f} to {hist[-1]:.2f} over "
        f"{len(hist)} days (forecast {cell['ndvi_forecast'][0]:.2f} -> "
        f"{cell['ndvi_forecast'][-1]:.2f})"
    )

    return ModelOutput(
        predicted_class=predicted,
        confidence=conf,
        attention_weights=attention,
        tabular_top_features=top_features,
        gradcam_summary=(
            "No leaf image branch ran in the digital twin: Stage 3B feeds "
            "weather + NDVI only, so there is no Grad-CAM for this cell."
        ),
        ndvi_trend=ndvi_trend,
        ndvi_is_proxy=bool(cell.get("ndvi_is_proxy", True)),
        true_class=cell.get("true_class"),
        sample_id=f"twin_field_{cell['field_id']}_day_{cell['day']}",
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def render_drilldown(cell: dict[str, Any], use_llm: bool = False) -> None:
    """Render the selected cell: class, confidence, attribution, diagnosis."""
    st.markdown("### 🔎 Cell drill-down")
    field_id = int(cell["field_id"])
    day = int(cell["day"])
    predicted = str(cell["predicted_class"])
    confidence = float(cell["confidence"])
    source = cell["source"]  # "observed" | "forecast"

    # --- class + confidence -----------------------------------------------
    badge = st.container(border=True)
    with badge:
        left, right = st.columns([2, 1])
        with left:
            st.markdown(
                f"**Field {field_id} · day {day} · "
                f"{CLASS_LABELS[predicted]}**"
            )
            st.caption(
                f"source: {source} · NDVI "
                f"{float(cell['ndvi_at_day']):.3f} · "
                f"{'PROXY' if cell.get('ndvi_is_proxy', True) else 'real'} series"
                + (
                    " · confidence shown = model P(same class tomorrow); "
                    "the label itself is observed"
                    if source == "observed" else ""
                )
            )
        with right:
            if predicted == "healthy" and confidence >= 0.8:
                st.success(f"confidence {confidence:.0%}")
            elif predicted != "healthy" and confidence >= 0.8:
                st.error(f"confidence {confidence:.0%}")
            else:
                st.warning(f"confidence {confidence:.0%}")

    if not DIAGNOSIS_AVAILABLE:
        st.warning(
            "Stage-2 diagnosis engine is unavailable "
            f"(`{_DIAGNOSIS_ERROR}`), so the full-diagnosis button is disabled. "
            "Restore `diagnosis_engine.py` / `knowledge_base.json` to enable it.",
            icon="⚠️",
        )
        return

    # --- model output + Layer 1 (+ optional Layer 2) -----------------------
    model = cell["model"]
    x = cell["input"]                        # (1, T, D) standardised
    step = int(cell.get("step", 0))
    class_idx = int(cell.get("class_idx", UNIFIED_INDEX[predicted]))
    channel_names, signed = attribute_channels(model, x, step, class_idx)

    output = build_model_output(cell, channel_names, signed)
    if output is None:
        st.warning("Could not build the model output for this cell.", icon="⚠️")
        return

    # --- attribution bar chart --------------------------------------------
    st.markdown("**Why the model said that** (gradient x input)")
    if channel_names:
        labels = [n.replace("_", " ") for n in channel_names]
        values = [round(float(v) * 100, 2) for v in signed]
        colors = ["#C62828" if v < 0 else "#2E7D32" for v in values]
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=colors),
                hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis=dict(title="contribution to the predicted logit (%)",
                       gridcolor="#E8E8E8"),
            yaxis=dict(autorange="reversed"),
            height=230,
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="#FAFAF7",
            paper_bgcolor="#FAFAF7",
        )
        st.plotly_chart(fig, width="stretch", key=f"attrib_{field_id}_{day}")
        st.caption(
            "positive (green) supports the prediction, negative (red) argues "
            "against it; `image` share is 0 by construction - no leaf photo "
            "in this path."
        )
    else:
        st.caption("Attribution unavailable for this cell (fell back to defaults).")

    # --- Layer 1 diagnosis -------------------------------------------------
    knowledge = cell.get("knowledge") or {}
    try:
        layer1 = diagnose(output, knowledge or None)
    except Exception as err:
        st.error(f"Diagnosis engine failed on this cell: {err}", icon="❌")
        return

    with st.expander("📋 Layer 1 structured diagnosis (technical draft)",
                     expanded=True):
        st.info(layer1.draft)
        c1, c2, c3 = st.columns(3)
        c1.metric("Severity", layer1.severity.upper())
        c2.metric("Driver", layer1.driver_modality.upper())
        c3.metric("Driver share", f"{layer1.driver_share:.0%}")

    # --- Layer 2 (optional) -------------------------------------------------
    polished: str | None = None
    if use_llm:
        with st.spinner("Asking the LLM to rewrite this for the farmer..."):
            try:
                polished = polish(layer1.draft)
            except Exception:
                polished = None  # polish() already swallows; belt and braces
    if polished:
        st.success(f"**Farmer-friendly advice (LLM):**\n\n{polished}")
    else:
        st.success(f"**Recommended action:** {layer1.actions[0] if layer1.actions else 'Monitor the field.'}")
        if use_llm:
            st.caption(f"LLM polish unavailable — kept Layer 1 text. "
                       f"({describe_provider()})")

    st.caption(f"Escalation: {layer1.escalation}")
