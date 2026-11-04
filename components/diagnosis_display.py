"""Stage 2B Component - Modern Multi-modal Diagnosis Presentation.

Displays:
1. Executive Diagnostic Banner with confidence badge, risk level, and quick highlights.
2. Two-column interactive panel:
   - Visual Evidence: Toggleable Grad-CAM heatmap overlay with high-res styling.
   - Multimodal Attribution: Dynamic Plotly bar chart + 20-step NDVI temporal curve.
3. Agronomist Advisory Card: Farmer-friendly actionable guidance with bulleted steps.
4. Raw Model Output Expander with clean formatting and JSON export.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from explainability import generate_heatmap_image, overlay_cam_on_image

MODALITY_COLORS = {
    "tabular": "#3b82f6",   # Modern Blue (Sensors)
    "image": "#10b981",     # Emerald Green (Visual)
    "ndvi": "#f59e0b",      # Amber (Vegetation Dynamics)
}

CLASS_FRIENDLY_NAMES = {
    "healthy": "Healthy Crop Foliage",
    "water_stress": "Drought / Water Stress",
    "nutrient_stress": "Nutrient Deficiency Stress",
    "disease_stress": "Leaf Disease / Pathogen Stress",
    "pest_stress": "Pest Infestation Damage",
    "stress_generic": "Generalized Environmental Stress",
}


def build_attention_figure(weights: dict[str, float]) -> go.Figure:
    """Create a sleek Plotly horizontal bar chart for modality attention."""
    labels = ["Soil/Weather Telemetry", "Leaf Visual Symptoms", "Canopy NDVI Trajectory"]
    keys = ["tabular", "image", "ndvi"]
    values = [float(weights.get(k, 0.0)) for k in keys]
    colors = [MODALITY_COLORS.get(k, "#64748b") for k in keys]

    total = sum(values) if sum(values) > 0 else 1.0
    percentages = [(v / total) * 100 for v in values]

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=labels,
            orientation="h",
            marker=dict(color=colors, cornerradius=6),
            text=[f"<b>{p:.1f}%</b>" for p in percentages],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=12, family="Inter, system-ui, sans-serif"),
            hovertemplate="<b>%{y}</b><br>Cross-Attention Share: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="<b>Multi-Head Cross-Attention Share</b>",
            font=dict(size=13, color="#1e293b", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            title="",
            range=[0, 105],
            showgrid=True,
            gridcolor="#f1f5f9",
            showticklabels=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color="#334155", family="Inter, system-ui, sans-serif"),
        ),
        height=190,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


def build_ndvi_sparkline(ndvi_series: np.ndarray | list[float], is_proxy: bool = True) -> go.Figure:
    """Create a clean Plotly sparkline showing the 20-step NDVI vegetative trajectory."""
    values = [float(x) for x in ndvi_series]
    steps = list(range(1, len(values) + 1))
    min_idx = int(np.argmin(values))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=values,
            mode="lines+markers",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=5, color="#047857"),
            name="NDVI",
            hovertemplate="Step %{x}: NDVI %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[steps[min_idx]],
            y=[values[min_idx]],
            mode="markers",
            marker=dict(color="#ef4444", size=8, symbol="diamond"),
            name="Trough",
            hovertemplate="Lowest Dip (Step %{x}): %{y:.3f}<extra></extra>",
        )
    )

    badge = " (Class-Conditional Proxy)" if is_proxy else " (Sentinel-2 Real)"
    fig.update_layout(
        title=dict(
            text=f"<b>Temporal Canopy Vigor (NDVI)</b><span style='font-size:10px; color:#64748b'>{badge}</span>",
            font=dict(size=13, color="#1e293b", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(title="20-Day Sequence", tickmode="linear", dtick=4, gridcolor="#f1f5f9"),
        yaxis=dict(title="Vigor", range=[0.0, 1.0], gridcolor="#f1f5f9"),
        height=190,
        margin=dict(l=10, r=10, t=30, b=20),
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


def render_banner(predicted_class: str, confidence: float, severity: str) -> None:
    """Display an executive diagnosis summary card."""
    percent = confidence * 100
    display_title = CLASS_FRIENDLY_NAMES.get(predicted_class, predicted_class.replace("_", " ").title())

    if predicted_class == "healthy" and confidence >= 0.75:
        theme_bg = "#ecfdf5"
        border_col = "#6ee7b7"
        text_col = "#065f46"
        badge_bg = "#10b981"
        status_icon = "🌿"
        headline = "Crop Foliage Appears Healthy & Balanced"
    elif severity == "high" or (predicted_class != "healthy" and confidence >= 0.8):
        theme_bg = "#fef2f2"
        border_col = "#fca5a5"
        text_col = "#991b1b"
        badge_bg = "#ef4444"
        status_icon = "🚨"
        headline = f"Urgent Stress Detected: {display_title}"
    else:
        theme_bg = "#fffbeb"
        border_col = "#fde68a"
        text_col = "#92400e"
        badge_bg = "#f59e0b"
        status_icon = "⚠️"
        headline = f"Stress Anomaly Detected: {display_title}"

    st.markdown(
        f"""
        <div style="background: {theme_bg}; border: 1.5px solid {border_col}; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 2.2rem;">{status_icon}</span>
                    <div>
                        <h2 style="margin: 0; color: {text_col} !important; font-size: 1.45rem; font-weight: 800;">
                            {headline}
                        </h2>
                        <span style="color: {text_col}; font-size: 0.9rem; font-weight: 500;">
                            Primary Prediction: <b>{display_title}</b>
                        </span>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: {badge_bg}; color: white; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.9rem;">
                        {percent:.1f}% Confidence
                    </div>
                    <div style="background: white; border: 1px solid {border_col}; color: {text_col}; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.9rem;">
                        Severity: {severity.upper()}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnosis_view(
    result: dict[str, Any],
    key: str = "diag_view",
) -> None:
    """Render the polished modern multi-modal diagnosis result view."""
    output = result["output"]
    layer1 = result["layer1"]
    layer2 = result.get("layer2")
    cam = np.asarray(result["cam"], dtype=np.float32)
    original_image = result["image"]

    predicted_class = getattr(output, "predicted_class", str(output.get("predicted_class", "unknown")))
    confidence = float(getattr(output, "confidence", output.get("confidence", 0.0)))
    severity = getattr(layer1, "severity", "medium").lower()

    # 1. Top Executive Banner
    render_banner(predicted_class, confidence, severity)

    # 2. Key Metadata Pill Bar
    st.markdown(
        f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;">
            <span style="background: #f1f5f9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                ⚡ Latency: {result.get('seconds', 0.0):.2f}s
            </span>
            <span style="background: #f1f5f9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                🎯 Dominant Driver: {getattr(layer1, 'driver_modality', 'multi-modal').title()} Stream
            </span>
            <span style="background: #f1f5f9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                📄 Sample ID: {getattr(output, 'sample_id', 'dashboard_sample')}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Two-Column Inspection (Left: Visual Grad-CAM | Right: Multimodal Attention & NDVI)
    c_left, c_right = st.columns([1, 1], gap="medium")

    with c_left:
        with st.container(border=True):
            st.markdown(
                """
                <h4 style="margin: 0 0 10px 0; font-size: 1.05rem; color: #1e293b; font-weight: 700;">
                    🔬 Visual Symptoms & Grad-CAM Localization
                </h4>
                """,
                unsafe_allow_html=True,
            )

            cam_toggle = st.radio(
                "Visualization Mode:",
                options=["Blended Saliency Overlay", "Original Leaf Photo", "Thermal Heatmap Only"],
                horizontal=True,
                key=f"{key}_cam_view",
            )

            if cam_toggle == "Blended Saliency Overlay":
                blended = overlay_cam_on_image(original_image, cam, alpha=0.45)
                st.image(blended, caption=f"Grad-CAM Attention Overlay for {predicted_class}", width="stretch")
            elif cam_toggle == "Original Leaf Photo":
                st.image(original_image, caption="Original Input Photograph", width="stretch")
            else:
                heatmap_only = generate_heatmap_image(cam, size=(original_image.width, original_image.height))
                st.image(heatmap_only, caption="Raw Saliency Heatmap Activation", width="stretch")

            gradcam_desc = getattr(output, "gradcam_summary", None) or "Highlighted regions denote leaf areas driving neural activations."
            st.caption(f"🔍 **Visual Insight:** {gradcam_desc}")

    with c_right:
        with st.container(border=True):
            st.markdown(
                """
                <h4 style="margin: 0 0 10px 0; font-size: 1.05rem; color: #1e293b; font-weight: 700;">
                    📊 Multimodal Evidence & Canopy Trend
                </h4>
                """,
                unsafe_allow_html=True,
            )

            attention_dict = getattr(output, "attention_weights", {})
            if attention_dict:
                fig_attn = build_attention_figure(attention_dict)
                st.plotly_chart(fig_attn, width="stretch")

            ndvi_arr = result.get("ndvi_series")
            if ndvi_arr is None and hasattr(output, "ndvi_values"):
                ndvi_arr = getattr(output, "ndvi_values", None)
            if ndvi_arr is None:
                ndvi_arr = np.linspace(0.65, 0.35, 20) if predicted_class != "healthy" else np.linspace(0.6, 0.75, 20)

            is_proxy = getattr(output, "ndvi_is_proxy", True)
            fig_ndvi = build_ndvi_sparkline(ndvi_arr, is_proxy=is_proxy)
            st.plotly_chart(fig_ndvi, width="stretch")

    # 4. Actionable Advisory & Expert Guidance Card
    with st.container(border=True):
        st.markdown(
            """
            <h3 style="margin: 0 0 12px 0; color: #0f172a; font-size: 1.25rem; font-weight: 800;">
                🌾 Agronomist Recommendations & Field Plan
            </h3>
            """,
            unsafe_allow_html=True,
        )

        if layer2:
            st.markdown(
                f"""
                <div style="background: #f8fafc; border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 6px; font-size: 0.95rem; color: #1e293b; line-height: 1.6; margin-bottom: 16px;">
                    <b>Farmer-Friendly Advisory:</b><br>{layer2}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: #f8fafc; border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 6px; font-size: 0.95rem; color: #1e293b; line-height: 1.6; margin-bottom: 16px;">
                    <b>Agronomist Assessment:</b><br>{getattr(layer1, 'draft', '')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        actions = getattr(layer1, "actions", ["Monitor field weekly", "Verify soil moisture manually"])
        escalation = getattr(layer1, "escalation", "Consult your nearest Krishi Vigyan Kendra (KVK) if symptoms spread.")

        c_act1, c_act2 = st.columns([3, 2], gap="large")
        with c_act1:
            st.markdown("##### 🎯 Recommended Remediation Steps:")
            for idx, act in enumerate(actions, start=1):
                st.markdown(f"**{idx}.** {act}")

        with c_act2:
            st.markdown("##### 🚨 Escalation Protocol:")
            st.warning(escalation)

    # 5. Raw Model Output Expander (JSON)
    with st.expander("🛠️ Inspect Raw Model Logits & Feature Embeddings"):
        top_feats = []
        if hasattr(output, "tabular_top_features"):
            for tf in output.tabular_top_features:
                top_feats.append({
                    "feature": tf.feature,
                    "value": round(float(tf.value), 3),
                    "contribution": round(float(tf.contribution), 4),
                })

        raw_json_dict = {
            "sample_id": getattr(output, "sample_id", "dashboard_sample"),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "severity": severity,
            "severity_reasons": getattr(layer1, "severity_reasons", []),
            "driver_modality": getattr(layer1, "driver_modality", "unknown"),
            "attention_weights": {k: round(float(v), 4) for k, v in getattr(output, "attention_weights", {}).items()},
            "tabular_top_features": top_feats,
            "gradcam_summary": getattr(output, "gradcam_summary", "N/A"),
            "ndvi_trend": getattr(output, "ndvi_trend", "N/A"),
            "ndvi_is_proxy": getattr(output, "ndvi_is_proxy", True),
            "inference_time_seconds": round(float(result.get("seconds", 0.0)), 3),
            "layer2_llm_polished": bool(layer2 is not None),
        }
        st.json(raw_json_dict)
