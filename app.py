"""Stage 2B - KisanGPT Crop Stress Doctor Interactive Dashboard (Streamlit).

A modern, responsive, high-performance web application providing:
1. Multi-modal crop stress diagnosis: leaf image + optional 5-sensor telemetry.
2. Explainable AI: Grad-CAM heatmap overlays and Plotly attention attribution charts.
3. Natural-language farmer-friendly advice (Stage 2A template + optional LLM polish).
4. Continuous learning: agronomist feedback logging to feedback_log.csv.
5. Multi-tab navigation: Diagnose, History, Batch Mode, and System About / Disclosures.

Run locally:
    python -m streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

import dataset as data
import diagnosis_cli
import stage1_config as cfg
import tabular_xgb as tx
from components.diagnosis_display import render_diagnosis_view
from components.feedback import (
    get_feedback_counts,
    load_feedback_log,
    render_feedback,
)
from components.image_upload import collect_image
from components.sensor_input import collect_sensors, get_sensor_dataframe
from diagnosis_engine import load_knowledge_base
from llm_polish import describe_provider, polish
from mobilenet_cnn import build_transforms
from stage1_config import UNIFIED_CLASSES, UNIFIED_INDEX

#: Primary and fallback model checkpoint paths
PRIMARY_MODEL_PATH = Path("artifacts/unified_model.pth")
FALLBACK_MODEL_PATH = Path("artifacts/attention_fusion.pt")
KNOWLEDGE_BASE_PATH = Path("knowledge_base.json")

# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KisanGPT — Multi-Modal Crop Doctor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* -------------------------------------------------------------
         KisanGPT Modern Agronomy Theme (Clean, High-Contrast Light)
         Enforces crisp visibility regardless of browser theme settings.
         ------------------------------------------------------------- */

      /* 1. Global Page Background & Header */
      .stApp {
          background-color: #f8fafc !important;
          color: #0f172a !important;
      }
      header[data-testid="stHeader"] {
          background-color: #f8fafc !important;
      }
      header[data-testid="stHeader"] svg {
          fill: #475569 !important;
      }

      /* 2. Global Typography & Text Colors */
      .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {
          color: #0f172a;
      }
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
          color: #0f172a !important;
      }

      /* 3. Sidebar Container & Elements */
      [data-testid="stSidebar"] {
          background-color: #ffffff !important;
          border-right: 1px solid #e2e8f0 !important;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5 {
          color: #0f172a !important;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label p,
      [data-testid="stSidebar"] [data-testid="stRadio"] label span {
          color: #1e293b !important;
          font-weight: 500 !important;
      }
      [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
          color: #64748b !important;
          font-weight: 600 !important;
          font-size: 0.82rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stMetricValue"] div {
          color: #0f172a !important;
          font-weight: 700 !important;
      }
      [data-testid="stSidebar"] hr {
          border-color: #e2e8f0 !important;
          margin: 14px 0 !important;
      }
      [data-testid="stSidebar"] .stCaption,
      [data-testid="stSidebar"] .stCaption p {
          color: #64748b !important;
      }

      /* 4. Native Container Cards (st.container(border=True)) */
      [data-testid="stVerticalBlockBorderWrapper"] {
          background-color: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          border-radius: 12px !important;
          padding: 16px 20px !important;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
      }

      /* 5. Widget Labels (Universal) */
      [data-testid="stWidgetLabel"],
      [data-testid="stWidgetLabel"] label,
      [data-testid="stWidgetLabel"] p,
      [data-testid="stWidgetLabel"] span {
          color: #0f172a !important;
          font-weight: 600 !important;
          font-size: 0.92rem !important;
      }

      /* 6. Radio Buttons (Preset/Upload/Random & Views) */
      [data-testid="stRadio"] div[role="radiogroup"] label {
          color: #1e293b !important;
          font-weight: 500 !important;
      }
      [data-testid="stRadio"] div[role="radiogroup"] label p,
      [data-testid="stRadio"] div[role="radiogroup"] label span,
      [data-testid="stRadio"] div[role="radiogroup"] label div {
          color: #1e293b !important;
          font-weight: 500 !important;
          font-size: 0.9rem !important;
      }
      [data-testid="stRadio"] div[role="radiogroup"] label:hover p {
          color: #059669 !important;
      }

      /* 7. Sliders (Telemetry Panel) */
      [data-testid="stSlider"] label,
      [data-testid="stSlider"] label p,
      [data-testid="stSlider"] label span {
          color: #0f172a !important;
          font-weight: 600 !important;
          font-size: 0.88rem !important;
      }
      [data-testid="stSlider"] [data-testid="stSliderThumbValue"],
      [data-testid="stSlider"] [data-testid="stSliderThumbValue"] div {
          color: #0f172a !important;
          font-weight: 700 !important;
          font-size: 0.88rem !important;
      }
      [data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
      [data-testid="stSlider"] [data-testid="stSliderTickBarMax"] {
          color: #64748b !important;
          font-size: 0.78rem !important;
      }

      /* 8. Selectboxes & Dropdowns */
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
          background-color: #ffffff !important;
          color: #0f172a !important;
          border: 1px solid #cbd5e1 !important;
          border-radius: 8px !important;
          min-height: 40px !important;
      }
      [data-testid="stSelectbox"] div[data-baseweb="select"] span {
          color: #0f172a !important;
          font-weight: 500 !important;
      }
      [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
          fill: #475569 !important;
      }
      div[data-baseweb="popover"] ul {
          background-color: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
      }
      div[data-baseweb="popover"] li {
          color: #0f172a !important;
          font-weight: 500 !important;
      }
      div[data-baseweb="popover"] li:hover {
          background-color: #f1f5f9 !important;
          color: #059669 !important;
      }

      /* 9. Expanders */
      [data-testid="stExpander"] {
          background-color: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          border-radius: 10px !important;
      }
      [data-testid="stExpander"] summary {
          color: #0f172a !important;
          font-weight: 600 !important;
      }
      [data-testid="stExpander"] summary svg {
          fill: #475569 !important;
      }

      /* 10. Secondary / Default Buttons */
      .stButton > button:not([kind="primary"]) {
          background-color: #ffffff !important;
          color: #1e293b !important;
          border: 1px solid #cbd5e1 !important;
          border-radius: 8px !important;
          font-weight: 600 !important;
          transition: all 0.2s ease !important;
      }
      .stButton > button:not([kind="primary"]):hover {
          background-color: #f0fdf4 !important;
          border-color: #10b981 !important;
          color: #10b981 !important;
      }

      /* 11. Primary Buttons */
      .stButton > button[kind="primary"] {
          background-color: #10b981 !important;
          color: #ffffff !important;
          border: none !important;
          border-radius: 8px !important;
          font-weight: 700 !important;
          padding: 10px 24px !important;
          font-size: 1rem !important;
          box-shadow: 0 1px 3px rgba(16, 185, 129, 0.3) !important;
          transition: all 0.2s ease !important;
      }
      .stButton > button[kind="primary"]:hover {
          background-color: #059669 !important;
          box-shadow: 0 4px 6px rgba(16, 185, 129, 0.4) !important;
      }

      /* 12. Captions & Info Text */
      .stCaption,
      [data-testid="stCaptionContainer"],
      [data-testid="stCaptionContainer"] p {
          color: #64748b !important;
          font-size: 0.85rem !important;
      }

      /* 13. File Uploader */
      [data-testid="stFileUploader"] {
          background-color: #ffffff !important;
          border: 1px dashed #cbd5e1 !important;
          border-radius: 10px !important;
      }
      [data-testid="stFileUploader"] span,
      [data-testid="stFileUploader"] small {
          color: #64748b !important;
      }

      /* 14. DataFrames & Tables */
      [data-testid="stDataFrame"] {
          border: 1px solid #e2e8f0 !important;
          border-radius: 8px !important;
          overflow: hidden !important;
      }

      /* 15. Toggles & Checkboxes */
      [data-testid="stCheckbox"] label,
      [data-testid="stCheckbox"] label span,
      [data-testid="stCheckbox"] label p,
      [data-testid="stToggle"] label,
      [data-testid="stToggle"] label span,
      [data-testid="stToggle"] label p {
          color: #0f172a !important;
          font-weight: 500 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------------------------
def resolve_model_path() -> Path | None:
    """Find valid model checkpoint path."""
    if PRIMARY_MODEL_PATH.exists():
        return PRIMARY_MODEL_PATH
    if FALLBACK_MODEL_PATH.exists():
        return FALLBACK_MODEL_PATH
    return None


@st.cache_resource(show_spinner="Loading KisanGPT Multi-Modal Model...")
def load_app_bundle() -> dict[str, Any]:
    """Load model, feature preprocessors, and knowledge base once per session."""
    ckpt_path = resolve_model_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            "Unified model checkpoint missing. Please train or verify: "
            "'artifacts/unified_model.pth' or 'artifacts/attention_fusion.pt'"
        )

    if not tx.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Tabular preprocessor missing: '{tx.PREPROCESSOR_PATH}'. Run Stage 0 training first."
        )

    preprocessor = json.loads(tx.PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    feature_names = list(preprocessor["feature_names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        knowledge = load_knowledge_base()
    except Exception:
        knowledge = {}

    model = diagnosis_cli.load_unified_model(ckpt_path, device, tabular_dim=len(feature_names))

    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "value_map": diagnosis_cli.build_value_map(preprocessor),
        "device": device,
        "model": model,
        "knowledge": knowledge,
        "eval_transform": build_transforms()[1],
        "ckpt_path": ckpt_path,
    }


# ---------------------------------------------------------------------------
# Core Inference Runner
# ---------------------------------------------------------------------------
def run_diagnosis(
    image: Image.Image,
    sensor_df: pd.DataFrame,
    sample_id: str,
    use_llm: bool,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Execute end-to-end multi-modal inference."""
    start_time = time.perf_counter()

    # 1. Prepare inputs
    vector = tx.transform_features(sensor_df, bundle["preprocessor"]).to_numpy(np.float32)
    tabular = torch.from_numpy(vector.astype(np.float32))
    image_tensor = bundle["eval_transform"](image)

    # 2. Pass 1: neutral proxy NDVI vector
    neutral_ndvi = np.zeros((1, cfg.NDVI_STEPS), dtype=np.float32)
    first_pass = diagnosis_cli.analyze_batch(
        bundle["model"],
        tabular,
        image_tensor.unsqueeze(0),
        torch.from_numpy(neutral_ndvi),
        bundle["device"],
    )

    # 3. Pass 2: matched proxy curve
    pred_class_idx = int(first_pass["pred"][0])
    matched_ndvi = data.build_proxy_ndvi(np.asarray([pred_class_idx], dtype=np.int64), seed=cfg.SEED)

    analysis = diagnosis_cli.analyze_batch(
        bundle["model"],
        tabular,
        image_tensor.unsqueeze(0),
        torch.from_numpy(matched_ndvi),
        bundle["device"],
    )

    context = {
        "row_index": -1,
        "image": sample_id,
        "true_class": None,
        "ndvi": matched_ndvi[0],
        "ndvi_is_proxy": bool(cfg.NDVI_IS_PROXY),
        "split": "dashboard",
        "index": 0,
    }

    output = diagnosis_cli.build_model_output(
        0,
        "dashboard",
        context,
        analysis,
        0,
        bundle["feature_names"],
        bundle["value_map"],
    )
    output = replace(output, sample_id=sample_id)

    # 4. Layer 1 diagnosis
    if bundle["knowledge"]:
        layer1 = diagnosis_cli.diagnose(output, bundle["knowledge"])
    else:
        from diagnosis_engine import Diagnosis
        layer1 = Diagnosis(
            sample_id=sample_id,
            predicted_class=output.predicted_class,
            confidence=output.confidence,
            severity="medium",
            driver_modality="image",
            actions=["Monitor crop condition closely", "Test soil pH and moisture"],
            escalation="Consult a local agronomist.",
            severity_reasons=["Knowledge base unavailable; default advisory applied."],
            draft=f"Detected {output.predicted_class} with {output.confidence*100:.1f}% confidence.",
            structured={},
        )

    # 5. Layer 2 LLM polish
    layer2 = None
    if use_llm:
        try:
            layer2 = polish(layer1.draft)
        except Exception:
            layer2 = None

    elapsed = time.perf_counter() - start_time

    return {
        "output": output,
        "layer1": layer1,
        "layer2": layer2,
        "cam": np.asarray(analysis["cams"][0], dtype=np.float32),
        "image": image,
        "ndvi_series": matched_ndvi[0],
        "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Sidebar Navigation & Telemetry
# ---------------------------------------------------------------------------
def render_sidebar(model_ready: bool, kb_ready: bool) -> tuple[str, bool]:
    """Render sleek modern sidebar."""
    st.sidebar.markdown(
        """
        <div style="padding: 12px 4px 16px 4px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.8rem;">🌾</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.3rem; color: #0f172a; font-weight: 800;">KisanGPT</h2>
                    <span style="font-size: 0.75rem; color: #64748b; font-weight: 600;">Precision Crop Doctor</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_tab = st.sidebar.radio(
        "Navigation",
        options=["🩺 Live Diagnosis", "📜 Case History", "📦 Batch Processing", "ℹ️ System & Disclosures"],
        index=0,
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("##### ⚙️ Engine Telemetry")
    if model_ready:
        st.sidebar.markdown("🟢 **Unified Fusion:** `Active` (CUDA RTX 5060)")
    else:
        st.sidebar.markdown("🔴 **Unified Fusion:** `No Model`")

    if kb_ready:
        st.sidebar.markdown("🟢 **Agronomy Rules:** `Loaded`")
    else:
        st.sidebar.markdown("🟡 **Agronomy Rules:** `Fallback`")

    st.sidebar.markdown("---")

    given_count, feedback_count = get_feedback_counts()
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Diagnoses", given_count)
    c2.metric("Feedbacks", feedback_count)

    st.sidebar.markdown("---")

    st.sidebar.markdown("##### 🤖 Agronomist LLM")
    use_llm = st.sidebar.toggle(
        "Generate Conversational Advisory",
        value=True,
        help="Use Layer 2 LLM translation (Ollama, OpenAI, or Anthropic) for farmer-friendly prose.",
    )
    st.sidebar.caption(f"Backend Provider: `{describe_provider()}`")

    return active_tab, use_llm


# ---------------------------------------------------------------------------
# Tab 1: Single Sample Diagnosis
# ---------------------------------------------------------------------------
def render_diagnose_tab(bundle: dict[str, Any], use_llm: bool) -> None:
    """Render the primary interactive leaf + sensor diagnosis workflow."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 800; color: #0f172a;">
                🌾 Multimodal Crop Stress Diagnosis
            </h1>
            <p style="margin: 0; color: #475569; font-size: 0.95rem;">
                Fuse high-resolution leaf imagery, soil/weather sensor telemetry, and temporal NDVI canopy dynamics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two-column layout with clean card wrappers
    c_img, c_sens = st.columns([1, 1], gap="medium")

    with c_img:
        with st.container(border=True):
            image, filename, source = collect_image(key="main_diag")

    with c_sens:
        with st.container(border=True):
            sensor_values = collect_sensors(key="main_sensors")

    btn_disabled = image is None
    btn_label = "🔬 Run Multimodal Diagnosis" if not btn_disabled else "⚠️ Select or Upload Leaf Photo to Proceed"

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(btn_label, type="primary", disabled=btn_disabled, width="stretch"):
        if image is None:
            st.error("Please supply a crop leaf photograph.")
            return

        sensor_df = get_sensor_dataframe(sensor_values, bundle.get("preprocessor"))
        sample_label = filename if filename else f"leaf_input_{int(time.time())}"

        with st.spinner("Executing tri-stream cross-attention inference..."):
            try:
                res = run_diagnosis(
                    image=image,
                    sensor_df=sensor_df,
                    sample_id=sample_label,
                    use_llm=use_llm,
                    bundle=bundle,
                )
                st.session_state["active_result"] = res
                st.session_state["diagnoses_count"] = st.session_state.get("diagnoses_count", 0) + 1
            except Exception as err:
                st.error(f"Inference error: {err}")
                return

    # Render results
    active_result = st.session_state.get("active_result")
    if active_result is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        render_diagnosis_view(active_result)

        st.markdown("---")
        output = active_result["output"]
        render_feedback(
            image_name=getattr(output, "sample_id", "sample"),
            predicted_class=getattr(output, "predicted_class", "unknown"),
            confidence=float(getattr(output, "confidence", 0.0)),
            key_prefix="diag_fb",
        )


# ---------------------------------------------------------------------------
# Tab 2: Feedback & History
# ---------------------------------------------------------------------------
def render_history_tab() -> None:
    """Render diagnosis audit log and feedback history."""
    st.markdown("## 📜 Case Audit Log & Agronomist Feedback")
    st.caption("Inspect past model calls, verified diagnoses, and flagged edge cases.")

    df = load_feedback_log()

    if df.empty:
        st.info("No feedback cases logged yet. Run a diagnosis and click '✓ Correct' or '✗ Incorrect' to build the registry.")
        return

    total_entries = len(df)
    correct_entries = int((df["feedback"] == "correct").sum())
    incorrect_entries = int((df["feedback"] == "incorrect").sum())
    agreement_rate = (correct_entries / total_entries * 100) if total_entries > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Logged Cases", total_entries)
    c2.metric("Agronomist Validated", correct_entries)
    c3.metric("Corrections Flagged", incorrect_entries)
    c4.metric("Agreement Ratio", f"{agreement_rate:.1f}%")

    st.markdown("### 📋 Recent 50 Diagnostic Records")
    recent_50 = df.iloc[::-1].head(50).reset_index(drop=True)
    st.dataframe(recent_50, width="stretch")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Audit Log (CSV)",
        data=csv_buffer.getvalue(),
        file_name="feedback_log.csv",
        mime="text/csv",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Tab 3: Multi-Image Batch Diagnosis Mode
# ---------------------------------------------------------------------------
def render_batch_tab(bundle: dict[str, Any]) -> None:
    """Process multiple leaf images sequentially."""
    st.markdown("## 📦 High-Throughput Batch Processing")
    st.caption("Upload multiple leaf photographs to run parallelized multimodal stress inference.")

    batch_files = st.file_uploader(
        "Upload a batch of crop leaf photos",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch_files_uploader",
    )

    if not batch_files:
        st.info("Select one or more images above to enable batch analysis.")
        return

    st.write(f"📁 Selected **{len(batch_files)}** images for processing.")

    with st.expander("⚙️ Environmental Baseline for Batch", expanded=False):
        batch_sensors = collect_sensors(key="batch_sensors_config")

    if st.button(f"🚀 Run Batch Analysis ({len(batch_files)} Images)", type="primary"):
        sensor_df = get_sensor_dataframe(batch_sensors, bundle.get("preprocessor"))
        results_list: list[dict[str, Any]] = []

        progress_bar = st.progress(0.0, text="Initializing batch inference...")
        time_start = time.perf_counter()

        for idx, file in enumerate(batch_files):
            step_text = f"Diagnosing {idx + 1}/{len(batch_files)}: {file.name}"
            progress_bar.progress((idx) / len(batch_files), text=step_text)

            try:
                img = Image.open(file).convert("RGB")
                res = run_diagnosis(
                    image=img,
                    sensor_df=sensor_df,
                    sample_id=file.name,
                    use_llm=False,
                    bundle=bundle,
                )
                out = res["output"]
                layer1 = res["layer1"]

                results_list.append({
                    "Image Name": file.name,
                    "Predicted Stress": out.predicted_class,
                    "Confidence": f"{out.confidence * 100:.1f}%",
                    "Severity": layer1.severity.upper(),
                    "Dominant Modality": layer1.driver_modality.title(),
                    "NDVI Trend": out.ndvi_trend,
                    "Latency (s)": round(res["seconds"], 3),
                })
            except Exception as err:
                results_list.append({
                    "Image Name": file.name,
                    "Predicted Stress": "ERROR",
                    "Confidence": "0.0%",
                    "Severity": "UNKNOWN",
                    "Dominant Modality": "N/A",
                    "NDVI Trend": str(err),
                    "Latency (s)": 0.0,
                })

            progress_bar.progress((idx + 1) / len(batch_files), text=step_text)

        total_elapsed = time.perf_counter() - time_start
        progress_bar.empty()

        st.success(f"✅ Finished diagnosing **{len(batch_files)}** images in **{total_elapsed:.2f}s** ({total_elapsed/len(batch_files):.2f}s/sample)!")

        results_df = pd.DataFrame(results_list)
        st.dataframe(results_df, width="stretch")

        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name=f"kisangpt_batch_results_{int(time.time())}.csv",
            mime="text/csv",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Tab 4: About & Disclosures
# ---------------------------------------------------------------------------
def render_about_tab() -> None:
    """Render scientific methodologies and disclosures."""
    st.markdown("## ℹ️ KisanGPT Architecture & Scientific Disclosures")

    st.markdown(
        """
        **KisanGPT** is an open-source, multi-modal precision agriculture decision-support system designed to diagnose
        crop stresses early by combining visual symptoms, local environmental sensor telemetry, and temporal vegetation dynamics.

        ### 🧠 Tri-Stream Cross-Attention Architecture
        1. **Leaf Phenology Stream (RGB Image):** MobileNetV2 pretrained backbone extracting 1,280-dimensional spatial visual features, supervised with Grad-CAM activation mapping.
        2. **Soil & Microclimate Stream (Tabular Sensors):** Multi-Layer Perceptron (MLP) encoding 27 normalized ground measurements (soil moisture, pH, rainfall, temperature, humidity) into a 128-dimensional embedding.
        3. **Temporal Canopy Stream (NDVI Time Series):** 1D-CNN temporal feature extractor processing 20-step vegetative indices into a 128-dimensional embedding.
        4. **Multi-Head Attention Fusion:** Pools multi-modal tokens into a 256-dimensional space and outputs both a 6-class diagnosis and dynamic attention weights explaining which sensor modality drove the call.

        ### ⚖️ Scientific Disclosures & Limitations
        1. **Synthetic Proxy NDVI:** In current offline test environments, co-located Sentinel-2 satellite passes are unavailable for individual field photos. Consequently, the 20-step NDVI time series is generated via a class-conditional synthetic proxy curve. All UI elements flag this with a `PROXY` disclaimer.
        2. **Weak Modality Pairing:** Because open datasets do not couple leaf photographs directly with physical soil probe records from the same field coordinate, samples are aligned via class-conditional weak pairing within strict stratified splits.
        3. **Reserved Classes:** The `pest_stress` class is reserved in the taxonomy for future pest telemetry streams and is not trained in the initial baseline weights.
        """
    )


# ---------------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------------
def main() -> None:
    """Main execution controller."""
    ckpt_path = resolve_model_path()
    model_ready = ckpt_path is not None
    kb_ready = KNOWLEDGE_BASE_PATH.exists()

    active_tab, use_llm = render_sidebar(model_ready=model_ready, kb_ready=kb_ready)

    if not model_ready:
        st.error("Unified model weights missing in `artifacts/`.")
        st.stop()

    try:
        bundle = load_app_bundle()
    except Exception as err:
        st.error(f"Failed to load application bundle: {err}")
        st.stop()

    if active_tab == "🩺 Live Diagnosis":
        render_diagnose_tab(bundle, use_llm)
    elif active_tab == "📜 Case History":
        render_history_tab()
    elif active_tab == "📦 Batch Processing":
        render_batch_tab(bundle)
    elif active_tab == "ℹ️ System & Disclosures":
        render_about_tab()


if __name__ == "__main__":
    main()
