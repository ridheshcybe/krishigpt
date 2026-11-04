# 🌱 KisanGPT — Crop Stress Doctor Dashboard (Stage 2B)

An interactive, demo-ready **Streamlit web application** for multi-modal crop stress diagnosis. KisanGPT unifies RGB leaf imagery, environmental soil/atmosphere telemetry, and vegetative temporal signatures to deliver instant diagnoses, visual explainability (Grad-CAM), modality attention attribution, and actionable farmer advice within 2 seconds.

---

## 🌟 Key Features

1. **Instant Multi-Modal Diagnosis (<2s):**
   - **Leaf Image:** Drag-and-drop file upload (`.jpg`, `.png`) or 1-click sampling from real held-out test splits (`data/paddy_images/test/`).
   - **Environmental Telemetry:** 5 interactive sliders (Soil Moisture, Soil pH, Temperature, Relative Humidity, 10-day Rainfall) inside an expander with CHES dataset training medians and class-conditional presets.
2. **Visual & Attributed Explainability:**
   - **Grad-CAM Overlay:** Interactive visualization of MobileNetV2 spatial activations localized on crop lesions.
   - **Attention Bar Chart:** Interactive Plotly horizontal chart depicting which modality (Tabular Sensors, Leaf Image, NDVI Canopy) drove the prediction.
   - **NDVI Trajectory Sparkline:** Interactive 20-step vegetative time-series curve.
3. **Agronomist & Farmer-Friendly Advisory:**
   - Multi-tier diagnosis: Layer 1 rule-based template + optional Layer 2 LLM polish (supports Ollama, OpenAI, Anthropic with automatic silent fallback).
   - Severity badges (Low / Medium / High) and concrete action steps with escalation guidelines.
4. **Active Learning Feedback Loop:**
   - Side-by-side **✓ Correct** and **✗ Incorrect** buttons.
   - Correction capture (actual class selection & custom notes) appended to `feedback_log.csv`.
   - Dynamic sidebar counters: *X diagnoses given, Y feedback received*.
5. **Multi-Tab Architecture:**
   - **🩺 Diagnose:** Primary single-sample inference & feedback.
   - **📜 History:** Audit log of past diagnoses (last 50 rows) with summary metrics and CSV export.
   - **📦 Batch Mode:** Sequential multi-image diagnosis with progress tracking and CSV download.
   - **ℹ️ About:** Scientific citations, architecture diagram, and honest proxy/pairing disclosures.

---

## 🏗️ Architecture Overview

```
                                  +-----------------------------+
                                  |   RGB Leaf Photo (224x224)  |
                                  +--------------+--------------+
                                                 |
                                                 v
+----------------------------+        +--------------------+        +---------------------------+
| 27-Dim Tabular Sensors     |        | MobileNetV2 Trunk  |        | 20-Step NDVI Time Series  |
| (Soil, Temp, Rain, etc.)   |        | (Spatial Conv CAM) |        | (Proxy Vegetative Curve)  |
+--------------+-------------+        +----------+---------+        +-------------+-------------+
               |                                 |                                |
               v                                 v                                v
       +---------------+                 +---------------+                +---------------+
       |  MLP Encoder  |                 | Image Encoder |                | 1D-CNN Encoder|
       |    (128-d)    |                 |   (1280-d)    |                |    (128-d)    |
       +-------+-------+                 +-------+-------+                +-------+-------+
               |                                 |                                |
               +---------------------------------+--------------------------------+
                                                 | (1536-d Token Split)
                                                 v
                               +-----------------------------------+
                               | Multi-Head Self-Attention (4-head)|
                               | Cross-Modality Fusion (256-d)     |
                               +-----------------+-----------------+
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
                         v                                               v
               +-------------------+                           +-------------------+
               | Classification    |                           | Attention Weights |
               | 6-Class Logits    |                           | (Tabular/Img/NDVI)|
               +---------+---------+                           +---------+---------+
                         |                                               |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  | Stage 2A Diagnosis Engine   |
                                  |  (Layer 1 KB + Layer 2 LLM) |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  | Streamlit Web Dashboard UI  |
                                  +-----------------------------+
```

---

## 📋 Prerequisites & Installation

### 1. Requirements
Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies verified:
- `streamlit>=1.36`
- `plotly>=5.20`
- `pillow>=10.0`
- `torch>=2.0`
- `torchvision>=0.15`
- `numpy>=1.24`
- `pandas>=2.0`
- `scikit-learn>=1.3`
- `xgboost>=2.0`
- `matplotlib>=3.7`

### 2. Verify Artifacts
The app expects the following files:
- Model Checkpoint: `artifacts/unified_model.pth` or `artifacts/attention_fusion.pt`
- Feature Preprocessor: `artifacts/xgb_preprocessor.json`
- Knowledge Base: `knowledge_base.json`
- Test Images: `data/paddy_images/test/`

If model checkpoints are not yet trained:
```bash
python attention_fusion.py
```

---

## 🚀 How to Run

### Default Port (8501)
```bash
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

### Custom Port Support
You can specify custom ports using `--server.port`:
```bash
streamlit run app.py --server.port 8502
```
or to run in headless background mode:
```bash
streamlit run app.py --server.port 8080 --server.headless true
```

---

## 🖥️ UI Walkthrough & Layout

### 1. 🩺 Diagnose Tab
- **Input Area:** Left column previews leaf uploads or randomly rolled test-set samples. Right column provides sliders for `Soil Moisture`, `pH`, `Temperature`, `Humidity`, and `Rainfall`.
- **Top Banner:** Colored dynamically:
  - 🟢 **Green (`st.success`)**: Healthy foliage with high confidence (>80%).
  - 🟡 **Yellow (`st.warning`)**: Moderate confidence or early-stage stress (50%–80%).
  - 🚨 **Red (`st.error`)**: High-confidence severe stress (>80%).
- **Explainability Columns:**
  - *Left:* Interactive toggle between Grad-CAM overlay, original leaf, and raw Jet heatmap.
  - *Right:* Plotly horizontal bar chart of attention weights + Plotly NDVI vegetative sparkline.
- **Agronomist Advisory:** Clean farmer-friendly guidance box with bulleted actions and escalation thresholds.
- **Auditing:** Expandable JSON viewer showing complete raw model output dictionary.
- **Feedback:** "✓ Correct" / "✗ Incorrect" buttons that record directly to `feedback_log.csv`.

### 2. 📜 History Tab
- Displays summary metrics: Total diagnoses, confirmed count, corrections flagged, and agreement rate.
- Interactive table showing the last 50 diagnoses with timestamps, filenames, predicted classes, confidences, and user verdicts.
- **Download CSV Button:** Export all feedback records for model retraining.

### 3. 📦 Batch Mode Tab
- Upload dozens of crop photos at once.
- Configure default or custom environmental sensor baselines.
- Sequential execution with an interactive progress bar.
- Export tabulated results directly to CSV.

### 4. ℹ️ About Tab
- Explains the multi-modal neural architecture.
- Full citations for Paddy Doctor and CHES datasets.
- Transparent disclosures regarding synthetic proxy NDVI and weak sample pairing.

---

## 🔧 Troubleshooting Guide

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `Unified model checkpoint missing` | Missing `.pth` or `.pt` weight file | Run `python attention_fusion.py` to train the unified model. |
| `No test images found` | Missing `data/paddy_images/test/` directory | Switch to "Upload Image" mode or ensure the dataset folder exists. |
| `Port 8501 is already in use` | Another process is occupying the default port | Run with a different port: `streamlit run app.py --server.port 8505` |
| `LLM Provider Offline` | Ollama or API key not configured | System automatically falls back to Stage 2A Layer 1 template diagnosis without crashing. |
| `Corrupt/Unreadable Image` | Unsupported image encoding | An alert will appear prompting the user to upload a standard JPG or PNG. |

---

## 📈 Active Learning & Retraining Loop

Every feedback interaction generates an entry in `feedback_log.csv`:
```csv
timestamp,image_name,predicted_class,confidence,user_actual_class,feedback
2026-10-29 15:30:10,BLB_leaf_12.jpg,disease_stress,0.9240,,correct
2026-10-29 15:32:45,upload_sample.jpg,water_stress,0.5820,nutrient_stress,incorrect
```

These flagged edge cases and confirmed samples can be imported directly into the retraining pipeline for semi-supervised fine-tuning.
