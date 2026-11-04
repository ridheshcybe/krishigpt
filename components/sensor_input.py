"""Stage 2B Component - Environmental Sensor Inputs.

Provides intuitive, responsive controls for:
- Soil Moisture (%)
- Soil pH
- Temperature (°C)
- Relative Humidity (%)
- 24-hr Rainfall (mm)

Includes:
- Quick Agro-Climatic Presets (Drought, Waterlogged, Humid Fungal, Optimal)
- CHES Dataset Baseline Reversion
- Visual gauge indicators
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

#: CHES training set medians from artifacts/xgb_preprocessor.json
CHES_MEDIANS: dict[str, float] = {
    "soil_moisture": 20.08,
    "ph": 6.50,
    "temperature": 24.99,
    "humidity": 59.99,
    "rainfall": 13.80,
}

#: Profile configurations for real-world stress testing
SENSOR_PRESETS: dict[str, dict[str, Any]] = {
    "🌱 Optimal Healthy Conditions": {
        "values": {"soil_moisture": 52.0, "ph": 6.6, "temperature": 26.0, "humidity": 65.0, "rainfall": 120.0},
        "description": "Standard paddy growth conditions; balanced water & nutrients.",
    },
    "☀️ Severe Drought (Water Stress)": {
        "values": {"soil_moisture": 12.0, "ph": 7.0, "temperature": 37.0, "humidity": 28.0, "rainfall": 2.0},
        "description": "Depleted soil moisture (<15%) and elevated heat typical of drought stress.",
    },
    "🌧️ Waterlogged Soil (Nutrient Leaching)": {
        "values": {"soil_moisture": 68.0, "ph": 5.2, "temperature": 25.0, "humidity": 75.0, "rainfall": 160.0},
        "description": "Saturated root zone causing nutrient chlorosis and root asphyxiation.",
    },
    "🌫️ High Humidity / Fungal Vulnerability": {
        "values": {"soil_moisture": 40.0, "ph": 6.5, "temperature": 29.0, "humidity": 92.0, "rainfall": 190.0},
        "description": "Sustained >85% humidity accelerating blast and bacterial pathogen spread.",
    },
    "⚖️ CHES Dataset Baseline (Medians)": {
        "values": CHES_MEDIANS,
        "description": "Neutral multi-year median telemetry from CHES sensor stations.",
    },
}

PREPROCESSOR_PATH = Path("artifacts/xgb_preprocessor.json")


def collect_sensors(key: str = "sensor_panel") -> dict[str, float]:
    """Render polished soil & microclimate telemetry panel."""
    # Seed initial session states cleanly
    for field, default_val in CHES_MEDIANS.items():
        state_key = f"{key}_{field}"
        if state_key not in st.session_state:
            st.session_state[state_key] = float(default_val)

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <h4 style="margin: 0; font-size: 1.15rem; color: #1e293b; font-weight: 700;">
                🧪 2. Soil & Microclimate Telemetry
            </h4>
            <span style="font-size: 0.8rem; background: #e2e8f0; color: #475569; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">
                CHES Tabular Stream
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset Selector Box
    selected_preset = st.selectbox(
        "Load Agro-Climatic Scenario:",
        options=list(SENSOR_PRESETS.keys()),
        index=0,
        key=f"{key}_scenario_select",
        help="Quickly populate realistic field sensor telemetry.",
    )

    # Apply scenario on change
    last_scenario_key = f"{key}_active_scenario"
    if st.session_state.get(last_scenario_key) != selected_preset:
        st.session_state[last_scenario_key] = selected_preset
        preset_info = SENSOR_PRESETS[selected_preset]
        for f_name, f_val in preset_info["values"].items():
            st.session_state[f"{key}_{f_name}"] = float(f_val)
        st.rerun()

    st.caption(f"💡 *{SENSOR_PRESETS[selected_preset]['description']}*")

    # Sensor Sliders in 2 clean columns
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.session_state[f"{key}_soil_moisture"] = st.slider(
            "💧 Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state[f"{key}_soil_moisture"]),
            step=0.5,
            key=f"{key}_slider_sm",
            help="Volumetric water content in the root zone.",
        )
        st.session_state[f"{key}_ph"] = st.slider(
            "⚗️ Soil pH",
            min_value=3.0,
            max_value=10.0,
            value=float(st.session_state[f"{key}_ph"]),
            step=0.1,
            key=f"{key}_slider_ph",
            help="Soil acidity/alkalinity level. Rice thrives at 6.0-7.0.",
        )
        st.session_state[f"{key}_rainfall"] = st.slider(
            "🌧️ Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=float(st.session_state[f"{key}_rainfall"]),
            step=1.0,
            key=f"{key}_slider_rain",
            help="24-hour precipitation accumulation.",
        )

    with col_b:
        st.session_state[f"{key}_temperature"] = st.slider(
            "🌡️ Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=float(st.session_state[f"{key}_temperature"]),
            step=0.5,
            key=f"{key}_slider_temp",
            help="Ambient field temperature.",
        )
        st.session_state[f"{key}_humidity"] = st.slider(
            "💨 Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state[f"{key}_humidity"]),
            step=0.5,
            key=f"{key}_slider_hum",
            help="Ambient relative humidity.",
        )

        st.write("")
        if st.button("↺ Reset to CHES Medians", key=f"{key}_reset_btn", width="stretch"):
            for field, val in CHES_MEDIANS.items():
                st.session_state[f"{key}_{field}"] = float(val)
            st.session_state[last_scenario_key] = "⚖️ CHES Dataset Baseline (Medians)"
            st.rerun()

    return {
        "soil_moisture": float(st.session_state[f"{key}_soil_moisture"]),
        "ph": float(st.session_state[f"{key}_ph"]),
        "temperature": float(st.session_state[f"{key}_temperature"]),
        "humidity": float(st.session_state[f"{key}_humidity"]),
        "rainfall": float(st.session_state[f"{key}_rainfall"]),
    }


def get_sensor_dataframe(
    sensor_dict: dict[str, float],
    preprocessor: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a complete, 27-column CHES compatible single-row DataFrame."""
    feature_names = [
        "soil_moisture", "ph", "temperature", "humidity", "rainfall",
        "soil_temperature", "solar_radiation", "wind_speed", "wind_direction",
        "ec", "nitrogen", "phosphorus", "potassium", "organic_matter",
        "battery_voltage", "internal_temp", "signal_strength", "pressure",
        "vpd", "dew_point", "co2", "leaf_wetness", "uv_index",
        "par", "heat_index", "gust_speed", "et0",
    ]
    if preprocessor and "feature_names" in preprocessor:
        feature_names = list(preprocessor["feature_names"])

    medians: dict[str, float] = {}
    if preprocessor and "medians" in preprocessor:
        medians = preprocessor["medians"]

    row_data: dict[str, float] = {}
    for feat in feature_names:
        if feat in sensor_dict:
            row_data[feat] = float(sensor_dict[feat])
        elif feat in medians:
            row_data[feat] = float(medians[feat])
        elif feat in CHES_MEDIANS:
            row_data[feat] = float(CHES_MEDIANS[feat])
        else:
            row_data[feat] = 0.0

    return pd.DataFrame([row_data], columns=feature_names)
