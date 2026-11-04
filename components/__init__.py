"""KisanGPT Streamlit Dashboard Components (Stage 2B).

Modular UI components for:
- image_upload: Leaf image selection via drag-and-drop or random test-set sampling.
- sensor_input: 5 environmental sensor sliders with CHES medians & class presets.
- diagnosis_display: Rich diagnosis presentation (banners, Grad-CAM, Plotly charts).
- feedback: Farmer/agronomist feedback collection logging to CSV.
"""

from components.diagnosis_display import render_diagnosis_view
from components.feedback import get_feedback_counts, load_feedback_log, render_feedback
from components.image_upload import collect_image, load_test_image_pool
from components.sensor_input import (
    CHES_MEDIANS,
    collect_sensors,
    get_sensor_dataframe,
)

__all__ = [
    "collect_image",
    "load_test_image_pool",
    "collect_sensors",
    "get_sensor_dataframe",
    "CHES_MEDIANS",
    "render_diagnosis_view",
    "render_feedback",
    "get_feedback_counts",
    "load_feedback_log",
]
