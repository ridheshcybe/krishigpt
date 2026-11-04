"""Stage 2B Component - Feedback Collection and Logging.

Collects user validation on diagnoses:
- Two buttons side by side: "✓ Correct" / "✗ Incorrect"
- If "Incorrect": optional text input "What was the actual condition?" + dropdown to pick actual class.
- On click: appends row to feedback_log.csv with:
  timestamp, image_name, predicted_class, confidence, user_actual_class, feedback
- Shows st.toast or st.success confirmation.
- Updates session-state counters dynamically.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from stage1_config import UNIFIED_CLASSES

FEEDBACK_LOG_PATH = Path("feedback_log.csv")
FIELDNAMES = [
    "timestamp",
    "image_name",
    "predicted_class",
    "confidence",
    "user_actual_class",
    "feedback",
]


def ensure_feedback_log() -> None:
    """Create feedback_log.csv with standard headers if not present, or migrate older format."""
    if not FEEDBACK_LOG_PATH.exists() or FEEDBACK_LOG_PATH.stat().st_size == 0:
        with FEEDBACK_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        return

    # Check header
    with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if first_line != ",".join(FIELDNAMES):
        try:
            # Try to read and normalize
            with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) <= 1:
                # Just header, rewrite
                with FEEDBACK_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()
            else:
                # Re-parse
                rows = []
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) >= 6:
                        rows.append({
                            "timestamp": parts[0],
                            "image_name": parts[1],
                            "predicted_class": parts[2],
                            "confidence": parts[3],
                            "user_actual_class": parts[4],
                            "feedback": parts[5],
                        })
                    elif len(parts) == 5:
                        rows.append({
                            "timestamp": parts[0],
                            "image_name": parts[1],
                            "predicted_class": parts[2],
                            "confidence": "",
                            "user_actual_class": parts[3],
                            "feedback": parts[4],
                        })
                with FEEDBACK_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception:
            with FEEDBACK_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()


def load_feedback_log() -> pd.DataFrame:
    """Load the full feedback log into a pandas DataFrame."""
    ensure_feedback_log()
    try:
        df = pd.read_csv(FEEDBACK_LOG_PATH)
        for col in FIELDNAMES:
            if col not in df.columns:
                df[col] = ""
        return df[FIELDNAMES]
    except Exception:
        return pd.DataFrame(columns=FIELDNAMES)


def get_feedback_counts() -> tuple[int, int]:
    """Return running counts: (diagnoses_given, feedback_received)."""
    given = st.session_state.get("diagnoses_count", 0)
    received = st.session_state.get("feedback_count", 0)

    df = load_feedback_log()
    if not df.empty:
        log_feedback_count = int((df["feedback"].fillna("").astype(str).str.lower().isin(["correct", "incorrect"])).sum())
        received = max(received, log_feedback_count)
        given = max(given, len(df))

    st.session_state["diagnoses_count"] = given
    st.session_state["feedback_count"] = received
    return given, received


def append_feedback_record(
    image_name: str,
    predicted_class: str,
    confidence: float,
    user_actual_class: str,
    verdict: str,
) -> None:
    """Append a complete feedback entry to feedback_log.csv."""
    ensure_feedback_log()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp_str,
        "image_name": str(image_name),
        "predicted_class": str(predicted_class),
        "confidence": round(float(confidence), 4),
        "user_actual_class": str(user_actual_class).strip(),
        "feedback": str(verdict).lower(),
    }
    with FEEDBACK_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    st.session_state["feedback_count"] = st.session_state.get("feedback_count", 0) + 1


def render_feedback(
    image_name: str,
    predicted_class: str,
    confidence: float,
    key_prefix: str = "feedback",
) -> None:
    """Render the user feedback collection interface.

    Args:
        image_name: Filename or identifier of the evaluated leaf.
        predicted_class: Stress class predicted by model.
        confidence: Prediction confidence score in [0, 1].
        key_prefix: Unique widget key namespace.
    """
    st.markdown("### 💬 Provide Agronomist Feedback")
    st.caption("Help retrain and calibrate future versions of the unified multi-modal network.")

    submitted_key = f"{key_prefix}_submitted_{image_name}"
    if st.session_state.get(submitted_key, False):
        st.success("✅ Thank you! Your feedback for this diagnosis has been recorded.")
        return

    # Two buttons side by side
    col_correct, col_incorrect = st.columns(2)

    with col_correct:
        btn_correct = st.button(
            "✓ Correct Diagnosis",
            key=f"{key_prefix}_btn_correct",
            type="primary",
            width="stretch",
            help="Confirm that the predicted condition matches the crop's real state.",
        )

    with col_incorrect:
        btn_incorrect = st.button(
            "✗ Incorrect Diagnosis",
            key=f"{key_prefix}_btn_incorrect",
            width="stretch",
            help="Flag this prediction as incorrect and provide the ground truth condition.",
        )

    flag_incorrect_key = f"{key_prefix}_show_incorrect_form"
    if btn_incorrect:
        st.session_state[flag_incorrect_key] = True

    if btn_correct:
        append_feedback_record(
            image_name=image_name,
            predicted_class=predicted_class,
            confidence=confidence,
            user_actual_class="",
            verdict="correct",
        )
        st.session_state[submitted_key] = True
        try:
            st.toast("✅ Feedback logged: Marked as Correct!", icon="🌱")
        except Exception:
            pass
        st.success("✅ Feedback logged: Marked as Correct! Running counters updated.")
        st.rerun()

    if st.session_state.get(flag_incorrect_key, False):
        with st.container(border=True):
            st.markdown("##### 📝 Ground Truth Correction")
            c_text, c_drop = st.columns([1, 1])

            with c_text:
                custom_actual = st.text_input(
                    "What was the actual condition?",
                    placeholder="e.g. nitrogen deficiency, brown spot",
                    key=f"{key_prefix}_custom_actual",
                )

            with c_drop:
                class_options = ["(Select Class)"] + UNIFIED_CLASSES
                selected_actual = st.selectbox(
                    "Or pick standard unified class:",
                    options=class_options,
                    key=f"{key_prefix}_select_actual",
                )

            resolved_actual = ""
            if selected_actual != "(Select Class)":
                resolved_actual = selected_actual
            elif custom_actual.strip():
                resolved_actual = custom_actual.strip()

            if st.button("Submit Correction", key=f"{key_prefix}_btn_submit_corr", type="primary"):
                append_feedback_record(
                    image_name=image_name,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    user_actual_class=resolved_actual,
                    verdict="incorrect",
                )
                st.session_state[submitted_key] = True
                st.session_state[flag_incorrect_key] = False
                try:
                    st.toast("⚠️ Feedback logged: Marked as Incorrect with correction.", icon="📝")
                except Exception:
                    pass
                st.success("⚠️ Correction recorded to feedback_log.csv! Thank you for the training signal.")
                st.rerun()
