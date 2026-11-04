"""Stage 3B - animated field map (the "aww" visual).

Renders the farm as a 10x10 grid where **each cell is one field record** from
the Stage-3A trajectory dataset (see ``components_twin.__init__`` for the data
model).  Cells are coloured by stress class and change as the day scrubber
moves across history and forecast.

Two views:

``Grid heatmap``      one Plotly heatmap, one cell per field, discrete colour
                      scale using the exact spec palette, hover shows class /
                      NDVI / source, click selects a cell.
``Small multiples``   100 miniature NDVI charts (history solid, forecast
                      dashed) - the "all fields" overview mode.

Controls are returned as plain values so ``dashboard_twin.py`` can keep them
in the sidebar: this module never mutates global state on its own beyond the
click-selection session key.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stage1_config import UNIFIED_CLASSES

#: Exact stress-class palette from the Stage-3B spec, in unified-taxonomy order
#: (index 0..5), which is what the heatmap's discrete colourscale consumes.
CLASS_COLORS: dict[str, str] = {
    "healthy": "#2E7D32",        # green
    "water_stress": "#F9A825",   # yellow
    "nutrient_stress": "#EF6C00",  # orange
    "disease_stress": "#C62828",  # red
    "pest_stress": "#6A1B9A",    # purple
    "stress_generic": "#757575",  # gray
}

CLASS_LABELS: dict[str, str] = {
    name: name.replace("_", " ").title() for name in UNIFIED_CLASSES
}

#: Grid geometry: 10x10 cells per page of fields.
GRID_SIDE: int = 10
CELLS_PER_PAGE: int = GRID_SIDE * GRID_SIDE

#: How long one frame stays on screen during "Play" (seconds).
PLAY_FRAME_SECONDS: float = 0.45


# ---------------------------------------------------------------------------
# Colour scale helpers
# ---------------------------------------------------------------------------
def class_colorscale() -> list[tuple[float, str]]:
    """Discrete colourscale mapping class index -> the spec palette."""
    stops: list[tuple[float, str]] = []
    n = len(UNIFIED_CLASSES)
    for i, name in enumerate(UNIFIED_CLASSES):
        color = CLASS_COLORS[name]
        stops.append((i / n, color))
        stops.append(((i + 1) / n, color))
    return stops


def colorbar_tickvals() -> list[float]:
    """Colourbar tick positions (centre of each class band)."""
    n = len(UNIFIED_CLASSES)
    return [(i + 0.5) * (n / len(UNIFIED_CLASSES)) for i in range(n)]


# ---------------------------------------------------------------------------
# Day scrubber
# ---------------------------------------------------------------------------
def scrubber_bounds(mode: str, t_history: int, t_total: int) -> tuple[int, int]:
    """Day range offered by the scrubber for the chosen view mode."""
    today = t_history - 1
    if mode == "Historical":
        return 0, today
    if mode == "Forecast":
        return t_history, t_total - 1
    return 0, t_total - 1  # "Both"


def render_day_scrubber(
    mode: str,
    t_history: int,
    t_total: int,
    key: str = "twin_day",
) -> int:
    """Render the day slider plus Play/Stop buttons; return the active day.

    ``Play`` advances the day by one per rerun with a short sleep so cells
    visibly change colour over time (the spec's "streamlit slider + rerun"
    animation path).

    ORDER MATTERS: Streamlit forbids writing to a widget's key *after* that
    widget has been instantiated in the same run, so the buttons and the
    animation step are rendered **before** ``st.slider``.  Getting this wrong
    raises ``cannot be modified after the widget ... is instantiated``.
    """
    lo, hi = scrubber_bounds(mode, t_history, t_total)
    today = t_history - 1

    # 1) controls first (they may write ``key`` below, before the slider exists)
    col_play, col_stop = st.columns(2)
    with col_play:
        if st.button("▶ Play", width="stretch",
                     help="Animate the grid through the selected day range"):
            st.session_state["twin_playing"] = True
    with col_stop:
        if st.button("⏹ Stop", width="stretch"):
            st.session_state["twin_playing"] = False

    # 2) initial value + clamp, both before the widget is created
    if key not in st.session_state:
        st.session_state[key] = min(max(today, lo), hi)
    if st.session_state[key] < lo or st.session_state[key] > hi:
        st.session_state[key] = min(max(st.session_state[key], lo), hi)

    finished = False
    if st.session_state.get("twin_playing", False):
        if int(st.session_state[key]) >= hi:
            st.session_state["twin_playing"] = False
            finished = True
        else:
            st.session_state[key] = int(st.session_state[key]) + 1

    # 3) the slider itself
    day = st.slider(
        "Day",
        min_value=lo,
        max_value=hi,
        step=1,
        key=key,
        help="0 = first observation, day "
        f"{today} = today, days {t_history}-{t_total - 1} are forecast.",
    )

    if day < today:
        st.caption(f"day {day} - historical")
    elif day == today:
        st.caption(f"day {day} - **today**")
    else:
        st.caption(f"day {day} - forecast (+{day - today}d)")
    if finished:
        st.info("▶ Animation finished", icon="✅")

    # 4) pace the animation, after every widget in this run exists
    if st.session_state.get("twin_playing", False):
        time.sleep(PLAY_FRAME_SECONDS)
        st.rerun()

    return int(day)


# ---------------------------------------------------------------------------
# Per-day state matrix
# ---------------------------------------------------------------------------
def day_state(
    field_ids: np.ndarray,
    day: int,
    labels: np.ndarray,
    pred_classes: np.ndarray,
    t_history: int,
) -> np.ndarray:
    """Class index per cell for ``day``: stored labels in history, model argmax
    for forecast days.

    ``labels`` is indexed with the **global** field id (the full dataset),
    while ``pred_classes`` is a page-local array (one row per cell in view),
    so it is indexed with the position inside ``field_ids``.
    """
    if day < t_history:
        return labels[field_ids, day].astype(int)
    step = day - t_history
    if step >= pred_classes.shape[1]:
        step = pred_classes.shape[1] - 1
    pos = np.arange(len(field_ids))
    return pred_classes[pos, step].astype(int)


def day_ndvi(
    field_ids: np.ndarray,
    day: int,
    ndvi: np.ndarray,
    ndvi_pred: np.ndarray,
    t_history: int,
) -> np.ndarray:
    """NDVI per cell for ``day`` (observed in history, forecast afterwards).

    Same indexing split as :func:`day_state`: ``ndvi`` is global, ``ndvi_pred``
    is page-local.
    """
    if day < t_history:
        return ndvi[field_ids, day]
    step = min(day - t_history, ndvi_pred.shape[1] - 1)
    pos = np.arange(len(field_ids))
    return ndvi_pred[pos, step]


# ---------------------------------------------------------------------------
# Heatmap view
# ---------------------------------------------------------------------------
def build_heatmap(
    page_fields: np.ndarray,
    day: int,
    labels: np.ndarray,
    pred_classes: np.ndarray,
    ndvi: np.ndarray,
    ndvi_pred: np.ndarray,
    t_history: int,
    confidences: np.ndarray | None = None,
) -> go.Figure:
    """Build the 10x10 discrete-colour heatmap for one day."""
    states = day_state(page_fields, day, labels, pred_classes, t_history)
    ndvis = day_ndvi(page_fields, day, ndvi, ndvi_pred, t_history)

    grid = states.reshape(GRID_SIDE, GRID_SIDE)
    ndvi_grid = ndvis.reshape(GRID_SIDE, GRID_SIDE)

    source = "forecast" if day >= t_history else "observed"
    step = max(day - t_history, 0)
    name_grid = np.array(
        [
            [UNIFIED_CLASSES[grid[r, c]] for c in range(GRID_SIDE)]
            for r in range(GRID_SIDE)
        ]
    )
    custom = np.empty((GRID_SIDE, GRID_SIDE), dtype=object)
    for r in range(GRID_SIDE):
        for c in range(GRID_SIDE):
            idx = r * GRID_SIDE + c
            fid = int(page_fields[idx])
            conf_txt = ""
            if confidences is not None and day >= t_history:
                conf_txt = f"<br>confidence {confidences[idx, step]:.0%}"
            custom[r, c] = (
                f"<b>Field {fid}</b><br>{CLASS_LABELS[name_grid[r, c]]}"
                f"<br>NDVI {ndvi_grid[r, c]:.3f} ({source}){conf_txt}"
            )

    fig = go.Figure(
        go.Heatmap(
            z=grid + 0.5,
            x=list(range(GRID_SIDE)),
            y=list(range(GRID_SIDE)),
            zmin=0.0,
            zmax=float(len(UNIFIED_CLASSES)),
            colorscale=class_colorscale(),
            showscale=True,
            customdata=custom,
            hovertemplate="%{customdata}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Farm stress map - day {day}</b>"
                f"<span style='font-size:11px;color:#8D6E63'>"
                f" ({source}) | each cell = one field record</span>"
            ),
            font=dict(size=15),
        ),
        xaxis=dict(title="", showgrid=False, zeroline=False, constrain="domain"),
        yaxis=dict(title="", showgrid=False, zeroline=False, autorange="reversed"),
        coloraxis_colorbar=dict(
            title="class",
            tickvals=colorbar_tickvals(),
            ticktext=[CLASS_LABELS[c] for c in UNIFIED_CLASSES],
            thickness=14,
            len=0.9,
        ),
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="#FAFAF7",
        paper_bgcolor="#FAFAF7",
        clickmode="event+select",
    )
    return fig


# ---------------------------------------------------------------------------
# Small-multiples view
# ---------------------------------------------------------------------------
def build_small_multiples(
    page_fields: np.ndarray,
    ndvi: np.ndarray,
    ndvi_pred: np.ndarray,
    t_history: int,
    today: int,
) -> go.Figure:
    """100 miniature NDVI charts: history solid green, forecast dashed red."""
    fig = make_subplots(
        rows=GRID_SIDE,
        cols=GRID_SIDE,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.012,
        vertical_spacing=0.018,
    )
    hist_days = np.arange(t_history)
    fut_days = np.arange(today, today + ndvi_pred.shape[1] + 1)

    for i, fid in enumerate(page_fields):
        r, c = divmod(i, GRID_SIDE)
        f = int(fid)
        fig.add_trace(
            go.Scatter(
                x=hist_days, y=ndvi[f, :t_history],
                mode="lines", line=dict(color=CLASS_COLORS["healthy"], width=1.4),
                showlegend=False, hoverinfo="skip",
            ),
            row=r + 1, col=c + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=fut_days,
                y=np.concatenate([[ndvi[f, t_history - 1]], ndvi_pred[i]]),
                mode="lines",
                line=dict(color=CLASS_COLORS["disease_stress"], width=1.4, dash="dash"),
                showlegend=False,
                hovertemplate=f"Field {f} day %{{x}}: NDVI %{{y:.3f}}<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(
        title=dict(
            text="<b>All fields - NDVI small multiples</b>"
                  "<span style='font-size:11px;color:#8D6E63'>"
                  " green = history, red dashed = forecast</span>",
            font=dict(size=15),
        ),
        height=700,
        margin=dict(l=5, r=5, t=55, b=5),
        plot_bgcolor="#FAFAF7",
        paper_bgcolor="#FAFAF7",
    )
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].update(visible=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[0.0, 1.0])
    return fig


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------
def render_field_map(
    page_fields: np.ndarray,
    day: int,
    labels: np.ndarray,
    pred_classes: np.ndarray,
    ndvi: np.ndarray,
    ndvi_pred: np.ndarray,
    t_history: int,
    view: str = "Grid heatmap",
    confidences: np.ndarray | None = None,
    key: str = "twin_map",
) -> int | None:
    """Draw the field map and return the field id the user clicked (or None).

    Clicks come from Plotly's ``on_select`` (Streamlit >= 1.35); if the host
    Streamlit does not return selection events the function degrades to a
    no-click return instead of crashing, and ``dashboard_twin.py`` still offers
    the sidebar dropdown.
    """
    selected: int | None = None

    if view == "Small multiples":
        fig = build_small_multiples(page_fields, ndvi, ndvi_pred, t_history,
                                    t_history - 1)
        st.plotly_chart(fig, width="stretch", key=f"{key}_sm")
        return None

    fig = build_heatmap(
        page_fields, day, labels, pred_classes, ndvi, ndvi_pred,
        t_history, confidences,
    )
    try:
        state = st.plotly_chart(
            fig,
            width="stretch",
            key=f"{key}_hm",
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:  # very old Streamlit without on_select
        st.plotly_chart(fig, width="stretch", key=f"{key}_hm")
        return None

    points = []
    if state is not None:
        try:
            points = state.selection.points
        except AttributeError:
            points = []
    if points:
        try:
            col = int(points[0]["x"])
            row = int(points[0]["y"])
            idx = row * GRID_SIDE + col
            if 0 <= idx < len(page_fields):
                selected = int(page_fields[idx])
                st.session_state["twin_selected_field"] = selected
        except (KeyError, TypeError, ValueError):
            selected = None
    elif "twin_selected_field" in st.session_state:
        selected = int(st.session_state["twin_selected_field"])

    return selected
