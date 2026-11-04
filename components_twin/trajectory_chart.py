"""Stage 3B - NDVI trajectory chart (history + forecast).

Plotly line chart for one field:

* historical NDVI, **solid** green
* predicted NDVI, **dashed** red, joined to today's value so the join is
  visible as a change of line style rather than a gap
* shaded uncertainty band of ``config.TRAJ_BAND`` (default +/-0.05) - the
  Stage-3A model has no variance head, so a fixed band is declared rather
  than invented per point (flagged in the caption)
* vertical "today" marker separating observed from forecast
* stress-onset markers: where the predicted class sequence first becomes
  non-healthy (and any onset already present in history, drawn hollow)

Hover, zoom and pan come free with Plotly; nothing here recomputes the model.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import config as cfg
from components_twin.field_map import CLASS_COLORS, CLASS_LABELS
from stage1_config import UNIFIED_CLASSES


def render_trajectory_chart(
    field_id: int,
    ndvi_history: np.ndarray,
    ndvi_forecast: np.ndarray,
    labels_history: np.ndarray,
    pred_classes: np.ndarray,
    pred_probs: np.ndarray,
    ndvi_is_proxy: bool = True,
    band: float | None = None,
) -> go.Figure:
    """Build the history-vs-forecast NDVI figure for one field.

    Args:
        field_id: cell/field identifier, used in the title and hover text.
        ndvi_history: observed NDVI for days ``0..t_history-1``.
        ndvi_forecast: predicted NDVI for the ``t_future`` days after today.
        labels_history: day-level classes for the observed window.
        pred_classes: predicted class index per future day.
        pred_probs: ``(t_future, 6)`` per-day class probabilities.
        ndvi_is_proxy: whether the series is PROXY (drives the badge).
        band: half-width of the uncertainty band; defaults to
            ``config.TRAJ_BAND``.

    Returns:
        A ``plotly.graph_objects.Figure`` (not yet drawn - the caller decides
        the container/key so Streamlit can deduplicate reruns).
    """
    band = cfg.TRAJ_BAND if band is None else float(band)
    t_history = len(ndvi_history)
    today = t_history - 1
    t_future = len(ndvi_forecast)

    hist_x = np.arange(t_history)
    fut_x = np.arange(today, today + t_future + 1)
    fut_y = np.concatenate([[ndvi_history[-1]], ndvi_forecast])

    fig = go.Figure()

    # --- uncertainty band (declared constant width, no variance head) ------
    fig.add_trace(
        go.Scatter(
            x=fut_x,
            y=np.clip(fut_y + band, 0.0, 1.0),
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name="band+",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fut_x,
            y=np.clip(fut_y - band, 0.0, 1.0),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(198,40,40,0.15)",
            hoverinfo="skip",
            showlegend=False,
            name=f"forecast band +/-{band}",
        )
    )

    # --- history (solid) ----------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=hist_x,
            y=ndvi_history,
            mode="lines+markers",
            line=dict(color=CLASS_COLORS["healthy"], width=2.6),
            marker=dict(size=4),
            name="historical NDVI",
            hovertemplate="day %{x}: NDVI %{y:.3f}<extra>observed</extra>",
        )
    )

    # --- forecast (dashed), joined at today ---------------------------------
    fig.add_trace(
        go.Scatter(
            x=fut_x,
            y=fut_y,
            mode="lines+markers",
            line=dict(color=CLASS_COLORS["disease_stress"], width=2.6, dash="dash"),
            marker=dict(size=4),
            name="predicted NDVI",
            hovertemplate="day %{x}: NDVI %{y:.3f} (forecast)<extra></extra>",
        )
    )

    # --- today marker -------------------------------------------------------
    fig.add_vline(
        x=today,
        line_width=1.6,
        line_dash="dot",
        line_color="#455A64",
    )
    fig.add_annotation(
        x=today,
        y=1.02,
        text="<b>today</b>",
        showarrow=False,
        font=dict(size=11, color="#455A64"),
        xanchor="left",
    )

    # --- onset annotations --------------------------------------------------
    # Historical onsets (already happening): hollow amber markers.
    hist_onsets = np.flatnonzero(
        (labels_history[1:] != 0) & (labels_history[:-1] == 0)
    ) + 1
    if labels_history[0] != 0:
        hist_onsets = np.concatenate([[0], hist_onsets])
    if hist_onsets.size:
        fig.add_trace(
            go.Scatter(
                x=hist_onsets,
                y=ndvi_history[hist_onsets],
                mode="markers",
                marker=dict(
                    size=12, symbol="circle-open", line=dict(width=2),
                    color=CLASS_COLORS["water_stress"],
                ),
                name="onset in history",
                hovertemplate=(
                    "day %{x}: stress onset (observed)<extra>%{customdata}</extra>"
                ),
                customdata=[
                    UNIFIED_CLASSES[labels_history[i]] for i in hist_onsets
                ],
            )
        )

    # Predicted onsets: first day a non-healthy class wins, filled marker.
    pred_onsets = np.flatnonzero(
        (pred_classes != 0)
        & np.concatenate([[True], pred_classes[1:] == 0])
    )
    if pred_onsets.size:
        first = int(pred_onsets[0])
        first_day = t_history + first
        cls = UNIFIED_CLASSES[int(pred_classes[first])]
        conf = float(pred_probs[first, int(pred_classes[first])])
        fig.add_trace(
            go.Scatter(
                x=[t_history + int(i) for i in pred_onsets],
                y=ndvi_forecast[pred_onsets],
                mode="markers",
                marker=dict(
                    size=13, symbol="diamond",
                    color=CLASS_COLORS.get(cls, "#757575"),
                    line=dict(width=1.5, color="#212121"),
                ),
                name="predicted onset",
                hovertemplate=(
                    "day %{x}: predicted onset<extra>%{customdata}</extra>"
                ),
                customdata=[UNIFIED_CLASSES[int(pred_classes[i])] for i in pred_onsets],
            )
        )
        fig.add_annotation(
            x=first_day,
            y=float(ndvi_forecast[first]),
            text=(
                f"{CLASS_LABELS[cls]} onset<br>in {first_day - today} day(s), "
                f"{conf:.0%} conf."
            ),
            showarrow=True,
            arrowhead=2,
            ax=28,
            ay=-34,
            font=dict(size=10, color="#212121"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#BDBDBD",
            borderwidth=1,
        )

    badge = (
        "PROXY series - prototype data, not a satellite readout"
        if ndvi_is_proxy
        else "real NDVI export"
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Field {field_id} - NDVI trajectory</b>"
                f"<span style='font-size:11px;color:#8D6E63'> [{badge}]</span>"
            ),
            font=dict(size=15),
        ),
        xaxis=dict(title="day", range=[-0.5, today + t_future + 0.5], gridcolor="#E8E8E8"),
        yaxis=dict(title="NDVI", range=[0.0, 1.0], gridcolor="#E8E8E8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        hovermode="x unified",
        height=420,
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="#FAFAF7",
        paper_bgcolor="#FAFAF7",
    )
    return fig
