"""Stage 3A - stress trajectory model (temporal digital twin network).

Encoder -> two decoder heads, exactly as specified:

    input (B, T_history, D)   D = 1 (NDVI only) or 5 (NDVI + 4 weather)
        |
        v
    [Option A] Temporal Transformer  - linear projection to d_model=128,
        sinusoidal positional encoding, N-layer TransformerEncoder (nhead=4),
        pooled (mean or last step, ``config.TRAJ_POOLING``) -> (B, 128)
                                                 <- TRAJ_ENCODER="transformer"
    [Option B] BiLSTM(128, 2 layers, bidirectional) -> context (B, 128)
                                                 <- TRAJ_ENCODER="bilstm"
        |
        +--> HEAD 1  NDVI regression:  Linear(128,64) ReLU Linear(64,T_future)
        |            -> (B, T_future), trained with MSE (Huber selectable)
        |
        +--> HEAD 2  Stress classes:   Linear(128,64) ReLU Linear(64,T_future*6)
                     -> logits (B, T_future, 6), softmax per time step at eval
                     -> trained with CrossEntropyLoss averaged over time steps

Encoder choice lives in ``config.TRAJ_ENCODER`` (Option A is the documented
default; Option B stays available for ablation).

NOTE on the forward contract: ``forward()`` returns *logits* for the class head
because CrossEntropyLoss expects raw logits; pass ``return_probs=True`` (or call
``predict()``) to receive the per-day softmax probabilities the spec describes.

ALERT LOGIC (post-processing, no extra parameters)
    ``detect_onset_alerts()`` scans the per-day class probabilities for the
    first non-healthy class within ``TRAJ_ALERT_HORIZON`` days and emits
    ``"water_stress onset predicted in X days"`` carrying that step's
    softmax probability as the confidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import config as cfg
from stage1_config import N_CLASSES, UNIFIED_CLASSES


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int = cfg.TRAJ_D_MODEL, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to a ``(B, T, d_model)`` sequence."""
        return x + self.pe[: x.size(1)].unsqueeze(0)


# ---------------------------------------------------------------------------
# Encoders (Option A preferred, Option B configurable)
# ---------------------------------------------------------------------------
class AttentivePooling(nn.Module):
    """Learned attention pooling over temporal history tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class TemporalTransformerEncoder(nn.Module):
    """Option A - TransformerEncoder over time with adaptive attention pooling.

    Combines the final horizon token (immediate state at day 21) with an attentive
    summary of historical drought and heatwave accumulation.

    Output: ``(B, d_model)`` context vector.
    """

    def __init__(
        self,
        input_dim: int = cfg.TRAJ_INPUT_DIM,
        d_model: int = cfg.TRAJ_D_MODEL,
        nhead: int = cfg.TRAJ_NHEAD,
        n_layers: int = cfg.TRAJ_TRANSFORMER_LAYERS,
        ff_dim: int = cfg.TRAJ_FF_DIM,
        dropout: float = cfg.TRAJ_DROPOUT,
        pooling: str = cfg.TRAJ_POOLING,
    ) -> None:
        super().__init__()
        pooling = pooling.lower()
        if pooling not in {"mean", "last", "attentive", "attention"}:
            raise ValueError(f"unknown pooling '{pooling}' (expected 'mean'|'last'|'attention')")
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.attentive_pool = AttentivePooling(d_model)
        self.pool_blend = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D)`` -> ``(B, d_model)`` pooled context."""
        h = self.dropout(self.pos(self.proj(x)))
        h = self.encoder(h)
        if self.pooling == "mean":
            return self.norm(h.mean(dim=1))
        if self.pooling in {"attention", "attentive"}:
            return self.norm(self.attentive_pool(h))

        last_tok = h[:, -1, :]
        if hasattr(self, "attentive_pool") and hasattr(self, "pool_blend"):
            attn_tok = self.attentive_pool(h)
            # Zero-initialized blend: identically equals last_tok at initialization
            blended = last_tok + torch.tanh(self.pool_blend) * (attn_tok - last_tok)
            return self.norm(blended)
        return self.norm(last_tok)


class BiLSTMEncoder(nn.Module):
    """Option B - BiLSTM(128, 2 layers, bidirectional), last-layer outputs fused.

    Output: ``(B, d_model)`` context vector.
    """

    def __init__(
        self,
        input_dim: int = cfg.TRAJ_INPUT_DIM,
        hidden: int = cfg.TRAJ_BILSTM_HIDDEN,
        n_layers: int = cfg.TRAJ_BILSTM_LAYERS,
        d_model: int = cfg.TRAJ_D_MODEL,
        dropout: float = cfg.TRAJ_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D)`` -> ``(B, d_model)``."""
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, hidden) - fuse the final layer's directions.
        forward = h_n[-2]
        backward = h_n[-1]
        return self.out(torch.cat([forward, backward], dim=-1))


def build_encoder(
    kind: str | None = None,
    pooling: str | None = None,
    input_dim: int = cfg.TRAJ_INPUT_DIM,
    d_model: int = cfg.TRAJ_D_MODEL,
    nhead: int = cfg.TRAJ_NHEAD,
    n_layers: int = cfg.TRAJ_TRANSFORMER_LAYERS,
    ff_dim: int = cfg.TRAJ_FF_DIM,
    dropout: float = cfg.TRAJ_DROPOUT,
) -> nn.Module:
    """Instantiate the encoder named by ``cfg.TRAJ_ENCODER`` (default Option A)."""
    chosen = (kind or cfg.TRAJ_ENCODER).lower()
    if chosen == "transformer":
        return TemporalTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            pooling=pooling or cfg.TRAJ_POOLING,
        )
    if chosen == "bilstm":
        return BiLSTMEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
    raise ValueError(f"unknown TRAJ_ENCODER '{chosen}' (expected 'transformer'|'bilstm')")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class TrajectoryModel(nn.Module):
    """Encoder + NDVI regression head + per-day stress-class head."""

    def __init__(
        self,
        t_future: int = cfg.TRAJ_T_FUTURE,
        input_dim: int = cfg.TRAJ_INPUT_DIM,
        n_classes: int = N_CLASSES,
        encoder_kind: str | None = None,
        pooling: str | None = None,
        d_model: int = cfg.TRAJ_D_MODEL,
        nhead: int = cfg.TRAJ_NHEAD,
        n_layers: int = cfg.TRAJ_TRANSFORMER_LAYERS,
        ff_dim: int = cfg.TRAJ_FF_DIM,
        dropout: float = cfg.TRAJ_DROPOUT,
    ) -> None:
        super().__init__()
        self.t_future = int(t_future)
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.encoder_kind = (encoder_kind or cfg.TRAJ_ENCODER).lower()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.n_layers = int(n_layers)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.pooling = str(pooling or cfg.TRAJ_POOLING)
        self.encoder = build_encoder(
            kind=self.encoder_kind,
            pooling=self.pooling,
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        )

        # HEAD 1 - NDVI regression with persistence-residual continuity
        self.ndvi_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.t_future),
        )
        self.ndvi_skip_scale = nn.Parameter(torch.zeros(1))

        # HEAD 2 - stress class sequence
        self.class_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.t_future * self.n_classes),
        )

        # HEAD 3 - early warning binary classifier (stress onset within K days)
        self.early_warning_head = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        last_ndvi: torch.Tensor | None = None,
        return_probs: bool = False,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """``x: (B, T_history, D)`` -> ``(ndvi_pred, class_head_output)``.

        ``class_head_output`` is raw logits by default (for ``CrossEntropyLoss``)
        or per-day softmax probabilities when ``return_probs=True``.
        """
        context = self.encoder(x)  # (B, d_model)
        pred_delta = self.ndvi_head(context)  # (B, T_future)

        if last_ndvi is None:
            last_observed = x[:, -1, 0:1]
        else:
            last_observed = last_ndvi.view(-1, 1)

        # Residual continuity: anchor forecast to the last observed NDVI value
        # Zero-initialized gating: produces identical output at init, learns physical continuity
        if hasattr(self, "ndvi_skip_scale") and self.ndvi_skip_scale != 0.0:
            gate = torch.sigmoid(self.ndvi_skip_scale)
            ndvi_trend = last_observed + (pred_delta - pred_delta[:, 0:1])
            ndvi_pred = torch.clamp((1.0 - gate) * pred_delta + gate * ndvi_trend, 0.0, 1.0)
        else:
            ndvi_pred = pred_delta

        logits = self.class_head(context).view(-1, self.t_future, self.n_classes)
        early_logit = self.early_warning_head(context).squeeze(-1) if hasattr(self, "early_warning_head") else None

        if return_dict:
            return {
                "ndvi": ndvi_pred,
                "delta": pred_delta,
                "logits": logits,
                "probs": torch.softmax(logits, dim=-1),
                "early_logit": early_logit,
            }

        if return_probs:
            return ndvi_pred, torch.softmax(logits, dim=-1)
        return ndvi_pred, logits

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, last_ndvi: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Inference helper: NDVI forecast + per-day class probabilities + early warning."""
        self.eval()
        out = self(x, last_ndvi=last_ndvi, return_dict=True)
        probs = out["probs"]
        res = {
            "ndvi": out["ndvi"],
            "probs": probs,
            "classes": probs.argmax(dim=-1),
        }
        if out["early_logit"] is not None:
            res["early_prob"] = torch.sigmoid(out["early_logit"])
        return res


# ---------------------------------------------------------------------------
# Alert logic (post-processing)
# ---------------------------------------------------------------------------
def detect_onset_alerts(
    probs: torch.Tensor | Any,
    horizon: int = cfg.TRAJ_ALERT_HORIZON,
    min_conf: float = cfg.TRAJ_ALERT_MIN_CONF,
    classes: list[str] | None = None,
    early_probs: torch.Tensor | np.ndarray | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Scan per-day class probabilities and/or binary early-warning probabilities.

    Args:
        probs: ``(B, T_future, n_classes)`` per-day softmax probabilities.
        horizon: look only at the first ``horizon`` predicted days (default 7).
        min_conf: optional confidence floor.
        classes: class names for reporting.
        early_probs: optional ``(B,)`` early-warning probability from binary head.
        threshold: optional threshold for alert decision (overrides min_conf).

    Returns:
        One dict per sample: ``{"alerted", "class", "class_idx", "day",
        "confidence", "message"}`` where ``day`` is days-from-today (0-based).
    """
    names = classes or UNIFIED_CLASSES
    arr = probs.detach().cpu() if isinstance(probs, torch.Tensor) else torch.as_tensor(probs)
    arr = arr.float()
    if arr.dim() == 2:  # a single sample (T, n_classes)
        arr = arr.unsqueeze(0)

    ep_arr = None
    if early_probs is not None:
        ep_arr = early_probs.detach().cpu().numpy() if isinstance(early_probs, torch.Tensor) else np.asarray(early_probs)
        if ep_arr.ndim == 0:
            ep_arr = ep_arr.reshape(1)

    limit = min(int(horizon), arr.size(1))
    thresh = threshold if threshold is not None else min_conf
    results: list[dict[str, Any]] = []

    for b in range(arr.size(0)):
        alerted = False
        record: dict[str, Any] = {
            "alerted": False,
            "class": None,
            "class_idx": None,
            "day": None,
            "confidence": None,
            # Every record carries its origin, including no-fire records.
            "alert_origin": "binary_head" if ep_arr is not None else "fallback_scan",
            "message": f"No stress onset predicted in the next {limit} days.",
        }

        # Dedicated binary early-warning head takes strict precedence
        if ep_arr is not None:
            p_alert = float(ep_arr[b])
            if p_alert >= thresh:
                # Identify the most likely stress class within horizon for display/messaging only
                sub_arr = arr[b, :limit]
                non_healthy = sub_arr[:, 1:].sum(dim=-1)
                best_d = int(non_healthy.argmax().item())
                cls_idx = int(sub_arr[best_d, 1:].argmax().item()) + 1
                conf = p_alert
                record.update({
                    "alerted": True,
                    "class": names[cls_idx] if cls_idx < len(names) else "stress_generic",
                    "class_idx": cls_idx,
                    "day": 0,  # Fired at evaluation day (t=0)
                    "alert_origin": "binary_head",
                    "confidence": round(conf, 4),
                    "message": (
                        f"EARLY WARNING: {names[cls_idx] if cls_idx < len(names) else 'stress'} "
                        f"onset predicted within {limit} days (confidence {conf:.0%})"
                    ),
                })
                alerted = True
            else:
                record.update({
                    "alerted": False,
                    "alert_origin": "binary_head",
                    "confidence": round(p_alert, 4),
                    "message": f"No stress onset predicted in the next {limit} days.",
                })
        else:
            # Fallback for models without dedicated binary early warning head
            for day in range(limit):
                step = arr[b, day]
                cls_idx = int(step.argmax().item())
                conf = float(step[cls_idx].item())
                if cls_idx != 0 and conf >= thresh:
                    ahead = int(day) + 1
                    record.update({
                        "alerted": True,
                        "class": names[cls_idx] if cls_idx < len(names) else "stress_generic",
                        "class_idx": cls_idx,
                        "day": int(day),
                        "predicted_onset_step": int(day),
                        "predicted_onset_day": ahead,  # 1-based day matching actual_onsets
                        "alert_origin": "fallback_scan",
                        "confidence": round(conf, 4),
                        "message": (
                            f"{names[cls_idx] if cls_idx < len(names) else 'stress'} onset predicted in {ahead} day"
                            f"{'s' if ahead != 1 else ''} (confidence {conf:.0%})"
                        ),
                    })
                    alerted = True
                    break

        record["alerted"] = alerted
        results.append(record)
    return results


# ---------------------------------------------------------------------------
# Persistence helper (shared by evaluate_trajectory / Stage 3B dashboard)
# ---------------------------------------------------------------------------
def load_trajectory_model(
    checkpoint: str | Path = cfg.TRAJ_CKPT,
    device: torch.device | str | None = None,
) -> tuple[TrajectoryModel, dict[str, Any]]:
    """Load ``artifacts/trajectory_model.pth`` with its config metadata.

    Raises a clear, actionable error if the checkpoint has not been trained yet.
    """
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory checkpoint not found: {path}. "
            "Train it first: python train_trajectory.py --mode train"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob.get("config", {})
    classes_list = meta.get("classes", UNIFIED_CLASSES)
    model = TrajectoryModel(
        t_future=int(meta.get("t_future", cfg.TRAJ_T_FUTURE)),
        input_dim=int(meta.get("input_dim", cfg.TRAJ_INPUT_DIM)),
        n_classes=len(classes_list),
        encoder_kind=str(meta.get("encoder", cfg.TRAJ_ENCODER)),
        pooling=str(meta.get("pooling", cfg.TRAJ_POOLING)),
        d_model=int(meta.get("d_model", cfg.TRAJ_D_MODEL)),
        nhead=int(meta.get("nhead", cfg.TRAJ_NHEAD)),
        n_layers=int(meta.get("n_layers", cfg.TRAJ_TRANSFORMER_LAYERS)),
        ff_dim=int(meta.get("ff_dim", cfg.TRAJ_FF_DIM)),
        dropout=float(meta.get("dropout", cfg.TRAJ_DROPOUT)),
    )
    # Ensure legacy checkpoints without new parameter keys evaluate with zero perturbation
    if "ndvi_skip_scale" not in blob["model_state"]:
        model.ndvi_skip_scale.data.zero_()
    if hasattr(model.encoder, "pool_blend") and "encoder.pool_blend" not in blob["model_state"]:
        model.encoder.pool_blend.data.zero_()

    model.load_state_dict(blob["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, blob

    model.load_state_dict(blob["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, blob
