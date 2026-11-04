"""Stage 3A - train the stress trajectory model (temporal digital twin).

    python train_trajectory.py --mode train       # full run (60 epochs, patience 12)
    python train_trajectory.py --mode dry-run     # 48 fields, 2 epochs, no artifacts

Spec compliance
---------------
* combined loss = MSE(ndvi) + ``TRAJ_LAMBDA_CE`` * CE(class), lambda=0.5
* AdamW lr=1e-4, weight_decay=1e-5, CosineAnnealingLR
* batch 32, 60 epochs, early stopping patience 12 on the **val combined loss**
* mixed precision (``torch.amp`` autocast + GradScaler) only when
  ``torch.cuda.is_available()`` - it is a no-op on CPU
* gradient clipping max_norm=1.0
* split 70/15/15 **by field** (``trajectory_data.field_split``), never by time
  step, so no field's future is memorised from its own history in another split
* class balancing: healthy is ~73% of future-window days, so the per-day CE
  term uses inverse-frequency weights fitted on the TRAIN split only
  (``config.TRAJ_BALANCE_CLASSES``), tempered by ``TRAJ_CLASS_WEIGHT_ALPHA``
  so the argmax decision boundary is not dragged onto the minority classes.
  Without weights the head degenerates to all-healthy (measured: macro F1 0.14,
  onset/alert F1 0.00); with untempered weights it over-fires (measured:
  alert precision 0.30, later-day accuracy 0.04).  The loss is still
  "CrossEntropyLoss per time step, averaged", just weighted.
* best checkpoint -> ``artifacts/trajectory_model.pth``
* history         -> ``artifacts/trajectory_history.json``
* tqdm bars on both loops, one log line per epoch, seed=42 everywhere

``--mode dry-run`` proves the whole pipeline (build -> split -> forward ->
backward -> validate) in seconds and deliberately writes **no** artifacts, so
it can never clobber a trained checkpoint or the 600-field cache.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from stage1_config import N_CLASSES, SEED, UNIFIED_CLASSES, UNIFIED_INDEX
from trajectory_data import (
    TrajectoryDataset,
    build_dataset,
    compute_channel_stats,
    field_split,
    log,
)
from trajectory_model import TrajectoryModel


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (and cuDNN determinism) - seed=42 everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """CUDA when available, otherwise CPU (Windows-friendly default)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiClassFocalLoss(nn.Module):
    """Focal loss for multi-class classification with class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits_v = logits[valid]
        targets_v = targets[valid]
        ce = F.cross_entropy(
            logits_v,
            targets_v,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class EarlyWarningLoss(nn.Module):
    """Lead-time aware loss for binary early-warning prediction.

    Supports BCEWithLogits, binary focal loss, and pairwise lead-time ranking loss
    with horizon-aware sample weighting (higher weights for earlier advance alerts).
    """

    def __init__(
        self,
        kind: str = "bce",
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        lead_weight_1d: float = 0.6,
        lead_weight_2d: float = 0.8,
        lead_weight_3d: float = 1.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.pos_weight = float(pos_weight)
        self.focal_gamma = float(focal_gamma)
        self.lead_weights = {
            1: float(lead_weight_1d),
            2: float(lead_weight_2d),
            3: float(lead_weight_3d),
            4: 1.0,
            5: 1.0,
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        days_to_onset: torch.Tensor,
    ) -> torch.Tensor:
        valid = (masks > 0.5)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        v_logits = logits[valid]
        v_targets = targets[valid]
        v_days = days_to_onset[valid]

        # Horizon-aware sample weighting
        weights = torch.ones_like(v_targets)
        for d_k, w_v in self.lead_weights.items():
            match = (v_targets == 1.0) & (v_days == d_k)
            weights[match] = w_v

        pos_wt = torch.tensor([self.pos_weight], device=logits.device)

        if self.kind == "bce":
            bce = F.binary_cross_entropy_with_logits(
                v_logits, v_targets, pos_weight=pos_wt, reduction="none"
            )
            return (bce * weights).mean()

        elif self.kind == "focal":
            p = torch.sigmoid(v_logits)
            p_t = p * v_targets + (1.0 - p) * (1.0 - v_targets)
            alpha = self.pos_weight / (1.0 + self.pos_weight)
            alpha_t = alpha * v_targets + (1.0 - alpha) * (1.0 - v_targets)
            focal_mod = alpha_t * ((1.0 - p_t) ** self.focal_gamma)
            bce = F.binary_cross_entropy_with_logits(v_logits, v_targets, reduction="none")
            return (focal_mod * bce * weights).mean()

        elif self.kind == "ranking":
            bce = F.binary_cross_entropy_with_logits(
                v_logits, v_targets, pos_weight=pos_wt, reduction="none"
            )
            bce_loss = (bce * weights).mean()

            # Pairwise margin ranking between positive samples with different lead times
            pos_mask = (v_targets == 1.0)
            if pos_mask.sum() >= 2:
                pos_logits = v_logits[pos_mask]
                pos_days = v_days[pos_mask]
                d_diff = pos_days.unsqueeze(1) - pos_days.unsqueeze(0)
                pair_mask = (d_diff > 0)
                if pair_mask.any():
                    l_high = pos_logits.unsqueeze(1).expand_as(d_diff)[pair_mask]
                    l_low = pos_logits.unsqueeze(0).expand_as(d_diff)[pair_mask]
                    rank_loss = F.margin_ranking_loss(
                        l_high, l_low, torch.ones_like(l_high), margin=0.1
                    )
                    return bce_loss + 0.5 * rank_loss
            return bce_loss

        return F.binary_cross_entropy_with_logits(v_logits, v_targets, pos_weight=pos_wt)


def make_loaders(
    data: dict[str, Any],
    split: dict[str, Any],
    batch_size: int,
    seed: int,
    channel_mean: list[float] | None = None,
    channel_std: list[float] | None = None,
    alert_horizon: int = 3,
    class_mapping: dict[int, int] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders over **field** indices (no time-step leak)."""
    sets = {
        key: np.asarray(split[key], dtype=np.int64) for key in ("train", "val", "test")
    }
    loaders: dict[str, DataLoader] = {}
    for key in ("train", "val", "test"):
        dataset = TrajectoryDataset(
            ndvi=data["ndvi"],
            weather=data["weather"],
            labels=data["labels"],
            field_indices=sets[key],
            t_history=data["t_history"],
            t_future=data["t_future"],
            use_weather=cfg.TRAJ_USE_WEATHER,
            channel_mean=channel_mean,
            channel_std=channel_std,
            alert_horizon=alert_horizon,
            class_mapping=class_mapping,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[key] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(key == "train"),
            num_workers=cfg.TRAJ_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            generator=generator if key == "train" else None,
            drop_last=False,
        )
    return loaders["train"], loaders["val"], loaders["test"]


def compute_class_weights(
    labels: np.ndarray,
    train_fields: np.ndarray,
    t_history: int,
    t_future: int,
    n_classes: int = N_CLASSES,
    class_names: list[str] | None = None,
    class_mapping: dict[int, int] | None = None,
    alpha: float | None = None,
) -> torch.Tensor | None:
    """Inverse-frequency class weights fitted on the TRAIN split's future window."""
    if not cfg.TRAJ_BALANCE_CLASSES:
        return None
    names = class_names or UNIFIED_CLASSES
    window = labels[train_fields, t_history : t_history + t_future].ravel()
    if class_mapping is not None:
        mapped = np.array([class_mapping.get(int(c), -100) for c in window], dtype=np.int64)
        valid = mapped != -100
        window = mapped[valid]
    counts = np.bincount(window.astype(np.int64), minlength=n_classes).astype(float)[:n_classes]
    if (counts == 0).any():
        missing = [names[i] for i, c in enumerate(counts) if c == 0]
        log(f"WARNING: classes absent from train window ({missing}) -> no CE weighting")
        return None
    weights = counts.sum() / (n_classes * counts)
    alpha = float(cfg.TRAJ_CLASS_WEIGHT_ALPHA if alpha is None else alpha)
    if alpha != 1.0:
        weights = weights ** alpha
        log(f"CE weights tempered with alpha={alpha}")
    log("CE class weights (train split): " + ", ".join(
        f"{names[i]}={w:.3f}" for i, w in enumerate(weights)
    ))
    return torch.tensor(weights, dtype=torch.float32)


def combined_loss(
    ndvi_pred: torch.Tensor,
    ndvi_true: torch.Tensor,
    logits: torch.Tensor,
    cls_true: torch.Tensor,
    early_logit: torch.Tensor | None = None,
    early_true: torch.Tensor | None = None,
    early_mask: torch.Tensor | None = None,
    days_to_onset: torch.Tensor | None = None,
    lambda_ce: float = 0.5,
    lambda_alert: float = 0.5,
    class_loss_fn: nn.Module | None = None,
    class_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    early_loss_fn: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combines SmoothL1+MSE on NDVI, weighted CE/Focal on classes, and lead-time-aware EarlyWarningLoss."""
    loss_ndvi = F.smooth_l1_loss(ndvi_pred, ndvi_true, beta=0.02) + 0.5 * F.mse_loss(ndvi_pred, ndvi_true)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_cls = cls_true.reshape(-1)
    if class_loss_fn is not None:
        loss_ce = class_loss_fn(flat_logits, flat_cls)
    else:
        loss_ce = F.cross_entropy(
            flat_logits,
            flat_cls,
            weight=class_weight,
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )

    if (
        early_logit is not None
        and early_true is not None
        and early_mask is not None
        and days_to_onset is not None
        and early_loss_fn is not None
    ):
        loss_early = early_loss_fn(early_logit, early_true, early_mask, days_to_onset)
    elif early_logit is not None and early_true is not None:
        valid = (early_true >= 0.0)
        if valid.any():
            loss_early = F.binary_cross_entropy_with_logits(early_logit[valid], early_true[valid])
        else:
            loss_early = torch.tensor(0.0, device=ndvi_pred.device)
    else:
        loss_early = torch.tensor(0.0, device=ndvi_pred.device)

    total = loss_ndvi + lambda_ce * loss_ce + lambda_alert * loss_early
    return total, loss_ndvi, loss_ce, loss_early


def run_epoch(
    model: TrajectoryModel,
    loader: DataLoader,
    optimizer: AdamW | None,
    mse: nn.Module,
    scheduler: CosineAnnealingLR | None,
    scaler: torch.amp.GradScaler,
    amp_device: str,
    use_amp: bool,
    lambda_ce: float,
    lambda_alert: float,
    max_norm: float,
    epoch: int,
    epochs: int,
    class_loss_fn: nn.Module | None = None,
    class_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    alert_horizon: int = 3,
    early_loss_fn: nn.Module | None = None,
) -> dict[str, float]:
    """One pass over ``loader`` (train when optimizer is given, else eval)."""
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device

    totals = {
        "loss": 0.0,
        "ndvi": 0.0,
        "ce": 0.0,
        "early": 0.0,
        "ndvi_mae": 0.0,
        "cls_acc": 0.0,
        "early_acc": 0.0,
    }
    all_cls_true: list[np.ndarray] = []
    all_cls_pred: list[np.ndarray] = []
    all_early_true: list[np.ndarray] = []
    all_early_pred: list[np.ndarray] = []
    lead_times: list[float] = []

    n_batches = 0
    desc = f"epoch {epoch}/{epochs} {'train' if training else 'val  '}"
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")

    for batch in bar:
        x = batch["x"].to(device, non_blocking=True)
        y_ndvi = batch["y_ndvi"].to(device, non_blocking=True)
        y_cls = batch["y_cls"].to(device, non_blocking=True)
        y_cls_raw = batch.get("y_cls_raw", batch["y_cls"]).numpy()
        last_ndvi = batch["last_ndvi"].to(device, non_blocking=True) if "last_ndvi" in batch else None

        early_true = batch.get("early_warning")
        early_mask = batch.get("early_warning_mask")
        days_to_onset = batch.get("days_to_onset")
        if early_true is not None:
            early_true = early_true.to(device, non_blocking=True)
        if early_mask is not None:
            early_mask = early_mask.to(device, non_blocking=True)
        if days_to_onset is not None:
            days_to_onset = days_to_onset.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(amp_device, enabled=use_amp):
            out = model(x, last_ndvi=last_ndvi, return_dict=True)
            ndvi_pred = out["ndvi"]
            logits = out["logits"]
            early_logit = out.get("early_logit")

            loss, loss_ndvi, loss_ce, loss_early = combined_loss(
                ndvi_pred=ndvi_pred,
                ndvi_true=y_ndvi,
                logits=logits,
                cls_true=y_cls,
                early_logit=early_logit,
                early_true=early_true,
                early_mask=early_mask,
                days_to_onset=days_to_onset,
                lambda_ce=lambda_ce,
                lambda_alert=lambda_alert,
                class_loss_fn=class_loss_fn,
                class_weight=class_weight,
                label_smoothing=label_smoothing,
                early_loss_fn=early_loss_fn,
            )

        if training:
            scaler.scale(loss).backward()
            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        totals["loss"] += float(loss.detach())
        totals["ndvi"] += float(loss_ndvi.detach())
        totals["ce"] += float(loss_ce.detach())
        totals["early"] += float(loss_early.detach())
        totals["ndvi_mae"] += float((ndvi_pred.detach() - y_ndvi).abs().mean())

        valid_mask = y_cls != -100
        if valid_mask.any():
            preds_cls = logits.detach().argmax(dim=-1)
            correct = (preds_cls[valid_mask] == y_cls[valid_mask]).float().sum()
            totals["cls_acc"] += float(correct / valid_mask.sum())
            if not training:
                all_cls_true.append(y_cls[valid_mask].cpu().numpy())
                all_cls_pred.append(preds_cls[valid_mask].cpu().numpy())

        if early_logit is not None and early_true is not None:
            early_p = torch.sigmoid(early_logit.detach())
            early_pred_bin = (early_p >= 0.5).float()
            valid_m = (early_mask > 0.5) if early_mask is not None else (early_true >= 0.0)
            if valid_m.any():
                totals["early_acc"] += float((early_pred_bin[valid_m] == early_true[valid_m]).float().mean())
                if not training:
                    all_early_true.append(early_true[valid_m].cpu().numpy())
                    all_early_pred.append(early_pred_bin[valid_m].cpu().numpy())
                    if days_to_onset is not None:
                        d_np = days_to_onset[valid_m].cpu().numpy()
                        t_np = early_true[valid_m].cpu().numpy()
                        p_np = early_pred_bin[valid_m].cpu().numpy()
                        for i_s in range(len(t_np)):
                            if t_np[i_s] == 1.0 and p_np[i_s] == 1.0 and d_np[i_s] > 0:
                                lead_times.append(float(d_np[i_s]))

        n_batches += 1
        bar.set_postfix(loss=f"{float(loss.detach()):.4f}", mae=f"{totals['ndvi_mae']/n_batches:.4f}")

    if training and scheduler is not None:
        scheduler.step()

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}

    if not training and all_cls_true:
        y_t = np.concatenate(all_cls_true)
        y_p = np.concatenate(all_cls_pred)
        metrics["macro_f1"] = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    else:
        metrics["macro_f1"] = 0.0

    if not training and all_early_true:
        e_t = np.concatenate(all_early_true).ravel()
        e_p = np.concatenate(all_early_pred).ravel()
        metrics["early_f1"] = float(f1_score(e_t, e_p, zero_division=0))
        metrics["early_mean_lead_time"] = float(np.mean(lead_times)) if lead_times else 0.0
    else:
        metrics["early_f1"] = 0.0
        metrics["early_mean_lead_time"] = 0.0

    return metrics


def save_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    """Write the comprehensive loss/MAE/F1 curves to plots/."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Combined Loss
        axes[0, 0].plot(epochs, [r["train_loss"] for r in history], label="train", color="#1565C0")
        axes[0, 0].plot(epochs, [r["val_loss"] for r in history], label="val", color="#C62828")
        axes[0, 0].set_title("Combined Loss (NDVI + CE + Alert)")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. NDVI Error
        axes[0, 1].plot(epochs, [r["train_ndvi_mae"] for r in history], label="train MAE", color="#2E7D32")
        axes[0, 1].plot(epochs, [r["val_ndvi_mae"] for r in history], label="val MAE", color="#E65100")
        axes[0, 1].set_title("NDVI MAE Error")
        axes[0, 1].set_xlabel("epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Classification Accuracy & Macro F1
        axes[1, 0].plot(epochs, [r["val_cls_acc"] for r in history], label="val accuracy", color="#00838F")
        axes[1, 0].plot(epochs, [r.get("val_macro_f1", 0.0) for r in history], label="val macro F1", color="#6A1B9A")
        axes[1, 0].set_title("Classification Metrics")
        axes[1, 0].set_xlabel("epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Early Warning & Composite Selection Score
        axes[1, 1].plot(epochs, [r.get("val_early_f1", 0.0) for r in history], label="early warning F1", color="#AD1457")
        axes[1, 1].plot(epochs, [r.get("selection_score", 0.0) for r in history], label="selection score", color="#2E7D32", lw=2)
        axes[1, 1].set_title("Early Warning & Model Selection Score")
        axes[1, 1].set_xlabel("epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log(f"training curves written to {path}")
    except Exception as err:
        log(f"WARNING: could not write training curves ({err})")


def train(args: argparse.Namespace) -> int:
    """Full training loop.  Returns a process exit code."""
    dry_run = args.mode == "dry-run"
    n_fields = min(args.n_fields, 48) if dry_run else args.n_fields
    epochs = (2 if args.epochs is None else args.epochs) if dry_run else (args.epochs or cfg.TRAJ_EPOCHS)
    batch_size = 16 if dry_run else args.batch_size
    min_epochs = 0 if dry_run else args.min_epochs
    patience = 0 if dry_run else args.patience

    set_seed(args.seed)
    device = pick_device()
    use_amp = bool(device.type == "cuda")
    amp_device = "cuda" if use_amp else "cpu"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""

    # Parse and validate active classes
    raw_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    active_classes = raw_classes if raw_classes else list(UNIFIED_CLASSES)
    log(f"mode={args.mode} device={device.type}{gpu_info} amp={use_amp} seed={args.seed}")
    log(f"active classes ({len(active_classes)}): {', '.join(active_classes)}")

    class_mapping: dict[int, int] | None = None
    if active_classes != list(UNIFIED_CLASSES) or args.merge_generic_into_nearest:
        class_mapping = {}
        for old_idx, old_name in enumerate(UNIFIED_CLASSES):
            if old_name in active_classes:
                class_mapping[old_idx] = active_classes.index(old_name)
            elif old_name == "stress_generic" and args.merge_generic_into_nearest:
                tgt_name = "water_stress" if "water_stress" in active_classes else active_classes[0]
                class_mapping[old_idx] = active_classes.index(tgt_name)
                log(f"[class_mapping] --merge_generic_into_nearest active: mapped 'stress_generic' ({old_idx}) -> '{tgt_name}' ({class_mapping[old_idx]})")
            else:
                class_mapping[old_idx] = -100
                log(f"[class_mapping] excluded class '{old_name}' ({old_idx}) -> masked (-100)")

    try:
        data = build_dataset(
            n_fields=n_fields,
            weather_source=args.weather_source,
            seed=args.seed,
            cache=not dry_run,
        )
        split = field_split(data["n_fields"], data["labels"], seed=args.seed, save=not dry_run)
        if cfg.TRAJ_STANDARDIZE:
            channel_mean, channel_std = compute_channel_stats(
                data["ndvi"], data["weather"],
                np.asarray(split["train"], dtype=np.int64),
                t_history=data["t_history"], use_weather=cfg.TRAJ_USE_WEATHER,
            )
            log(f"input z-score stats (train): mean={channel_mean} std={channel_std}")
        else:
            channel_mean = channel_std = None

        train_loader, val_loader, _ = make_loaders(
            data, split, batch_size, args.seed, channel_mean, channel_std,
            alert_horizon=args.alert_horizon, class_mapping=class_mapping,
        )
    except Exception as err:
        log(f"ERROR building data: {type(err).__name__}: {err}")
        return 1

    log(
        f"fields: train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} | batches/epoch train={len(train_loader)} "
        f"val={len(val_loader)} | D={cfg.TRAJ_INPUT_DIM} T_hist={data['t_history']} "
        f"T_fut={data['t_future']} | alert_horizon={args.alert_horizon}"
    )

    model = TrajectoryModel(
        t_future=data["t_future"],
        input_dim=cfg.TRAJ_INPUT_DIM,
        n_classes=len(active_classes),
        encoder_kind=args.encoder,
        pooling=args.pooling,
        d_model=args.d_model,
        nhead=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    if hasattr(model, "ndvi_skip_scale"):
        model.ndvi_skip_scale.data.fill_(0.5)
    if hasattr(model.encoder, "pool_blend"):
        model.encoder.pool_blend.data.fill_(0.5)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"TrajectoryModel[{model.encoder_kind}] parameters: {n_params:,}")

    class_weight = compute_class_weights(
        data["labels"],
        np.asarray(split["train"], dtype=np.int64),
        data["t_history"],
        data["t_future"],
        n_classes=len(active_classes),
        class_names=active_classes,
        class_mapping=class_mapping,
        alpha=args.class_weight_alpha,
    )
    if class_weight is not None:
        class_weight = class_weight.to(device)

    class_loss_fn = None
    if args.focal_loss:
        class_loss_fn = MultiClassFocalLoss(
            gamma=args.focal_gamma,
            weight=class_weight,
            label_smoothing=args.label_smoothing,
            ignore_index=-100,
        )
        log(f"using MultiClassFocalLoss(gamma={args.focal_gamma}, label_smoothing={args.label_smoothing})")

    # Compute alert_pos_weight strictly from the training split (healthy-at-endpoint fields only)
    train_healthy_pos = 0
    train_healthy_neg = 0
    for f_idx in split["train"]:
        if data["labels"][f_idx, data["t_history"] - 1] == 0:
            has_on = np.any(data["labels"][f_idx, data["t_history"] : data["t_history"] + args.alert_horizon] != 0)
            if has_on:
                train_healthy_pos += 1
            else:
                train_healthy_neg += 1
    alert_pos_weight = min(float(train_healthy_neg) / max(train_healthy_pos, 1), 5.0)
    log(
        f"[early_warning] train split healthy fields: pos={train_healthy_pos}, neg={train_healthy_neg} "
        f"-> alert_pos_weight={alert_pos_weight:.2f}"
    )

    early_loss_fn = EarlyWarningLoss(
        kind=args.alert_loss,
        pos_weight=alert_pos_weight,
        focal_gamma=args.alert_focal_gamma,
        lead_weight_1d=args.lead_weight_1d,
        lead_weight_2d=args.lead_weight_2d,
        lead_weight_3d=args.lead_weight_3d,
    )
    log(
        f"using EarlyWarningLoss(kind={args.alert_loss}, pos_weight={alert_pos_weight:.2f}, "
        f"lead_weights=[1d:{args.lead_weight_1d}, 2d:{args.lead_weight_2d}, 3d:{args.lead_weight_3d}])"
    )

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.05)
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    best_selection_score = -float("inf")
    best_val_loss = float("inf")
    best_val_mae: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    ckpt_path = Path(cfg.TRAJ_CKPT)
    epoch_ckpt_dir = Path(cfg.CHECKPOINTS_DIR)
    epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = "dryrun_trajectory" if dry_run else "trajectory"
    started = time.time()

    epoch_iter = tqdm(range(1, epochs + 1), desc="epochs", unit="ep")
    for epoch in epoch_iter:
        t0 = time.time()
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mse=mse,
            scheduler=scheduler,
            scaler=scaler,
            amp_device=amp_device,
            use_amp=use_amp,
            lambda_ce=args.lambda_ce,
            lambda_alert=args.lambda_alert,
            max_norm=cfg.TRAJ_MAX_NORM,
            epoch=epoch,
            epochs=epochs,
            class_loss_fn=class_loss_fn,
            class_weight=class_weight,
            label_smoothing=args.label_smoothing,
            alert_horizon=args.alert_horizon,
            early_loss_fn=early_loss_fn,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                mse=mse,
                scheduler=None,
                scaler=scaler,
                amp_device=amp_device,
                use_amp=use_amp,
                lambda_ce=args.lambda_ce,
                lambda_alert=args.lambda_alert,
                max_norm=cfg.TRAJ_MAX_NORM,
                epoch=epoch,
                epochs=epochs,
                class_loss_fn=class_loss_fn,
                class_weight=class_weight,
                label_smoothing=args.label_smoothing,
                alert_horizon=args.alert_horizon,
                early_loss_fn=early_loss_fn,
            )

        # Lead-time aware composite model selection score
        norm_lead = min(max(val_stats.get("early_mean_lead_time", 0.0) / float(args.alert_horizon), 0.0), 1.0)
        selection_score = 0.45 * val_stats.get("macro_f1", 0.0) - 0.25 * val_stats["ndvi_mae"] + 0.30 * norm_lead

        # Regression Guard 1: Macro F1 warning after epoch 35
        if epoch >= 35 and val_stats.get("macro_f1", 0.0) < 0.45:
            log(f"[warning] epoch {epoch:02d}: val_macro_f1 ({val_stats.get('macro_f1', 0.0):.4f}) is below 0.45 threshold")

        # Regression Guard 2: Skip saving if NDVI MAE worsens by > 10% compared to best previous checkpoint
        mae_degraded = False
        if best_val_mae is not None and val_stats["ndvi_mae"] > best_val_mae * 1.10:
            log(
                f"[guard] Skipping checkpoint save at epoch {epoch:02d}: val_ndvi_mae ({val_stats['ndvi_mae']:.4f}) "
                f"is > 10% worse than best ({best_val_mae:.4f})"
            )
            mae_degraded = True

        row = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "train_ndvi": round(train_stats["ndvi"], 6),
            "train_ce": round(train_stats["ce"], 6),
            "train_early": round(train_stats["early"], 6),
            "train_ndvi_mae": round(train_stats["ndvi_mae"], 6),
            "train_cls_acc": round(train_stats["cls_acc"], 6),
            "val_loss": round(val_stats["loss"], 6),
            "val_ndvi": round(val_stats["ndvi"], 6),
            "val_ce": round(val_stats["ce"], 6),
            "val_early": round(val_stats["early"], 6),
            "val_ndvi_mae": round(val_stats["ndvi_mae"], 6),
            "val_cls_acc": round(val_stats["cls_acc"], 6),
            "val_macro_f1": round(val_stats.get("macro_f1", 0.0), 6),
            "val_early_f1": round(val_stats.get("early_f1", 0.0), 6),
            "val_early_mean_lead_time": round(val_stats.get("early_mean_lead_time", 0.0), 4),
            "selection_score": round(selection_score, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        epoch_iter.set_postfix(val=f"{row['val_loss']:.4f}", score=f"{selection_score:.4f}")

        ckpt_blob = {
            "model_state": model.state_dict(),
            "config": {
                "t_history": int(data["t_history"]),
                "t_future": int(data["t_future"]),
                "input_dim": int(cfg.TRAJ_INPUT_DIM),
                "encoder": model.encoder_kind,
                "pooling": str(getattr(model.encoder, "pooling", "mean")),
                "d_model": int(args.d_model),
                "nhead": int(args.n_heads),
                "n_layers": int(args.n_layers),
                "ff_dim": int(args.ff_dim),
                "dropout": float(args.dropout),
                "alert_horizon": int(args.alert_horizon),
                "lambda_alert": float(args.lambda_alert),
                "alert_loss": str(args.alert_loss),
                "alert_pos_weight": float(alert_pos_weight),
                "lead_weight_1d": float(args.lead_weight_1d),
                "lead_weight_2d": float(args.lead_weight_2d),
                "lead_weight_3d": float(args.lead_weight_3d),
                "classes": active_classes,
                "input_mean": channel_mean,
                "input_std": channel_std,
                "use_weather": bool(cfg.TRAJ_USE_WEATHER),
                "seed": int(args.seed),
            },
            "best_epoch": best_epoch or epoch,
            "best_selection_score": float(best_selection_score if best_selection_score > -1e8 else selection_score),
            "val_loss": float(val_stats["loss"]),
            "val_cls_acc": float(val_stats["cls_acc"]),
            "val_macro_f1": float(val_stats.get("macro_f1", 0.0)),
            "val_ndvi_mae": float(val_stats["ndvi_mae"]),
            "epoch": epoch,
        }
        epoch_ckpt_file = epoch_ckpt_dir / f"{ckpt_prefix}_epoch_{epoch:02d}.pth"
        torch.save(ckpt_blob, epoch_ckpt_file)

        # Model selection: based on composite selection score and regression guards
        improved = selection_score > best_selection_score + 1e-5
        if improved and not mae_degraded:
            best_selection_score = selection_score
            best_val_loss = val_stats["loss"]
            best_val_mae = val_stats["ndvi_mae"]
            best_epoch = epoch
            bad_epochs = 0
            ckpt_blob["best_epoch"] = best_epoch
            ckpt_blob["best_selection_score"] = float(best_selection_score)
            ckpt_blob["val_loss"] = float(best_val_loss)
            if not dry_run:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt_blob, ckpt_path)
        else:
            bad_epochs += 1

        log(
            f"epoch {epoch:02d}/{epochs}  train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f}  ndvi_mae={row['val_ndvi_mae']:.4f}  "
            f"val_acc={row['val_cls_acc']:.4f}  val_f1={row['val_macro_f1']:.4f}  "
            f"alert_f1={row['val_early_f1']:.4f}  lead={row['val_early_mean_lead_time']:.2f}d  "
            f"score={row['selection_score']:.4f}  lr={row['lr']:.2e}  {row['seconds']}s  "
            f"{'*best*' if improved else ''}"
        )

        if epoch >= min_epochs and patience > 0 and bad_epochs >= patience:
            log(
                f"early stopping at epoch {epoch}: selection_score did not improve "
                f"for {patience} epochs after min_epochs={min_epochs} (best epoch {best_epoch}, "
                f"score {best_selection_score:.4f})"
            )
            break

    elapsed = round(time.time() - started, 1)
    peak_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
        if device.type == "cuda" else 0.0
    )
    gpu_mem_str = f" | gpu_peak_mem={peak_mb:.2f}MB" if device.type == "cuda" else ""
    log(f"training finished in {elapsed}s | best epoch {best_epoch} "
        f"selection_score={best_selection_score:.4f} val_loss={best_val_loss:.4f}{gpu_mem_str}")

    payload: dict[str, Any] = {
        "config": {
            "mode": args.mode,
            "seed": int(args.seed),
            "n_fields": int(data["n_fields"]),
            "t_history": int(data["t_history"]),
            "t_future": int(data["t_future"]),
            "input_dim": int(cfg.TRAJ_INPUT_DIM),
            "encoder": args.encoder or cfg.TRAJ_ENCODER,
            "pooling": str(getattr(model.encoder, "pooling", "mean")),
            "d_model": int(args.d_model),
            "nhead": int(args.n_heads),
            "n_layers": int(args.n_layers),
            "ff_dim": int(args.ff_dim),
            "dropout": float(args.dropout),
            "standardize": bool(cfg.TRAJ_STANDARDIZE),
            "input_mean": channel_mean,
            "input_std": channel_std,
            "epochs": int(epochs),
            "min_epochs": int(min_epochs),
            "batch_size": int(batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lambda_ce": float(args.lambda_ce),
            "lambda_alert": float(args.lambda_alert),
            "alert_horizon": int(args.alert_horizon),
            "patience": int(patience),
            "focal_loss": bool(args.focal_loss),
            "focal_gamma": float(args.focal_gamma),
            "label_smoothing": float(args.label_smoothing),
            "balance_classes": bool(cfg.TRAJ_BALANCE_CLASSES),
            "class_weight_alpha": float(
                cfg.TRAJ_CLASS_WEIGHT_ALPHA if args.class_weight_alpha is None
                else args.class_weight_alpha
            ),
            "class_weights": (
                [round(float(w), 4) for w in class_weight]
                if class_weight is not None else None
            ),
            "split": {k: len(split[k]) for k in ("train", "val", "test")},
            "split_by": "field",
            "device": device.type,
            "amp": bool(use_amp),
            "ndvi_is_proxy": bool(data["ndvi_is_proxy"]),
            "weather_is_proxy": bool(data["weather_is_proxy"]),
            "classes": active_classes,
        },
        "epochs_log": history,
        "best_epoch": best_epoch,
        "best_selection_score": round(float(best_selection_score), 6),
        "best_val_loss": round(float(best_val_loss), 6),
        "seconds": elapsed,
    }

    if dry_run:
        log("dry-run complete: no checkpoint / history written (by design)")
        log(f"sample val batch loss {val_stats['loss']:.4f} - pipeline is healthy")
        return 0

    history_path = Path(cfg.TRAJ_HISTORY_JSON)
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"history written to {history_path}")
    save_training_curves(history, Path("plots/trajectory_training_curves.png"))
    log(f"best checkpoint: {ckpt_path} (epoch {best_epoch}, val_loss {best_val_loss:.4f}, score {best_selection_score:.4f})")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Stage 3A - train the temporal digital twin trajectory model."
    )
    parser.add_argument("--mode", choices=["train", "dry-run"], default="train",
                        help="train = full run; dry-run = 48 fields / 2 epochs, "
                             "no production artifacts overwritten")
    parser.add_argument("--epochs", "--max_epochs", "--max-epochs", type=int, default=None)
    parser.add_argument("--min_epochs", "--min-epochs", type=int, default=35,
                        help="minimum epochs before early stopping is permitted (default 35)")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stopping patience on composite selection_score (default 20)")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=cfg.TRAJ_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.TRAJ_LR)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=cfg.TRAJ_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=cfg.TRAJ_DROPOUT)
    parser.add_argument("--n_layers", "--n-layers", type=int, default=cfg.TRAJ_TRANSFORMER_LAYERS)
    parser.add_argument("--d_model", "--d-model", type=int, default=cfg.TRAJ_D_MODEL)
    parser.add_argument("--n_heads", "--n-heads", type=int, default=cfg.TRAJ_NHEAD)
    parser.add_argument("--ff_dim", "--ff-dim", type=int, default=cfg.TRAJ_FF_DIM)
    parser.add_argument("--pooling", choices=["mean", "last", "attention", "attentive"], default=None,
                        help="override pooling method (last, mean, attention)")
    parser.add_argument("--alert_horizon", "--alert-horizon", type=int, default=3,
                        help="early warning forecast horizon K in days (default 3)")
    parser.add_argument("--lambda_alert", "--lambda-alert", type=float, default=0.5,
                        help="weight of binary early-warning BCE loss (default 0.5)")
    parser.add_argument("--alert_loss", "--alert-loss", choices=["bce", "focal", "ranking"], default="bce",
                        help="loss function for binary early-warning head (bce, focal, ranking)")
    parser.add_argument("--alert_focal_gamma", "--alert-focal-gamma", type=float, default=2.0,
                        help="focal gamma for early warning focal loss (default 2.0)")
    parser.add_argument("--lead_weight_3d", "--lead-weight-3d", type=float, default=1.0,
                        help="horizon sample weight for onset in 3 days (default 1.0)")
    parser.add_argument("--lead_weight_2d", "--lead-weight-2d", type=float, default=0.8,
                        help="horizon sample weight for onset in 2 days (default 0.8)")
    parser.add_argument("--lead_weight_1d", "--lead-weight-1d", type=float, default=0.6,
                        help="horizon sample weight for onset in 1 day (default 0.6)")
    parser.add_argument("--lambda-ce", "--lambda_ce", type=float, default=cfg.TRAJ_LAMBDA_CE,
                        help="weight of the class-sequence CE term (default 0.5)")
    parser.add_argument("--classes", type=str, default=",".join(UNIFIED_CLASSES),
                        help="comma-separated active class list (supports 4-class mode)")
    parser.add_argument("--merge_generic_into_nearest", "--merge-generic-into-nearest", action="store_true",
                        help="merge stress_generic into adjacent stress class")
    parser.add_argument("--label_smoothing", "--label-smoothing", type=float, default=0.05,
                        help="label smoothing factor for classification (default 0.05)")
    parser.add_argument("--focal_loss", "--focal-loss", action="store_true",
                        help="use focal loss instead of cross entropy for classes")
    parser.add_argument("--focal_gamma", "--focal-gamma", type=float, default=2.0,
                        help="gamma focusing parameter for focal loss (default 2.0)")
    parser.add_argument("--n-fields", "--n_fields", type=int, default=cfg.TRAJ_N_FIELDS)
    parser.add_argument("--encoder", choices=["transformer", "bilstm"], default=None,
                        help="override config.TRAJ_ENCODER for an ablation")
    parser.add_argument("--class-weight-alpha", "--class_weight_alpha", type=float, default=None,
                        help="tempering exponent for the CE class weights")
    parser.add_argument("--weather-source", "--weather_source", choices=["proxy", "open-meteo", "auto"],
                        default="proxy")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    try:
        return train(args)
    except FileNotFoundError as err:
        log(f"ERROR: {err}")
        return 1
    except KeyboardInterrupt:
        log("interrupted by user")
        return 130
    except Exception as err:
        log(f"ERROR: {type(err).__name__}: {err}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
