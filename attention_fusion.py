"""Stage 1B - unified multi-modal crop-stress model with attention fusion.

One network ingests three modalities at once and reports both a 6-class
stress prediction and the attention weights that explain which modality
drove it.  It consumes the Stage-1A data contract instead of re-implementing
it:

* taxonomy and label mappings -- ``stage1_config`` (single source of truth;
  note the Stage-1 order swaps water/nutrient versus the Stage-0 rule-fusion
  vector, so every cross-stage conversion here goes through class *names*,
  never raw indices);
* ingestion, weak pairing, PROXY NDVI, splits -- ``dataset.py``
  (rationale and limitations documented in PAIRING_STRATEGY.md).

Architecture
------------
1. RGB leaf image  (224x224) -> MobileNetV2 encoder -> 1280-d
2. Tabular sensor row        -> MLP encoder         -> 128-d
3. NDVI series (20 steps)    -> 1D-CNN encoder      -> 128-d   [PROXY input]

The three embeddings concatenate to 1536-d, are sliced back into three
modality tokens, projected to a shared 256-d space and passed through
multi-head self-attention (4 heads) with a learned query token.  The
attention-weighted pooled vector (256-d) feeds:

* classification head: Linear(256, 6) -> logits (softmax at reporting time;
  CrossEntropyLoss consumes logits - the Stage-0 convention);
* attention head: per-modality weights, exposed by ``forward()``.

Honest caveats (full text in PAIRING_STRATEGY.md and report_stage1.txt)
-----------------------------------------------------------------------
* Weak pairing: rows and photos are never jointly observed; images are a
  same-binary-class random draw, split-restricted so no photo crosses splits.
* PROXY NDVI: no real paired Sentinel-2 series exists, so the third
  modality is class-conditional synthetic data.  It can recover the label by
  construction - keep the PROXY flag attached to every number derived from it.
* pest_stress is reserved and untrained (0 samples in every split).
* disease_stress labels come from image-sourced samples (CHES has no disease
  label), so the image branch and the rule-fusion baseline are strong there
  while XGBoost cannot predict that class at all.

Usage
-----
    python attention_fusion.py                        # full run (50 epochs)
    python attention_fusion.py --epochs 2 --limit_batches 20   # smoke test
    python attention_fusion.py --eval_only            # reuse a checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import dataset as data
import stage1_config as cfg
import tabular_xgb as tx
from fusion_inference import (
    UNIFIED_CLASSES as STAGE0_CLASSES,
    image_to_unified,
    load_image_bundle,
    load_tabular_bundle,
    predict_image_probs,
    predict_tabular_probs,
    rule_fusion,
    tabular_to_unified,
)
from mobilenet_cnn import (
    IMG_SIZE,
    build_transforms,
    plot_confusion_matrix,
    save_json,
)
from stage1_config import N_CLASSES, UNIFIED_CLASSES, UNIFIED_INDEX

# ---------------------------------------------------------------------------
# Configuration (taxonomy lives in stage1_config; the rest is training-only)
# ---------------------------------------------------------------------------
SEED = cfg.SEED
ARTIFACTS_DIR = tx.ARTIFACTS_DIR
PLOTS_DIR = Path("plots")
EXPLAIN_DIR = PLOTS_DIR / "explainability"
CKPT_PATH = ARTIFACTS_DIR / "attention_fusion.pt"
METRICS_PATH = ARTIFACTS_DIR / "stage1_metrics.json"
ATTENTION_PATH = ARTIFACTS_DIR / "stage1_attention.npz"
REPORT_PATH = Path("report_stage1.txt")

NDVI_STEPS = cfg.NDVI_STEPS
D_MODEL = 256                             # pooled representation width
N_HEADS = 4                               # 2-4 allowed by the spec
MODALITY_DIMS = (1280, 128, 128)          # image, tabular, ndvi -> 1536 total
MODALITY_NAMES = ("image", "tabular", "ndvi")

DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_STABILITY_TOL = 0.01              # +-1% metric stability tolerance
DEFAULT_STABILITY_PATIENCE = 2            # consecutive stable steps within +-1%
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"


def log(message: str) -> None:
    """Print a timestamped progress line (same style as the Stage-0 scripts)."""
    print(f"[stage1b {time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stage0_index_to_stage1(stage0_vector: np.ndarray) -> int:
    """Convert a Stage-0 unified vector (rule-fusion order) to a Stage-1 index.

    Both taxonomies hold the same class *names* in different positions
    (water/nutrient swap), so the conversion is name-based and never index
    arithmetic on assumed orders.
    """
    name = STAGE0_CLASSES[int(np.argmax(np.asarray(stage0_vector, dtype=float)))]
    return UNIFIED_INDEX[name]


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """MobileNetV2 trunk -> 1280-d embedding, last feature map kept for Grad-CAM."""

    def __init__(self, trainable: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features
        stage0_ckpt = ARTIFACTS_DIR / "mobilenet_paddy.pth"
        if trainable and stage0_ckpt.exists():
            try:
                blob = torch.load(stage0_ckpt, map_location="cpu", weights_only=False)
                state = blob.get("state_dict", {})
                feat_state = {
                    k[len("features."):]: v
                    for k, v in state.items()
                    if k.startswith("features.")
                }
                if feat_state:
                    self.features.load_state_dict(feat_state, strict=False)
            except Exception:  # noqa: BLE001
                pass
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280
        self.activations: torch.Tensor | None = None
        self.features[-1].register_forward_hook(self._capture)
        if not trainable:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            # Freeze early stem blocks to speed up fine-tuning and preserve low-level filters;
            # keep top 4 blocks trainable so Grad-CAM and disease adaptation remain active.
            for block in list(self.features.children())[:-4]:
                for param in block.parameters():
                    param.requires_grad = False

    def _capture(self, _module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        self.activations = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class TabularEncoder(nn.Module):
    """MLP encoder for the CHES sensor row -> 128-d embedding."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(128)
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class NdviEncoder(nn.Module):
    """1D-CNN over the (PROXY) NDVI curve -> 128-d embedding."""

    def __init__(self, steps: int = NDVI_STEPS) -> None:
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 128)
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.unsqueeze(1)).squeeze(-1))


class AttentionFusion(nn.Module):
    """Three modality tokens -> multi-head attention -> pooled 256-d -> heads."""

    def __init__(self, tabular_dim: int = cfg.TABULAR_DIM) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder(in_dim=tabular_dim)
        self.ndvi_encoder = NdviEncoder()
        dims = (
            self.image_encoder.out_dim,
            self.tabular_encoder.out_dim,
            self.ndvi_encoder.out_dim,
        )
        assert dims == MODALITY_DIMS, dims

        self.token_proj = nn.ModuleList([nn.Linear(d, D_MODEL) for d in dims])
        self.query = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.normal_(self.query, std=0.02)

        # Cross-modality contextual interaction among tokens
        self.cross_attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=0.1, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(D_MODEL)
        self.cross_scale = nn.Parameter(torch.zeros(1))

        # Executive decision query pooling
        self.attention = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=0.1, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(D_MODEL, N_CLASSES),
        )

    def encode_from_image_emb(
        self, img_emb: torch.Tensor, tabular: torch.Tensor, ndvi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, per-modality attention weights (B, 3)) from image embeddings."""
        tab_emb = self.tabular_encoder(tabular)
        has_ndvi = (ndvi.abs().sum(dim=-1, keepdim=True) > 1e-5).to(img_emb.dtype)
        ndvi_emb = self.ndvi_encoder(ndvi) * has_ndvi
        if self.training and not torch.jit.is_tracing():
            # Modality dropout during training prevents single-modality collapse onto
            # the synthetic PROXY NDVI curve and forces image + tabular branches to
            # remain predictive both jointly and when NDVI is absent/neutral at inference.
            bsz = img_emb.size(0)
            r = torch.rand(bsz, device=img_emb.device)
            drop_ndvi = (r < 0.25).unsqueeze(1).to(ndvi_emb.dtype)
            drop_tab = ((r >= 0.25) & (r < 0.35)).unsqueeze(1).to(tab_emb.dtype)
            ndvi_emb = ndvi_emb * (1.0 - drop_ndvi)
            tab_emb = tab_emb * (1.0 - drop_tab)

        merged = torch.cat([img_emb, tab_emb, ndvi_emb], dim=-1)  # (B, 1536)
        tokens = torch.split(merged, MODALITY_DIMS, dim=-1)
        tokens = torch.stack(
            [proj(tok) for proj, tok in zip(self.token_proj, tokens)], dim=1
        )                                                        # (B, 3, 256)

        # 1. Cross-modality token interaction (zero-scaled at init for 100% backward compatibility)
        if hasattr(self, "cross_attn") and hasattr(self, "cross_scale"):
            c_out, _ = self.cross_attn(tokens, tokens, tokens)
            tokens = tokens + self.cross_scale * self.cross_attn_norm(c_out)

        query = self.query.expand(tokens.size(0), -1, -1)
        pooled, weights = self.attention(
            query, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        pooled = self.pool_norm(pooled.squeeze(1))               # (B, 256)
        return self.classifier(pooled), weights.squeeze(1)       # weights: (B, 3)

    def encode(
        self, images: torch.Tensor, tabular: torch.Tensor, ndvi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, per-modality attention weights (B, 3))."""
        return self.encode_from_image_emb(self.image_encoder(images), tabular, ndvi)

    def forward(
        self, images: torch.Tensor, tabular: torch.Tensor, ndvi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(images, tabular, ndvi)


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------
def inverse_frequency_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights, mean-normalised to 1.0."""
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    weights = len(labels) / (N_CLASSES * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _extract_unique_image_embeddings(
    model: AttentionFusion,
    ds: data.MultiModalStressDataset,
    device: torch.device,
    use_amp: bool,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, np.ndarray]:
    """Extract 1280-d embeddings for the unique images in ``ds`` on ``device``."""
    meta = getattr(ds, "_unique_img_meta", None)
    if meta is None:
        sample_paths = ds.table["image"][ds.indices]
        path_to_uid: dict[Any, int] = {}
        unique_paths: list[Path] = []
        sample_uid = np.empty(len(sample_paths), dtype=np.int64)
        for i, p in enumerate(sample_paths):
            uid = path_to_uid.get(p)
            if uid is None:
                uid = len(unique_paths)
                path_to_uid[p] = uid
                unique_paths.append(Path(p))
            sample_uid[i] = uid
        meta = (unique_paths, sample_uid)
        setattr(ds, "_unique_img_meta", meta)
    else:
        unique_paths, sample_uid = meta

    was_training = model.image_encoder.training
    model.image_encoder.eval()
    embs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(unique_paths), chunk_size):
            batch_paths = unique_paths[start : start + chunk_size]
            batch_imgs = torch.stack(
                [ds._eval_tf(ds._load_image(p)) for p in batch_paths], dim=0
            ).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                emb = model.image_encoder(batch_imgs)
            embs.append(emb.float())
    if was_training:
        model.image_encoder.train(True)
    return torch.cat(embs, dim=0), sample_uid


def _attention_balance_loss(attn: torch.Tensor) -> torch.Tensor:
    """Light regulariser keeping all three modality attentions active and non-degenerate."""
    prior = torch.tensor([0.28, 0.36, 0.36], device=attn.device, dtype=attn.dtype)
    mean_attn = attn.mean(dim=0)
    balance = (mean_attn - prior).pow(2).sum()
    entropy = -(attn * attn.clamp_min(1e-6).log()).sum(dim=-1).mean()
    return 0.15 * balance - 0.02 * entropy


def run_epoch(
    model: AttentionFusion,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
    max_grad_norm: float = 1.0,
    max_batches: int | None = None,
    desc: str | None = None,
) -> dict[str, Any]:
    """One pass over ``loader``; trains when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen, batches = 0.0, 0, 0, 0

    ds = loader.dataset
    if max_batches is None and isinstance(ds, data.MultiModalStressDataset) and len(ds) > 0:
        if desc is None:
            desc = "train" if training else "val  "

        # 1. End-to-end fine-tuning step on balanced raw-image mini-batches during training
        if training:
            labels_np = ds.table["label"][ds.indices]
            active_classes = [c for c in range(N_CLASSES) if np.any(labels_np == c)]
            per_cls = max(1, (32 * 40) // max(len(active_classes), 1))
            chosen: list[np.ndarray] = []
            for c in active_classes:
                cls_pos = np.where(labels_np == c)[0]
                pick = np.random.choice(cls_pos, size=min(per_cls, len(cls_pos)), replace=False)
                chosen.append(pick)
            ft_pos = np.concatenate(chosen)
            np.random.shuffle(ft_pos)
            for start in range(0, len(ft_pos), 32):
                batch_pos = ft_pos[start : start + 32]
                samples = ds.indices[batch_pos]
                rows = ds.table["row"][samples]
                imgs = torch.stack(
                    [ds._load_tensor(Path(ds.table["image"][s])) for s in samples], dim=0
                ).to(device, non_blocking=True)
                tabs = torch.from_numpy(ds.features[rows]).to(device, non_blocking=True)
                ndvis = torch.from_numpy(ds.ndvi[samples]).to(device, non_blocking=True)
                lbls = torch.from_numpy(ds.table["label"][samples]).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits, attn = model(imgs, tabs, ndvis)
                    loss = criterion(logits, lbls) + _attention_balance_loss(attn)
                scaler.scale(loss).backward()
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        # 2. Precompute unique image embeddings for the split & run fast batched pass
        unique_embs, sample_uid = _extract_unique_image_embeddings(
            model, ds, device, use_amp
        )
        n_samples = len(ds)
        bsz = max(int(loader.batch_size or DEFAULT_BATCH), 256)
        order = np.random.permutation(n_samples) if training else np.arange(n_samples)

        samples_all = ds.indices[order]
        rows_all = ds.table["row"][samples_all]
        uids_all = torch.from_numpy(sample_uid[order]).to(device, non_blocking=True)
        tabs_all = torch.from_numpy(ds.features[rows_all]).to(device, non_blocking=True)
        ndvis_all = torch.from_numpy(ds.ndvi[samples_all]).to(device, non_blocking=True)
        lbls_all = torch.from_numpy(ds.table["label"][samples_all]).to(device, non_blocking=True)

        total_steps = (n_samples + bsz - 1) // bsz
        bar = tqdm(range(0, n_samples, bsz), total=total_steps, desc=desc, leave=False, unit="batch")
        for start in bar:
            end = min(start + bsz, n_samples)
            img_emb = unique_embs[uids_all[start:end]]
            if training:
                noise = torch.randn_like(img_emb) * 0.02
                img_emb = img_emb + noise
            tabular = tabs_all[start:end]
            ndvi = ndvis_all[start:end]
            labels = lbls_all[start:end]

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits, _attn = model.encode_from_image_emb(img_emb, tabular, ndvi)
                    ce_loss = criterion(logits, labels)
                    loss = ce_loss + (_attention_balance_loss(_attn) if training else 0.0)
            if training:
                scaler.scale(loss).backward()
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            loss_val = float(ce_loss.detach())
            total_loss += loss_val * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += labels.size(0)
            batches += 1
            if batches % 50 == 0 or end == n_samples:
                bar.set_postfix(
                    loss=f"{total_loss / max(seen, 1):.4f}",
                    acc=f"{correct / max(seen, 1):.3f}",
                )

        return {
            "loss": total_loss / max(seen, 1),
            "accuracy": correct / max(seen, 1),
            "samples": seen,
            "batches": batches,
        }

    total_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    if desc is None:
        desc = "train" if training else "val  "
    bar = tqdm(loader, total=total_steps, desc=desc, leave=False, unit="batch")

    for step, (tabular, images, ndvi, labels) in enumerate(bar):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        ndvi = ndvi.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, _attn = model(images, tabular, ndvi)
                ce_loss = criterion(logits, labels)
                loss = ce_loss + (_attention_balance_loss(_attn) if training else 0.0)
        if training:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        loss_val = float(ce_loss.detach())
        total_loss += loss_val * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += labels.size(0)
        batches += 1
        bar.set_postfix(
            loss=f"{total_loss / max(seen, 1):.4f}",
            acc=f"{correct / max(seen, 1):.3f}",
        )

    return {
        "loss": total_loss / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "samples": seen,
        "batches": batches,
    }


@torch.no_grad()
def predict(
    model: AttentionFusion,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval",
) -> dict[str, np.ndarray]:
    """Return labels, predictions and attention weights for a loader."""
    model.eval()
    ds = loader.dataset
    if isinstance(ds, data.MultiModalStressDataset) and len(ds) > 0:
        unique_embs, sample_uid = _extract_unique_image_embeddings(
            model, ds, device, use_amp=False
        )
        n_samples = len(ds)
        bsz = max(int(loader.batch_size or DEFAULT_BATCH), 256)
        samples_all = ds.indices
        rows_all = ds.table["row"][samples_all]
        uids_all = torch.from_numpy(sample_uid).to(device, non_blocking=True)
        tabs_all = torch.from_numpy(ds.features[rows_all]).to(device, non_blocking=True)
        ndvis_all = torch.from_numpy(ds.ndvi[samples_all]).to(device, non_blocking=True)

        y_pred_list, attn_list = [], []
        for start in range(0, n_samples, bsz):
            end = min(start + bsz, n_samples)
            logits, weights = model.encode_from_image_emb(
                unique_embs[uids_all[start:end]],
                tabs_all[start:end],
                ndvis_all[start:end],
            )
            y_pred_list.append(logits.argmax(dim=1).cpu().numpy())
            attn_list.append(weights.cpu().numpy())
        return {
            "y_true": ds.table["label"][samples_all].astype(np.int64),
            "y_pred": np.concatenate(y_pred_list),
            "attention": np.concatenate(attn_list),
        }

    y_true, y_pred, attn = [], [], []
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for tabular, images, ndvi, labels in bar:
        logits, weights = model(
            images.to(device), tabular.to(device), ndvi.to(device)
        )
        y_true.append(labels.numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        attn.append(weights.cpu().numpy())
    return {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "attention": np.concatenate(attn),
    }


def train_model(
    model: AttentionFusion,
    loaders: dict[str, DataLoader],
    labels_train: np.ndarray,
    epochs: int,
    patience: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    limit_batches: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
    epoch_ckpt_dir: Path | None = CHECKPOINTS_DIR,
    ckpt_prefix: str = "attention_fusion",
    stability_tol: float = DEFAULT_STABILITY_TOL,
    stability_patience: int = DEFAULT_STABILITY_PATIENCE,
) -> dict[str, Any]:
    """Train with AdamW + cosine schedule, per-epoch checkpoints & ±1% stability early exit."""
    criterion = nn.CrossEntropyLoss(
        weight=inverse_frequency_weights(labels_train).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_loss, best_epoch, bad = math.inf, -1, 0
    best_state: dict[str, torch.Tensor] | None = None
    prev_val_acc: float | None = None
    prev_val_loss: float | None = None
    stable_count = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_stats = run_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler,
            use_amp, max_batches=limit_batches,
            desc=f"epoch {epoch:02d}/{epochs} train",
        )
        val_stats = run_epoch(
            model, loaders["val"], criterion, None, device, scaler, use_amp,
            max_batches=limit_batches,
            desc=f"epoch {epoch:02d}/{epochs} val  ",
        )
        scheduler.step()
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(elapsed, 1),
            }
        )
        log(
            f"epoch {epoch:02d}/{epochs}  train loss {train_stats['loss']:.4f} "
            f"acc {train_stats['accuracy']:.4f} | val loss {val_stats['loss']:.4f} "
            f"acc {val_stats['accuracy']:.4f} | {elapsed:.0f}s"
        )

        epoch_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        if epoch_ckpt_dir is not None:
            epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
            epoch_ckpt_file = epoch_ckpt_dir / f"{ckpt_prefix}_epoch_{epoch:02d}.pt"
            torch.save(
                {
                    "state_dict": epoch_state,
                    "class_order": UNIFIED_CLASSES,
                    "epoch": epoch,
                    "val_loss": val_stats["loss"],
                    "val_accuracy": val_stats["accuracy"],
                    "history": history,
                    **(checkpoint_extra or {}),
                },
                epoch_ckpt_file,
            )
            log(f"checkpoint saved -> {epoch_ckpt_file}")

        if val_stats["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, bad = val_stats["loss"], epoch, 0
            best_state = epoch_state
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": best_state,
                        "class_order": UNIFIED_CLASSES,
                        "epoch": epoch,
                        "best_val_loss": best_loss,
                        **(checkpoint_extra or {}),
                    },
                    checkpoint_path,
                )
        else:
            bad += 1

        # Check ±1% metric stability (accuracy delta <= stability_tol and loss relative delta bounded)
        if prev_val_acc is not None and prev_val_loss is not None:
            acc_delta = abs(val_stats["accuracy"] - prev_val_acc)
            loss_rel_delta = abs(val_stats["loss"] - prev_val_loss) / max(abs(prev_val_loss), 1e-8)
            if acc_delta <= stability_tol and loss_rel_delta <= max(stability_tol * 5.0, 0.05):
                stable_count += 1
            else:
                stable_count = 0

            if stability_patience > 0 and stable_count >= stability_patience:
                log(
                    f"early exit at epoch {epoch}: model metric stable within "
                    f"+-{stability_tol * 100:.1f}% (delta_val_acc={acc_delta * 100:.2f}%, "
                    f"delta_val_loss_rel={loss_rel_delta * 100:.2f}% for {stable_count} "
                    f"consecutive epochs; best epoch {best_epoch})"
                )
                break

        prev_val_acc = val_stats["accuracy"]
        prev_val_loss = val_stats["loss"]

        if bad >= patience:
            log(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss}


# ---------------------------------------------------------------------------
# 3. Evaluation, baselines, attention analysis
# ---------------------------------------------------------------------------
def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> dict[str, Any]:
    """Accuracy / macro F1 / per-class report for one system."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=UNIFIED_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "title": title,
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in UNIFIED_CLASSES
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(N_CLASSES))
        ).tolist(),
    }


def baseline_predictions(
    bundle_tabular: dict[str, Any],
    bundle_image: dict[str, Any],
    frame: Any,
    table: dict[str, Any],
    test_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score the three Stage-0 systems on the same paired test samples.

    Stage-0 models emit Stage-0 label spaces; every prediction is converted
    to the Stage-1 taxonomy through class names (see
    ``stage0_index_to_stage1``), never by assuming index positions.
    """
    rows = frame.iloc[table["row"][test_indices]]
    image_paths = table["image"][test_indices]

    tab_probs = predict_tabular_probs(bundle_tabular, rows)
    tab_class_order = bundle_tabular["class_order"]
    tab_to_s1 = np.asarray([UNIFIED_INDEX[c] for c in tab_class_order], dtype=np.int64)
    xgb_pred = tab_to_s1[tab_probs.argmax(axis=1)]

    unique_paths = sorted({str(p) for p in image_paths})
    _, eval_tf = build_transforms()
    img_model = bundle_image["model"]
    img_device = bundle_image["device"]
    img_classes = bundle_image["classes"]
    image_unified_map: dict[str, np.ndarray] = {}
    image_s1_map: dict[str, int] = {}
    with torch.no_grad():
        for start in range(0, len(unique_paths), 64):
            batch_p = unique_paths[start : start + 64]
            tensors = []
            for p in batch_p:
                with Image.open(p) as im:
                    tensors.append(eval_tf(im.convert("RGB")))
            batch_t = torch.stack(tensors, dim=0).to(img_device)
            probs_b = torch.softmax(img_model(batch_t), dim=1).cpu().numpy()
            for p, pr in zip(batch_p, probs_b):
                iu = image_to_unified(pr, img_classes)
                image_unified_map[p] = iu
                image_s1_map[p] = stage0_index_to_stage1(iu)

    image_pred = np.empty(len(image_paths), dtype=np.int64)
    rule_pred = np.empty(len(image_paths), dtype=np.int64)
    for i, path in enumerate(image_paths):
        p_str = str(path)
        image_unified = image_unified_map[p_str]
        image_pred[i] = image_s1_map[p_str]
        label, _rule = rule_fusion(
            tabular_to_unified(tab_probs[i], tab_class_order),
            image_unified,
        )
        rule_pred[i] = UNIFIED_INDEX[label]

    return {
        "xgb": xgb_pred,
        "cnn": image_pred,
        "rule": rule_pred,
        "truth": table["label"][test_indices],
    }


def attention_by_class(attention: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean attention weight per modality per class (rows = classes)."""
    table = np.zeros((N_CLASSES, len(MODALITY_NAMES)))
    for cls in range(N_CLASSES):
        mask = labels == cls
        if mask.sum():
            table[cls] = attention[mask].mean(axis=0)
    return table


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def plot_comparison(results: dict[str, dict[str, Any]], path: Path) -> None:
    """Bar chart of accuracy and macro F1 across the four systems."""
    names = list(results)
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["macro_f1"] for n in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, acc, width, label="accuracy", color="#2b7bba")
    ax.bar(x + width / 2, f1, width, label="macro F1", color="#f39c12")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("Stage 1: unified attention fusion vs Stage-0 baselines (6-class)")
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_attention_heatmap(
    table: np.ndarray, classes: list[str], path: Path, title: str
) -> None:
    """Heatmap of mean attention weight per modality per class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(table, cmap="viridis", vmin=0.0, vmax=max(0.5, table.max()))
    ax.set_xticks(range(len(MODALITY_NAMES)))
    ax.set_xticklabels(MODALITY_NAMES)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            ax.text(
                c, r, f"{table[r, c]:.2f}", ha="center", va="center",
                color="white" if table[r, c] < table.max() * 0.6 else "black",
                fontsize=9,
            )
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sample_attention(
    attention: np.ndarray, preds: np.ndarray, n: int, path: Path
) -> None:
    """Per-sample attention heatmaps for the first ``n`` test predictions."""
    n = min(n, len(attention))
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.28 * n)))
    ax.imshow(attention[:n], cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(MODALITY_NAMES)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"#{i} -> {UNIFIED_CLASSES[preds[i]]}" for i in range(n)], fontsize=7
    )
    ax.set_title("Per-sample modality attention (test set, first rows)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def grad_cam_image(
    model: AttentionFusion,
    image: torch.Tensor,
    tabular: torch.Tensor,
    ndvi: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """Grad-CAM over MobileNetV2's last feature map for one sample."""
    model.eval()
    image = image.unsqueeze(0).to(device)
    tabular = tabular.unsqueeze(0).to(device)
    ndvi = ndvi.unsqueeze(0).to(device)
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logits, _ = model(image, tabular, ndvi)
        score = logits[0, target_class]
        acts = model.image_encoder.activations
        grad = torch.autograd.grad(score, acts, retain_graph=False)[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def save_gradcam_overlays(
    model: AttentionFusion,
    dataset: data.MultiModalStressDataset,
    preds: np.ndarray,
    attention: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    count: int = 8,
) -> list[Path]:
    """Save Grad-CAM overlays for the first ``count`` test samples."""
    written: list[Path] = []
    for i in range(min(count, len(indices))):
        sample = int(indices[i])
        path = Path(dataset.table["image"][sample])
        tabular, image, ndvi, label = dataset[i]
        cam = grad_cam_image(model, image, tabular, ndvi, int(preds[i]), device)

        pil = dataset._load_image(path).resize((IMG_SIZE, IMG_SIZE))
        base = np.asarray(pil, dtype=np.float32) / 255.0
        heatmap = plt.get_cmap("jet")(cam)[..., :3]
        overlay = np.clip(0.45 * base + 0.55 * heatmap, 0, 1)

        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        ax[0].imshow(base)
        ax[0].set_title(f"input: {path.name}", fontsize=8)
        ax[1].imshow(heatmap)
        ax[1].set_title("Grad-CAM (image branch)", fontsize=8)
        ax[2].imshow(overlay)
        attn_txt = ", ".join(
            f"{n}={w:.2f}" for n, w in zip(MODALITY_NAMES, attention[i])
        )
        ax[2].set_title(
            f"pred {UNIFIED_CLASSES[preds[i]]}\ntrue {UNIFIED_CLASSES[label]} | {attn_txt}",
            fontsize=8,
        )
        for axis in ax:
            axis.axis("off")
        fig.tight_layout()
        out = EXPLAIN_DIR / f"gradcam_{i:02d}_{UNIFIED_CLASSES[preds[i]]}.png"
        EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        written.append(out)
    return written


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
def write_report(payload: dict[str, Any], path: Path) -> None:
    """Write report_stage1.txt (metrics, attention analysis, caveats)."""
    lines: list[str] = []
    rule = "=" * 88
    lines += [rule, "STAGE 1 - UNIFIED MULTI-MODAL ATTENTION FUSION (KisanGPT)", rule, ""]

    lines += ["1. DATA AND PAIRING (Stage-1A contract; details in PAIRING_STRATEGY.md)"]
    d = payload["data"]
    lines += [
        f"  tabular rows ............ {d['rows']:,} (CHES, real) "
        f"-> {d['feature_dim']}-dim model vector",
        f"  leaf images ............. {d['images']:,} (paddy, real)",
        f"  samples ................. {d['pairs']:,} "
        f"(row-sourced {d['rows']:,} + image-sourced disease_stress "
        f"{d['pairs'] - d['rows']:,})",
        f"  splits .................. train {d['train']:,} / val {d['val']:,} / "
        f"test {d['test']:,} (stratified on the unified 6-class label, seed 42; "
        f"pairing is split-restricted - no photo or row crosses splits)",
        f"  class order ............. {', '.join(UNIFIED_CLASSES)} "
        f"(stage1_config.CLASS_MAP; water/nutrient swap vs Stage 0)",
        f"  pairing ................. WEAK - healthy row -> healthy photo, "
        f"stress row -> random disease photo; disease_stress comes from "
        f"image-sourced samples (CHES has no disease label)",
        f"  NDVI .................... PROXY = {d['ndvi_proxy']}, class-conditional "
        f"({d['ndvi_steps']} steps) - can encode the label by construction; "
        f"prototype only",
        f"  pest_stress ............. RESERVED, untrained "
        f"(support {d['pest_support']:,} in every split)",
        f"  binary=0 out-of-window .. {d['binary0_outside']:,} rows routed to "
        f"stress_generic (Option B, documented - not ignored)",
        "  class counts: " + ", ".join(f"{k}={v:,}" for k, v in d["class_counts"].items()),
        "",
    ]

    lines += ["2. ARCHITECTURE"]
    a = payload["architecture"]
    lines += [
        f"  image ................... MobileNetV2 -> {a['image_dim']}d",
        f"  tabular ................. MLP {a['tabular_mlp']} -> {a['tabular_out']}d",
        f"  ndvi .................... 1D-CNN -> {a['ndvi_dim']}d (PROXY input)",
        f"  fusion .................. concat 1536d -> 3 tokens -> "
        f"{a['heads']}-head attention -> pooled {a['pooled']}d",
        f"  heads ................... Linear({a['pooled']}, {a['classes']}) logits "
        f"(softmax at inference) + per-modality attention weights",
        "",
    ]

    lines += ["3. TRAINING"]
    t = payload["training"]
    lines += [
        f"  loss .................... CrossEntropyLoss (inverse-frequency, mean-normalised)",
        f"  optimizer ............... AdamW lr={t['lr']} weight_decay={t['weight_decay']}",
        f"  scheduler ............... CosineAnnealingLR",
        f"  batch / epochs .......... {t['batch']} / {t['epochs']} "
        f"(early stopping patience {t['patience']} on val loss)",
        f"  mixed precision ......... {'on (cuda)' if t['amp'] else 'off (cpu host)'}, "
        f"gradient clipping max_norm={t['grad_clip']}",
        f"  best epoch .............. {t['best_epoch']} "
        f"(val loss {t['best_val_loss']:.4f})"
        if (isinstance(t["best_val_loss"], (int, float)) and t["best_val_loss"] >= 1e-3)
        else f"  best epoch .............. {t['best_epoch']} "
        f"(val loss {t['best_val_loss']:.4e})",
        "",
    ]

    lines += ["4. RESULTS (6-class, shared held-out test samples)"]
    lines += ["", f"  {'system':<36}{'accuracy':>10}{'macro F1':>10}"]
    lines += [f"  {'-' * 36}{'-' * 10}{'-' * 10}"]
    for name, block in payload["results"].items():
        lines.append(f"  {name:<36}{block['accuracy']:>10.4f}{block['macro_f1']:>10.4f}")

    best = payload["results"]["attention fusion (Stage 1)"]
    lines += [
        "",
        "  Per-class (attention fusion):",
        f"  {'class':<18}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for name in UNIFIED_CLASSES:
        s = best["per_class"][name]
        lines.append(
            f"  {name:<18}{s['precision']:>10.3f}{s['recall']:>10.3f}"
            f"{s['f1']:>10.3f}{int(s['support']):>10}"
        )

    lines += ["", "5. ATTENTION ANALYSIS (mean weight per modality per class)"]
    table = np.asarray(payload["attention"]["by_class"])
    lines.append("  " + f"{'class':<18}" + "".join(f"{m:>12}" for m in MODALITY_NAMES))
    for i, name in enumerate(UNIFIED_CLASSES):
        lines.append("  " + f"{name:<18}" + "".join(f"{v:>12.4f}" for v in table[i]))
    dominant = [
        (UNIFIED_CLASSES[i], MODALITY_NAMES[int(np.argmax(table[i]))])
        for i in range(N_CLASSES)
        if table[i].sum() > 0
    ]
    lines += [
        "",
        "  dominant modality per class: "
        + ", ".join(f"{c} -> {m}" for c, m in dominant),
    ]

    lines += ["", "6. LIMITATIONS (read before citing any number above)"]
    lines += [
        "  - Weak pairing: rows and photos are never jointly observed; the",
        "    image branch sees a random same-binary-class leaf. Attention",
        "    weights describe the model's use of that evidence, not field-",
        "    measured importance.",
        "  - PROXY NDVI is class-conditional, so the third modality alone can",
        "    recover the label by construction. Any gain attributed to it is a",
        "    property of the proxy; replace dataset.build_proxy_ndvi() with a",
        "    real Sentinel-2 export before drawing satellite conclusions.",
        "  - disease_stress labels come from image-sourced samples, so the",
        "    image-only and rule-fusion baselines are structurally strong there",
        "    while XGBoost cannot predict that class at all.",
        "  - pest_stress is reserved and untrained (0 support): its precision/",
        "    recall/F1 are 0 by construction, not a model failure.",
        "  - See PAIRING_STRATEGY.md for the full pairing rationale, verified",
        "    invariants and split policy.",
        "",
    ]
    lines += [rule]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage-1 training/evaluation options."""
    parser = argparse.ArgumentParser(
        description="Stage 1B: unified multi-modal attention fusion model"
    )
    parser.add_argument("--data_path", type=Path, default=tx.DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", type=Path, default=Path("data/paddy_images"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--stability_tol", type=float, default=DEFAULT_STABILITY_TOL,
                        help="early exit when val metric changes by <= this fraction (default 0.01 = ±1%%)")
    parser.add_argument("--stability_patience", type=int, default=DEFAULT_STABILITY_PATIENCE,
                        help="consecutive stable epochs within ±stability_tol to trigger early exit")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit_batches", type=int, default=None,
                        help="cap batches per epoch (smoke testing)")
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--gradcam_count", type=int, default=8)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="smoke test on smaller data without writing artifacts")
    return parser.parse_args(argv)


def verify_saved_splits(saved_indices: np.ndarray) -> bool:
    """Cross-check rebuilt indices against artifacts/split_indices.json."""
    if not data.SPLIT_PATH.exists():
        return True
    try:
        saved = json.loads(data.SPLIT_PATH.read_text(encoding="utf-8"))
        rebuilt = np.asarray(saved["splits"]["test"]["sample_indices"], dtype=np.int64)
        return bool(
            len(rebuilt) == len(saved_indices)
            and np.array_equal(rebuilt[: len(saved_indices)], saved_indices)
        )
    except Exception:                                    # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> int:
    """Build the Stage-1A dataset, train, evaluate, compare, explain."""
    args = parse_args(argv)
    set_seed(args.seed)
    started = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    log(f"device={device.type}{gpu_info} amp={'on' if use_amp else 'off'}")

    # ---- Stage-1A data contract -----------------------------------------
    tabular = data.load_tabular(args.data_path)
    row_splits = data.split_rows(tabular["labels"], seed=args.seed)
    pools = data.build_image_pools(args.image_dir, seed=args.seed)
    table = data.build_sample_table(
        tabular["labels"], row_splits, pools, seed=args.seed
    )

    real_ndvi = data.load_real_ndvi(tabular["frame"], cfg.NDVI_STEPS)
    ndvi_is_proxy = real_ndvi is None
    if real_ndvi is not None:
        ndvi = real_ndvi
        log("using real NDVI export")
    else:
        log(f"PROXY NDVI in use (class-conditional, {cfg.NDVI_STEPS} steps)")
        ndvi = data.build_proxy_ndvi(table["label"], seed=args.seed)

    split_index = {
        name: np.where(table["split"] == name)[0]
        for name in ("train", "val", "test")
    }
    if not verify_saved_splits(split_index["test"]):
        log("WARNING: rebuilt test indices differ from artifacts/split_indices.json "
            "(seed changed?) - results will not be comparable to Stage 1A runs")

    if args.max_train:
        split_index["train"] = split_index["train"][: args.max_train]
    if args.max_val:
        split_index["val"] = split_index["val"][: args.max_val]
    if args.max_test:
        split_index["test"] = split_index["test"][: args.max_test]

    train_tf, eval_tf = build_transforms()
    cache: dict[Path, Image.Image] = {}
    datasets = {
        "train": data.MultiModalStressDataset(
            split_index["train"], table, tabular["features"], ndvi, train_tf, cache
        ),
        "val": data.MultiModalStressDataset(
            split_index["val"], table, tabular["features"], ndvi, eval_tf, cache
        ),
        "test": data.MultiModalStressDataset(
            split_index["test"], table, tabular["features"], ndvi, eval_tf, cache
        ),
    }
    pin = device.type == "cuda"
    loaders = {
        name: DataLoader(
            ds, batch_size=args.batch_size, shuffle=(name == "train"),
            num_workers=args.num_workers, pin_memory=pin,
        )
        for name, ds in datasets.items()
    }
    log(
        f"loaders ready: train {len(datasets['train']):,} / "
        f"val {len(datasets['val']):,} / test {len(datasets['test']):,} "
        f"(proxy={ndvi_is_proxy})"
    )

    # ---- model ----------------------------------------------------------
    model = AttentionFusion(tabular_dim=tabular["features"].shape[1]).to(device)
    log(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    history: dict[str, Any] = {"history": [], "best_epoch": None, "best_val_loss": None}
    if not args.eval_only:
        history = train_model(
            model, loaders, table["label"][split_index["train"]],
            epochs=args.epochs, patience=args.patience, device=device,
            lr=args.lr, weight_decay=args.weight_decay, use_amp=use_amp,
            limit_batches=args.limit_batches,
            checkpoint_path=None if args.dry_run else CKPT_PATH,
            checkpoint_extra={
                "config": {
                    "ndvi_steps": cfg.NDVI_STEPS, "d_model": D_MODEL,
                    "heads": N_HEADS, "classes": N_CLASSES,
                    "ndvi_is_proxy": ndvi_is_proxy,
                }
            },
            epoch_ckpt_dir=CHECKPOINTS_DIR,
            ckpt_prefix="dryrun_attention_fusion" if args.dry_run else "attention_fusion",
            stability_tol=args.stability_tol,
            stability_patience=args.stability_patience,
        )
        if not args.dry_run:
            ckpt_blob = {
                "state_dict": model.state_dict(),
                "class_order": UNIFIED_CLASSES,
                "config": {
                    "ndvi_steps": cfg.NDVI_STEPS, "d_model": D_MODEL,
                    "heads": N_HEADS, "classes": N_CLASSES,
                    "ndvi_is_proxy": ndvi_is_proxy,
                },
                "history": history,
            }
            torch.save(ckpt_blob, CKPT_PATH)
            unified_alias = ARTIFACTS_DIR / "unified_model.pth"
            torch.save(ckpt_blob, unified_alias)
            log(f"checkpoint saved -> {CKPT_PATH} (and {unified_alias})")
        else:
            peak_mb = (
                round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
                if device.type == "cuda" else 0.0
            )
            gpu_mem_str = f" | gpu_peak_mem={peak_mb:.2f}MB" if device.type == "cuda" else ""
            log(f"dry-run complete: no checkpoint / metrics written (by design){gpu_mem_str}")
            return 0
    elif CKPT_PATH.exists():
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        raw_hist = checkpoint.get("history", history)
        if isinstance(raw_hist, dict):
            history = {
                "history": raw_hist.get("history", []),
                "best_epoch": raw_hist.get("best_epoch") or checkpoint.get("epoch"),
                "best_val_loss": raw_hist.get("best_val_loss") or checkpoint.get("best_val_loss"),
            }
        else:
            history = {
                "history": raw_hist if isinstance(raw_hist, list) else [],
                "best_epoch": checkpoint.get("epoch"),
                "best_val_loss": checkpoint.get("best_val_loss"),
            }
        unified_alias = ARTIFACTS_DIR / "unified_model.pth"
        torch.save(checkpoint, unified_alias)
        log(f"loaded existing checkpoint (synced -> {unified_alias})")

    # ---- evaluation ------------------------------------------------------
    test_stats = predict(model, loaders["test"], device)
    results: dict[str, dict[str, Any]] = {
        "attention fusion (Stage 1)": metrics_block(
            test_stats["y_true"], test_stats["y_pred"], "attention fusion (Stage 1)"
        )
    }

    if not args.skip_baselines and args.max_test is None:
        log("scoring Stage-0 baselines on the same test samples ...")
        bundle_tab = load_tabular_bundle()
        bundle_img = load_image_bundle(device=device)
        base = baseline_predictions(
            bundle_tab, bundle_img, tabular["frame"], table, split_index["test"]
        )
        results["XGBoost tabular-only"] = metrics_block(
            base["truth"], base["xgb"], "XGBoost tabular-only"
        )
        results["MobileNetV2 image-only"] = metrics_block(
            base["truth"], base["cnn"], "MobileNetV2 image-only"
        )
        results["rule-based fusion (post-fix)"] = metrics_block(
            base["truth"], base["rule"], "rule-based fusion (post-fix)"
        )
    elif not args.skip_baselines:
        log("WARNING: --max_test set; skipping baseline comparison to avoid "
            "train/test contamination")

    # ---- attention analysis + explainability ----------------------------
    table_attn = attention_by_class(test_stats["attention"], test_stats["y_true"])
    plot_comparison(results, PLOTS_DIR / "stage1_comparison.png")
    plot_confusion_matrix(
        np.asarray(results["attention fusion (Stage 1)"]["confusion_matrix"]),
        UNIFIED_CLASSES,
        PLOTS_DIR / "stage1_confusion_matrix.png",
    )
    plot_attention_heatmap(
        table_attn, UNIFIED_CLASSES, EXPLAIN_DIR / "attention_by_class.png",
        "Mean attention weight per modality per class (Stage 1)",
    )
    plot_sample_attention(
        test_stats["attention"], test_stats["y_pred"], 40,
        EXPLAIN_DIR / "attention_samples.png",
    )
    np.savez_compressed(
        ATTENTION_PATH,
        sample_index=split_index["test"],
        row_index=table["row"][split_index["test"]],
        image=[str(p) for p in table["image"][split_index["test"]]],
        attention=test_stats["attention"],
        y_true=test_stats["y_true"],
        y_pred=test_stats["y_pred"],
        modalities=np.asarray(MODALITY_NAMES),
    )
    gradcam_files = save_gradcam_overlays(
        model, datasets["test"], test_stats["y_pred"], test_stats["attention"],
        split_index["test"], device, count=args.gradcam_count,
    )
    log(f"explainability: {len(gradcam_files)} Grad-CAM overlays + 2 attention plots")

    # ---- report ----------------------------------------------------------
    coverage = data.coverage_report(table)
    payload = {
        "data": {
            "rows": tabular["meta"]["rows"],
            "feature_dim": tabular["meta"]["feature_dim"],
            "images": sum(len(v) for s in pools.values() for v in s.values()),
            "pairs": int(len(table["label"])),
            "train": int(len(split_index["train"])),
            "val": int(len(split_index["val"])),
            "test": int(len(split_index["test"])),
            "ndvi_steps": int(ndvi.shape[1]),
            "ndvi_proxy": bool(ndvi_is_proxy),
            "binary0_outside": tabular["meta"]["binary0_outside_window"],
            "pest_support": int(
                np.sum(table["label"][split_index["test"]] == UNIFIED_INDEX["pest_stress"])
            ),
            "class_counts": data.label_counts(table["label"]),
            "coverage": coverage,
        },
        "architecture": {
            "image_dim": 1280,
            "tabular_mlp": [tabular["features"].shape[1], 256, 128],
            "tabular_out": 128,
            "ndvi_dim": 128,
            "heads": N_HEADS,
            "pooled": D_MODEL,
            "classes": N_CLASSES,
        },
        "training": {
            "lr": args.lr, "weight_decay": args.weight_decay,
            "batch": args.batch_size, "epochs": args.epochs,
            "patience": args.patience, "amp": use_amp,
            "grad_clip": args.grad_clip,
            "best_epoch": history.get("best_epoch"),
            "best_val_loss": history.get("best_val_loss") or float("nan"),
        },
        "results": results,
        "attention": {
            "by_class": table_attn.tolist(),
            "modalities": list(MODALITY_NAMES),
        },
        "history": history.get("history", []),
        "artifacts": {
            "checkpoint": str(CKPT_PATH),
            "attention": str(ATTENTION_PATH),
            "comparison_plot": str(PLOTS_DIR / "stage1_comparison.png"),
            "confusion_plot": str(PLOTS_DIR / "stage1_confusion_matrix.png"),
            "attention_plot": str(EXPLAIN_DIR / "attention_by_class.png"),
            "gradcam": [str(p) for p in gradcam_files],
        },
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    save_json(payload, METRICS_PATH)
    write_report(payload, REPORT_PATH)
    log(f"metrics -> {METRICS_PATH}")
    log(f"report  -> {REPORT_PATH}")
    log(f"done in {payload['runtime_seconds']}s")
    for name, block in results.items():
        log(f"  {name:<36} acc {block['accuracy']:.4f}  macro F1 {block['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
