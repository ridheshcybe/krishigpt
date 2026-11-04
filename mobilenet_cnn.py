"""MobileNetV2 classifier for the Kaggle Plant Stress Identification dataset.

Dataset: 2,200 RGB paddy-leaf images in five classes
(``BLB``, ``blast``, ``healthy``, ``hispa``, ``leaf_spot``).  The provided
train/val/test folders are merged and re-split 70/15/15 with stratified
sampling, as required, using a fixed seed.

Training is two-phase:
  * phase 1 -- freeze the ImageNet backbone, train the 128-unit head for 30
    epochs;
  * phase 2 -- unfreeze the top backbone blocks (requested: 53; torchvision
    MobileNetV2 exposes 19 blocks, so all are unfrozen), fine-tune for 20
    epochs.

Both phases use Adam(1e-3), ``ReduceLROnPlateau(factor=0.5, patience=5)`` on
validation loss, and inverse-frequency class weights.

The model emits logits; softmax is applied by ``predict_proba`` and the fusion
stage.  Applying softmax inside the network together with ``CrossEntropyLoss``
would double-normalise the probabilities and break training.

Usage
-----
    python mobilenet_cnn.py --image_dir data/paddy_images

Artifacts
---------
* ``artifacts/mobilenet_paddy.pth`` -- best checkpoint (weights + metadata)
* ``artifacts/cnn_metrics.json``     -- test metrics and training history
* ``plots/confusion_matrix_cnn.png``
* ``plots/cnn_training_curves.png``
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_IMAGE_DIR = DATA_DIR / "paddy_images"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PLOTS_DIR = PROJECT_DIR / "plots"
MODEL_PATH = ARTIFACTS_DIR / "mobilenet_paddy.pth"
METRICS_PATH = ARTIFACTS_DIR / "cnn_metrics.json"

KAGGLE_SLUG = "ritikbompilwar/plantstressidentification"
KAGGLE_CACHE = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "ritikbompilwar"
    / "plantstressidentification"
)

CLASSES = ["BLB", "blast", "healthy", "hispa", "leaf_spot"]
HEALTHY_INDEX = CLASSES.index("healthy")
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42

BATCH_SIZE = 32
HEAD_EPOCHS = 30
FINETUNE_EPOCHS = 20
HEAD_UNITS = 128
DROPOUT = 0.3
LEARNING_RATE = 1e-3
UNFREEZE_LAYERS = 53
STABILITY_TOL = 0.01
STABILITY_PATIENCE = 2
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

# Folder spelling -> canonical class name.
CLASS_ALIASES = {
    "blb": "BLB",
    "bacterial_leaf_blight": "BLB",
    "blast": "blast",
    "healthy": "healthy",
    "hispa": "hispa",
    "leaf_spot": "leaf_spot",
    "leafspot": "leaf_spot",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[save] {path}")


def _class_from_dir(name: str) -> str | None:
    """Return the canonical class for a folder name, or None."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(key)


def _contains_class_dirs(root: Path) -> bool:
    """True when ``root`` or any descendant is a recognisable class folder.

    The dataset ships ``train/<class>``, ``val/<class>`` and ``test/<class>``
    sub-folders; the parent of those split folders must be returned so that
    all 2,200 images are merged before re-splitting.
    """
    if not root.is_dir():
        return False
    for dirpath, _dirnames, _filenames in os.walk(root):
        if _class_from_dir(Path(dirpath).name) is not None:
            return True
    return False


def resolve_image_dir(image_dir: Path | None) -> Path:
    """Locate the real paddy dataset directory.

    Search order: the CLI path, ``data/paddy_images``, the local kagglehub
    cache, then a kagglehub download (requires Kaggle credentials).  Raises a
    clear error rather than fabricating images.
    """
    search_roots: list[Path] = []
    if image_dir is not None:
        search_roots.append(image_dir)
    search_roots.append(DEFAULT_IMAGE_DIR)

    for root in search_roots:
        if _contains_class_dirs(root):
            return root

    if KAGGLE_CACHE.is_dir():
        for version in sorted(KAGGLE_CACHE.glob("versions/*"), reverse=True):
            for candidate in [version / "Dataset", version, *version.glob("*/Dataset")]:
                if _contains_class_dirs(candidate):
                    return candidate

    try:
        import kagglehub

        downloaded = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        for candidate in [downloaded, downloaded / "Dataset", *downloaded.glob("*/Dataset")]:
            if _contains_class_dirs(candidate):
                return candidate
    except Exception as exc:  # noqa: BLE001 - surfaced in the message below
        download_error = str(exc)
    else:
        download_error = "downloaded archive has no class folders"

    raise FileNotFoundError(
        "Paddy-leaf images not found. Download the Kaggle 'Plant Stress "
        "Identification' dataset and place its train/val/test folders under "
        f"{DEFAULT_IMAGE_DIR} (or pass --image_dir). Kaggle auto-download "
        f"failed: {download_error}"
    )


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
class PaddyDataset(Dataset):
    """RGB paddy-leaf dataset with configurable transforms."""

    def __init__(self, samples: list[tuple[Path, int]], transform: Callable) -> None:
        """Store the ``(path, label)`` samples and the transform to apply."""
        if not samples:
            raise ValueError("PaddyDataset requires at least one image")
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one RGB image, apply the transform, and return it with its label."""
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:  # noqa: BLE001 - include the offending path
            raise RuntimeError(f"Corrupted or unreadable image: {path} ({exc})") from exc
        return tensor, label


_SCAN_CACHE: dict[str, list[tuple[Path, int]]] = {}


def scan_images(root: Path) -> list[tuple[Path, int]]:
    """Collect labelable, readable images and skip corrupted files with a log."""
    cache_key = str(root.resolve())
    if cache_key in _SCAN_CACHE:
        return list(_SCAN_CACHE[cache_key])
    samples: list[tuple[Path, int]] = []
    corrupted = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        folder_class = _class_from_dir(Path(dirpath).name)
        if folder_class is None:
            continue
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:  # noqa: BLE001 - bad file, skip
                corrupted += 1
                print(f"[data] WARNING: skipping corrupted image {path}")
                continue
            samples.append((path, CLASSES.index(folder_class)))

    if not samples:
        raise RuntimeError(
            f"No valid paddy images found under {root}. Expected class folders "
            f"named one of {sorted(set(CLASS_ALIASES.values()))}."
        )
    if corrupted:
        print(f"[data] skipped {corrupted} corrupted image(s)")
    _SCAN_CACHE[cache_key] = samples
    return list(samples)


def stratified_split(
    samples: list[tuple[Path, int]], seed: int = SEED
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split samples 70/15/15 with stratification on the five classes."""
    labels = np.asarray([label for _path, label in samples])
    counts = np.bincount(labels, minlength=len(CLASSES))
    if counts.min() < 2:
        raise ValueError(f"Each class needs at least 2 images; got {counts.tolist()}")

    train_idx, temp_idx = train_test_split(
        np.arange(len(samples)), test_size=0.30, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=seed
    )
    to_samples = lambda idx: [samples[i] for i in idx]  # noqa: E731
    return to_samples(train_idx), to_samples(val_idx), to_samples(test_idx)


def build_transforms() -> tuple[Callable, Callable]:
    """Return (train, eval) transforms with the required augmentation."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize]
    )
    return train_transform, eval_transform


def build_loaders(
    image_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int = SEED,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Scan, validate, split, and wrap the real images in DataLoaders."""
    samples = scan_images(image_dir)
    if max_samples is not None and 0 < max_samples < len(samples):
        labels = np.asarray([label for _path, label in samples])
        keep_idx, _ = train_test_split(
            np.arange(len(samples)),
            train_size=max_samples,
            stratify=labels,
            random_state=seed,
        )
        samples = [samples[i] for i in sorted(keep_idx)]
    train_samples, val_samples, test_samples = stratified_split(samples, seed=seed)
    train_tf, eval_tf = build_transforms()

    pin_memory = torch.cuda.is_available()
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        PaddyDataset(train_samples, train_tf), shuffle=True, **common
    )
    val_loader = DataLoader(PaddyDataset(val_samples, eval_tf), shuffle=False, **common)
    test_loader = DataLoader(PaddyDataset(test_samples, eval_tf), shuffle=False, **common)

    def class_counts(part: list[tuple[Path, int]]) -> dict[str, int]:
        """Return per-class image counts for one split."""
        counts = np.bincount([label for _p, label in part], minlength=len(CLASSES))
        return {name: int(counts[i]) for i, name in enumerate(CLASSES)}

    split_meta = {
        "source": str(image_dir),
        "total": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "class_counts": {
            "all": class_counts(samples),
            "train": class_counts(train_samples),
            "val": class_counts(val_samples),
            "test": class_counts(test_samples),
        },
    }
    print(
        f"[data] {len(samples):,} images -> train={len(train_samples):,} "
        f"val={len(val_samples):,} test={len(test_samples):,}  counts={split_meta['class_counts']['all']}"
    )
    return train_loader, val_loader, test_loader, split_meta


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class PaddyMobileNetV2(nn.Module):
    """ImageNet MobileNetV2 with the specified 5-class classification head."""

    def __init__(self, num_classes: int = len(CLASSES), pretrained: bool = True) -> None:
        """Build the ImageNet backbone and the specified 5-class classifier head."""
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            base = models.mobilenet_v2(weights=weights)
            self.pretrained_loaded = pretrained
        except Exception as exc:  # noqa: BLE001 - offline fallback, clearly logged
            print(
                f"[model] WARNING: could not load ImageNet weights ({exc}); "
                "training from random initialisation"
            )
            base = models.mobilenet_v2(weights=None)
            self.pretrained_loaded = False

        self.features = base.features
        in_features = base.last_channel
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, HEAD_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_UNITS, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits (softmax is applied at inference)."""
        return self.classifier(self.features(x))


def set_backbone_trainable(model: PaddyMobileNetV2, trainable: bool) -> None:
    """Freeze or unfreeze all backbone feature parameters."""
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def unfreeze_top_layers(model: PaddyMobileNetV2, count: int) -> int:
    """Unfreeze the last ``count`` backbone blocks; return how many were unfrozen."""
    blocks = list(model.features.children())
    selected = blocks[-count:] if count > 0 else []
    for block in selected:
        for parameter in block.parameters():
            parameter.requires_grad = True
    return len(selected)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    counts = np.clip(counts, 1.0, None)
    weights = len(y) / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    phase: str,
) -> tuple[float, float]:
    """Run one train or evaluation epoch and return (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(
        loader,
        desc=f"{phase} [{'train' if training else 'val'}]",
        leave=False,
        unit="batch",
    ) as progress:
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.set_grad_enabled(training):
                logits = model(images)
                loss = criterion(logits, labels)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += batch_size
            progress.set_postfix(
                loss=f"{total_loss / max(total, 1):.4f}",
                acc=f"{correct / max(total, 1):.4f}",
            )

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, probabilities) on ``loader``."""
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="test", leave=False, unit="batch"):
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)
            all_prob.append(probabilities.cpu().numpy())
            all_pred.extend(probabilities.argmax(dim=1).cpu().tolist())
            all_true.extend(labels.tolist())
    return np.asarray(all_true), np.asarray(all_pred), np.vstack(all_prob)


def train(
    image_dir: Path,
    head_epochs: int = HEAD_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    device: torch.device | None = None,
    max_samples: int | None = None,
    dry_run: bool = False,
    stability_tol: float = STABILITY_TOL,
    stability_patience: int = STABILITY_PATIENCE,
) -> dict[str, Any]:
    """Run the two-phase training pipeline and persist model + metrics."""
    if not dry_run:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[train] device={device}{gpu_info}  torch={torch.__version__}")

    train_loader, val_loader, test_loader, split_meta = build_loaders(
        image_dir, batch_size, num_workers, max_samples=max_samples
    )
    train_labels = np.asarray([label for _p, label in train_loader.dataset.samples])
    weights = class_weights(train_labels).to(device)
    print(f"[train] class weights: {weights.cpu().numpy().round(4).tolist()}")

    model = PaddyMobileNetV2().to(device)
    print(f"[train] ImageNet weights loaded: {model.pretrained_loaded}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    global_epoch = 0
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = "dryrun_mobilenet_paddy" if dry_run else "mobilenet_paddy"

    def _save_epoch_ckpt(phase_num: int, ep: int, v_loss: float, v_acc: float) -> Path:
        state_copy = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        ckpt_file = CHECKPOINTS_DIR / f"{ckpt_prefix}_epoch_{global_epoch:02d}.pth"
        blob = {
            "state_dict": state_copy,
            "classes": CLASSES,
            "arch": "mobilenet_v2",
            "img_size": IMG_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "phase": phase_num,
            "epoch": ep,
            "global_epoch": global_epoch,
            "val_loss": v_loss,
            "val_acc": v_acc,
            "best_epoch": best_epoch or global_epoch,
        }
        torch.save(blob, ckpt_file)
        return ckpt_file

    # ---- phase 1: frozen backbone ----------------------------------------
    set_backbone_trainable(model, False)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    print(f"[phase 1] frozen backbone, {head_epochs} epochs")
    prev_val_acc: float | None = None
    prev_val_loss: float | None = None
    stable_epochs = 0
    for epoch in range(1, head_epochs + 1):
        global_epoch += 1
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, "phase1"
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None, "phase1")
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "phase": 1,
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = global_epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if not dry_run:
                torch.save(
                    {
                        "state_dict": best_state,
                        "classes": CLASSES,
                        "arch": "mobilenet_v2",
                        "img_size": IMG_SIZE,
                        "mean": IMAGENET_MEAN,
                        "std": IMAGENET_STD,
                        "best_epoch": best_epoch,
                        "val_loss": best_val_loss,
                    },
                    MODEL_PATH,
                )
        ckpt_file = _save_epoch_ckpt(1, epoch, val_loss, val_acc)
        print(
            f"[phase1] {epoch:02d}/{head_epochs} lr={lr:.2e} "
            f"loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ckpt={ckpt_file.name}"
        )
        if prev_val_acc is not None and prev_val_loss is not None:
            acc_delta = abs(val_acc - prev_val_acc)
            loss_rel_delta = abs(val_loss - prev_val_loss) / max(abs(prev_val_loss), 1e-8)
            if acc_delta <= stability_tol and loss_rel_delta <= max(stability_tol * 5.0, 0.05):
                stable_epochs += 1
            else:
                stable_epochs = 0
            if stability_patience > 0 and stable_epochs >= stability_patience:
                print(
                    f"[phase1] early exit at epoch {epoch}: metric stable within "
                    f"+-{stability_tol * 100:.1f}% (delta_val_acc={acc_delta * 100:.2f}%, "
                    f"delta_val_loss_rel={loss_rel_delta * 100:.2f}% for {stable_epochs} "
                    f"consecutive epochs)"
                )
                break
        prev_val_acc = val_acc
        prev_val_loss = val_loss

    # ---- phase 2: fine-tune the top backbone layers ----------------------
    unfrozen = unfreeze_top_layers(model, UNFREEZE_LAYERS)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    print(f"[phase 2] unfroze {unfrozen} backbone blocks, {finetune_epochs} epochs")
    prev_val_acc = None
    prev_val_loss = None
    stable_epochs = 0
    for epoch in range(1, finetune_epochs + 1):
        global_epoch += 1
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, "phase2"
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None, "phase2")
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "phase": 2,
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = global_epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if not dry_run:
                torch.save(
                    {
                        "state_dict": best_state,
                        "classes": CLASSES,
                        "arch": "mobilenet_v2",
                        "img_size": IMG_SIZE,
                        "mean": IMAGENET_MEAN,
                        "std": IMAGENET_STD,
                        "best_epoch": best_epoch,
                        "val_loss": best_val_loss,
                    },
                    MODEL_PATH,
                )
        ckpt_file = _save_epoch_ckpt(2, epoch, val_loss, val_acc)
        print(
            f"[phase2] {epoch:02d}/{finetune_epochs} lr={lr:.2e} "
            f"loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ckpt={ckpt_file.name}"
        )
        if prev_val_acc is not None and prev_val_loss is not None:
            acc_delta = abs(val_acc - prev_val_acc)
            loss_rel_delta = abs(val_loss - prev_val_loss) / max(abs(prev_val_loss), 1e-8)
            if acc_delta <= stability_tol and loss_rel_delta <= max(stability_tol * 5.0, 0.05):
                stable_epochs += 1
            else:
                stable_epochs = 0
            if stability_patience > 0 and stable_epochs >= stability_patience:
                print(
                    f"[phase2] early exit at epoch {epoch}: metric stable within "
                    f"+-{stability_tol * 100:.1f}% (delta_val_acc={acc_delta * 100:.2f}%, "
                    f"delta_val_loss_rel={loss_rel_delta * 100:.2f}% for {stable_epochs} "
                    f"consecutive epochs)"
                )
                break
        prev_val_acc = val_acc
        prev_val_loss = val_loss

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train] restoring best checkpoint from global epoch {best_epoch}")

    # ---- test evaluation -------------------------------------------------
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(CLASSES)
    }
    binary_true = (y_true != HEALTHY_INDEX).astype(int)
    binary_pred = (y_pred != HEALTHY_INDEX).astype(int)
    metrics: dict[str, Any] = {
        "dataset": split_meta,
        "classes": CLASSES,
        "device": str(device),
        "pretrained_imagenet": bool(model.pretrained_loaded),
        "hyperparameters": {
            "epochs_head": head_epochs,
            "epochs_finetune": finetune_epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
            "loss": "CrossEntropyLoss(weight=inverse_frequency)",
            "unfrozen_blocks": unfrozen,
        },
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float(
                precision_recall_fscore_support(
                    y_true, y_pred, average="weighted", zero_division=0
                )[2]
            ),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=range(len(CLASSES))
            ).tolist(),
        },
        "binary_healthy_vs_disease": {
            "accuracy": float(accuracy_score(binary_true, binary_pred)),
            "macro_f1": float(
                precision_recall_fscore_support(
                    binary_true, binary_pred, average="macro", zero_division=0
                )[2]
            ),
        },
        "history": history,
    }

    print("\n=== TEST METRICS (MobileNetV2, 5-class) ===")
    print(f"accuracy        : {metrics['test']['accuracy']:.4f}")
    print(f"macro precision : {metrics['test']['macro_precision']:.4f}")
    print(f"macro recall    : {metrics['test']['macro_recall']:.4f}")
    print(f"macro F1        : {metrics['test']['macro_f1']:.4f}")
    print(f"weighted F1     : {metrics['test']['weighted_f1']:.4f}")
    print("\nPer-class metrics:")
    print(f"{'class':<12}{'precision':>10}{'recall':>9}{'f1':>9}{'support':>9}")
    for name, values in per_class.items():
        print(
            f"{name:<12}{values['precision']:>10.4f}{values['recall']:>9.4f}"
            f"{values['f1']:>9.4f}{values['support']:>9,}"
        )

    if dry_run:
        peak_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            if device.type == "cuda" else 0.0
        )
        print(
            f"[dry-run] complete: no checkpoint/plots overwritten | "
            f"device={device} gpu_peak_mem={peak_mb:.2f}MB"
        )
        return metrics

    plot_confusion_matrix(
        np.asarray(metrics["test"]["confusion_matrix"]),
        CLASSES,
        PLOTS_DIR / "confusion_matrix_cnn.png",
    )
    plot_training_curves(history, PLOTS_DIR / "cnn_training_curves.png")
    # Spec-named alias; the historical filename is kept for compatibility.
    plot_training_curves(history, PLOTS_DIR / "training_curves_cnn.png")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": CLASSES,
            "arch": "mobilenet_v2",
            "img_size": IMG_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "best_epoch": best_epoch,
            "test_accuracy": metrics["test"]["accuracy"],
        },
        MODEL_PATH,
    )
    print(f"[save] {MODEL_PATH}")
    save_json(metrics, METRICS_PATH)
    return metrics


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    """Save an annotated confusion-matrix figure."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Greens")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="MobileNetV2 paddy-leaf confusion matrix (test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j]):,}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    """Save loss/accuracy curves across both training phases."""
    phase1_epochs = max(
        (row["epoch"] for row in history if row["phase"] == 1), default=0
    )
    epochs = [
        row["epoch"] + (0 if row["phase"] == 1 else phase1_epochs)
        for row in history
    ]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_acc.plot(epochs, [r["train_acc"] for r in history], label="train")
    ax_acc.plot(epochs, [r["val_acc"] for r in history], label="val")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 on the paddy-leaf stress dataset."
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help=f"Dataset root (default: {DEFAULT_IMAGE_DIR}, then Kaggle cache)",
    )
    parser.add_argument("--epochs_head", type=int, default=HEAD_EPOCHS)
    parser.add_argument("--epochs_ft", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0 if os.name == "nt" else 4,
        help="DataLoader workers (0 is safest on Windows)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Stratified subset size for quick testing",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without overwriting saved checkpoints/plots",
    )
    parser.add_argument(
        "--stability_tol",
        type=float,
        default=STABILITY_TOL,
        help="Early exit when val metric changes <= tol (default 0.01 = ±1%%)",
    )
    parser.add_argument(
        "--stability_patience",
        type=int,
        default=STABILITY_PATIENCE,
        help="Consecutive stable epochs within ±tol before early exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = parse_args(argv)
    try:
        image_dir = resolve_image_dir(args.image_dir)
        train(
            image_dir=image_dir,
            head_epochs=args.epochs_head,
            finetune_epochs=args.epochs_ft,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            dry_run=args.dry_run,
            stability_tol=args.stability_tol,
            stability_patience=args.stability_patience,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
