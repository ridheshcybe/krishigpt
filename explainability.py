"""Stage 1C / Explainability module for KisanGPT unified crop stress diagnosis.

Provides multi-modal explainability utilities:
1. Grad-CAM for the MobileNetV2 RGB image branch.
2. Gradient x Input attribution for tabular sensor features.
3. Attention weight extraction and normalization across modalities.
4. Overlay generation and visualization helpers.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    tabular: torch.Tensor,
    ndvi: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Run a forward + backward pass to extract Grad-CAM and tabular attributions.

    Args:
        model: Unified model (AttentionFusion).
        images: Image tensor (B, 3, 224, 224).
        tabular: Tabular sensor tensor (B, tabular_dim).
        ndvi: NDVI time series tensor (B, T).
        device: Torch device (CPU or CUDA).

    Returns:
        dict containing:
            - pred: Class predictions (B,)
            - confidence: Class probabilities (B,)
            - attention: Modality attention weights (B, 3)
            - contrib: Tabular feature gradient x input attributions (B, tabular_dim)
            - tabular_values: Raw tabular input values (B, tabular_dim)
            - cams: List of spatial Grad-CAM activation maps (B, H, W)
            - probs: Full probability distribution over all classes (B, N_CLASSES)
    """
    model.eval()
    tab = tabular.clone().detach().to(device).requires_grad_(True)
    imgs = images.to(device)
    ts_ndvi = ndvi.to(device)

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logits, attention = model(imgs, tab, ts_ndvi)
        probs = F.softmax(logits.detach(), dim=1)
        pred = probs.argmax(dim=1)
        confidence = probs.gather(1, pred[:, None]).squeeze(1)
        scores = logits.gather(1, pred[:, None]).squeeze(1)

        activations = model.image_encoder.activations
        grad_tab, grad_acts = torch.autograd.grad(
            scores.sum(), [tab, activations], retain_graph=False, allow_unused=True
        )

    contrib = (grad_tab * tab).detach().cpu().numpy()
    tabular_values = tab.detach().cpu().numpy()

    cams: list[np.ndarray] = []
    if grad_acts is not None and activations is not None:
        weights = grad_acts.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()
        cams = [cam[i] for i in range(cam.shape[0])]
    else:
        cams = [np.zeros((7, 7), dtype=np.float32) for _ in range(logits.size(0))]

    return {
        "pred": pred.detach().cpu().numpy(),
        "confidence": confidence.detach().cpu().numpy(),
        "attention": attention.detach().cpu().numpy(),
        "contrib": contrib,
        "tabular_values": tabular_values,
        "cams": cams,
        "probs": probs.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
    }


def overlay_cam_on_image(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a 2D Grad-CAM heatmap with a PIL RGB image.

    Args:
        image: Original PIL Image.
        cam: 2D numpy array of activations.
        alpha: Heatmap blend opacity in [0, 1].
        colormap: Matplotlib colormap name.

    Returns:
        Blended PIL RGB Image resized to image dimensions.
    """
    w, h = image.size
    cam_arr = np.asarray(cam, dtype=np.float32)
    # Normalize between 0 and 1
    c_min, c_max = float(cam_arr.min()), float(cam_arr.max())
    if c_max > c_min:
        cam_norm = (cam_arr - c_min) / (c_max - c_min)
    else:
        cam_norm = np.zeros_like(cam_arr)

    # Resize to original image resolution
    cam_pil = Image.fromarray((cam_norm * 255.0).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BILINEAR
    )
    cam_resized = np.asarray(cam_pil, dtype=np.float32) / 255.0

    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam_resized)[..., :3]  # (H, W, 3) in [0, 1]

    img_arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    overlay = np.clip((1.0 - alpha) * img_arr + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray((overlay * 255.0).astype(np.uint8))


def generate_heatmap_image(
    cam: np.ndarray,
    size: tuple[int, int] = (224, 224),
    colormap: str = "jet",
) -> Image.Image:
    """Generate a standalone colormapped heatmap image from a Grad-CAM array."""
    cam_arr = np.asarray(cam, dtype=np.float32)
    c_min, c_max = float(cam_arr.min()), float(cam_arr.max())
    if c_max > c_min:
        cam_norm = (cam_arr - c_min) / (c_max - c_min)
    else:
        cam_norm = np.zeros_like(cam_arr)

    cam_pil = Image.fromarray((cam_norm * 255.0).astype(np.uint8)).resize(
        size, Image.Resampling.BILINEAR
    )
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(np.asarray(cam_pil, dtype=np.float32) / 255.0)[..., :3]
    return Image.fromarray((heatmap * 255.0).astype(np.uint8))
