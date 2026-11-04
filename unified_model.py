"""Stage 1B Unified Multi-Modal Crop Stress Model module.

Exposes the AttentionFusion architecture and model loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch

from attention_fusion import (
    AttentionFusion,
    ImageEncoder,
    TabularEncoder,
    NdviEncoder,
    CKPT_PATH as ATTENTION_FUSION_PATH,
)
import diagnosis_cli

#: Canonical aliases
UnifiedModel = AttentionFusion
NDVIEncoder = NdviEncoder

#: Primary checkpoint path
UNIFIED_MODEL_PATH = Path("artifacts/unified_model.pth")
FALLBACK_MODEL_PATH = Path("artifacts/attention_fusion.pt")


def load_model(
    checkpoint_path: Path | str | None = None,
    device: torch.device | None = None,
    tabular_dim: int = 27,
) -> AttentionFusion:
    """Load unified model checkpoint (either unified_model.pth or attention_fusion.pt)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        if UNIFIED_MODEL_PATH.exists():
            checkpoint_path = UNIFIED_MODEL_PATH
        elif FALLBACK_MODEL_PATH.exists():
            checkpoint_path = FALLBACK_MODEL_PATH
        else:
            checkpoint_path = UNIFIED_MODEL_PATH

    path = Path(checkpoint_path)
    return diagnosis_cli.load_unified_model(path, device, tabular_dim=tabular_dim)
