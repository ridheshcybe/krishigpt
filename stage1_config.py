"""Shared Stage-1 configuration - single source of truth for the taxonomy.

Stage 1 unifies three modalities (tabular sensors, RGB leaf photos, NDVI time
series) under ONE 6-class label space.  Every Stage-1 module must import the
class order from here instead of re-declaring it, because index order is
exactly the kind of detail that silently corrupts a multi-modal pipeline.

Unified taxonomy (CLASS_MAP)
----------------------------
    0 healthy       1 water_stress   2 nutrient_stress
    3 disease_stress  4 pest_stress  5 stress_generic

NOTE ON ORDER: this differs from the Stage-0 rule-based fusion vector
(``fusion_inference.UNIFIED_CLASSES`` = healthy, nutrient, water, disease,
pest, generic) - water and nutrient swap places.  The Stage-1A spec fixes the
order above for the new pipeline; ``unified_index_from_stage0()`` converts
Stage-0 indices when the two meet (Stage 1B wires ``attention_fusion.py`` onto
this config).

Class sources
-------------
* tabular (CHES rules)   -> healthy, water_stress, nutrient_stress,
                            stress_generic
* images (5 -> 2 collapse)-> healthy, disease_stress
* pest_stress            -> RESERVED, no data source yet (untrained class;
                            documented in PAIRING_STRATEGY.md and the report)
"""

from __future__ import annotations

import numpy as np

# --- taxonomy --------------------------------------------------------------
CLASS_MAP: dict[int, str] = {
    0: "healthy",
    1: "water_stress",
    2: "nutrient_stress",
    3: "disease_stress",
    4: "pest_stress",
    5: "stress_generic",
}
UNIFIED_CLASSES: list[str] = [CLASS_MAP[i] for i in range(len(CLASS_MAP))]
N_CLASSES: int = len(UNIFIED_CLASSES)
UNIFIED_INDEX: dict[str, int] = {name: i for i, name in CLASS_MAP.items()}

# Classes that can appear in each modality (documented in the report).
TABULAR_CLASSES: list[str] = ["healthy", "water_stress", "nutrient_stress", "stress_generic"]
IMAGE_CLASSES: list[str] = ["BLB", "blast", "healthy", "hispa", "leaf_spot"]  # bacterial_leaf_blight, ...
RESERVED_CLASSES: list[str] = ["pest_stress"]

# --- modalities ------------------------------------------------------------
SEED = 42
NDVI_STEPS = 20                # spec allows 10-30 time steps
NDVI_IS_PROXY: bool = True     # PROXY flag: no real paired Sentinel-2 series
                               # exists for these samples (see dataset.py)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TABULAR_DIM = 27               # 25 scaled group features + 2 ordinal encodings
RAW_COLUMN_COUNT = 32          # raw CSV width (label + annotations excluded
                               # from TABULAR_DIM - see dataset.py docstring)

# --- Stage-0 index conversions --------------------------------------------
#: fusion_inference.UNIFIED_CLASSES order -> this config's order.
#: Stage-0: healthy 0, nutrient 1, water 2, disease 3, pest 4, generic 5
#: Stage-1: healthy 0, water 1, nutrient 2, disease 3, pest 4, generic 5
_STAGE0_TO_STAGE1 = np.asarray([0, 2, 1, 3, 4, 5], dtype=np.int64)


def unified_index_from_stage0(y: np.ndarray) -> np.ndarray:
    """Convert labels from the Stage-0 unified order to the Stage-1 order."""
    y = np.asarray(y, dtype=np.int64)
    return _STAGE0_TO_STAGE1[y]


def tabular_label_to_unified(y_tabular: np.ndarray) -> np.ndarray:
    """Map the tabular 4-class labels (tabular_xgb.CLASS_ORDER) to unified 6.

    ``tabular_xgb.CLASS_ORDER = [healthy, nutrient_stress, stress_generic,
    water_stress]`` -> ``[0, 2, 5, 1]`` in the unified space.
    """
    mapping = {
        0: UNIFIED_INDEX["healthy"],
        1: UNIFIED_INDEX["nutrient_stress"],
        2: UNIFIED_INDEX["stress_generic"],
        3: UNIFIED_INDEX["water_stress"],
    }
    y = np.asarray(y_tabular, dtype=np.int64)
    out = np.empty_like(y)
    for source, target in mapping.items():
        out[y == source] = target
    return out


def image_label_to_unified(y_image: np.ndarray) -> np.ndarray:
    """Map the 5-class image labels to unified 6 (collapse to healthy/disease).

    ``mobilenet_cnn.CLASSES = [BLB, blast, healthy, hispa, leaf_spot]`` ->
    healthy stays healthy, the four disease subtypes collapse to
    ``disease_stress``.
    """
    mapping = {
        0: UNIFIED_INDEX["disease_stress"],   # BLB
        1: UNIFIED_INDEX["disease_stress"],   # blast
        2: UNIFIED_INDEX["healthy"],
        3: UNIFIED_INDEX["disease_stress"],   # hispa
        4: UNIFIED_INDEX["disease_stress"],   # leaf_spot
    }
    y = np.asarray(y_image, dtype=np.int64)
    out = np.empty_like(y)
    for source, target in mapping.items():
        out[y == source] = target
    return out
