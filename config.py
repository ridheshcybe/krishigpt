"""Compatibility wrapper for KisanGPT configuration.

Re-exports all constants, class mappings, and modality specs from stage1_config,
then appends the Stage-3A (temporal digital twin) configuration.  Stage 3A
requires "reuse config.py - extend, don't duplicate", so the trajectory knobs
live here next to the Stage-1 taxonomy instead of in a second config module.
"""

from __future__ import annotations

from stage1_config import *

# ===========================================================================
# Stage 3A - temporal digital twin (trajectory pipeline)
# ===========================================================================
# Nothing above this line is modified: Stage 1/2 code keeps its Stage-1 config.

# --- horizons (days) -------------------------------------------------------
TRAJ_T_HISTORY: int = 21          # observed window (spec: 14-30)
TRAJ_T_FUTURE: int = 14           # forecast horizon (spec: 7-14)
TRAJ_T_TOTAL: int = TRAJ_T_HISTORY + TRAJ_T_FUTURE   # 35 days (spec: 30-60)

#: Input channels: D = 5 (NDVI + 4 weather vars) or D = 1 (NDVI only).
TRAJ_USE_WEATHER: bool = True
TRAJ_WEATHER_VARS: list[str] = ["temperature", "humidity", "rainfall", "soil_moisture"]
TRAJ_INPUT_DIM: int = (1 + len(TRAJ_WEATHER_VARS)) if TRAJ_USE_WEATHER else 1

# --- data ------------------------------------------------------------------
TRAJ_N_FIELDS: int = 600                # synthetic/real fields to build
TRAJ_SPLIT: tuple[float, float, float] = (0.70, 0.15, 0.15)  # by FIELD
#: Real field-level NDVI export, if a Sentinel-2/GEE extract is ever dropped in.
TRAJ_NDVI_CSV = "data/field_ndvi_timeseries.csv"
#: PROXY flags - mirrored into artifacts so reports/UI can never forget them.
TRAJ_NDVI_IS_PROXY: bool = True         # cleared automatically when the CSV loads
TRAJ_WEATHER_IS_PROXY: bool = True      # cleared automatically for open-meteo

# --- encoder (Option A chosen; Option B selectable) ------------------------
TRAJ_ENCODER: str = "transformer"       # "transformer" (preferred) | "bilstm"
TRAJ_D_MODEL: int = 128
TRAJ_NHEAD: int = 4
TRAJ_TRANSFORMER_LAYERS: int = 3        # spec allows 2-4
TRAJ_BILSTM_HIDDEN: int = 128           # Option B
TRAJ_BILSTM_LAYERS: int = 2             # Option B
TRAJ_DROPOUT: float = 0.10
TRAJ_FF_DIM: int = 256                  # transformer feed-forward width

#: Context pooling: "mean" (spec option 1) or "last" (spec option 2).
#: "last" is the default because mean-pooling averages 21 days together and
#: dilutes short-lived triggers (measured: a 3-day soil>58% nutrient episode
#: averaged over 21 days never won the argmax - 1 prediction out of 5,880
#: training days - while long dry spells survived averaging and scored 100%).
TRAJ_POOLING: str = "last"

#: Z-score the input channels using TRAIN-split statistics.  Raw channels span
#: very different ranges (NDVI ~0.5, humidity ~85, soil ~40, rainfall ~5), so
#: an unnormalised projection is badly conditioned and the encoder ends up
#: reading only the largest-magnitude channels.  Stats are fitted on the train
#: fields only and stored in the checkpoint, so val/test never leak.
TRAJ_STANDARDIZE: bool = True

# --- training --------------------------------------------------------------
TRAJ_BATCH_SIZE: int = 32
TRAJ_EPOCHS: int = 60
TRAJ_PATIENCE: int = 12                 # early stopping on val combined loss
STABILITY_TOL: float = 0.01             # +-1% metric stability tolerance for early exit
STABILITY_PATIENCE: int = 2             # consecutive stable epochs within +-1% before early exit
CHECKPOINTS_DIR: str = "artifacts/checkpoints"
TRAJ_LR: float = 1e-4
TRAJ_WEIGHT_DECAY: float = 1e-5
TRAJ_LAMBDA_CE: float = 0.5             # combined loss = MSE + lambda * CE
TRAJ_MAX_NORM: float = 1.0              # gradient clipping
TRAJ_NUM_WORKERS: int = 0               # 0 = Windows-friendly default

#: Inverse-frequency weighting of the per-day CE term.  Healthy is ~73% of all
#: future-window days, and with plain CE the class head collapses to
#: all-healthy (measured: macro F1 0.14, onset/alert F1 0.00).  Weighted CE is
#: still CrossEntropyLoss per time step - it only re-balances the classes - and
#: the weights are fitted on the TRAIN split only (no test leakage).
TRAJ_BALANCE_CLASSES: bool = True

#: Tempering exponent for those weights: ``w = balanced ** ALPHA``.
#: alpha=1.0 is full inverse-frequency weighting, which shifts every argmax
#: toward the minority classes (measured: days 6-13 accuracy fell to 0.04-0.15
#: and every field raised an alert); alpha=0.0 is the collapsing unweighted CE.
#: 0.75 was picked on the VALIDATION split (not test) from
#: {0, 0.5, 0.75, 1.0}: it gave the best macro F1 (0.234) and best onset F1
#: (0.659) while keeping alert precision at 0.88; alpha=1.0 scored higher on
#: alert F1 only by firing on every field (precision 0.63 = the base rate).
TRAJ_CLASS_WEIGHT_ALPHA: float = 0.75

# --- alerts (post-processing) ----------------------------------------------
TRAJ_ALERT_HORIZON: int = 7             # look for onset within the next 7 days
TRAJ_ALERT_MIN_CONF: float = 0.0        # spec default: report prob, no filter
TRAJ_BAND: float = 0.05                 # +/- uncertainty band when no variance head

# --- artifact paths --------------------------------------------------------
TRAJ_DATA_NPZ = "artifacts/trajectory_data.npz"
TRAJ_SPLIT_JSON = "artifacts/trajectory_split.json"
TRAJ_CKPT = "artifacts/trajectory_model.pth"
TRAJ_HISTORY_JSON = "artifacts/trajectory_history.json"
TRAJ_METRICS_JSON = "artifacts/trajectory_metrics.json"
