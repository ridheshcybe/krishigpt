"""Stage 3A - field-level NDVI/weather trajectory dataset (the digital twin's data).

Builds ``(N_fields, T_total)`` day-level stress labels, an NDVI series per field
and a ``(N_fields, T_total, 4)`` weather stack, then slices them into
(history window -> future horizon) pairs for ``trajectory_model.py``.

CAUSAL ORDER (this is what makes the task *predictable*)
---------------------------------------------------------
    weather drivers  ->  stress labels  ->  NDVI curve

1. Weather first (``generate_weather``): autocorrelated daily temperature,
   rainfall, soil moisture and humidity - Markov dry/wet spells, an
   Ornstein-Uhlenbeck temperature with occasional heat waves, a lagged
   water-balance soil moisture.  Because the drivers are autocorrelated, a dry
   or humid spell that has begun is still there tomorrow: that persistence is
   what a forecaster can learn, and it is what turns "water_stress onset
   predicted in X days" into a real statement.

2. Labels next (``derive_labels``): documented agronomic thresholds with
   enter/exit hysteresis, applied in priority order:

       water_stress    soil moisture < 18%   (exit > 24% for 2 days)
       disease_stress  humidity > 80% for 3+ days at 22-34C (exit < 72%)
       nutrient_stress soil moisture > 58% for 3+ days - waterlogging leaches
                       nutrients (exit < 50%)
       pest_stress     rare 4-7 day burst (reserved class, deliberately rare)
       stress_generic  heat wave > 34C for 2+ days (exit < 32C)

   The enter windows (2-3 days) and exit windows (2-3 days) create the
   autocorrelation a sequence model needs; thresholds mirror the
   ``favorable_conditions`` ranges in Stage-2A's knowledge_base.json.

3. NDVI last (``render_proxy_ndvi``): a class-conditional PROXY curve rendered
   *from* the labels, so "labels derived from NDVI trajectory patterns" holds
   by construction and ``decode_labels_from_ndvi`` can re-derive them.

DATA SOURCES (declared, no local collection)
--------------------------------------------
NDVI  - **PROXY by default**.  A real Sentinel-2 / Google Earth Engine export
        can be dropped at ``data/field_ndvi_timeseries.csv`` (one row per
        field, one numeric column per day); if it loads, the PROXY flag clears
        automatically.  Otherwise class-conditional curves over 30-60 days are
        generated and flagged PROXY in three places: ``cfg.TRAJ_NDVI_IS_PROXY``,
        the stats file and this docstring (mirroring Stage-1A's
        ``dataset.NDVI_IS_PROXY`` convention).

        healthy          stable 0.66-0.78
        water_stress     0.60 -> 0.30 decline
        nutrient_stress  0.70 -> 0.40 gradual decline
        disease_stress   baseline with a sudden lesion-like dip at onset
        pest_stress      irregular spikes/dips
        stress_generic   noisy moderate decline 0.60 -> 0.45

Weather- **PROXY by default** (temperature, humidity, rainfall, soil_moisture).
        ``--weather-source open-meteo`` fetches real daily history over the
        free Open-Meteo API (stdlib HTTP, no key); labels are then derived from
        the *real* weather by the same documented rules.  Any network/shape
        failure falls back to PROXY with a logged warning, never a crash.
        Flag lands in ``cfg.TRAJ_WEATHER_IS_PROXY``.

LEAKAGE CONTROL
---------------
``field_split()`` splits 70/15/15 **by field**, never by time step, so no
field's history window can appear in two splits.  ``--validate-only`` asserts
the three index sets are disjoint.

CLI
---
    python trajectory_data.py --validate-only
    python trajectory_data.py --n-fields 600 --weather-source proxy
    python trajectory_data.py --weather-source open-meteo   # real, falls back
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

import config as cfg
from stage1_config import N_CLASSES, SEED, UNIFIED_CLASSES, UNIFIED_INDEX

# ---------------------------------------------------------------------------
# Thresholds for the label state machine (documented, agronomic).
# Enter windows create persistence; exit windows are wider than enter windows
# so a state does not flicker on for a single day (hysteresis).
# ---------------------------------------------------------------------------
WATER_ENTER, WATER_EXIT = 18.0, 24.0          # soil moisture %
NUTRIENT_ENTER, NUTRIENT_EXIT = 58.0, 50.0    # soil moisture % (waterlogging)
DISEASE_HUM_ENTER, DISEASE_HUM_EXIT = 80.0, 72.0
DISEASE_T_LO, DISEASE_T_HI = 22.0, 34.0
GENERIC_TEMP_ENTER, GENERIC_TEMP_EXIT = 34.0, 32.0
ENTER_DAYS = {1: 2, 2: 3, 3: 3, 4: 4, 5: 2}    # days sustained to enter a state
EXIT_DAYS = {1: 2, 2: 2, 3: 2, 4: 3, 5: 2}     # days sustained to leave it
PEST_BURST_P = 0.12                           # fraction of fields with a burst
HEATWAVE_P = 0.25                             # fraction of fields with a heat wave

#: Priority when several triggers are true on the same day.
PRIORITY = [
    UNIFIED_INDEX["water_stress"],
    UNIFIED_INDEX["disease_stress"],
    UNIFIED_INDEX["nutrient_stress"],
    UNIFIED_INDEX["pest_stress"],
    UNIFIED_INDEX["stress_generic"],
]

#: Tolerance when a real export has fewer columns than we need.
MIN_DAYS = cfg.TRAJ_T_HISTORY + cfg.TRAJ_T_FUTURE


def log(message: str) -> None:
    """Print a timestamped progress line (Stage-3A tag, matches Stage 1/2)."""
    print(f"[stage3a {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Weather drivers (PROXY by default)
# ---------------------------------------------------------------------------
def generate_weather(n_fields: int, t_total: int, seed: int = SEED) -> np.ndarray:
    """Build the declared-PROXY ``(N, T, 4)`` daily weather stack.

    Columns: temperature (C), humidity (%), rainfall (mm/day), soil moisture (%).

    The series are deliberately **autocorrelated** - Markov dry/wet spells, an
    OU temperature with injected heat waves, a lagged water balance and
    rain-coupled humidity - because a forecaster can only predict stress that
    is still building when the forecast is issued.  Deterministic per field via
    ``default_rng([seed, 4000 + field_id])``.
    """
    weather = np.zeros((n_fields, t_total, 4), dtype=np.float32)

    for i in tqdm(range(n_fields), desc="weather (proxy)", leave=False):
        rng = np.random.default_rng([seed, 4000 + i])

        # --- temperature: OU process around 27C + optional heat wave --------
        temp = np.empty(t_total, dtype=np.float64)
        temp[0] = 27.0 + rng.normal(0.0, 2.0)
        for d in range(1, t_total):
            temp[d] = temp[d - 1] + 0.25 * (27.0 - temp[d - 1]) + rng.normal(0.0, 1.4)
        if rng.random() < HEATWAVE_P:
            start = int(rng.integers(0, max(1, t_total - 6)))
            length = int(rng.integers(4, 7))
            bump = float(rng.uniform(7.0, 10.0))
            temp[start : start + length] += bump

        # --- rainfall: Markov dry/wet spells -------------------------------
        rain = np.zeros(t_total, dtype=np.float64)
        wet = bool(rng.random() < 0.35)
        for d in range(t_total):
            if wet:
                rain[d] = float(rng.exponential(9.0))
                if rng.random() < 0.25:      # mean wet spell ~4 days
                    wet = False
            elif rng.random() < 0.08:        # mean dry spell ~12 days
                wet = True

        # --- soil moisture: water balance (rain in, evaporation + drainage out)
        # Oscillates around 30-55: a dry spell of ~a week crosses 18% (water
        # stress), sustained heavy rain crosses 56% (waterlogging).  Heat waves
        # raise evaporation, which is why drought and heat co-occur.
        soil = np.empty(t_total, dtype=np.float64)
        soil[0] = float(rng.uniform(35.0, 55.0))
        for d in range(1, t_total):
            evap = 1.6 + 0.16 * max(0.0, temp[d] - 25.0)
            drainage = 0.30 * max(0.0, soil[d - 1] - 55.0)
            recharge = 0.03 * (40.0 - soil[d - 1])   # capillary rise / deeper reserve
            soil[d] = soil[d - 1] + 0.85 * rain[d] - evap - drainage + recharge
            soil[d] += rng.normal(0.0, 0.6)
            soil[d] = min(max(soil[d], 3.0), 75.0)

        # --- humidity: AR(1) spells - wet days stay humid for days ---------
        humid = np.empty(t_total, dtype=np.float64)
        level = 55.0 + rng.normal(0.0, 5.0)
        for d in range(t_total):
            base = 56.0 - 1.4 * (temp[d] - 27.0) + 34.0 * (rain[d] > 0.5)
            level = level + 0.65 * (base - level) + rng.normal(0.0, 2.5)
            humid[d] = float(np.clip(level, 8.0, 99.0))

        weather[i, :, 0] = temp
        weather[i, :, 1] = humid
        weather[i, :, 2] = rain
        weather[i, :, 3] = soil

    return weather


def fetch_open_meteo(t_total: int, lat: float = 26.85, lon: float = 80.95) -> np.ndarray | None:
    """Fetch ``t_total`` real daily weather days from Open-Meteo (no API key).

    Returns ``(t_total, 4)`` float32 for (temperature, humidity, rainfall,
    soil_moisture) aggregated from hourly history, or ``None`` on any network,
    quota or schema problem - callers then fall back to PROXY.
    """
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=t_total + 7)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,soil_moisture_0_to_1cm"
        "&timezone=auto"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # stdlib only
            payload = json.loads(resp.read().decode("utf-8"))
        hourly = payload["hourly"]
        frame = pd.DataFrame(
            {
                "temperature": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
                "humidity": pd.to_numeric(hourly["relative_humidity_2m"], errors="coerce"),
                "rainfall": pd.to_numeric(hourly["precipitation"], errors="coerce"),
                "soil_moisture": pd.to_numeric(hourly["soil_moisture_0_to_1cm"], errors="coerce"),
            }
        )
        frame.index = pd.to_datetime(hourly["time"])
        daily = pd.DataFrame(
            {
                "temperature": frame["temperature"].resample("D").mean(),
                "humidity": frame["humidity"].resample("D").mean(),
                "rainfall": frame["rainfall"].resample("D").sum(),
                "soil_moisture": frame["soil_moisture"].resample("D").mean(),
            }
        ).dropna()
        if len(daily) < t_total:
            log(f"WARNING: Open-Meteo returned {len(daily)} days < {t_total}; using PROXY")
            return None

        # Convert volumetric soil moisture (m^3/m^3, e.g. 0.08 to 0.45) to percentage (8.0% to 45.0%)
        # to match the agronomic state machine thresholds (WATER_ENTER=18.0%, NUTRIENT_ENTER=58.0%).
        if daily["soil_moisture"].max() <= 1.5:
            daily["soil_moisture"] = daily["soil_moisture"] * 100.0

        return daily.tail(t_total).to_numpy(np.float32)
    except (urllib.error.URLError, KeyError, ValueError, OSError, TimeoutError) as err:
        log(f"WARNING: Open-Meteo fetch failed ({err}); falling back to PROXY weather")
        return None


def build_weather(
    labels: np.ndarray | None,
    n_fields: int,
    t_total: int,
    source: str = "proxy",
    seed: int = SEED,
) -> tuple[np.ndarray, bool]:
    """Build ``(N, T, 4)`` weather.  Returns ``(weather, is_proxy)``.

    ``source``: ``proxy`` (declared synthetic, deterministic), ``open-meteo``
    (real, falls back to proxy on failure) or ``auto`` (open-meteo first).
    """
    real: np.ndarray | None = None
    if source in ("open-meteo", "auto"):
        log(f"requesting real daily weather from Open-Meteo (source={source})")
        real = fetch_open_meteo(t_total)
    if real is not None:
        # Real Open-Meteo macro-climate enriched with realistic per-field
        # soil drainage and micro-topographic variability (deterministic per field seed).
        fields_weather = np.empty((n_fields, t_total, 4), dtype=np.float32)
        for i in range(n_fields):
            rng = np.random.default_rng([seed, 7000 + i])
            t_offset = float(rng.normal(0.0, 0.8))       # Local temp lapse/exposure +-1.5C
            h_offset = float(rng.normal(0.0, 3.0))       # Local relative humidity +-5%
            s_factor = float(rng.uniform(0.85, 1.15))    # Soil drainage/texture factor
            
            w = real.copy()
            w[:, 0] = np.clip(w[:, 0] + t_offset, 5.0, 50.0)
            w[:, 1] = np.clip(w[:, 1] + h_offset, 15.0, 100.0)
            w[:, 3] = np.clip(w[:, 3] * s_factor, 2.0, 80.0)
            fields_weather[i] = w
        return fields_weather, False

    weather = generate_weather(n_fields, t_total, seed=seed)
    return weather, True


# ---------------------------------------------------------------------------
# 2. Labels - derived from the weather drivers (hysteresis state machine)
# ---------------------------------------------------------------------------
def _sustained(mask: np.ndarray, days: int) -> np.ndarray:
    """True where the last ``days`` entries are all True (padded with False)."""
    out = np.zeros(len(mask), dtype=bool)
    for d in range(days - 1, len(mask)):
        out[d] = bool(mask[d - days + 1 : d + 1].all())
    return out


def derive_labels(weather: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Derive day-level stress labels from the weather drivers.

    A per-field state machine: a state is entered only once its trigger has
    held for ``ENTER_DAYS`` consecutive days and it is left only once its exit
    condition has held for ``EXIT_DAYS`` days (hysteresis).  When several
    triggers are live, ``PRIORITY`` decides: water > disease > nutrient >
    pest > generic.

    Returns ``(N, T_total)`` int64 with ``0 = healthy``.
    """
    n_fields, t_total = weather.shape[0], weather.shape[1]
    labels = np.zeros((n_fields, t_total), dtype=np.int64)

    for i in tqdm(range(n_fields), desc="labels (rules)", leave=False):
        rng = np.random.default_rng([seed, 6000 + i])
        temp, humid, rain, soil = (weather[i, :, k] for k in range(4))

        enter = {
            1: _sustained(soil < WATER_ENTER, ENTER_DAYS[1]),
            2: _sustained(
                (humid > DISEASE_HUM_ENTER) & (temp >= DISEASE_T_LO) & (temp <= DISEASE_T_HI),
                ENTER_DAYS[2],
            ),
            3: _sustained(soil > NUTRIENT_ENTER, ENTER_DAYS[3]),
            4: np.zeros(t_total, dtype=bool),   # pest burst, filled below
            5: _sustained(temp > GENERIC_TEMP_ENTER, ENTER_DAYS[5]),
        }
        exit_ok = {
            1: _sustained(soil > WATER_EXIT, EXIT_DAYS[1]),
            2: _sustained(humid < DISEASE_HUM_EXIT, EXIT_DAYS[2]),
            3: _sustained(soil < NUTRIENT_EXIT, EXIT_DAYS[3]),
            4: np.zeros(t_total, dtype=bool),
            5: _sustained(temp < GENERIC_TEMP_EXIT, EXIT_DAYS[5]),
        }

        # Reserved class: rare, short pest burst on a minority of fields.
        if rng.random() < PEST_BURST_P:
            start = int(rng.integers(0, max(1, t_total - 7)))
            length = int(rng.integers(4, 8))
            enter[4][start : start + length] = True
            exit_ok[4][start + length :] = True

        state = 0
        for d in range(t_total):
            if state != 0:
                # _sustained already requires EXIT_DAYS[state] consecutive days.
                # Single check here eliminates the previous double-counting of exit duration.
                if exit_ok[state][d]:
                    state = 0
            else:
                for candidate in PRIORITY:
                    if enter[candidate][d]:
                        state = candidate
                        break
            labels[i, d] = state

    return labels


# ---------------------------------------------------------------------------
# 3. NDVI - real export if present, otherwise declared PROXY curves
# ---------------------------------------------------------------------------
def load_real_ndvi(n_fields: int, t_total: int) -> np.ndarray | None:
    """Return a real ``(N, T)`` NDVI matrix from a field-level export, else None."""
    path = Path(cfg.TRAJ_NDVI_CSV)
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path)
    except Exception as err:  # pragma: no cover - corrupt file
        log(f"WARNING: could not read {path} ({err}); using PROXY curves")
        return None
    numeric = table.select_dtypes(include=[np.number])
    if len(table) != n_fields or numeric.shape[1] < t_total:
        log(
            f"WARNING: {path} is {table.shape} but need ({n_fields}, {t_total}); "
            "using PROXY curves"
        )
        return None
    log(f"real field-level NDVI loaded from {path} -> PROXY flag cleared")
    return numeric.to_numpy(np.float32)[:, :t_total]


def _runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    """Split one label row into ``[(start, end_exclusive, class), ...]`` runs."""
    out: list[tuple[int, int, int]] = []
    start = 0
    for j in range(1, len(row)):
        if row[j] != row[j - 1]:
            out.append((start, j, int(row[start])))
            start = j
    out.append((start, len(row), int(row[start])))
    return out


def render_proxy_ndvi(labels: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Render class-conditional PROXY NDVI curves causally per day.

    PROXY - prototyping only. Causal daily update:
    - On day d, NDVI depends strictly on current state and elapsed days in that state,
      never on future episode length or future end boundaries.
    - Smoothing is trailing-only (_smooth_causal), ensuring input day 20 does NOT
      leak tomorrow's (day 21) onset or NDVI values.
    - Deterministic per field and day via seeded RNG.
    """
    n_fields, t_total = labels.shape
    curves = np.zeros((n_fields, t_total), dtype=np.float32)

    for i in tqdm(range(n_fields), desc="ndvi (proxy)", leave=False):
        base_rng = np.random.default_rng([seed, 7000 + i])
        base = float(base_rng.uniform(0.66, 0.78))
        curve = np.empty(t_total, dtype=np.float64)

        current_state = int(labels[i, 0])
        state_start = 0

        for d in range(t_total):
            cls = int(labels[i, d])
            if cls != current_state:
                current_state = cls
                state_start = d
            k = d - state_start  # days elapsed in current state

            if cls == UNIFIED_INDEX["healthy"]:
                val = base
            elif cls == UNIFIED_INDEX["water_stress"]:
                val = max(0.30, 0.60 - 0.025 * k)
            elif cls == UNIFIED_INDEX["nutrient_stress"]:
                val = max(0.40, 0.70 - 0.015 * k)
            elif cls == UNIFIED_INDEX["disease_stress"]:
                if k == 0:
                    val = 0.38
                elif k == 1:
                    val = 0.44
                else:
                    val = min(0.57, 0.52 + 0.005 * (k - 2))
            elif cls == UNIFIED_INDEX["pest_stress"]:
                pest_rng = np.random.default_rng([seed, 8000 + i, d])
                val = 0.58 + float(pest_rng.normal(0.0, 0.05))
            else:  # stress_generic
                val = max(0.45, 0.60 - 0.020 * k)

            # Causal per-day observation noise (strictly indexed by field and day)
            obs_rng = np.random.default_rng([seed, 9000 + i, d])
            val += float(obs_rng.normal(0.0, 0.015))
            curve[d] = val

        # Trailing-only (causal) moving average: values[d] never accesses future days
        smoothed = _smooth_causal(curve, window=3)
        curves[i] = np.clip(smoothed, 0.05, 1.0).astype(np.float32)
    return curves


def _smooth_causal(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Trailing-only (causal) moving average.

    values[d] is smoothed using strictly past and current values:
        out[d] = mean(values[max(0, d - window + 1) : d + 1])
    No future values (values[d + 1 :]) are ever accessed.
    """
    if window <= 1 or values.size < 2:
        return values.copy()
    out = np.empty_like(values)
    for d in range(len(values)):
        start = max(0, d - window + 1)
        out[d] = float(np.mean(values[start : d + 1]))
    return out


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Backward-compatible alias, strictly executing trailing-only causal smoothing."""
    return _smooth_causal(values, window=window)


def decode_labels_from_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """Rule-based decoder: re-derive day-level classes from NDVI alone.

    Documented thresholds on the 5-day level, 5-day slope, short-window
    volatility and single-day drop.  Order matters: healthy is decided first
    (level > 0.62), then generic, then the decline classes, then pest/disease
    spikes.  Calibrated once against the generated proxy series (agreement
    0.66, per-class recall reported by ``--validate-only``).

    Honest limitation: nutrient and water are both smooth declines offset by
    ~0.10 NDVI, so a rolling window cannot separate them reliably (recall
    ~0.01 for nutrient).  That is precisely *why* labels are derived from the
    weather drivers instead of from NDVI alone - this function is a
    consistency diagnostic, not a model, and never feeds training.
    """
    x = np.atleast_2d(np.asarray(ndvi, dtype=np.float64))
    n, t = x.shape
    level = np.zeros_like(x)
    slope = np.zeros_like(x)
    vol = np.zeros_like(x)

    for d in range(t):
        lo = max(0, d - 4)
        level[:, d] = x[:, lo : d + 1].mean(axis=1)
        if d >= 5:
            slope[:, d] = x[:, d] - x[:, d - 5]
            vol[:, d] = np.std(x[:, d - 4 : d + 1] - x[:, d - 5 : d], axis=1)

    daily = np.zeros_like(x)
    daily[:, 1:] = x[:, 1:] - x[:, :-1]

    out = np.full((n, t), UNIFIED_INDEX["healthy"], dtype=np.int64)
    out[level <= 0.60] = UNIFIED_INDEX["stress_generic"]
    water = (slope <= -0.05) | ((level <= 0.50) & (slope <= -0.03))
    out[water] = UNIFIED_INDEX["water_stress"]
    nutrient = (
        (out != UNIFIED_INDEX["water_stress"])
        & (level <= 0.58)
        & (slope <= -0.015)
    )
    out[nutrient] = UNIFIED_INDEX["nutrient_stress"]
    out[vol >= 0.05] = UNIFIED_INDEX["pest_stress"]
    out[daily <= -0.10] = UNIFIED_INDEX["disease_stress"]
    return out


# ---------------------------------------------------------------------------
# 4. Dataset cache + field-level split
# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """Slices field series into ``(history window) -> (future targets)`` pairs.

    Item dict:
        ``x``       ``(T_history, D)`` float32 - NDVI (+ weather if enabled)
        ``y_ndvi``  ``(T_future,)`` float32    - future NDVI ground truth
        ``y_cls``    ``(T_future,)`` int64      - future day-level class indices
        ``field_id`` scalar long                - provenance (leakage checks)
    """

    def __init__(
        self,
        ndvi: np.ndarray,
        weather: np.ndarray,
        labels: np.ndarray,
        field_indices: np.ndarray | list[int],
        t_history: int = cfg.TRAJ_T_HISTORY,
        t_future: int = cfg.TRAJ_T_FUTURE,
        use_weather: bool = cfg.TRAJ_USE_WEATHER,
        channel_mean: np.ndarray | list[float] | None = None,
        channel_std: np.ndarray | list[float] | None = None,
        alert_horizon: int = 3,
        class_mapping: dict[int, int] | None = None,
    ) -> None:
        self.ndvi = ndvi
        self.weather = weather
        self.labels = labels
        self.field_indices = np.asarray(field_indices, dtype=np.int64)
        self.t_history = int(t_history)
        self.t_future = int(t_future)
        self.use_weather = bool(use_weather)
        self.alert_horizon = int(alert_horizon)
        self.class_mapping = class_mapping
        #: Optional z-score normalisation (train-split stats, see config).
        self.channel_mean = (
            None if channel_mean is None
            else np.asarray(channel_mean, dtype=np.float32)
        )
        self.channel_std = (
            None if channel_std is None
            else np.asarray(channel_std, dtype=np.float32)
        )
        if t_history + t_future > ndvi.shape[1]:
            raise ValueError(
                f"history {t_history} + future {t_future} exceeds series length {ndvi.shape[1]}"
            )

    def __len__(self) -> int:
        return int(self.field_indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        field = int(self.field_indices[index])
        h, k = self.t_history, self.t_future
        parts = [self.ndvi[field, :h, None]]
        if self.use_weather:
            parts.append(self.weather[field, :h, :])
        x = np.concatenate(parts, axis=1).astype(np.float32)
        if self.channel_mean is not None and self.channel_std is not None:
            x = (x - self.channel_mean) / self.channel_std

        last_ndvi_val = float(self.ndvi[field, h - 1])
        future_ndvi_arr = self.ndvi[field, h : h + k].astype(np.float32)
        delta_ndvi_arr = (future_ndvi_arr - last_ndvi_val).astype(np.float32)
        future_labels_arr = self.labels[field, h : h + k].astype(np.int64)

        if self.class_mapping is not None:
            mapped_cls = np.array([self.class_mapping.get(int(c), -100) for c in future_labels_arr], dtype=np.int64)
        else:
            mapped_cls = future_labels_arr.copy()

        early_horizon = min(self.alert_horizon, k)
        endpoint_label = int(self.labels[field, h - 1])
        is_endpoint_stressed = (endpoint_label != 0)

        # Strict pre-onset early-warning target:
        # A field is positive (1.0) ONLY if:
        # 1. It is healthy (not stressed) at the history endpoint (day h-1).
        # 2. A stress onset occurs strictly after the history endpoint, within early_horizon days.
        # If the field is already stressed at the endpoint, it is masked (-1.0, mask=0.0)
        # because ongoing stress is a current diagnosis, not an advance pre-onset early warning.
        if is_endpoint_stressed:
            early_warn_val = -1.0
            early_warn_mask = 0.0
            days_to_onset = -1
        else:
            early_warn_mask = 1.0
            future_sub = future_labels_arr[:early_horizon]
            has_onset = np.any(future_sub != 0)
            if has_onset:
                first_day_1based = int(np.flatnonzero(future_sub != 0)[0]) + 1  # 1, 2, ..., early_horizon
                early_warn_val = 1.0
                days_to_onset = first_day_1based
            else:
                early_warn_val = 0.0
                days_to_onset = 0

        return {
            "x": torch.from_numpy(x),
            "y_ndvi": torch.from_numpy(future_ndvi_arr),
            "y_cls": torch.from_numpy(mapped_cls),
            "y_cls_raw": torch.from_numpy(future_labels_arr),
            "field_id": torch.tensor(field, dtype=torch.long),
            "last_ndvi": torch.tensor([last_ndvi_val], dtype=torch.float32),
            "delta_ndvi": torch.from_numpy(delta_ndvi_arr),
            "early_warning": torch.tensor(early_warn_val, dtype=torch.float32),
            "early_warning_mask": torch.tensor(early_warn_mask, dtype=torch.float32),
            "days_to_onset": torch.tensor(days_to_onset, dtype=torch.long),
            "is_endpoint_stressed": torch.tensor(1.0 if is_endpoint_stressed else 0.0, dtype=torch.float32),
            "endpoint_label": torch.tensor(endpoint_label, dtype=torch.long),
        }


def build_dataset(
    n_fields: int = cfg.TRAJ_N_FIELDS,
    t_history: int = cfg.TRAJ_T_HISTORY,
    t_future: int = cfg.TRAJ_T_FUTURE,
    weather_source: str = "proxy",
    seed: int = SEED,
    force: bool = False,
    cache: bool = True,
    cache_path: str | Path | None = None,
    read_only: bool = False,
) -> dict[str, Any]:
    """Build (or load the cached) trajectory arrays.

    Order: weather -> labels -> NDVI (see module docstring).  Cached to
    ``artifacts/trajectory_data.npz`` so train/evaluate/dashboard all consume
    byte-identical arrays instead of regenerating independently.
    """
    t_total = t_history + t_future
    resolved_cache_path = Path(cache_path) if cache_path is not None else Path(cfg.TRAJ_DATA_NPZ)

    if cache and not force and resolved_cache_path.exists():
        try:
            with np.load(resolved_cache_path, allow_pickle=True) as blob:
                cached_n_fields = int(blob["meta"][0]) if "meta" in blob else None
                cached_t_hist = int(blob["meta"][1]) if "meta" in blob else None
                cached_t_fut = int(blob["meta"][2]) if "meta" in blob else None
                
                cached_seed = None
                if "seed" in blob:
                    cached_seed = int(blob["seed"])
                elif "meta" in blob and len(blob["meta"]) >= 4:
                    cached_seed = int(blob["meta"][3])

                cached_source = str(blob["weather_source"]) if "weather_source" in blob else None

                # Strict cache validation: dimensions, seed, and weather source must match
                mismatches = []
                if cached_n_fields != n_fields:
                    mismatches.append(f"n_fields ({cached_n_fields} vs {n_fields})")
                if cached_t_hist != t_history:
                    mismatches.append(f"t_history ({cached_t_hist} vs {t_history})")
                if cached_t_fut != t_future:
                    mismatches.append(f"t_future ({cached_t_fut} vs {t_future})")
                if cached_seed is not None and cached_seed != seed:
                    mismatches.append(f"seed ({cached_seed} vs {seed})")
                elif cached_seed is None:
                    mismatches.append("seed (missing in legacy cache)")
                if cached_source is not None and cached_source != weather_source and weather_source != "auto":
                    mismatches.append(f"weather_source ({cached_source} vs {weather_source})")
                elif cached_source is None:
                    mismatches.append("weather_source (missing in legacy cache)")

                if not mismatches:
                    ndvi = blob["ndvi"]
                    weather = blob["weather"]
                    labels = blob["labels"]
                    flags = blob["flags"]
                    log(
                        f"trajectory cache loaded from {resolved_cache_path} "
                        f"(N={n_fields}, T={t_total}, seed={cached_seed}, source={cached_source})"
                    )
                    field_meta = []
                    for i in range(n_fields):
                        events = [
                            {
                                "class": UNIFIED_CLASSES[cls],
                                "class_idx": cls,
                                "onset": int(start),
                                "end": int(end),
                            }
                            for start, end, cls in _runs(labels[i])
                            if cls != 0
                        ]
                        field_meta.append({"field_id": i, "events": events})

                    return {
                        "ndvi": ndvi,
                        "weather": weather,
                        "labels": labels,
                        "ndvi_is_proxy": bool(flags[0]),
                        "weather_is_proxy": bool(flags[1]),
                        "weather_source": str(cached_source),
                        "n_fields": n_fields,
                        "t_history": t_history,
                        "t_future": t_future,
                        "seed": int(cached_seed),
                        "field_meta": field_meta,
                        "loaded_path": str(resolved_cache_path.resolve()),
                    }
                else:
                    msg = f"trajectory cache invalidated ({', '.join(mismatches)})"
                    if read_only:
                        raise ValueError(f"Strict read-only load failed: {msg}")
                    log(f"{msg}; rebuilding")
        except Exception as err:
            if read_only:
                raise
            log(f"WARNING: trajectory cache unreadable ({err}); rebuilding")

    if read_only:
        raise FileNotFoundError(
            f"Strict read-only load failed: dataset cache not found or unreadable at {resolved_cache_path}."
        )

    log(
        f"building {n_fields} field trajectories: T_history={t_history}, "
        f"T_future={t_future} (total {t_total} days), seed={seed}"
    )

    # 1) weather drivers first (they are what makes the future predictable)
    weather, weather_is_proxy = build_weather(
        None, n_fields, t_total, source=weather_source, seed=seed
    )
    log(
        f"PROXY flags: weather={'YES' if weather_is_proxy else 'no'} "
        f"(source={weather_source})"
    )

    # 2) labels derived from the drivers, with hysteresis
    labels = derive_labels(weather, seed=seed)

    # 3) NDVI rendered from the labels (or a real export, if one was supplied)
    real_ndvi = load_real_ndvi(n_fields, t_total)
    ndvi_is_proxy = real_ndvi is None
    if ndvi_is_proxy:
        log("no real NDVI export found -> building PROXY class-conditional curves")
        ndvi = render_proxy_ndvi(labels, seed=seed)
    else:
        ndvi = np.asarray(real_ndvi, dtype=np.float32)
    log(f"PROXY flags: ndvi={'YES' if ndvi_is_proxy else 'no'}")

    counts = np.bincount(labels.ravel(), minlength=N_CLASSES)
    log(
        "class day share: "
        + ", ".join(
            f"{UNIFIED_CLASSES[c]}={counts[c] / max(labels.size, 1):.1%}"
            for c in range(N_CLASSES)
        )
    )

    # Field-level event summary (for the report / dashboards).
    field_meta: list[dict[str, Any]] = []
    for i in range(n_fields):
        events = [
            {
                "class": UNIFIED_CLASSES[cls],
                "class_idx": cls,
                "onset": int(start),
                "end": int(end),
            }
            for start, end, cls in _runs(labels[i])
            if cls != 0
        ]
        field_meta.append({"field_id": i, "events": events})

    data = {
        "ndvi": ndvi,
        "weather": weather,
        "labels": labels,
        "ndvi_is_proxy": ndvi_is_proxy,
        "weather_is_proxy": weather_is_proxy,
        "weather_source": ("proxy" if weather_is_proxy else weather_source),
        "n_fields": n_fields,
        "t_history": t_history,
        "t_future": t_future,
        "seed": seed,
        "field_meta": field_meta,
        "loaded_path": str(resolved_cache_path.resolve()) if cache else None,
    }

    if cache:
        resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resolved_cache_path,
            ndvi=ndvi,
            weather=weather,
            labels=labels,
            flags=np.asarray([ndvi_is_proxy, weather_is_proxy], dtype=np.int8),
            meta=np.asarray([n_fields, t_history, t_future, seed], dtype=np.int64),
            seed=np.asarray(seed, dtype=np.int64),
            weather_source=np.asarray(str(data["weather_source"])),
        )
        log(f"trajectory cache written to {resolved_cache_path} (seed={seed}, source={data['weather_source']})")
    return data


def field_split(
    n_fields: int,
    labels: np.ndarray,
    seed: int = SEED,
    save: bool = True,
) -> dict[str, list[int]]:
    """Stratified 70/15/15 split **by field** (never by time step).

    Stratum = the field's first non-healthy class (``0`` when a field stays
    healthy all the way through).  Falls back to an unstratified split if any
    stratum is too small to stratify, so a small ``--n-fields`` dry run still
    works.  Writes ``artifacts/trajectory_split.json`` for reuse by
    ``train_trajectory.py`` / ``evaluate_trajectory.py``.
    """
    rng_strata = np.zeros(n_fields, dtype=np.int64)
    for i in range(n_fields):
        nonzero = np.flatnonzero(labels[i] != 0)
        rng_strata[i] = int(labels[i, nonzero[0]]) if nonzero.size else 0

    indices = np.arange(n_fields, dtype=np.int64)
    train_frac, val_frac, test_frac = cfg.TRAJ_SPLIT
    temp_frac = val_frac + test_frac

    def _can_stratify(y: np.ndarray) -> bool:
        _, counts = np.unique(y, return_counts=True)
        return counts.size >= 2 and int(counts.min()) >= 2

    strat = rng_strata if _can_stratify(rng_strata) else None
    if strat is None:
        log("WARNING: strata too small for stratified split -> plain random split")
    train_idx, temp_idx = train_test_split(
        indices, test_size=temp_frac, random_state=seed, stratify=strat
    )
    temp_strat = rng_strata[temp_idx] if strat is not None else None
    if temp_strat is not None and not _can_stratify(temp_strat):
        temp_strat = None
    rel_val = val_frac / temp_frac
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1.0 - rel_val, random_state=seed, stratify=temp_strat
    )

    split = {
        "train": sorted(int(x) for x in train_idx),
        "val": sorted(int(x) for x in val_idx),
        "test": sorted(int(x) for x in test_idx),
        "seed": int(seed),
        "split_by": "field",
        "fractions": [train_frac, val_frac, test_frac],
        "n_fields": int(n_fields),
    }

    # Leakage guard: the three sets must be disjoint and cover every field.
    sets = [set(split[k]) for k in ("train", "val", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("field split leakage detected: overlapping field sets")
    if len(set.union(*sets)) != n_fields:
        raise RuntimeError("field split does not cover every field exactly once")

    if save:
        path = Path(cfg.TRAJ_SPLIT_JSON)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        log(
            f"field split saved to {path}: train={len(split['train'])} "
            f"val={len(split['val'])} test={len(split['test'])} (disjoint, verified)"
        )
    return split


def compute_channel_stats(
    ndvi: np.ndarray,
    weather: np.ndarray,
    field_indices: np.ndarray,
    t_history: int = cfg.TRAJ_T_HISTORY,
    use_weather: bool = cfg.TRAJ_USE_WEATHER,
    eps: float = 1e-6,
) -> tuple[list[float], list[float]]:
    """Per-channel z-score stats over the **history windows of given fields**.

    Fit on train fields only (no leakage); returned as plain lists so they can
    be stored in the checkpoint alongside the model config.
    """
    parts = [ndvi[field_indices, :t_history, None]]
    if use_weather:
        parts.append(weather[field_indices, :t_history, :])
    x = np.concatenate(parts, axis=2).astype(np.float64)
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return [round(float(v), 6) for v in mean], [round(float(v), 6) for v in std]


def _validate_split_readonly(split: dict[str, Any], path: Path, dataset_path: str | Path | None) -> None:
    """Fail-closed structural validation for read-only split loading."""
    required = ("train", "val", "test")
    missing = [k for k in required if k not in split]
    if missing:
        raise ValueError(f"Strict read-only load failed: split missing keys {missing} in {path}")
    parts: dict[str, set[int]] = {}
    for k in required:
        vals = split[k]
        if not isinstance(vals, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in vals):
            raise ValueError(
                f"Strict read-only load failed: split '{k}' must be a list of integer field indices in {path}"
            )
        if any(i < 0 for i in vals):
            raise ValueError(f"Strict read-only load failed: negative field index in split '{k}' in {path}")
        if len(set(vals)) != len(vals):
            raise ValueError(f"Strict read-only load failed: duplicate field indices in split '{k}' in {path}")
        parts[k] = set(vals)
    if parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"]:
        raise ValueError(f"Strict read-only load failed: split partitions overlap in {path}")
    # Bound against the ACTUAL dataset when one is explicitly paired.
    if dataset_path is not None:
        try:
            with np.load(dataset_path, allow_pickle=False) as blob:
                keys = set(blob.files)
                if "ndvi" not in keys:
                    raise ValueError(
                        f"Strict read-only load failed: dataset at {dataset_path} has no ndvi array for split validation"
                    )
                shape = blob["ndvi"].shape
                if len(shape) != 2 or shape[0] < 1:
                    raise ValueError(
                        f"Strict read-only load failed: dataset ndvi has invalid shape {shape} at {dataset_path}"
                    )
                actual_n = int(shape[0])
                if "meta" in keys and len(blob["meta"]) >= 1 and int(blob["meta"][0]) != actual_n:
                    raise ValueError(
                        f"Strict read-only load failed: dataset meta n_fields {int(blob['meta'][0])} "
                        f"disagrees with ndvi shape {actual_n} at {dataset_path}"
                    )
        except ValueError:
            raise
        except Exception as err:
            raise ValueError(
                f"Strict read-only load failed: cannot read dataset for split validation at {dataset_path} ({err})"
            )
        all_idx = parts["train"] | parts["val"] | parts["test"]
        if all_idx and max(all_idx) >= actual_n:
            raise ValueError(
                f"Strict read-only load failed: split index out of range for dataset with {actual_n} fields in {path}"
            )
        declared = split.get("n_fields")
        if declared is not None:
            if isinstance(declared, bool) or not isinstance(declared, int):
                raise ValueError(
                    f"Strict read-only load failed: split n_fields must be an integer in {path}"
                )
            if declared != actual_n:
                raise ValueError(
                    f"Strict read-only load failed: split declares {declared} fields but dataset has {actual_n} in {path}"
                )


def load_split(
    seed: int = SEED,
    split_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    read_only: bool = False,
) -> dict[str, Any]:
    """Load the saved field split, rebuilding it if missing or seed mismatches."""
    path = Path(split_path) if split_path is not None else Path(cfg.TRAJ_SPLIT_JSON)
    if path.exists():
        try:
            split = json.loads(path.read_text(encoding="utf-8"))
            if split.get("seed") == seed:
                if read_only:
                    _validate_split_readonly(split, path, dataset_path)
                split["loaded_path"] = str(path.resolve())
                return split
            msg = f"split seed mismatch in {path} (cached {split.get('seed')} vs requested {seed})"
            if read_only:
                raise ValueError(f"Strict read-only load failed: {msg}")
            log(f"{msg}; recomputing split")
        except Exception as err:
            if read_only:
                raise
            log(f"WARNING: failed reading split ({err}); recomputing")

    if read_only:
        raise FileNotFoundError(
            f"Strict read-only load failed: split file not found or invalid at {path}."
        )

    data = build_dataset(seed=seed, cache_path=dataset_path)
    res = field_split(data["n_fields"], data["labels"], seed=seed)
    res["loaded_path"] = str(path.resolve())
    return res


# ---------------------------------------------------------------------------
# 5. Validation report (CLI)
# ---------------------------------------------------------------------------
def validate(data: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    """Print + persist a shape/flag/consistency summary.  Never raises on data."""
    ndvi, weather, labels = data["ndvi"], data["weather"], data["labels"]
    t_history, t_future = data["t_history"], data["t_future"]

    future = labels[:, t_history : t_history + t_future]
    counts = np.bincount(future.ravel(), minlength=N_CLASSES)
    class_share = {UNIFIED_CLASSES[c]: int(counts[c]) for c in range(N_CLASSES)}

    decoded = decode_labels_from_ndvi(ndvi)
    agreement = float((decoded == labels).mean())
    decoder_recall = {
        UNIFIED_CLASSES[c]: (
            round(float((decoded[labels == c] == c).mean()), 4)
            if (labels == c).any() else None
        )
        for c in range(N_CLASSES)
    }

    # How much of the future is knowable from the past: the fraction of fields
    # whose class at the forecast origin persists into day 0 of the horizon.
    origin_stress = int((labels[:, t_history - 1] != 0).sum())
    future_stress = int((future[:, 0] != 0).sum())
    onset_within_7 = sum(
        1
        for row in future
        if (np.flatnonzero(row != 0).size and int(np.flatnonzero(row != 0)[0]) < cfg.TRAJ_ALERT_HORIZON)
    )

    # Leakage check on the split actually in use.
    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    disjoint = not (tr & va or tr & te or va & te)

    stats: dict[str, Any] = {
        "n_fields": int(data["n_fields"]),
        "t_history": int(t_history),
        "t_future": int(t_future),
        "t_total": int(ndvi.shape[1]),
        "shapes": {
            "ndvi": list(ndvi.shape),
            "weather": list(weather.shape),
            "labels": list(labels.shape),
        },
        "input_dim": int(1 + (weather.shape[2] if cfg.TRAJ_USE_WEATHER else 0)),
        "ndvi_is_proxy": bool(data["ndvi_is_proxy"]),
        "weather_is_proxy": bool(data["weather_is_proxy"]),
        "weather_source": data["weather_source"],
        "ndvi_range": [float(ndvi.min()), float(ndvi.max())],
        "ndvi_mean": float(ndvi.mean()),
        "weather_ranges": {
            name: [float(weather[:, :, j].min()), float(weather[:, :, j].max())]
            for j, name in enumerate(cfg.TRAJ_WEATHER_VARS)
        },
        "future_class_day_counts": class_share,
        "label_derivation": "weather thresholds + hysteresis (see trajectory_data)",
        "stressed_at_forecast_origin": origin_stress,
        "stressed_at_horizon_day0": future_stress,
        "onset_within_alert_horizon": onset_within_7,
        "rule_decoder_agreement": round(agreement, 4),
        "rule_decoder_per_class_recall": decoder_recall,
        "split": {k: len(split[k]) for k in ("train", "val", "test")},
        "split_by_field_disjoint": bool(disjoint),
        "seed": int(data["seed"]),
    }

    log("=== Stage 3A dataset validation ===")
    log(f"fields={stats['n_fields']}  history={t_history}d  future={t_future}d  "
        f"T_total={stats['t_total']}d  D={stats['input_dim']}")
    log(f"shapes: ndvi={stats['shapes']['ndvi']} weather={stats['shapes']['weather']} "
        f"labels={stats['shapes']['labels']}")
    log(f"NDVI: range [{stats['ndvi_range'][0]:.3f}, {stats['ndvi_range'][1]:.3f}] "
        f"mean {stats['ndvi_mean']:.3f}  -> PROXY={'YES' if stats['ndvi_is_proxy'] else 'no'}")
    log(f"WEATHER source={stats['weather_source']} -> PROXY="
        f"{'YES' if stats['weather_is_proxy'] else 'no'}")
    for name, (lo, hi) in stats["weather_ranges"].items():
        log(f"   {name:<15} [{lo:.2f}, {hi:.2f}]")
    log("future-window class days: "
        + ", ".join(f"{k}={v}" for k, v in class_share.items()))
    log(f"stressed at origin={origin_stress} fields, at horizon day0={future_stress}, "
        f"onset within {cfg.TRAJ_ALERT_HORIZON}d={onset_within_7} fields")
    log(f"rule-decoder agreement with labels: {agreement:.3f} "
        "(proxy consistency diagnostic, not a model metric)")
    log("rule-decoder per-class recall: "
        + ", ".join(f"{k}={v}" for k, v in decoder_recall.items()))
    log(f"split by field: {stats['split']} disjoint={disjoint}")

    out = Path("artifacts/trajectory_data_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"stats written to {out}")
    return stats


def main() -> int:
    """CLI entrypoint: build the trajectory dataset (optionally validate only)."""
    parser = argparse.ArgumentParser(
        description="Stage 3A - build/validate the field-level trajectory dataset."
    )
    parser.add_argument("--validate-only", action="store_true",
                        help="build (or load cache), validate, report, do not train")
    parser.add_argument("--n-fields", type=int, default=cfg.TRAJ_N_FIELDS)
    parser.add_argument("--t-history", type=int, default=cfg.TRAJ_T_HISTORY)
    parser.add_argument("--t-future", type=int, default=cfg.TRAJ_T_FUTURE)
    parser.add_argument("--weather-source", choices=["proxy", "open-meteo", "auto"],
                        default="proxy",
                        help="proxy = declared synthetic (default, deterministic); "
                             "open-meteo = real daily history with proxy fallback")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--no-cache", action="store_true", help="skip the .npz cache")
    args = parser.parse_args()

    if args.t_history < 14 or args.t_history > 30:
        log(f"WARNING: T_history={args.t_history} outside spec range 14-30")
    if args.t_future < 7 or args.t_future > 14:
        log(f"WARNING: T_future={args.t_future} outside spec range 7-14")

    try:
        data = build_dataset(
            n_fields=args.n_fields,
            t_history=args.t_history,
            t_future=args.t_future,
            weather_source=args.weather_source,
            seed=args.seed,
            force=args.force,
            cache=not args.no_cache,
        )
        split = field_split(data["n_fields"], data["labels"], seed=args.seed)
        validate(data, split)
    except Exception as err:  # graceful, clear message, non-zero exit
        log(f"ERROR: {type(err).__name__}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
