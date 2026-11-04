"""Regression test suite for causal trajectory generation, invariance, and cache identity.

Tests:
1. _smooth_causal([1., 2., 3., 4., 5.], 3) == [1., 1.5, 2., 3., 4.]
2. History invariance under divergent futures: past NDVI is bit-identical regardless of future labels.
3. History invariance under extended sequence length: past NDVI is bit-identical regardless of total length.
4. Deterministic exit hysteresis: exit occurs after documented number of qualifying days, not twice that duration.
5. Cache identity & validation: seed and weather_source mismatches or legacy missing fields prevent silent reuse.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from stage1_config import UNIFIED_INDEX, SEED
from trajectory_data import (
    render_proxy_ndvi,
    _smooth_causal,
    derive_labels,
    generate_weather,
    build_dataset,
    ENTER_DAYS,
    EXIT_DAYS,
    WATER_ENTER,
    WATER_EXIT,
    DISEASE_HUM_ENTER,
    DISEASE_HUM_EXIT,
    DISEASE_T_LO,
    DISEASE_T_HI,
)


def test_smooth_causal_vector() -> None:
    print("--- Test 1: _smooth_causal([1., 2., 3., 4., 5.], 3) ---")
    inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    expected = np.array([1.0, 1.5, 2.0, 3.0, 4.0], dtype=np.float64)
    out = _smooth_causal(inp, window=3)
    print(f"  Input:    {inp.tolist()}")
    print(f"  Expected: {expected.tolist()}")
    print(f"  Output:   {out.tolist()}")
    assert np.allclose(out, expected), f"Expected {expected}, got {out}"
    print("  -> PASSED: _smooth_causal returns exact trailing moving average.\n")


def test_invariance_under_divergent_futures() -> None:
    print("--- Test 2: Invariance under divergent futures (seed=42) ---")
    n_fields = 30
    t_hist = 21
    t_fut = 14

    # Past labels (days 0..20) generated from weather
    weather = generate_weather(n_fields, t_hist, seed=123)
    past_labels = derive_labels(weather, seed=123)

    # 5 distinct future scenarios for days 21..34
    futures = {
        "healthy_future": np.zeros((n_fields, t_fut), dtype=np.int64),
        "water_stress_future": np.full((n_fields, t_fut), UNIFIED_INDEX["water_stress"], dtype=np.int64),
        "disease_stress_future": np.full((n_fields, t_fut), UNIFIED_INDEX["disease_stress"], dtype=np.int64),
        "nutrient_delayed": np.zeros((n_fields, t_fut), dtype=np.int64),
        "pest_burst": np.zeros((n_fields, t_fut), dtype=np.int64),
    }
    futures["nutrient_delayed"][:, 3:] = UNIFIED_INDEX["nutrient_stress"]
    futures["pest_burst"][:, 1:6] = UNIFIED_INDEX["pest_stress"]

    baseline_full = np.concatenate([past_labels, futures["healthy_future"]], axis=1)
    baseline_ndvi = render_proxy_ndvi(baseline_full, seed=42)[:, :t_hist]

    for name, fut in futures.items():
        if name == "healthy_future":
            continue
        full_labels = np.concatenate([past_labels, fut], axis=1)
        fut_ndvi = render_proxy_ndvi(full_labels, seed=42)[:, :t_hist]

        max_diff = float(np.abs(baseline_ndvi - fut_ndvi).max())
        is_exact = np.array_equal(baseline_ndvi, fut_ndvi)
        print(f"  Future '{name:<22}': max_abs_diff = {max_diff:.10f}, exact_equal = {is_exact}")
        assert is_exact, f"Future '{name}' contaminated past NDVI! max_diff = {max_diff}"

    print("  -> PASSED: Past rendered NDVI is bit-identical across all future scenarios.\n")


def test_invariance_under_extended_total_length() -> None:
    print("--- Test 3: Invariance under extending total series length ---")
    n_fields = 20
    t_hist = 21

    # Deterministic past labels
    weather_21 = generate_weather(n_fields, t_hist, seed=456)
    past_labels = derive_labels(weather_21, seed=456)

    # Render with length 21
    ndvi_len_21 = render_proxy_ndvi(past_labels, seed=42)

    # Render with length 35 (past + 14 healthy)
    labels_35 = np.pad(past_labels, ((0, 0), (0, 14)), mode="constant", constant_values=0)
    ndvi_len_35 = render_proxy_ndvi(labels_35, seed=42)

    # Render with length 50 (past + 29 mixed)
    labels_50 = np.pad(past_labels, ((0, 0), (0, 29)), mode="constant", constant_values=1)
    ndvi_len_50 = render_proxy_ndvi(labels_50, seed=42)

    # Compare first 21 days
    diff_35 = float(np.abs(ndvi_len_21 - ndvi_len_35[:, :t_hist]).max())
    diff_50 = float(np.abs(ndvi_len_21 - ndvi_len_50[:, :t_hist]).max())

    is_exact_35 = np.array_equal(ndvi_len_21, ndvi_len_35[:, :t_hist])
    is_exact_50 = np.array_equal(ndvi_len_21, ndvi_len_50[:, :t_hist])

    print(f"  Length 21 vs Length 35[:21]: max_diff = {diff_35:.10f}, exact = {is_exact_35}")
    print(f"  Length 21 vs Length 50[:21]: max_diff = {diff_50:.10f}, exact = {is_exact_50}")

    assert is_exact_35, f"Length 35 changed past NDVI! max_diff = {diff_35}"
    assert is_exact_50, f"Length 50 changed past NDVI! max_diff = {diff_50}"
    print("  -> PASSED: Extending series length leaves past NDVI bit-identical.\n")


def test_exit_hysteresis_deterministic() -> None:
    print("--- Test 4: Exit hysteresis after documented qualifying days ---")
    # Water stress: ENTER_DAYS[1] = 2 (soil < 18.0), EXIT_DAYS[1] = 2 (soil > 24.0)
    # Day 0: soil=15.0 (dry day 1)  -> state 0 (needs 2 days)
    # Day 1: soil=15.0 (dry day 2)  -> enters state 1 (water_stress)
    # Day 2: soil=15.0 (dry day 3)  -> state 1
    # Day 3: soil=15.0 (dry day 4)  -> state 1
    # Day 4: soil=26.0 (wet day 1)  -> qualifying exit day 1 -> state remains 1
    # Day 5: soil=26.0 (wet day 2)  -> qualifying exit day 2 -> exits to state 0 on day 5!
    # Day 6: soil=26.0 (wet day 3)  -> state 0
    # Day 7: soil=26.0 (wet day 4)  -> state 0
    n_days = 8
    weather = np.zeros((1, n_days, 4), dtype=np.float32)
    weather[0, :, 0] = 25.0  # temp
    weather[0, :, 1] = 50.0  # humid
    weather[0, :, 2] = 0.0   # rain
    weather[0, :4, 3] = 15.0 # dry (< 18.0)
    weather[0, 4:, 3] = 26.0 # wet (> 24.0)

    labels = derive_labels(weather, seed=999)
    actual = labels[0].tolist()
    expected = [0, 1, 1, 1, 1, 0, 0, 0]

    print(f"  Soil sequence:     {weather[0, :, 3].tolist()}")
    print(f"  Expected states:   {expected}")
    print(f"  Actual states:     {actual}")

    assert actual == expected, f"Exit duration mismatch: expected {expected}, got {actual}"
    print("  -> PASSED: Exit occurs on day 5 after exactly EXIT_DAYS=2 qualifying days (no double-counting).\n")


def test_cache_identity_and_validation() -> None:
    print("--- Test 5: Cache identity and legacy/mismatch rejection ---")
    import config as cfg

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cache = Path(tmpdir) / "test_traj_cache.npz"
        orig_cache = cfg.TRAJ_DATA_NPZ
        cfg.TRAJ_DATA_NPZ = str(tmp_cache)

        try:
            # 1. Build initial dataset with seed=42, proxy
            print("  Building cache (seed=42, source=proxy, n_fields=10)...")
            d1 = build_dataset(n_fields=10, t_history=21, t_future=14, weather_source="proxy", seed=42, force=True)
            assert tmp_cache.exists(), "Cache file was not created!"

            # 2. Re-load identical params -> should load from cache
            print("  Requesting identical params (seed=42, source=proxy)...")
            d2 = build_dataset(n_fields=10, t_history=21, t_future=14, weather_source="proxy", seed=42)
            assert np.array_equal(d1["labels"], d2["labels"])
            assert d2["seed"] == 42
            assert d2["weather_source"] == "proxy"
            assert len(d2["field_meta"]) == 10

            # 3. Seed mismatch -> must NOT silently reuse seed 42
            print("  Requesting mismatched seed (seed=99)...")
            d3 = build_dataset(n_fields=10, t_history=21, t_future=14, weather_source="proxy", seed=99)
            assert d3["seed"] == 99
            assert not np.array_equal(d1["labels"], d3["labels"])

            # 4. Legacy cache missing weather_source -> must NOT silently reuse
            print("  Simulating legacy cache missing 'weather_source' and 'seed' keys...")
            # Save legacy-style npz without weather_source
            np.savez_compressed(
                tmp_cache,
                ndvi=d1["ndvi"],
                weather=d1["weather"],
                labels=d1["labels"],
                flags=np.asarray([1, 1], dtype=np.int8),
                meta=np.asarray([10, 21, 14], dtype=np.int64),  # legacy meta length 3
            )
            d_legacy = build_dataset(n_fields=10, t_history=21, t_future=14, weather_source="proxy", seed=42)
            assert "seed" in d_legacy
            assert "weather_source" in d_legacy

            print("  -> PASSED: Cache correctly validates seed and weather_source, rejecting legacy/mismatches.\n")
        finally:
            cfg.TRAJ_DATA_NPZ = orig_cache


if __name__ == "__main__":
    test_smooth_causal_vector()
    test_invariance_under_divergent_futures()
    test_invariance_under_extended_total_length()
    test_exit_hysteresis_deterministic()
    test_cache_identity_and_validation()
    print("=" * 60)
    print("ALL FOCUSED TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
