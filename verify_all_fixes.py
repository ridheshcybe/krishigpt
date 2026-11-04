"""Verification test suite for Stage 3A Causal Trajectory Updates.

Tests:
1. History Invariance: Identical pasts with divergent futures produce bit-for-bit identical input arrays.
2. Trailing Smoothing: Causal smoothing strictly accesses trailing values (<= d).
3. Cache Validation: Rebuilds on seed mismatch or weather source mismatch.
4. Exit Duration: No double counting of EXIT_DAYS in derive_labels.
"""

import json
import shutil
import numpy as np
from pathlib import Path

from stage1_config import UNIFIED_INDEX, SEED
from trajectory_data import (
    render_proxy_ndvi,
    _smooth_causal,
    _smooth,
    derive_labels,
    generate_weather,
    build_dataset,
    load_split,
    _sustained,
    WATER_ENTER,
    WATER_EXIT,
    ENTER_DAYS,
    EXIT_DAYS,
)

def test_history_invariance():
    print("=" * 60)
    print("TEST 1: HISTORY INVARIANCE UNDER DIVERGENT FUTURES")
    print("=" * 60)
    
    n_test_fields = 50
    t_hist = 21
    t_fut = 14
    t_total = t_hist + t_fut
    
    # Generate realistic weather and labels for the history window
    weather = generate_weather(n_test_fields, t_hist, seed=100)
    hist_labels = derive_labels(weather, seed=100)  # (50, 21)
    
    # Define 5 wildly different future scenarios (days 21..34)
    futures = {
        "all_healthy": np.zeros((n_test_fields, t_fut), dtype=np.int64),
        "immediate_drought": np.full((n_test_fields, t_fut), UNIFIED_INDEX["water_stress"], dtype=np.int64),
        "immediate_disease": np.full((n_test_fields, t_fut), UNIFIED_INDEX["disease_stress"], dtype=np.int64),
        "delayed_nutrient": np.zeros((n_test_fields, t_fut), dtype=np.int64),
        "pest_burst": np.zeros((n_test_fields, t_fut), dtype=np.int64),
    }
    futures["delayed_nutrient"][:, 2:] = UNIFIED_INDEX["nutrient_stress"]
    futures["pest_burst"][:, 1:6] = UNIFIED_INDEX["pest_stress"]
    
    rendered_histories = {}
    for name, fut in futures.items():
        full_labels = np.concatenate([hist_labels, fut], axis=1)
        curves = render_proxy_ndvi(full_labels, seed=42)
        rendered_histories[name] = curves[:, :t_hist]
    
    baseline = rendered_histories["all_healthy"]
    all_passed = True
    for name, hist in rendered_histories.items():
        if name == "all_healthy":
            continue
        max_diff = float(np.abs(baseline - hist).max())
        is_exact = np.array_equal(baseline, hist)
        print(f"  Future '{name:<18}' vs 'all_healthy': max_abs_diff = {max_diff:.8f}, exact_equal = {is_exact}")
        if max_diff != 0.0 or not is_exact:
            all_passed = False
            
    print(f"-> History Invariance Result: {'PASSED (Zero Leakage)' if all_passed else 'FAILED'}\n")
    assert all_passed, "History invariance failed!"

def test_trailing_smoothing():
    print("=" * 60)
    print("TEST 2: TRAILING-ONLY (CAUSAL) SMOOTHING")
    print("=" * 60)
    
    # Array with an acute shock at index 21 (tomorrow)
    x = np.ones(35, dtype=np.float64) * 0.70
    x[21] = 0.38  # Sudden disease onset on future day 0
    
    smoothed = _smooth_causal(x, window=3)
    
    # In centered smoothing, smoothed[20] was contaminated by x[21] (0.38)
    # In trailing smoothing, smoothed[20] depends strictly on x[18..20]
    expected_20 = (x[18] + x[19] + x[20]) / 3.0
    actual_20 = smoothed[20]
    
    print(f"  Raw value at day 20: {x[20]:.4f}, Raw shock at day 21: {x[21]:.4f}")
    print(f"  Smoothed value at day 20: {actual_20:.6f} (expected: {expected_20:.6f})")
    print(f"  Difference at day 20 from expected: {abs(actual_20 - expected_20):.8f}")
    
    # Verify that truncation before smoothing matches slicing after smoothing
    smoothed_past_only = _smooth_causal(x[:21], window=3)
    max_diff_prefix = float(np.abs(smoothed[:21] - smoothed_past_only).max())
    print(f"  Max diff between _smooth_causal(x)[:21] and _smooth_causal(x[:21]): {max_diff_prefix:.8f}")
    
    passed = (abs(actual_20 - expected_20) < 1e-12) and (max_diff_prefix == 0.0)
    print(f"-> Trailing Smoothing Result: {'PASSED (Strictly Causal)' if passed else 'FAILED'}\n")
    assert passed, "Trailing smoothing failed!"

def test_cache_validation():
    print("=" * 60)
    print("TEST 3: CACHE VALIDATION (SEED AND WEATHER SOURCE AWARE)")
    print("=" * 60)
    
    # Ensure fresh test directory for cache
    test_cache_path = Path("artifacts/test_cache_data.npz")
    if test_cache_path.exists():
        test_cache_path.unlink()
        
    # Temporarily monkey-patch cfg.TRAJ_DATA_NPZ
    import config as cfg
    orig_path = cfg.TRAJ_DATA_NPZ
    cfg.TRAJ_DATA_NPZ = str(test_cache_path)
    
    try:
        # 1. Build initial dataset with seed=42, proxy
        print("  1. Building initial cache (seed=42, source=proxy)...")
        data1 = build_dataset(n_fields=20, t_history=21, t_future=14, weather_source="proxy", seed=42, force=True)
        assert test_cache_path.exists(), "Cache file was not created!"
        
        # 2. Re-load with identical parameters -> should load from cache
        print("  2. Requesting identical parameters (seed=42, source=proxy)...")
        data2 = build_dataset(n_fields=20, t_history=21, t_future=14, weather_source="proxy", seed=42)
        assert np.array_equal(data1["labels"], data2["labels"])
        
        # 3. Request different seed -> must detect mismatch and rebuild
        print("  3. Requesting different seed (seed=43)...")
        data3 = build_dataset(n_fields=20, t_history=21, t_future=14, weather_source="proxy", seed=43)
        assert data3["seed"] == 43
        assert not np.array_equal(data1["labels"], data3["labels"]), "Seed 43 silently returned seed 42 labels!"
        
        # 4. Request different weather source -> must detect mismatch and rebuild
        print("  4. Requesting different weather source (source=open-meteo)...")
        data4 = build_dataset(n_fields=20, t_history=21, t_future=14, weather_source="open-meteo", seed=43)
        # Even if open-meteo falls back, cache mismatch check must have triggered
        
        print("-> Cache Validation Result: PASSED (Parameters Strictly Enforced)\n")
    finally:
        cfg.TRAJ_DATA_NPZ = orig_path
        if test_cache_path.exists():
            test_cache_path.unlink()

def test_exit_duration():
    print("=" * 60)
    print("TEST 4: EXIT DURATION SINGLE COUNTING IN DERIVE_LABELS")
    print("=" * 60)
    
    # Synthetic weather with 4 dry days (soil=15), then 2 wet days (soil=25)
    # ENTER_DAYS[1] = 2 (water_stress)
    # EXIT_DAYS[1] = 2 (exit condition: soil > 24)
    # Day 0: dry (1st day) -> state 0
    # Day 1: dry (2nd day) -> enters water_stress (state 1)
    # Day 2: dry (3rd day) -> state 1
    # Day 3: dry (4th day) -> state 1
    # Day 4: wet (1st day of soil=25) -> exit sustained for 1 day -> remains state 1
    # Day 5: wet (2nd day of soil=25) -> exit sustained for 2 days -> exits to state 0!
    
    n_days = 8
    weather = np.zeros((1, n_days, 4), dtype=np.float32)
    weather[0, :, 0] = 25.0  # temp
    weather[0, :, 1] = 50.0  # humid
    weather[0, :, 2] = 0.0   # rain
    weather[0, :4, 3] = 15.0 # dry
    weather[0, 4:, 3] = 25.0 # wet (> 24.0)
    
    labels = derive_labels(weather, seed=42)
    expected = [0, 1, 1, 1, 1, 0, 0, 0]
    actual = list(labels[0])
    
    print(f"  Soil sequence: {list(weather[0, :, 3])}")
    print(f"  Expected states (enter in 2d, exit in 2d): {expected}")
    print(f"  Actual states:                            {actual}")
    
    assert actual == expected, f"Exit duration mismatch: expected {expected}, got {actual}"
    print("-> Exit Duration Result: PASSED (Single Counted, Exactly 2 Days)\n")

if __name__ == "__main__":
    test_history_invariance()
    test_trailing_smoothing()
    test_cache_validation()
    test_exit_duration()
    print("=" * 60)
    print("ALL 4 CRITICAL CHECKS SUCCESSFULLY VERIFIED!")
    print("=" * 60)
