"""Read-only audit of production trajectory cache compatibility.

Loads artifacts/trajectory_data.npz in read-only mode and verifies:
1. Exact label agreement between derive_labels(weather, seed) and cached labels.
2. Exact NDVI agreement between render_proxy_ndvi(cached_labels, seed) and cached NDVI.
3. Completeness of cache schema (versioning, weather sources, field_meta).
4. Logic analysis of weather_source='auto' cache reuse.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from trajectory_data import derive_labels, render_proxy_ndvi


def run_audit() -> None:
    cache_path = Path("artifacts/trajectory_data.npz")
    if not cache_path.exists():
        print(f"ERROR: Production cache {cache_path} does not exist.")
        sys.exit(1)

    print("=" * 70)
    print("PRODUCTION CACHE COMPATIBILITY AUDIT (READ-ONLY)")
    print(f"Target: {cache_path} ({cache_path.stat().st_size} bytes)")
    print("=" * 70)

    # 1. Load cache read-only
    with np.load(cache_path, allow_pickle=True, mmap_mode="r") as blob:
        keys = list(blob.files)
        print(f"\n1. Stored Keys: {keys}")

        # Check required metadata
        missing_meta = []
        if "weather" not in blob:
            missing_meta.append("weather")
        if "labels" not in blob:
            missing_meta.append("labels")
        if "ndvi" not in blob:
            missing_meta.append("ndvi")
        if "seed" not in blob and ("meta" not in blob or len(blob["meta"]) < 4):
            missing_meta.append("seed")

        if missing_meta:
            print(f"CRITICAL: Missing required metadata/arrays: {missing_meta}")
            sys.exit(1)

        weather = np.array(blob["weather"])
        cached_labels = np.array(blob["labels"])
        cached_ndvi = np.array(blob["ndvi"])

        seed = int(blob["seed"]) if "seed" in blob else int(blob["meta"][3])
        cached_source = str(blob["weather_source"]) if "weather_source" in blob else None
        meta = np.array(blob["meta"]) if "meta" in blob else None
        flags = np.array(blob["flags"]) if "flags" in blob else None

        print(f"   Shape: weather={weather.shape}, labels={cached_labels.shape}, ndvi={cached_ndvi.shape}")
        print(f"   Seed: {seed}")
        print(f"   Flags: ndvi_is_proxy={bool(flags[0]) if flags is not None else 'N/A'}, weather_is_proxy={bool(flags[1]) if flags is not None else 'N/A'}")
        print(f"   Stored weather_source: {cached_source}")

    # 2. Recompute labels with current derive_labels
    print("\n2. Recomputing labels using current derive_labels(weather, seed)...")
    recomputed_labels = derive_labels(weather, seed=seed)

    label_mismatches = int(np.sum(recomputed_labels != cached_labels))
    total_elements = cached_labels.size
    print(f"   Total label positions: {total_elements}")
    print(f"   Label mismatch count:  {label_mismatches}")
    if label_mismatches == 0:
        print("   -> Exact match: recomputed labels match cached labels bit-for-bit.")
    else:
        print(f"   -> Mismatch rate: {label_mismatches / total_elements:.4%}")
        # Find which fields and days differ
        diff_fields, diff_days = np.where(recomputed_labels != cached_labels)
        unique_fields = np.unique(diff_fields)
        print(f"   Differing in {len(unique_fields)} fields out of {weather.shape[0]}.")

    # 3. Recompute NDVI from cached labels using current render_proxy_ndvi
    print("\n3. Recomputing NDVI using current render_proxy_ndvi(cached_labels, seed)...")
    recomputed_ndvi = render_proxy_ndvi(cached_labels, seed=seed)

    abs_ndvi_diff = np.abs(recomputed_ndvi - cached_ndvi)
    max_abs_ndvi_diff = float(np.max(abs_ndvi_diff))
    mean_abs_ndvi_diff = float(np.mean(abs_ndvi_diff))
    exact_ndvi_match = np.array_equal(recomputed_ndvi, cached_ndvi)

    print(f"   Max absolute NDVI difference:  {max_abs_ndvi_diff:.10f}")
    print(f"   Mean absolute NDVI difference: {mean_abs_ndvi_diff:.10f}")
    print(f"   Exact bit-identical match:     {exact_ndvi_match}")
    if exact_ndvi_match:
        print("   -> Exact match: recomputed NDVI matches cached NDVI bit-for-bit.")
    else:
        print(f"   -> Difference detected! Maximum discrepancy: {max_abs_ndvi_diff}")

    # 4. Check schema attributes
    print("\n4. Cache Schema & Metadata Completeness Inspection:")
    has_schema_version = "schema_version" in keys or "version" in keys
    print(f"   - Generator/schema version recorded: {has_schema_version}")

    has_requested_source = "requested_weather_source" in keys
    has_resolved_source = "weather_source" in keys
    print(f"   - Requested weather source recorded:  {has_requested_source}")
    print(f"   - Resolved weather source recorded:   {has_resolved_source} ({cached_source})")

    has_field_meta = "field_meta" in keys
    print(f"   - Actual field_meta stored in npz:   {has_field_meta}")

    # 5. Inspection of weather_source='auto' behavior
    print("\n5. Analysis of weather_source='auto' Cache Reuse:")
    print("   Code in build_dataset:")
    print("     if cached_source is not None and cached_source != weather_source and weather_source != 'auto':")
    print("         mismatches.append(f'weather_source ({cached_source} vs {weather_source})')")
    print("   Impact:")
    print("     When weather_source='auto', the condition is bypassed completely.")
    print("     If a cached dataset exists with weather_source='proxy', calling build_dataset(weather_source='auto')")
    print("     will silently reuse the 'proxy' cache without attempting to fetch real weather from Open-Meteo,")
    print("     even if network connectivity and real data are available.")

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY:")
    print(f"  Labels: {'MATCH' if label_mismatches == 0 else f'MISMATCH ({label_mismatches} errors)'}")
    print(f"  NDVI:   {'MATCH (bit-identical)' if exact_ndvi_match else f'MISMATCH (max diff {max_abs_ndvi_diff:.6f})'}")
    print("=" * 70)


if __name__ == "__main__":
    run_audit()
