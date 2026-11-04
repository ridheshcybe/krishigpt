# Pairing strategy — Stage 1A multi-modal dataset

**Status:** prototyping / pipeline validation.
**Seed:** 42 everywhere (splits, pairing, proxy curves).
**Owners of truth:** `stage1_config.py` (taxonomy), `dataset.py` (pipeline).

---

## 1. Why weak pairing

The three modalities we need do not exist together for any real sample:

| Modality | Source | What it actually covers |
|---|---|---|
| Tabular sensors | CHES (`data/ches_dataset.csv`), 212,019 rows | soil moisture, pH, weather, canopy… plus a binary stress flag |
| RGB leaf photos | paddy dataset (`data/paddy_images/`), 2,200 images | 5 leaf classes: BLB, blast, healthy, hispa, leaf_spot |
| NDVI time series | — | **nothing real available** (CHES has a single scalar `NDVI` column, not a series) |

There is no dataset that records a sensor row *and* photographs the same
plant at the same time, and no paired Sentinel-2 export for these fields.
Since Stage 1 must ingest all three modalities in one forward pass, we pair
them **weakly**: by label compatibility rather than by joint observation.

Weak pairing lets us build, train and debug the full architecture now, and
swap in genuinely paired data later without touching the model.

---

## 2. The rules

### 2.1 Row-sourced samples (212,019 of 213,754)

Each CHES row becomes one sample:

* **label** — from the deterministic Stage-0 rules:
  * `healthy`: binary=0 **and** soil moisture in [20, 60] **and** rainfall in [50, 200]
  * `water_stress`: binary=1 **and** soil moisture < 20
  * `nutrient_stress`: binary=1 **and** (pH < 6.0 **or** pH > 7.5)
  * `stress_generic`: everything else
* **image** — a random photo from the *matching* class:
  * healthy row → a **healthy** leaf photo;
  * any stress row → a random photo from the four disease classes
    (BLB/blast/hispa/leaf_spot). These are the only stress images that
    exist, so this is a **same-binary-class fallback**, not a same-class
    match — documented deviation from "pair with an image of the same stress
    class", forced by the data.
* **NDVI** — class-conditional PROXY curve matching the row's label (§3).

### 2.2 Image-sourced samples (1,735) — source of `disease_stress`

CHES has no disease label, so `disease_stress` would have zero training data
if labels came only from tabular rows. Each of the 1,735 disease photos
(1,214 train / 260 val / 261 test) therefore becomes one sample:

* **label** — from the shared image mapping (5 classes → healthy / disease_stress);
* **tabular row** — a random stress row from the *same split*, used only as
  contextual input (**reverse weak pairing**);
* **NDVI** — disease-class PROXY curve.

### 2.3 Split policy (no leakage across splits)

1. CHES rows: stratified **70/15/15 on the unified 6-class label**, seed 42.
2. Images: stratified 70/15/15 per class (reuses `mobilenet_cnn.stratified_split`).
3. **Pairing is split-restricted**: a row only ever draws images from its own
   split's pool, and image-sourced samples only borrow rows from their own
   split's pool. *No photo and no sensor row appears in two splits.*

Verified invariants (asserted by a check run over the built table):

* no image path in two splits — **PASS**
* every sample's image belongs to its own split pool — **PASS** (0 violations)
* every row-sourced sample's row belongs to its own split — **PASS**
* same-class / binary-consistent pairing rule — **PASS**
* image-sourced contextual rows come from the same split — **PASS**

Split indices are persisted to `artifacts/split_indices.json` (sample indices
per split, seed, strategy, class order) so any run can be reproduced exactly.

---

## 3. NDVI is PROXY (declared, prototype-only)

No real paired series exists, so `dataset.py` generates **class-conditional
synthetic curves**, per the Stage 1A spec:

| Class | Shape (T = 20 steps) |
|---|---|
| healthy | 0.60–0.80, stable (±0.015) |
| water_stress | 0.60 → 0.30 linear decline |
| nutrient_stress | 0.70 → 0.40 gradual decline |
| disease_stress | 0.70 → 0.50 with 2–3 sudden lesion-like dips (0.15–0.30) |
| pest_stress | irregular walk with random spikes/dips |
| stress_generic | 0.60 → 0.45 noisy moderate decline |

Measured on the built dataset: healthy 0.697 → 0.696 (stable), water
0.599 → 0.300, nutrient 0.700 → 0.400, disease curves contain sudden drops
in 100% of samples, deterministic under seed 42.

**The PROXY flag appears in three places**, as required:

1. code — `stage1_config.NDVI_IS_PROXY = True` and the `build_proxy_ndvi`
   docstring;
2. dataset class — `MultiModalStressDataset.ndvi_is_proxy`;
3. outputs — `artifacts/stage1a_stats.json → ndvi.is_proxy = true`, the
   stdout validation block, and this document.

If a real export lands at `data/ndvi_timeseries.csv` (row-aligned with CHES,
≥ 20 numeric steps), `load_real_ndvi()` picks it up automatically and clears
the flag.

### Known limitation — label leakage by construction

Because the curve is **class-conditional**, the NDVI branch alone can recover
the label. This is intentional for prototyping (it exercises the third
modality end-to-end) but it means:

* any accuracy gain attributable to the NDVI branch is a property of the
  proxy, not of satellite imagery;
* attention weight on the NDVI modality must not be reported as
  "satellite importance" until a real series replaces the proxy.

---

## 4. Class coverage and reserved classes

Unified taxonomy (`stage1_config.CLASS_MAP`, single source of truth):

| # | Class | Source | train | val | test |
|---|---|---|---|---|---|
| 0 | healthy | tabular (+ healthy photos) | 1,791 | 383 | 384 |
| 1 | water_stress | tabular | 51,546 | 11,046 | 11,046 |
| 2 | nutrient_stress | tabular | 9,458 | 2,027 | 2,027 |
| 3 | disease_stress | **images only** | 1,214 | 260 | 261 |
| 4 | pest_stress | **none — RESERVED** | 0 | 0 | 0 |
| 5 | stress_generic | tabular | 85,618 | 18,347 | 18,346 |
| | **total** | | 149,627 | 32,063 | 32,064 |

* **`pest_stress` is reserved and currently untrained.** It has no data
  source in either dataset (no pest annotation in CHES, no pest class in the
  paddy images). The class index is kept so the head shape never changes;
  metrics for it will be 0/undefined and must be reported as such. A future
  stage can source it from a pest dataset or a declared pest rule.
* **Missing classes per split:** only `pest_stress` (all three splits).
  Reported explicitly by `dataset.py` (`MISSING=['pest_stress']`).
* Class order differs from the Stage-0 rule-fusion vector (water/nutrient are
  swapped); `stage1_config.unified_index_from_stage0()` converts between them.

---

## 5. Explicit decision: binary=0 rows outside the healthy window

61,263 of the 63,821 binary=0 rows fall outside the strict healthy
moisture/rainfall window. Decision: **Option B — routed to `stress_generic`
as a documented residual class** (not silently ignored, not relabelled
healthy).

*Code:* comment at the `binary=0` residual branch in `tabular_xgb.assign_classes`
plus `dataset.RESIDUAL_POLICY`.
*Report:* printed by `dataset.py` on every run, stored in
`artifacts/stage1a_stats.json → tabular.binary0_outside_window` and
`binary0_policy`, and echoed in `report.txt` / `report_stage1.txt`.

Option A (widening the window to moisture [10, 80] / rainfall [20, 400]) was
implemented and measured in Stage 0: the CHES binary flag is nearly
independent of the features inside *either* window (binary-vs-feature AUC
0.506), so widening only relabels rows — healthy precision ceiling moves
0.290 → ~0.30 — while silently changing every downstream metric.

---

## 6. Implications for downstream stages (read before training)

1. **Image evidence is partly decorative** for row-sourced samples: a
   water-stress row may carry a photo of a random diseased leaf. The model
   must learn *which modality to trust*, which is exactly what the attention
   fusion layer is for — but attention values describe model behaviour on
   weak pairs, not field-measured importance.
2. **`disease_stress` accuracy is driven by the image branch by
   construction** (its label comes from the photo), so the image-only
   baseline will be strong on that class; XGBoost cannot predict it at all.
3. **NDVI results are proxy results.** Keep the PROXY flag attached to every
   metric, checkpoint and plot derived from it.
4. **`pest_stress` is untrained** — expect 0 precision/recall/F1 and
   remember it is reserved, not absent by accident.
5. Replacing weak pairs with real paired data requires only re-implementing
   `build_sample_table()`; the dataset contract
   `(tabular[27], image[3,224,224], ndvi[T], label)` and the split file stay
   unchanged.
