# 🌾 KisanGPT — Digital Twin Dashboard (Stage 3B)

An interactive **Streamlit** view of the Stage-3A temporal digital twin: a farm
grid whose cells change colour across history *and* forecast, an NDVI
trajectory chart with an uncertainty band and onset markers, severity-coded
alerts, and a click-through drill-down that ends in the **Stage-2 diagnosis
engine**.

---

## 🚀 How to run

```bash
# 1. Stage 3A artifacts must exist (see Troubleshooting if not)
python trajectory_data.py --validate-only
python train_trajectory.py --mode train

# 2. launch
streamlit run dashboard_twin.py
```

| Variant | Command |
| --- | --- |
| Custom port | `streamlit run dashboard_twin.py --server.port 8502` |
| Headless (server / demo box) | `streamlit run dashboard_twin.py --server.headless true` |
| No browser auto-open | `streamlit run dashboard_twin.py --server.headless true --browser.gatherUsageStats false` |

Open **http://localhost:8501** (or the port you passed).

---

## 🗺️ Read this first: what a "cell" is

> **Each cell in the 10×10 grid is one field record** produced by
> `trajectory_data.build_dataset()` — it carries a real NDVI series, a weather
> series, day-level labels and Stage-3A model predictions. The grid is a
> **layout, not a contagion model**: cells do not infect one another, each one
> evolves from its own drivers. Nothing on this screen is invented.

Colour scale (from the Stage-3B spec):

| class | colour | hex |
| --- | --- | --- |
| healthy | green | `#2E7D32` |
| water_stress | yellow | `#F9A825` |
| nutrient_stress | orange | `#EF6C00` |
| disease_stress | red | `#C62828` |
| pest_stress | purple | `#6A1B9A` |
| stress_generic | gray | `#757575` |

**History vs forecast:** colours on days `0–20` are stored labels; days
`21–34` are the trajectory model's argmax. Day 20 is *today*.

---

## 🖥️ Walkthrough

### Sidebar
* **Status** — encoder/pooling/epoch, field count, and the PROXY flags for
  NDVI and weather.
* **Grid page** — 100 cells per page; 6 pages cover all 600 fields.
* **Selected cell / field** — dropdown; clicking a cell in the map drives the
  same selection.
* **Days to show** — `Historical` / `Forecast` / `Both` (changes the scrubber
  range).
* **Map view** — `Grid heatmap` or `Small multiples` (100 mini NDVI charts,
  green = history, red dashed = forecast).
* **Time** — day slider plus **▶ Play / ⏹ Stop**; Play advances one frame
  every 0.45 s and stops itself at the end of the range.
* **LLM polish (Layer 2)** — rewrites the Stage-2 diagnosis when a provider is
  reachable; otherwise silently keeps the Layer-1 template.
* **Data disclosure** — the mandatory PROXY warning.

### Main area
1. **Field map** (top) — animated 10×10 heatmap. Hover shows field id, class,
   NDVI and (on forecast days) confidence. **Click a cell to select it.**
2. **NDVI trajectory** (middle-left) — historical solid green, forecast dashed
   red, declared ±0.05 uncertainty band, a dotted *today* line, hollow markers
   for onsets already in history and a labelled diamond for the first
   **predicted** onset.
3. **Alert panel** (middle-right) — onset / escalation / coverage families,
   colour-coded green (watch) / yellow (act soon) / red (act now). Six alerts
   per family are shown, the rest summarised.
4. **Cell drill-down** (bottom) — current class + confidence, gradient×input
   attribution over the five input channels (`image` share is **0** by
   construction), then the Stage-2 Layer-1 diagnosis with severity, driver and
   escalation, plus optional LLM polish.

---

## 📸 Screenshots

> Placeholder — capture after `streamlit run dashboard_twin.py`:
>
> | File | What to capture |
> | --- | --- |
> | `screenshots/twin_grid.png` | Grid heatmap in `Both` mode, red/yellow cells visible |
> | `screenshots/twin_forecast.png` | `Forecast` mode with the day slider at 30 |
> | `screenshots/twin_small_multiples.png` | Small multiples overview |
> | `screenshots/twin_drilldown.png` | Drill-down with the Stage-2 diagnosis expanded |

---

## ⚠️ Data disclosure (also shown in the sidebar)

* NDVI and weather are **PROXY** — class-conditional synthetic series, because
  no paired Sentinel-2 export exists for these records. Replace the source in
  `trajectory_data.py` to clear the flags.
* Labels come from documented weather thresholds with hysteresis, so the
  mapping the model learns is reproducible, not field-verified agronomy.
* `pest_stress` proxy bursts are random with no weather signature, so the
  model essentially never predicts that class (reported at F1 = 0 in Stage 3A).
* The uncertainty band is a **declared** ±0.05 — the model has no variance
  head, so no per-point interval is fabricated.

---

## 🔧 Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `Trajectory model missing` | no `artifacts/trajectory_model.pth` | `python train_trajectory.py --mode train` |
| `Trajectory data unavailable` | no `artifacts/trajectory_data.npz` | `python trajectory_data.py --validate-only` |
| Stage-2 button disabled / warning | `diagnosis_engine.py` or `knowledge_base.json` missing | restore those Stage-2A files; the map keeps working without them |
| `LLM polish: unavailable` | no Ollama / API key configured | optional — Layer-1 text is shown instead; set `KISANGPT_LLM_PROVIDER`, `OPENAI_API_KEY` or run Ollama |
| Port already in use | another process owns 8501 | `streamlit run dashboard_twin.py --server.port 8502` |
| Cells never change colour | day slider parked on a history day | move past day 20, or switch **Days to show** to `Forecast`, and press **▶ Play** |
| Empty alert panel | no onset predicted for the visible cells | normal — it means the next 7 days look clear for that page |

---

## 🧩 Reuse map (nothing here is duplicated)

| Concern | Imported from |
| --- | --- |
| Trajectory data, labels, split | `trajectory_data.py` (Stage 3A) |
| Model + `detect_onset_alerts` | `trajectory_model.py` (Stage 3A) |
| Layer-1 diagnosis | `diagnosis_engine.py` (Stage 2A) |
| Layer-2 LLM polish | `llm_polish.py` (Stage 2A) |
| Class taxonomy, SEED, horizons | `stage1_config.py` / `config.py` |
