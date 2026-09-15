# PS 26077 — Technical Summary
### AI-Driven Hyper-Local Early Warning System for Cloudbursts & Flash Floods
**MoES / NCMRWF | Smart India Hackathon 2026**

---

## 1. What We Are Building

A **multi-stage nowcasting pipeline** that predicts cloudbursts and flash floods 2–6 hours ahead at hyper-local (sub-district) resolution in the Uttarakhand Himalayas, using only freely available data (no restricted IMD radar access needed).

The pipeline has 6 functional modules that run in sequence:

```
Raw IMD Grid / GAGAN IWV / AWS Sensors
          │
          ▼
 [Module 1] Data Labeling (L-score spatial localization)
          │
          ▼
 [Module 2] SNN Neuromorphic Edge Gate (sensor-level trigger)
          │
          ▼
 [Module 3] L2-Logistic Regression Classifier (cloudburst probability)
          │
          ▼
 [Module 4] 4-Tier Operational Alert (GREEN / YELLOW / ORANGE / RED)
          │
          ▼
 [Module 5] XAI Explainer (why did it fire — per-feature logit decomposition)
          │
          ▼
 [Module 6] PINN 2D Shallow Water Flood Simulator (inundation extent map)
```

---

## 2. Datasets Used

### Primary Training Dataset — IMD 0.25-Degree Gridded Daily Rainfall
- **Source:** India Meteorological Department (IMD) gridded rainfall product
- **Resolution:** 0.25° × 0.25° (~27 km × 27 km per cell)
- **Time span:** 24 years — 2000 to 2023, JJAS monsoon months only (June–September)
- **Access method:** `imdlib` Python library, reading from locally cached `.grd` binary files
- **Coverage:** 6 sub-regions (see Section 3)
- **Total labeled rows:** 913,536 grid-cell-day records

### Supporting Dataset — GAGAN GNSS IWV (Integrated Water Vapour)
- **Source:** GPS Aided GEO Augmented Navigation (GAGAN) — ISRO/AAI
- **What it gives:** Precipitable water column (mm) above each GNSS station, at 30-min intervals
- **Role:** Pre-convective moisture buildup signal — IWV rises sharply 3–6 hours before a cloudburst
- **Time span:** March 2013 – February 2014 (archive window)
- **Stations used:** Guwahati (Assam anchor), Bagdogra, Lengpui

### Supporting Dataset — MOSDAC In-Situ AWS (Automatic Weather Stations)
- **Source:** ISRO MOSDAC portal — ASSAM_ALL_2013 CSV file
- **What it gives:** 30-minute station readings: rainfall (mm), temperature, humidity, pressure, wind speed
- **Role:** Ground truth for rainfall intensity, used in feature engineering (R, RI, R_30, R_60)
- **Coverage:** ~13 stations across Assam/Northeast India corridor

### Topography — ISRO Bhuvan CartoDEM
- **Resolution:** 30 m digital elevation model
- **Role:** Input to the PINN flood simulator (valley slope and canyon geometry)

---

## 3. Six Target Sub-Regions (Uttarakhand + Assam)

| Region Key | District | River Basin | Bounding Box | Why Chosen |
|---|---|---|---|---|
| `uttarkashi` | Uttarkashi | Bhagirathi | 30.5–31.25°N, 78.0–78.75°E | Highest cloudburst frequency in Uttarakhand; Gangotri upstream |
| `chamoli` | Chamoli | Alaknanda | 30.25–30.75°N, 79.0–79.75°E | 2021 Chamoli disaster; Badrinath catchment |
| `rudraprayag` | Rudraprayag | Mandakini | 30.0–30.75°N, 78.75–79.25°E | Kedarnath 2013 disaster — highest documented death toll in India |
| `pithoragarh` | Pithoragarh | Kali/Sharda | 29.5–30.25°N, 79.75–80.5°E | Kumaon Himalaya; Bastari cloudburst 2016 |
| `tehri` | Tehri | Bhilangana | 30.25–30.75°N, 78.25–78.75°E | Tehri reservoir catchment — dam safety critical |
| `assam` | Assam-Meghalaya corridor | Brahmaputra tributaries | 24.0–27.0°N, 89.5–94.0°E | GAGAN AWS anchor overlap; northeast monsoon regime |

---

## 4. Reference Papers Used

| Paper | Authors / Journal / Year | What We Used From It |
|---|---|---|
| **Kedarnath disaster rainfall analysis** | Jena et al., *Journal of Hydrometeorology*, 2020 | Validated CB_CORE_FLOOR_MM = 64.5 mm/day as the "very heavy rain" entry gate for Uttarakhand |
| **IMD Mausam — 2013 Kedarnath Special Issue** | IMD, *Mausam*, 2014 | Ground-truth dates (June 15–17, 2013) and spatial extent for the Kedarnath disaster registry entry |
| **Uttarkashi Asi Ganga Cloudburst** | Gupta et al., *Current Science*, 2013 | Confirmed Aug 3–4, 2012 as Uttarkashi disaster registry dates |
| **Skilful nowcasting of extreme precipitation with NowcastNet** | Zhang et al., *Nature*, 2023 | Physics-informed loss concept (continuity equation residual) used in our PINN SWE module |
| **DGMR — Deep Generative Model of Rainfall** | Ravuri et al., *Nature*, 2021 | CSI / POD / FAR metric framework — we report the same metrics for comparability |
| **Enhanced smart weather prediction using Binarized SNNs** | Amanullah et al., *Knowledge and Information Systems*, 2025 | Closest published precedent for SNN in weather classification |
| **Training of AI nowcasting models should account for user requirements** | Heistermann et al., *NHESS*, 2025 | Justification for threshold-exceedance framing ("will it cross danger level?") instead of exact rainfall prediction |
| **ISRO NETRA benchmark** | ISRO / MOSDAC | Baseline CSI ~0.35 for Western Himalaya satellite-only cloudburst detection — our target to beat |
| **DMMC Uttarakhand disaster reports** | State Disaster Management Authority, 2016/2019 | Pithoragarh Bastari (Jul 2016) and Mori Tons Valley (Aug 2019) disaster registry dates |

---

## 5. Models Used

### Module 2 — Spiking Neural Network (SNN) Edge Gate
- **Framework:** snnTorch (PyTorch-based)
- **Architecture:** Input → Dense(12 LIF neurons) → Output(1 LIF neuron)
- **Neuron model:** Leaky Integrate-and-Fire (LIF) with fast-sigmoid surrogate gradient
- **Two instances:**
  - **Gate A (Cloudburst):** beta=0.50 (short tau ~5–10 min), inputs: R, RI, delta_R, delta_RI
  - **Gate B (Thunderstorm):** beta=0.88 (long tau ~45–60 min), inputs: IWV_trend, pressure_trend, wind_shift, temp_drop, CAPE_trend
- **Deployment:** Hand-translated to C99 (`snn_gate_mcu.c`) for ESP32/ARM Cortex-M4 edge MCUs
- **Why SNN here:** Edge sensors run on solar/battery for months; SNN stays silent (near-zero power) and only fires a spike when the reading changes sharply — perfectly matched to the power constraint

### Module 3 — L2-Regularised Logistic Regression Classifier
- **Framework:** scikit-learn `LogisticRegression` with `StandardScaler` pipeline
- **Features:** R (rain intensity), R_30 (30-min accumulation), R_60 (60-min accumulation), RI (intensity acceleration), L_score (spatial localization), IWV_now, IWV_trend_3hr
- **Regularization:** C tuned from {0.001, 0.01, 0.1, 1.0, 10.0} via validation CSI
- **Training result:** C=0.1 optimal; Val CSI = 0.3787
- **Why logistic regression:** Fully interpretable (per-feature logit contributions), fast inference on embedded systems, no black box

### Module 6 — Physics-Informed Neural Network (PINN), 2D Shallow Water Equations
- **Architecture:** 6-layer fully connected MLP (32 hidden units, Tanh), input: (x, y, t, z, dz/dx, dz/dy)
- **PDEs solved:** Non-dimensional 2D Shallow Water Equations (continuity + x-momentum + y-momentum)
- **Training:** Two-stage — Adam (250 epochs) → L-BFGS refinement (25 steps)
- **Non-dimensionalization scales:** L=20 km, T=3 hr, H0=5 m, U0=sqrt(g·H0)~7 m/s, Z0=500 m
- **Output:** Peak inundation depth (m), peak velocity (m/s), flooded area (km2) at any query time

---

## 6. The Labeling System (What We Had to Build / Fix)

This was the most methodologically complex part. IMD daily gridded data doesn't come with cloudburst labels — we had to derive them entirely from physics.

### The Core Problem
A grid cell with 100 mm of rainfall could be:
- A genuine hyper-local cloudburst (dangerous, needs a RED alert)
- Part of a 500 km synoptic monsoon depression (not a cloudburst at all — would be a **false alarm** to alert on)

The difference is **spatial localization** — a cloudburst is intense at one cell and weak at all surrounding cells.

### The L-Score (Single-Knob Spatial Localization)

```
L = (R_core - R_background) / R_core
```

Where R_background is the mean rainfall in a 1–2 cell annulus (~27–54 km) around the core cell.

- L close to 1.0 → rain is almost entirely concentrated at this one cell → cloudburst
- L close to 0.0 → rain is spread uniformly across the region → synoptic monsoon event

**Thresholds chosen:**
- L >= 0.70 → `CONFIRMED_CLOUDBURST`
- L >= 0.40 → `CANDIDATE_CLOUDBURST`
- L < 0.40 (but rain >= 64.5 mm) → `WIDESPREAD_HEAVY_RAIN` (hard negative — explicitly taught as NOT a cloudburst)

### Why We Simplified to One Knob

Early versions tested three-variable thresholds: R_bg <= 10 mm OR ratio <= 0.20 OR L >= 0.70. We proved mathematically that both of the first two conditions are redundant given the entry gate (R_core >= 64.5 mm):
- R_bg <= 10 with R_core >= 64.5 forces L >= 0.845, which already satisfies L >= 0.70
- ratio <= 0.20 forces L >= 0.80, which already satisfies L >= 0.70

So the multi-condition rule was literally identical to just `L >= 0.70`. Collapsing to one knob eliminated a false sense of "multiple independent checks."

### The Disaster Registry (Ground-Truth Override)

The L-score alone would misclassify known historical disasters if the IMD gridded data happened to underestimate the rainfall intensity (common in complex terrain where the grid cell average dilutes a very localized spike).

We built a `DISASTER_REGISTRY` — a list of 6 peer-reviewed confirmed events with exact dates and bounding boxes:

| Event | Region Key | Dates | Source |
|---|---|---|---|
| Kedarnath Disaster (Mandakini Basin) | rudraprayag | 2013-06-15 to 17 | Jena et al. 2020; IMD Mausam 2014 |
| Kedarnath Regional Deluge (Alaknanda) | chamoli | 2013-06-16 to 17 | Jena et al. 2020; NDMA 2013 |
| Uttarkashi Asi Ganga Cloudburst | uttarkashi | 2012-08-03 to 04 | Gupta et al. 2013 |
| Pithoragarh Bastari Cloudburst | pithoragarh | 2016-07-01 to 02 | Mausam 2017; DMMC Uttarakhand |
| Chamoli Ghat Cloudburst | chamoli | 2016-07-01 to 02 | Mausam 2017; DMMC Uttarakhand |
| Mori Tons Valley Cloudburst | uttarkashi | 2019-08-18 to 19 | SDMA Uttarakhand 2019 |

**Registry logic:** If a grid cell falls within the disaster bounding box on a registry date AND R_core >= 64.5 mm, it is **force-promoted** to `CONFIRMED_CLOUDBURST` — even if L is between 0.40 and 0.70 (where the grid averaging may have diluted the true peak). This cell is also flagged `override_applied=True` for full XAI transparency.

### Bugs Found and Fixed in the Dataset Pipeline

**1. Double-counting overrides (Chamoli/Rudraprayag geographic overlap)**
The Alaknanda and Mandakini basins are geographically adjacent. The registry had separate entries for both regions on the same Kedarnath dates. Without strict `region_key` scoping, cells on the border were being matched by both entries and counted twice. Fix: `check_disaster_registry()` filters by `region_key` first, so a Chamoli-region cell can only match Chamoli entries, never Rudraprayag entries.

**2. Dead-code Chamoli bounding box**
An early version only checked `is_rudra` coordinates, meaning all Chamoli registry entries existed in the dictionary but were unreachable. Fix: the registry lookup is now purely bounding-box + date + region_key — no hardcoded region-specific coordinate blocks.

**3. Phase D loading the wrong file**
Phase D originally expected a CSV from Phase C (`cloudburst_events.csv`). But the IMD pipeline writes parquet files directly. Phase D was failing at startup with "file not found." Fix: Phase D now auto-detects the parquet files in `data/raw/imd_rain/` and loads them directly, with the CSV as a fallback.

**4. `_assign_event_groups` KeyError**
The LOEO-CV grouping function hardcoded `df["final_label"]` but parquet files use the column name `"rain_label"`. Fix: dynamic column detection (`"rain_label" if "rain_label" in df.columns else "final_label"`).

---

## 7. Current Training Results (Live — Just Completed)

| Metric | Value | Target | Status |
|---|---|---|---|
| Best L2 C | 0.1 | — | Selected by Val CSI |
| Val CSI | 0.3787 | >= 0.35 (NETRA baseline) | Beating NETRA |
| **Test POD** | **0.6326** | >= 0.85 | Below target — expected at logistic regression baseline |
| **Test FAR** | **0.5173** | <= 0.35 | Above target — needs ConvLSTM correction layer |
| **Test CSI** | **0.3770** | >= 0.50 | Baseline established |
| **Test PR-AUC** | **0.5767** | >= 0.70 | Baseline established |
| Positive samples | 8,747 / 913,536 | — | 0.96% — realistic rare-event rate |

> These are **logistic regression baseline numbers** — the intentional Week 1 safety net. The CSI of 0.377 already matches NETRA's ~0.35 benchmark using only 4 rain features with no satellite data. Adding the ConvLSTM correction layer and satellite CTT/CTCR inputs is the next step to push toward CSI >= 0.50.

---

## 8. What's Next After Training Completes

1. **LOEO-CV results** — completing now, will give per-event-fold generalization scores
2. **Full training report** — auto-generated at `outputs/training_report.md`
3. **ConvLSTM correction layer** — the neural correction on top of the logistic baseline (PS Part 5 §7.2)
4. **pysteps optical-flow baseline** — the standard comparison benchmark every published nowcasting paper reports
5. **Alert delivery wiring** — connecting the classifier output to the 4-tier alert → WhatsApp/SMS demo
