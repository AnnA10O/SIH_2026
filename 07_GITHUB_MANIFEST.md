# GitHub Repository Manifest & Architecture Inventory
**Problem Statement PS 26077 | MoES / NCMRWF / ISRO MOSDAC**  
**Project**: AI-Driven Hyper-Local Cloudburst Nowcasting & PINN Hydrodynamic Flood Warning System  

---

## 1. Repository Overview & Git Upload Readiness

This manifest documents all files designated for upload to GitHub. The repository has undergone deep cleaning:
- **Zero Credentials Leaks**: All private tokens and API configs (`config.json`, `mosdac_api/config.json`) are strictly excluded. Clean configuration template `config.example.json` is provided.
- **No Data Bloat**: Massive raw data files (>636 MB raw datasets in `data/`, multi-gigabyte spreadsheets, and 1.1 GB Python virtual environment `nowcast/`) are safely excluded via `.gitignore`.
- **Clean ML Architecture**: Deprecated legacy models and hardcoded formulas have been cleaned out and updated to the production **1D-CNN + BiLSTM Deep Neural Nowcaster** ($POD = 88.75\%$, $CSI = 0.4154$, $\tau = 0.15$ on unseen 24-year holdout; diagnostic LOEO-CV: $CSI = 0.9316$) coupled with the **Dual Neuromorphic SNN Edge Gate** (91.39% edge power reduction) and **2D Hydrodynamic PINN Shallow Water Solver**.

---

## 2. Tracked Files Directory (Ready for GitHub)

### A. Core Machine Learning & Pipeline Engine (`src/`)
| File | Size | Role & Architecture Component |
|:---|:---:|:---|
| [`src/config.py`](file:///d:/SIH/src/config.py) | ~7 KB | Global paths, sensor thresholds, feature definitions, and model weight paths (`NEURAL_MODEL_PT`, `NEURAL_MODEL_JSON`). |
| [`src/train_neural_nowcaster.py`](file:///d:/SIH/src/train_neural_nowcaster.py) | ~8 KB | PyTorch 1D-CNN + BiLSTM model definition, temporal feature ordering `[R_60, R_30, R, RI]`, training loop, and weight exporter. |
| [`src/snn_gate.py`](file:///d:/SIH/src/snn_gate.py) | ~21 KB | Dual Neuromorphic SNN LIF Gates (Gate A: Cloudburst short $\tau$, Gate B: Thunderstorm long $\tau$), streaming `SNNNeuromorphicGate` class, and telemetry escalation logic. |
| [`src/snn_gate_mcu.c`](file:///d:/SIH/src/snn_gate_mcu.c) | ~7 KB | Hand-translated, verified C99 implementation of the SNN LIF update rule for low-power edge microcontrollers (ESP32/Cortex-M4). |
| [`src/snn_gate_mcu.h`](file:///d:/SIH/src/snn_gate_mcu.h) | ~1.5 KB | Header file for MCU C99 deployment with calibrated synaptic weights and thresholds. |
| [`src/spatial_fusion.py`](file:///d:/SIH/src/spatial_fusion.py) | ~6 KB | Module 5: 2D Spatial Fusion Grid with Inverse Distance Weighting (IDW) interpolation across sparse AWS station meshes. |
| [`src/pinn_handoff.py`](file:///d:/SIH/src/pinn_handoff.py) | ~4 KB | Module 6: PINN Flood Handoff coupler constructing excess runoff source terms $Q(x,y,t)$ when rainfall reaches cloudburst intensity. |
| [`src/pinn_swe.py`](file:///d:/SIH/src/pinn_swe.py) | ~16 KB | Physics-Informed Neural Network (PINN) solver for 2D Shallow Water Equations (SWE) predicting localized water depth $h$ and velocity fields $(u,v)$. |
| [`src/station_feature_engine.py`](file:///d:/SIH/src/station_feature_engine.py) | ~5 KB | Module 1: Streaming in-situ feature buffer calculating $R$, $RI$, $R_{30}$, $R_{60}$, and dewpoint depression over rolling 60-min windows. |
| [`src/xai_explainer.py`](file:///d:/SIH/src/xai_explainer.py) | ~9 KB | Module 7: Explainable AI engine decomposing precursor logit contributions and generating human-readable forecaster directives. |
| [`src/phase_a_region_discovery.py`](file:///d:/SIH/src/phase_a_region_discovery.py) | ~12 KB | Region discovery algorithm ranking 14 GAGAN GNSS receiver stations and identifying optimal mountain AWS sensor clusters. |
| [`src/phase_b_quality_gate.py`](file:///d:/SIH/src/phase_b_quality_gate.py) | ~14 KB | Multi-stage quality control gate (stuck-sensor detection, unphysical spike clip, crosstalk cross-correlation). |
| [`src/phase_c_labeling.py`](file:///d:/SIH/src/phase_c_labeling.py) | ~18 KB | Automated physical event labeling assigning 4-tier truth tags using spatial localization score ($L$-score) and GNSS IWV drawdown. |
| [`src/phase_d_training.py`](file:///d:/SIH/src/phase_d_training.py) | ~30 KB | 60-fold Leave-One-Event-Out (LOEO-CV) validation engine and linear benchmark evaluation. |
| [`src/utils.py`](file:///d:/SIH/src/utils.py) | ~7 KB | Haversine distance, coordinate projections, logging, and data manipulation helpers. |
| [`src/__init__.py`](file:///d:/SIH/src/__init__.py) | 0.1 KB | Package initializer. |

---

### B. Production Model Checkpoints (`models/`)
| File | Size | Role |
|:---|:---:|:---|
| [`models/cloudburst_cnn_bilstm.pt`](file:///d:/SIH/models/cloudburst_cnn_bilstm.pt) | 26.6 KB | Serialized PyTorch deep learning model checkpoint (1D-CNN + BiLSTM). |
| [`models/cloudburst_cnn_bilstm_weights.json`](file:///d:/SIH/models/cloudburst_cnn_bilstm_weights.json) | 149 KB | Exported neural weights and scaler parameters for client-side JavaScript and edge browser inference. |
| [`models/calibrated_nowcast_model.joblib`](file:///d:/SIH/models/calibrated_nowcast_model.joblib) | 1.6 KB | Serialized baseline regularized linear model benchmark. |

---

### C. Interactive Web GIS & Simulation Dashboard (`dashboard/`)
| File | Size | Role |
|:---|:---:|:---|
| [`dashboard/index.html`](file:///d:/SIH/dashboard/index.html) | 113 KB | Main dashboard application: Web GIS network map, HUD telemetry cards, interactive PINN inundation canvas, and scenario controls. |
| [`dashboard/css/dashboard.css`](file:///d:/SIH/dashboard/css/dashboard.css) | 19 KB | Modern glassmorphism UI stylesheet with dark theme, responsive grid, animated radar sweep, and alert badges. |
| [`dashboard/js/app.js`](file:///d:/SIH/dashboard/js/app.js) | 96 KB | Application controller handling real-time telemetry streaming, audio siren triggers, and interactive scenario playback. |
| [`dashboard/js/neural_inference.js`](file:///d:/SIH/dashboard/js/neural_inference.js) | 7.2 KB | Pure client-side JavaScript 1D-CNN + BiLSTM forward pass engine executing deep neural inference in real-time. |
| [`dashboard/js/pinn_canvas.js`](file:///d:/SIH/dashboard/js/pinn_canvas.js) | 56 KB | WebGL / 2D Canvas hydrodynamic flood renderer visualizing water depth contours, velocity vectors, and digital elevation models. |
| [`dashboard/js/snn_gate.js`](file:///d:/SIH/dashboard/js/snn_gate.js) | 5.4 KB | Neuromorphic SNN LIF membrane dynamics simulator running on browser sensor nodes. |
| [`dashboard/js/spatial_gate.js`](file:///d:/SIH/dashboard/js/spatial_gate.js) | 5.5 KB | Multi-station spatial localization ($L$-score) verification gate. |
| [`dashboard/js/stations_data.js`](file:///d:/SIH/dashboard/js/stations_data.js) | 10 KB | Geographic metadata and coordinate geometries for all AWS stations and GAGAN GNSS nodes. |
| [`dashboard/js/historical_events.js`](file:///d:/SIH/dashboard/js/historical_events.js) | 7.3 KB | Curated historical storm replay events (Guwahati 2013, Dharamshala, Uttarakhand flash floods). |

---

### D. Executable Pipeline Run & Simulation Entrypoints
| File | Size | Role |
|:---|:---:|:---|
| [`pipeline.py`](file:///d:/SIH/pipeline.py) | 2.6 KB | Master CLI runner executing Phase A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ D pipeline sequentially or by individual phase. |
| [`simulator.py`](file:///d:/SIH/simulator.py) | 20.7 KB | Real-time multi-station streaming simulator featuring live SNN spiking, neural nowcast scoring, and PINN handoff. |
| [`test_nowcasting_pipeline.py`](file:///d:/SIH/test_nowcasting_pipeline.py) | 7.9 KB | End-to-end integration test validating all 7 modules in a continuous operational loop. |
| [`mosdac_downloader.py`](file:///d:/SIH/mosdac_downloader.py) | 9.2 KB | Automated ISRO MOSDAC INSAT-3D/3DR satellite and in-situ AWS product downloader. |
| [`mosdac_api/mdapi.py`](file:///d:/SIH/mosdac_api/mdapi.py) | 35 KB | Python API client library for querying and downloading data from MOSDAC endpoints. |
| [`scan_all_regions.py`](file:///d:/SIH/scan_all_regions.py) | 4.4 KB | Multi-region spatial search utility across Himalayan river basins. |
| [`test_multi_region.py`](file:///d:/SIH/test_multi_region.py) | 0.4 KB | Quick verification test for multi-region GAGAN and AWS station discovery. |
| [`visualize_dataset.py`](file:///d:/SIH/visualize_dataset.py) | 11.7 KB | Visual chart generator for multi-region sensor distributions, rainfall histograms, and diurnal curves. |

---

### E. Showcase Reports & Visual Deliverables (`outputs/`)
| File | Size | Role |
|:---|:---:|:---|
| [`outputs/india_aws_network_map.html`](file:///d:/SIH/outputs/india_aws_network_map.html) | 92 KB | Interactive Folium Web GIS map of all IMD and ISRO AWS stations across India. |
| [`outputs/india_aws_network_map.png`](file:///d:/SIH/outputs/india_aws_network_map.png) | 666 KB | Rendered high-resolution geographic distribution map. |
| [`outputs/cloudburst_simulation_showcase.html`](file:///d:/SIH/outputs/cloudburst_simulation_showcase.html) | 20 KB | Standalone interactive HTML demo of the cloudburst simulation flow. |
| [`outputs/spatial_interpolation_demo.html`](file:///d:/SIH/outputs/spatial_interpolation_demo.html) | 11 KB | Interactive 2D Inverse Distance Weighting (IDW) interpolation demonstration. |
| [`outputs/dataset_dashboard.png`](file:///d:/SIH/outputs/dataset_dashboard.png) | 285 KB | Comprehensive multi-panel training dataset analysis and class imbalance visualization. |
| [`outputs/simulation_dashboard.png`](file:///d:/SIH/outputs/simulation_dashboard.png) | 162 KB | Live terminal telemetry dashboard screen capture. |
| [`outputs/spatial_interpolation_demo.png`](file:///d:/SIH/outputs/spatial_interpolation_demo.png) | 579 KB | Visual comparison of raw gauge point readings vs smoothed 2D spatial convective field. |
| [`outputs/Cloudburst_Simulation_Report.pdf`](file:///d:/SIH/outputs/Cloudburst_Simulation_Report.pdf) | 240 KB | Official executive simulation report in PDF format. |
| [`outputs/simulation_report.pdf`](file:///d:/SIH/outputs/simulation_report.pdf) | 377 KB | Comprehensive system validation report in PDF format. |
| [`outputs/presentation_slide_draft.md`](file:///d:/SIH/outputs/presentation_slide_draft.md) | 8.8 KB | Word-for-word competition pitching presentation script and literature benchmark comparison. |
| [`outputs/training_report.md`](file:///d:/SIH/outputs/training_report.md) | 6.4 KB | Full technical training report on the 913,536-sample multi-region dataset with 1D-CNN + BiLSTM metrics. |
| [`outputs/research_paper_benchmarks.md`](file:///d:/SIH/outputs/research_paper_benchmarks.md) | 9.3 KB | Peer-reviewed literature benchmark tables (HRRR, PySTEPS, DGMR, NowcastNet) and spatial double-penalty analysis. |
| [`outputs/simulation_run_report.md`](file:///d:/SIH/outputs/simulation_run_report.md) | 9.9 KB | Mathematical coupling and step-by-step logs from the multi-station simulation run. |
| [`outputs/station_quality_report.csv`](file:///d:/SIH/outputs/station_quality_report.csv) | 5.8 KB | Phase B sensor validation report showing pass/fail status and data completeness. |
| [`outputs/region_scores.csv`](file:///d:/SIH/outputs/region_scores.csv) | 1.3 KB | Phase A composite scores across all candidate Himalayan regions. |
| [`outputs/cloudburst_events.csv`](file:///d:/SIH/outputs/cloudburst_events.csv) | 500 KB | Labeled storm catalog with detected convective spikes and candidate cloudburst events. |
| [`outputs/all_india_aws_stations.csv`](file:///d:/SIH/outputs/all_india_aws_stations.csv) | 180 KB | Cleaned catalog of all operating AWS stations with coordinates and regional tags. |

---

### F. Configuration, Documentation & Diagnostics
| File | Size | Role |
|:---|:---:|:---|
| [`.gitignore`](file:///d:/SIH/.gitignore) | 1.8 KB | Comprehensive Git ignore rules protecting credentials and excluding large raw data archives. |
| [`config.example.json`](file:///d:/SIH/config.example.json) | 0.1 KB | Clean template for configuring MOSDAC API credentials without exposing private keys. |
| [`PS26077_Nowcasting_Research_and_Build_Plan.md`](file:///d:/SIH/PS26077_Nowcasting_Research_and_Build_Plan.md) | 33.5 KB | Comprehensive system architecture design, literature review, and MoES problem statement mapping. |
| [`research_paper_benchmarks.md`](file:///d:/SIH/research_paper_benchmarks.md) | 12.6 KB | Rigorous scientific benchmarks contextualizing in-situ point sensing against gridded radar models. |
| [`presentation_slide_draft.md`](file:///d:/SIH/presentation_slide_draft.md) | 8.8 KB | Competition slide deck draft and speaker notes. |
| [`training_report.md`](file:///d:/SIH/training_report.md) | 6.4 KB | High-level training report on the multi-region dataset. |
| [`SIH.txt`](file:///d:/SIH/SIH.txt) | 8.8 KB | Problem statement reference notes and team specifications. |
| [`scratch/`](file:///d:/SIH/scratch) | ~60 KB | Research diagnostic and benchmark scripts (`benchmark_snn.py`, `audit_pod.py`, `audit_fn.py`, `test_spatial_contrast.py`, etc.). |

---

## 3. Excluded Files & Operational Rationale

The following directories and files are deliberately excluded from Git tracking in `.gitignore`:

1. **`data/` (> 636 MB)**:
   - Raw IMD gridded rain parquets (`data/raw/imd_rain/`) and ERA5 NetCDF files (`data/raw/era5/`).
   - GAGAN raw IWV text archive (`data/raw/gagan_iwv_v1.txt`, 13.6 MB).
   - Reason: Exceeds GitHub recommended repository size and pushes would fail without Git LFS. Data can be re-downloaded via `mosdac_downloader.py`.
2. **Root Raw Spreadsheets & Large Files**:
   - `ASSAM_ALL_2013-02-01_2014-03-15_Sep2026_177056.csv` (21.9 MB).
   - `guwahati_weather_cleaned.xlsx` (12.4 MB), `open-meteo-27.59N79.10E164m.xlsx` (4.8 MB), `guwahati_weather_feb2013_mar2014.xlsx` (1.4 MB).
   - `AWS_LIST_IN_SITU.pdf` (6.1 MB).
   - Reason: Raw observational inputs that bloat Git history. All necessary features and extracted weights are preserved in `models/` and `outputs/`.
3. **`nowcast/` Virtual Environment (~1.1 GB, 34,300+ files)**:
   - Python virtual environment. Never committed to version control.
4. **`config.json` & `mosdac_api/config.json`**:
   - Private credentials and MOSDAC API tokens. Excluded for security.
5. **Caches & Scratch Logs**:
   - `__pycache__/`, `*.pyc`, `*.log`, `scan_stations.py`, `verify_data.py`.

---

## 4. End-to-End Verification Status

The repository code has been thoroughly verified:
- `test_nowcasting_pipeline.py`: **PASS** (Exit Code 0). All 7 modules execute end-to-end:
  - Module 1: Ingestion & 60-min feature buffer ($R = 100\text{ mm/hr}$, $RI = +160\text{ mm/hr}^2$).
  - Module 2: SNN Edge Gate triggers $\text{⚡ SPIKE}$, escalates telemetry to 5-min mode, requests satellite tile.
  - Module 3/4: 1D-CNN + BiLSTM Deep Neural Nowcaster evaluates convective risk $P(\text{CB})$.
  - Module 5: 2D Spatial Fusion Grid builds interpolated risk field across gauge gaps.
  - Module 6: PINN Hydrodynamic Handoff couples excess runoff $Q(x,y,t)$ to 2D Shallow Water Equations.
- `simulator.py`: **PASS** (Exit Code 0). Multi-station streaming, burst triggering, and PINN handoff verified.
