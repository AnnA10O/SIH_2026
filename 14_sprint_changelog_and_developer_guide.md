# Sprint Changelog & Developer Guide

This document summarizes the core development activities, bug fixes, architecture decisions, and data management tasks executed since the previous major repository push. It serves as a hand-off document for any incoming developers.

---

## 1. Architectural Changes & Feature Updates

### Real-Time Inference Integration
- **`src/inference_server.py`**: Built a lightweight inference orchestrator. It simultaneously loads the `CloudburstCNNBiLSTM` weights and the `SharedSWEPINN` simulator into GPU/CPU memory. 
- **IMD API Handoff**: Hooked the inference engine up to the official IMD AWS endpoint (`https://api.imd.gov.in/api/v1/aws`) using Python `requests`, allowing the model to perform forward passes on live telemetry rather than simulated CSV data.
- **Logging Pipeline**: Stripped out bare `print()` statements across the backend and replaced them with a standard `logging` implementation. All inference engine states, model loads, and HTTP errors are now cleanly piped to `logs/inference.log`.

### Dashboard & UI Restructure
- **UI Consolidation**: Re-architected the frontend directories. The legacy and ideation files were moved into `UI/dashboard/`, while the new live simulation UI was isolated into `UI/realtime/`.
- **API Proxy Launcher**: Upgraded `run_dashboard.py` from a simple HTTP file server to a hybrid launcher. It now prompts the user via a terminal menu to select a UI, and it natively proxies `/api/nowcast` GET requests to the PyTorch inference server without requiring an external framework like FastAPI.
- **Live UI Polling**: The `UI/realtime/index.html` frontend was stripped of all randomized mock functions (`randVal()`). It now utilizes `fetch()` to poll the `/api/nowcast` endpoint every 5 seconds. Map simulation bounding boxes (`L.rectangle`) were removed per design requirements to keep the UI clean.

### Environment & Reproducibility
- **`requirements.txt` & `README.md`**: Created a clean dependency manifest and a quickstart guide so that any developer cloning the repo can instantly spin up the environment.

---

## 2. Problems Encountered & Solutions

| Problem | Root Cause | Solution Implemented |
|---------|------------|----------------------|
| **Spatiotemporal Transformer failed to learn** | The multi-head cross-attention Transformer model evaluated severely poorly (ROC-AUC 0.50, CSI 0.0069). It degenerated to predicting 100% positive classes due to the lack of deep sequential temporal grids in the current feature subset. | Abandoned the Transformer. Confirmed that the current **1D-CNN + BiLSTM** remains our statistically superior and fully validated baseline for tabular AWS sequences. |
| **API returned `401 Unauthorized`** | The public IMD API Endpoint requires an authorized developer token. | Removed hardcoded data generation. Implemented a graceful failure state where the UI persistently shows "Loading...". Added an OS environment variable trap (`$env:IMD_API_KEY`) to dynamically inject the token into the `Authorization: Bearer` header. |
| **Frontend Map Clutter** | The PINN SWE simulation engine was drawing static bounding boxes across the UI map, cluttering the view for standard AWS point predictions. | Stripped `L.rectangle` generation from Leaflet map entirely; the map now cleanly focuses on the sensor network dots. |

---

## 3. Dataset Management

### Downloaded / Integrated
- **ISRO MOSDAC (INSAT-3D/3DR)**: Integrated `.nc` (NetCDF4) and `.h5` satellite swath datasets during the feature fusion phase.
- **IMD AWS Historicals**: Heavy historical tabular data (`guwahati_weather_cleaned.xlsx`, `ASSAM_ALL...csv`) remains standard for the 60-fold cross-validation pipeline.

### Deleted / Ignored
- **Mock Fallback Data**: Completely deleted the fallback dictionaries in `inference_server.py`. The system now strictly relies on truth data from the API.

### Reviewed
- **Feature Matrices**: The 4-feature temporal sequence `[R_60, R_30, R, RI]` was reviewed and confirmed to be highly sensitive to the `CNN+BiLSTM` architecture.

---

## 4. Planned Next Steps (Roadmap)

1. **MOSDAC API Integration**: The realtime engine currently pulls from IMD AWS. We need to integrate the `mosdac_api/mdapi.py` into `inference_server.py` to fetch live INSAT-3D Cloud Top Temperature (CTT) grids and fuse them before the forward pass.
2. **PINN Visualization**: The backend PINN engine calculates depth and velocity matrices ($h, u, v$), but they aren't currently being drawn on the frontend. We need to implement a WebGL or Canvas layer to render the flood inundation polygons when a red alert triggers.
3. **Edge Deployment**: Finalize the MCU C-code translation (`src/snn_gate_mcu.c`) for the neuromorphic SNN gates to be deployed on actual hardware nodes (e.g., ESP32/STM32).

---

## 5. CV1.2 Reporting Protocol Status

**Status: ~85% Completed**

The CV1.2 (Cross-Validation / Continuous Verification) protocol requires strict algorithmic benchmarking and false-alarm auditing. 

- **Completed Tasks:**
  - 60-fold Leave-One-Event-Out Cross Validation (LOEO-CV) is fully implemented in `src/phase_d_training.py`.
  - Strict meteorological verification metrics (CSI, FAR, POD, Brier Score) have been generated and documented in `outputs/training_report.md`.
  - The False Alarm Reduction Plan (`outputs/11_false_alarm_reduction_plan.md`) is documented and addresses the spatial clustering requirements.
  
- **Remaining Tasks:**
  - **Live Environment Audit**: The CV1.2 protocol requires evaluating FAR (False Alarm Ratio) in a live, streaming environment over a 72-hour sustained test. Now that the `inference_server.py` is hooked up to live IMD APIs, we need to run a 3-day daemon script to record the live predictions against actual rainfall to finalize the CV1.2 compliance report.

---

## 6. Crucial Developer Notes

- **Lazy Loading**: PyTorch (`torch`) is a massive library. To prevent the basic web server from taking 15 seconds to start up, `inference_server.py` is dynamically imported *only* when the `/api/nowcast` route is hit in `run_dashboard.py`.
- **API Keys**: Ensure `IMD_API_KEY` is in your environment variables. Without it, the application correctly assumes a degraded state.
- **Log Files**: Do not use `print()`. Use `logging.info()` or `logging.error()`. All outputs are written to `logs/inference.log`. When debugging the inference server, tail this log file: `Get-Content logs\inference.log -Wait`.
