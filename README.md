# SIH 2026 Cloudburst Early Warning System

Welcome to the Keraunos Cloudburst Early Warning System (PS26077). This repository contains the complete deep learning pipeline (1D-CNN + BiLSTM, PINN SWE, SNN) and real-time dashboard for predicting and simulating flash floods.

## Quickstart Guide

If you've freshly cloned this repository, follow these steps to run the real-time simulation UI:

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
To fetch real-time AWS telemetry data from the India Meteorological Department (IMD), you must provide a valid IMD API token.

**On Windows (PowerShell):**
```powershell
$env:IMD_API_KEY="your_imd_token_here"
```

**On Linux/Mac (Bash):**
```bash
export IMD_API_KEY="your_imd_token_here"
```
*(If you don't have a key, the backend will gracefully fail the live fetch and the UI will remain in a "Loading..." state.)*

### 3. Launch the Dashboard
Run the dashboard launcher to start the server:
```bash
python run_dashboard.py
```
Select **[3] Realtime Simulation UI** from the terminal menu. The browser will automatically open and connect to the local inference server.

## Project Structure
- `models/`: Pre-trained PyTorch weights for the CNN+BiLSTM and PINN models.
- `src/`: Core neural network architectures, data processing pipelines, and the `inference_server.py`.
- `UI/`: Frontend HTML/JS dashboards (Real-time and Mission Control).
- `outputs/`: Generated reports, metrics, plots, and performance analysis.
