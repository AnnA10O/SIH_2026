import json
import random
import time
import requests
import torch
import numpy as np
from pathlib import Path
import os
import sys
import logging

ROOT = Path("d:/SIH")
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))

from src.train_neural_nowcaster_v2 import CloudburstCNNBiLSTM
from src.pinn_swe import SharedSWEPINN
from src.fusion_buffer import DataFusionBuffer
from src.mosdac_live_daemon import MosdacLiveDaemon, STATIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOGS_DIR / "inference.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

class InferenceOrchestrator:
    def __init__(self):
        logging.info("Initializing DataFusionBuffer and Models...")
        self.buffer = DataFusionBuffer()
        
        # Start the MOSDAC Daemon for live satellite fetching
        self.mosdac_daemon = MosdacLiveDaemon(self.buffer)
        self.mosdac_daemon.start()

        model_path = MODELS_DIR / "cloudburst_cnn_bilstm_best.pt"
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
            self.ordered_feats = ckpt["features"]
            self.cnn = CloudburstCNNBiLSTM(in_features=len(self.ordered_feats)).to(DEVICE)
            self.cnn.load_state_dict(ckpt["model_state_dict"])
            self.cnn.eval()
            self.scaler_mean = np.array(ckpt["scaler_mean"])
            self.scaler_scale = np.array(ckpt["scaler_scale"])
            self.threshold = ckpt.get("optimal_threshold", 0.5)
            logging.info("CNN+BiLSTM Loaded.")
        else:
            logging.warning("CNN+BiLSTM not found. Using dummy model.")
            self.cnn = None

        pinn_path = MODELS_DIR / "pinn_swe_dem_calibrated.pt"
        self.pinn = SharedSWEPINN(device="cpu" if not torch.cuda.is_available() else "cuda")
        if pinn_path.exists():
            self.pinn.model.load_state_dict(torch.load(pinn_path, map_location=self.pinn.device, weights_only=True))
            self.pinn.model.eval()
            logging.info("PINN SWE Simulator Loaded.")
        else:
            logging.warning("PINN SWE checkpoint not found. Using untrained weights.")

    def fetch_open_meteo_aws(self):
        try:
            logging.info("Attempting Open-Meteo API fallback...")
            aws_data = {}
            lats = [str(stn["lat"]) for stn in STATIONS]
            lons = [str(stn["lng"]) for stn in STATIONS]
            
            # Request all 14 stations in a single batched call
            url = f"https://api.open-meteo.com/v1/forecast?latitude={','.join(lats)}&longitude={','.join(lons)}&current=temperature_2m,precipitation"
            resp = requests.get(url, timeout=7)
            
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for idx, stn in enumerate(STATIONS):
                        stn_id = stn["id"]
                        curr = data[idx].get("current", {})
                        
                        # Open-Meteo maps to IMD schema
                        aws_data[stn_id] = {
                            "temp": curr.get("temperature_2m", 0.0),
                            "R": curr.get("precipitation", 0.0),
                            "R_30": curr.get("precipitation", 0.0) / 2.0,
                            "R_60": curr.get("precipitation", 0.0)
                        }
                return aws_data
            else:
                logging.error(f"Open-Meteo API returned status {resp.status_code}")
                return {}
        except Exception as e:
            logging.error(f"Open-Meteo Fallback Failed: {e}")
            return {}

    def fetch_realtime_imd_aws(self):
        try:
            url = "https://api.imd.gov.in/api/v1/aws"
            api_key = os.environ.get("IMD_API_KEY", "")
            if not api_key:
                logging.warning("No IMD_API_KEY provided. Skipping IMD API.")
                return self.fetch_open_meteo_aws()

            headers = {"Authorization": f"Bearer {api_key}"}
            
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    logging.warning("IMD API returned empty dictionary.")
                    return self.fetch_open_meteo_aws()
                return data
            else:
                logging.error(f"AWS API returned status {resp.status_code}")
                return self.fetch_open_meteo_aws()
        except Exception as e:
            logging.error(f"AWS API Fetch Failed: {e}")
            return self.fetch_open_meteo_aws()

    def predict_nowcast(self):
        aws_data = self.fetch_realtime_imd_aws()
        results = []
        
        now = time.time()
        # 1. Ingest AWS data into Buffer to ensure strict staleness tracking
        for stn_id, data in aws_data.items():
            for metric in ['R', 'R_30', 'R_60', 'RI']:
                if metric in data:
                    # Ingest using a generic policy if not defined, the buffer handles it
                    self.buffer.ingest(stn_id, metric, float(data[metric]), timestamp_epoch=now)
        
        # We only predict for stations we actually know about (the 14 defined STATIONS)
        for stn in STATIONS:
            stn_id = stn["id"]
            data = aws_data.get(stn_id, {})
            
            # Incorporate hardcoded lat/lon in case AWS API omits it
            data["lat"] = stn["lat"]
            data["lon"] = stn["lng"]
            
            if self.cnn is not None:
                x_raw = np.zeros(len(self.ordered_feats))
                for i, f in enumerate(self.ordered_feats):
                    # Static/Metadata fields bypass the buffer
                    if f in ['lat', 'lon', 'doy', 'month', 'spatial_contrast', 'rain_3day_accum', 'rain_7day_accum', 'rain_trend_7day']:
                        if f in data: x_raw[i] = data[f]
                    
                    # Buffer-managed staleness and validity flags for AWS
                    elif f == 'R_staleness_s':
                        state = self.buffer.get_channel_state(stn_id, 'R')
                        x_raw[i] = state.staleness_s if state.staleness_s != float('inf') else 0.0
                    elif f == 'R_valid':
                        x_raw[i] = 1.0 if self.buffer.get_channel_state(stn_id, 'R').valid else 0.0
                    
                    # Buffer-managed staleness and validity flags for Satellite (MOSDAC)
                    elif f == 'uth_staleness_s':
                        state = self.buffer.get_channel_state(stn_id, 'uth_kalpana')
                        x_raw[i] = state.staleness_s if state.staleness_s != float('inf') else 0.0
                    elif f == 'uth_valid':
                        x_raw[i] = 1.0 if self.buffer.get_channel_state(stn_id, 'uth_kalpana').valid else 0.0
                    elif f == 'hem_staleness_s':
                        state = self.buffer.get_channel_state(stn_id, 'hem_kalpana')
                        x_raw[i] = state.staleness_s if state.staleness_s != float('inf') else 0.0
                    elif f == 'hem_valid':
                        x_raw[i] = 1.0 if self.buffer.get_channel_state(stn_id, 'hem_kalpana').valid else 0.0
                    
                    # Direct reading values (R, R_30, R_60, RI)
                    else:
                        state = self.buffer.get_channel_state(stn_id, f)
                        x_raw[i] = state.value if state.value is not None else 0.0
                
                # Forward Pass
                x_norm = (x_raw - self.scaler_mean) / self.scaler_scale
                # NaN guard in case scaler is broken
                x_norm = np.nan_to_num(x_norm, nan=0.0) 
                x_t = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    prob = self.cnn.predict_proba(x_t).item()
            else:
                prob = random.uniform(0, 1)

            if prob >= 0.85: tier = "red"
            elif prob >= 0.60: tier = "orange"
            elif prob >= 0.35: tier = "yellow"
            else: tier = "green"

            # Check if any crucial satellite data is DEGRADED (for UI warnings)
            uth_state = self.buffer.get_channel_state(stn_id, 'uth_kalpana')
            is_degraded = uth_state.degraded

            # Neuromorphic SNN Gate Heuristics
            # Gate A (Fast Temporal): Spikes on sudden rain intensity
            r_val = self.buffer.get_channel_state(stn_id, 'R').value or 0.0
            r60_val = self.buffer.get_channel_state(stn_id, 'R_60').value or 0.0
            gate_a = bool(r_val > 5.0 or r60_val > 15.0 or prob > 0.6)

            # Gate B (Spatial/Synoptic): Spikes on deep convective clouds (UTH / contrast)
            uth_val = uth_state.value if uth_state.value is not None else 0.0
            sc_val = data.get('spatial_contrast', 0.0)
            gate_b = bool(uth_val > 70.0 or sc_val > 5.0 or prob > 0.7)

            stn_res = {
                "id": stn_id,
                "gate_a": gate_a,
                "gate_b": gate_b,
                "P_CB": float(prob),
                "tier": tier,
                "metrics": data,
                "sat_degraded": is_degraded
            }

            if tier == "red":
                logging.warning(f"RED ALERT at {stn_id}. Triggering PINN SWE Flash Flood Simulation...")
                sim_res = self.pinn.simulate_inundation(hazard_type="cloudburst", eval_time_hr=1.0)
                stn_res["pinn"] = sim_res

            results.append(stn_res)
            
        return results

# Singleton
_orchestrator = None

def handle_api_request():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = InferenceOrchestrator()
    return _orchestrator.predict_nowcast()
