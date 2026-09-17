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

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))

from src.train_neural_nowcaster_v2 import CloudburstCNNBiLSTM
from src.pinn_swe import SharedSWEPINN
from src.fusion_buffer import DataFusionBuffer
from src.mosdac_live_daemon import MosdacLiveDaemon, ALL_STATIONS
from src.aws_live_daemon import AwsLiveDaemon
from src.snn_gate import CloudburstSNNGate, ThunderstormSNNGate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Main server logger
logging.basicConfig(
    filename=LOGS_DIR / "server.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Daemons logger
daemon_logger = logging.getLogger("daemons")
daemon_logger.setLevel(logging.INFO)
dh = logging.FileHandler(LOGS_DIR / "daemons.log")
dh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
daemon_logger.addHandler(dh)
daemon_logger.propagate = False

class InferenceOrchestrator:
    def __init__(self):
        logging.info("Initializing DataFusionBuffer and Models...")
        self.buffer = DataFusionBuffer()
        
        # Stateful Neuromorphic SNN Gates per station
        self.snn_gates = {}
        
        # Start the MOSDAC Daemon for live satellite fetching
        self.mosdac_daemon = MosdacLiveDaemon(self.buffer)
        self.mosdac_daemon.start()

        # Start the AWS Daemon for live ground truth and IDW Virtual Grid
        self.aws_daemon = AwsLiveDaemon(self.buffer)
        self.aws_daemon.start()

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

    def get_basin_for_station(self, stn_id: str) -> str:
        basin_map = {
            'UK-1': 'rudraprayag', 'UK-2': 'chamoli', 'UK-3': 'uttarkashi',
            'UK-4': 'pithoragarh', 'UK-5': 'tehri', 'UK-6': 'pauri', 
            'UK-7': 'nainital', 'UK-8': 'almora'
        }
        return basin_map.get(stn_id, 'rudraprayag')

    def predict_nowcast(self):
        results = []
        
        now = time.time()

        
        # We predict for all stations (physical + virtual grid)
        for stn in ALL_STATIONS:
            stn_id = stn["id"]
            
            data = {}
            # We don't have aws_data dict anymore. We get it straight from the buffer in the feature loop below.
            
            # Incorporate hardcoded lat/lon
            data["lat"] = stn["lat"]
            data["lon"] = stn["lng"]
            
            # Retrieve or initialize SNN Gates for this specific station
            if stn_id not in self.snn_gates:
                self.snn_gates[stn_id] = (CloudburstSNNGate(), ThunderstormSNNGate())
            cb_gate, ts_gate = self.snn_gates[stn_id]

            # Gate A: Neuromorphic Cloudburst SNN (Fast Temporal)
            r_val = self.buffer.get_channel_state(stn_id, 'R').value or 0.0
            ri_val = self.buffer.get_channel_state(stn_id, 'RI').value or 0.0
            gate_a_res = cb_gate.evaluate({"R": r_val, "RI": ri_val})
            gate_a = gate_a_res["fired_spike"]

            # Gate B: Neuromorphic Thunderstorm SNN (Spatial/Synoptic)
            uth_state = self.buffer.get_channel_state(stn_id, 'uth_kalpana')
            uth_val = uth_state.value if uth_state.value is not None else 0.0
            sc_val = data.get('spatial_contrast', 0.0)
            gate_b_res = ts_gate.evaluate({
                "IWV_trend": (uth_val / 100.0) * 5.0,
                "pressure_trend": (sc_val / 10.0) * 3.0,
                "wind_shift": 0.0,
                "temp_drop": 0.0,
                "CAPE_trend": 0.0
            })
            gate_b = gate_b_res["fired_spike"]

            # Only run the heavy CNN if one of the SNN gates fired
            prob = 0.0
            if gate_a or gate_b:
                if self.cnn is not None:
                    x_raw = np.zeros(len(self.ordered_feats))
                    for i, f in enumerate(self.ordered_feats):
                        if f in ['lat', 'lon', 'doy', 'month', 'spatial_contrast', 'rain_3day_accum', 'rain_7day_accum', 'rain_trend_7day']:
                            if f in data: x_raw[i] = data[f]
                        elif f == 'R_staleness_s':
                            state = self.buffer.get_channel_state(stn_id, 'R')
                            x_raw[i] = state.staleness_s if state.staleness_s != float('inf') else 0.0
                        elif f == 'R_valid':
                            x_raw[i] = 1.0 if self.buffer.get_channel_state(stn_id, 'R').valid else 0.0
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
                        else:
                            state = self.buffer.get_channel_state(stn_id, f)
                            x_raw[i] = state.value if state.value is not None else 0.0
                    
                    x_norm = (x_raw - self.scaler_mean) / self.scaler_scale
                    x_norm = np.nan_to_num(x_norm, nan=0.0) 
                    x_t = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    
                    with torch.no_grad():
                        prob = self.cnn.predict_proba(x_t).item()
                else:
                    prob = random.uniform(0.5, 1)

            if prob >= 0.85: tier = "red"
            elif prob >= 0.60: tier = "orange"
            elif prob >= 0.35: tier = "yellow"
            else: tier = "green"

            # Check if any crucial satellite data is DEGRADED (for UI warnings)
            is_degraded = uth_state.degraded

            r_state = self.buffer.get_channel_state(stn_id, 'R')
            is_virtual = r_state.is_virtual
            nearest_stn_dist_km = float(r_state.nearest_stn_dist_km) if r_state.nearest_stn_dist_km is not None else 0.0

            # Ensure actual IDW/Physical values are passed to frontend metrics
            # Ensure actual IDW/Physical values are passed to frontend metrics
            data["rain"] = float(r_state.value) if r_state.value is not None else 0.0
            data["rain_intensity"] = float(self.buffer.get_channel_state(stn_id, 'RI').value or 0.0)

            stn_res = {
                "id": stn_id,
                "lat": stn["lat"],
                "lng": stn["lng"],
                "gate_a": gate_a,
                "gate_b": gate_b,
                "P_CB": float(prob),
                "tier": tier,
                "is_virtual": is_virtual,
                "nearest_stn_dist_km": nearest_stn_dist_km,
                "metrics": data,
                "sat_degraded": is_degraded
            }

            if tier == "red" and not is_virtual:
                logging.warning(f"RED ALERT at {stn_id}. Triggering PINN SWE Flash Flood Simulation...")
                region = self.get_basin_for_station(stn_id)
                rain_int = data.get("rain_intensity", 60.0)
                sim_res = self.pinn.simulate_inundation(
                    hazard_type="cloudburst", 
                    eval_time_hr=1.0,
                    region=region,
                    rain_intensity=rain_int
                )
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
