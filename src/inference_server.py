import json
import random
import time
import requests
import torch
import numpy as np
from pathlib import Path

ROOT = Path("d:/SIH")
MODELS_DIR = ROOT / "models"

import sys
import logging
sys.path.insert(0, str(ROOT))
from src.train_neural_nowcaster_v2 import CloudburstCNNBiLSTM
from src.pinn_swe import SharedSWEPINN

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
        logging.info("Loading models into memory...")
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

    def fetch_realtime_imd_aws(self):
        try:
            import os
            # Attempt to fetch real-time data from IMD API (Endpoint #5/#9)
            url = "https://api.imd.gov.in/api/v1/aws"
            
            # The API returns 401 Unauthorized without a valid API key.
            # Set this via terminal: $env:IMD_API_KEY="your_token_here"
            api_key = os.environ.get("IMD_API_KEY", "")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                # Assuming the response is formatted as station_id: {metrics...}
                return data
            else:
                logging.error(f"API returned status {resp.status_code}")
                return {}
        except Exception as e:
            logging.error(f"API Fetch Failed: {e}")
            # Do NOT mock data. Return empty to trigger 'Loading...' state in UI.
            return {}

    def predict_nowcast(self):
        aws_data = self.fetch_realtime_imd_aws()
        results = []
        
        for stn_id, data in aws_data.items():
            if self.cnn is not None:
                x_raw = np.zeros(len(self.ordered_feats))
                for i, f in enumerate(self.ordered_feats):
                    if f in data: x_raw[i] = data[f]
                
                x_norm = (x_raw - self.scaler_mean) / self.scaler_scale
                x_t = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    prob = self.cnn.predict_proba(x_t).item()
            else:
                prob = random.uniform(0, 1)

            if prob >= 0.85: tier = "red"
            elif prob >= 0.60: tier = "orange"
            elif prob >= 0.35: tier = "yellow"
            else: tier = "green"

            stn_res = {
                "id": stn_id,
                "gate": "a",
                "P_CB": float(prob),
                "tier": tier,
                "metrics": data
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
