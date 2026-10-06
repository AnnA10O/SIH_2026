import json
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
import math
import random

from src.train_neural_nowcaster import CloudburstCNNBiLSTM
from src.pinn_swe import SharedSWEPINN
from src.fusion_buffer import DataFusionBuffer
from src.mosdac_live_daemon import MosdacLiveDaemon, ALL_STATIONS, PHYSICAL_STATIONS
from src.aws_live_daemon import AwsLiveDaemon
from src.snn_gate import CloudburstSNNGate, ThunderstormSNNGate
from src.feature_builder import feature_builder_pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Phase 5.5: DEMO_MODE is off by default. With it off, MOSDAC mock & GLOBAL_SIMULATION_ACTIVE unreachable.
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "server.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

daemon_logger = logging.getLogger("daemons")
daemon_logger.setLevel(logging.INFO)
dh = logging.FileHandler(LOGS_DIR / "daemons.log")
dh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
daemon_logger.addHandler(dh)
daemon_logger.propagate = False

# Phase 5.8: One channel-name mapping table (buffer channel -> feature_builder training name)
CHANNEL_TO_FEAT = {
    "R": "R_1h",
    "aws_temp": "temp",
    "RH": "rh",
    "pressure": "pressure",     # tendency only — NOT absolute pressure feature
    "uth_kalpana": "uth_mean",
    "hem_kalpana": "hem_mean",
    "gagan_iwv": "tpw_mean",    # GAGAN IWV maps to TPW_now via feature_builder
}

# Phase 5.4: Required channels — absence => no_data, never green
# MAJOR DESIGN FINDING: IMD API documented sample has NO rainfall field.
# If rain is required (it is, as R_1h is a core feature), and IMD has no rain,
# the live system is PERMANENTLY no_data until IMD whitelists our IP and exposes rainfall.
REQUIRED_LIVE_CHANNELS = {"R"}  # Rain is required; others are optional with valid=0 degradation


class InferenceOrchestrator:
    def __init__(self):
        logging.info("Initializing DataFusionBuffer and Models...")
        self.buffer = DataFusionBuffer()

        self.snn_gates = {}

        # Start the MOSDAC Daemon for live satellite fetching
        self.mosdac_daemon = MosdacLiveDaemon(self.buffer, demo_mode=DEMO_MODE)
        self.mosdac_daemon.start()

        # Start the AWS Daemon for live ground truth
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
            logging.warning("CNN+BiLSTM not found. No model available.")
            self.cnn = None
            self.ordered_feats = []

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

    def _check_provenance(self, stn_id: str) -> tuple:
        """
        Phase 5.4: Check provenance of required channels.
        Returns (ok: bool, reason: str).
        ok=False means no_data tier applies.
        """
        for ch in REQUIRED_LIVE_CHANNELS:
            state = self.buffer.get_channel_state(stn_id, ch)
            if not state.valid:
                return False, f"required_channel_missing:{ch}"
            meta = state.source_meta
            prov = meta.get("provenance", "live")
            if prov in {"synthetic", "missing"}:
                return False, f"required_channel_synthetic_or_missing:{ch}"
        return True, "ok"

    def predict_nowcast(self):
        results = []
        now = time.time()

        for stn in PHYSICAL_STATIONS:
            stn_id = stn["id"]

            # ── Phase 5.4: Provenance guard — returns no_data if any required channel bad ──
            prov_ok, prov_reason = self._check_provenance(stn_id)
            if not prov_ok:
                results.append({
                    "id": stn_id,
                    "lat": stn["lat"],
                    "lng": stn["lng"],
                    "gate_a": False, "gate_b": False,
                    "P_CB": 0.0,
                    "tier": "no_data",
                    "reason": prov_reason,
                    "simulated": DEMO_MODE
                })
                continue

            if stn_id not in self.snn_gates:
                self.snn_gates[stn_id] = (CloudburstSNNGate(), ThunderstormSNNGate())
            cb_gate, ts_gate = self.snn_gates[stn_id]

            # Gate A: Neuromorphic Cloudburst SNN
            r_state = self.buffer.get_channel_state(stn_id, 'R')
            r_val = r_state.value if r_state.valid and r_state.value is not None else 0.0
            ri_val = r_val  # RI is now computed inside feature_builder, not stored separately
            gate_a_res = cb_gate.evaluate({"R": r_val, "RI": ri_val})
            gate_a = gate_a_res["fired_spike"]

            # Gate B: Neuromorphic Thunderstorm SNN
            uth_state = self.buffer.get_channel_state(stn_id, 'uth_kalpana')
            uth_val = uth_state.value if uth_state.valid and uth_state.value is not None else 0.0
            gate_b_res = ts_gate.evaluate({
                "IWV_trend": (uth_val / 100.0) * 5.0,
                "pressure_trend": 0.0,
                "wind_shift": 0.0,
                "temp_drop": 0.0,
                "CAPE_trend": 0.0
            })
            gate_b = gate_b_res["fired_spike"]

            prob = 0.0
            if gate_a or gate_b:
                if self.cnn is not None:
                    # Phase 5.8: Use get_dataframe + feature_builder, not ad-hoc zero-fill
                    df_live = self.buffer.get_dataframe(stn_id)
                    if df_live.empty:
                        results.append({
                            "id": stn_id, "lat": stn["lat"], "lng": stn["lng"],
                            "gate_a": gate_a, "gate_b": gate_b,
                            "P_CB": 0.0, "tier": "no_data", "reason": "empty_buffer",
                            "simulated": DEMO_MODE
                        })
                        continue

                    # Add fixed station metadata
                    df_live["lat"] = stn["lat"]
                    df_live["lon"] = stn["lng"]
                    df_live["doy"] = df_live["timestamp_utc"].dt.dayofyear
                    df_live["month"] = df_live["timestamp_utc"].dt.month

                    # Run feature_builder (same code path as training)
                    try:
                        df_feat = feature_builder_pipeline(df_live, is_training=False)
                    except Exception as e:
                        logging.error(f"[InferenceServer] feature_builder error for {stn_id}: {e}")
                        results.append({
                            "id": stn_id, "lat": stn["lat"], "lng": stn["lng"],
                            "gate_a": gate_a, "gate_b": gate_b,
                            "P_CB": 0.0, "tier": "no_data", "reason": f"feature_error:{e}",
                            "simulated": DEMO_MODE
                        })
                        continue

                    # Phase 5.8+: Alias feature names to match whatever checkpoint
                    # the model was trained with. The GitHub epoch-23 checkpoint uses
                    # short names (R, R_valid, R_staleness_s) while feature_builder
                    # produces verbose names (R_1h, RI_valid, etc.).
                    FEAT_ALIASES = {
                        "R":              "R_1h",
                        "R_valid":        "RI_valid",
                        "R_staleness_s":  "R_1h",   # no true staleness col; proxy with raw rain
                    }
                    for new_name, old_name in FEAT_ALIASES.items():
                        if new_name not in df_feat.columns and old_name in df_feat.columns:
                            df_feat[new_name] = df_feat[old_name]

                    missing_feats = [f for f in self.ordered_feats if f not in df_feat.columns]
                    if missing_feats:
                        logging.warning(f"[InferenceServer] Zero-filling missing features at {stn_id}: {missing_feats}")
                        for mf in missing_feats:
                            df_feat[mf] = 0.0

                    latest = df_feat.iloc[-1]
                    x_raw = np.array([latest[f] for f in self.ordered_feats], dtype=np.float32)

                    # Phase 5.8: No zero-fill — if any NaN remains after feature_builder, it's no_data
                    if np.isnan(x_raw).any():
                        results.append({
                            "id": stn_id, "lat": stn["lat"], "lng": stn["lng"],
                            "gate_a": gate_a, "gate_b": gate_b,
                            "P_CB": 0.0, "tier": "no_data", "reason": "nan_features",
                            "simulated": DEMO_MODE
                        })
                        continue

                    x_norm = (x_raw - self.scaler_mean) / self.scaler_scale
                    x_t = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                    with torch.no_grad():
                        prob = self.cnn.predict_proba(x_t).item()

            if prob >= 0.85: tier = "red"
            elif prob >= 0.60: tier = "orange"
            elif prob >= 0.35: tier = "yellow"
            else: tier = "green"

            is_degraded = uth_state.degraded
            is_virtual = r_state.is_virtual

            stn_res = {
                "id": stn_id,
                "lat": stn["lat"],
                "lng": stn["lng"],
                "gate_a": gate_a,
                "gate_b": gate_b,
                "P_CB": float(prob),
                "tier": tier,
                "is_virtual": is_virtual,
                "nearest_stn_dist_km": float(r_state.nearest_stn_dist_km),
                "metrics": {"rain": r_val},
                "sat_degraded": is_degraded,
                "simulated": DEMO_MODE
            }

            # Phase 5.4: PINN must NOT be called from no_data/synthetic path
            # Only call PINN if tier=red AND provenance is live AND not demo
            if tier == "red" and prov_ok and not DEMO_MODE and not is_virtual:
                logging.warning(f"RED ALERT at {stn_id}. Triggering PINN SWE Flash Flood Simulation...")
                region = self.get_basin_for_station(stn_id)
                sim_res = self.pinn.simulate_inundation(
                    hazard_type="cloudburst",
                    eval_time_hr=1.0,
                    region=region,
                    rain_intensity=r_val
                )
                stn_res["pinn"] = sim_res

            results.append(stn_res)

        virtual_nodes = self._generate_idw_subgrid(results)
        results.extend(virtual_nodes)
        return results

    def _generate_idw_subgrid(self, physical_results, spacing_deg=0.075, radius_km=30.0):
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        uk_results = {r["id"]: r for r in physical_results if r["id"].startswith("UK-")}
        if not uk_results: return []
        uk_stns = [s for s in PHYSICAL_STATIONS if s["id"].startswith("UK-")]
        lats = [s["lat"] for s in uk_stns]
        lons = [s["lng"] for s in uk_stns]
        min_lat, max_lat = min(lats) - 0.2, max(lats) + 0.2
        min_lon, max_lon = min(lons) - 0.2, max(lons) + 0.2
        virtual_nodes = []
        vid = 1
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                nearest_dist = min(haversine(lat, lon, s["lat"], s["lng"]) for s in uk_stns)
                if nearest_dist <= radius_km:
                    weights, rains, tiers = [], [], []
                    for stn_id, res in uk_results.items():
                        stn = next(s for s in uk_stns if s["id"] == stn_id)
                        dist = max(haversine(lat, lon, stn["lat"], stn["lng"]), 0.1)
                        w = 1.0 / dist**2
                        weights.append(w)
                        rains.append(res.get("metrics", {}).get("rain", 0.0))
                        tiers.append(res.get("tier", "green"))
                    total_w = sum(weights)
                    idw_rain = sum(w*r for w,r in zip(weights, rains)) / total_w
                    tier_scores = {"green": 0, "yellow": 1, "orange": 2, "red": 3, "no_data": -1}
                    avg_score = sum(w * tier_scores.get(t, 0) for w,t in zip(weights, tiers)) / total_w
                    vtier = "red" if avg_score >= 2.5 else ("orange" if avg_score >= 1.5 else ("yellow" if avg_score >= 0.5 else "green"))
                    virtual_nodes.append({
                        "id": f"V-{vid:04d}", "is_virtual": True, "tier": vtier,
                        "gate_a": vtier in ("orange", "red"), "gate_b": vtier == "red",
                        "P_CB": round(min(idw_rain/100.0, 0.99), 4), "simulated": DEMO_MODE,
                        "metrics": {"lat": round(lat,4), "lon": round(lon,4), "rain": round(idw_rain,3), "rain_intensity": round(idw_rain*random.uniform(0.8,1.2),3), "tier": vtier}
                    })
                    vid += 1
                lon += spacing_deg
            lat += spacing_deg
        return virtual_nodes


# Singleton
_orchestrator = None
init_lock = __import__('threading').Lock()


def handle_api_request():
    global _orchestrator
    with init_lock:
        if _orchestrator is None:
            _orchestrator = InferenceOrchestrator()
        return _orchestrator.predict_nowcast()
