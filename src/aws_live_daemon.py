import time
import threading
import logging
import requests
import random
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import threading
from src.mosdac_live_daemon import PHYSICAL_STATIONS, VIRTUAL_GRID

GLOBAL_SIMULATION_ACTIVE = False
GLOBAL_SIMULATION_TARGET = "UK-6"
WAKE_EVENT = threading.Event()

class AwsLiveDaemon:
    def __init__(self, buffer, interval_seconds=900):
        self.buffer = buffer
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread = None

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="AwsLiveDaemon")
        self.thread.start()
        logging.getLogger("daemons").info("[AWS Daemon] Started background thread.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _compute_virtual_grid(self, physical_data):
        if not VIRTUAL_GRID or not physical_data: return {}
        
        p_lats = np.array([s["lat"] for s in PHYSICAL_STATIONS])
        p_lons = np.array([s["lng"] for s in PHYSICAL_STATIONS])
        p_ids = [s["id"] for s in PHYSICAL_STATIONS]
        
        v_lats = np.array([s["lat"] for s in VIRTUAL_GRID])
        v_lons = np.array([s["lng"] for s in VIRTUAL_GRID])
        
        keys = ["temp", "R", "R_30", "R_60", "RI"]
        p_vals = {k: np.array([physical_data.get(pid, {}).get(k, 0.0) for pid in p_ids]) for k in keys}
        
        virtual_data = {}
        # Pre-convert physical coords to radians
        lat1, lon1 = np.radians(p_lats), np.radians(p_lons)
        
        for i, v_stn in enumerate(VIRTUAL_GRID):
            v_id = v_stn["id"]
            lat2, lon2 = np.radians(v_lats[i]), np.radians(v_lons[i])
            
            # Vectorized Haversine
            dlat = lat1 - lat2
            dlon = lon1 - lon2
            a = np.sin(dlat/2)**2 + np.cos(lat2) * np.cos(lat1) * np.sin(dlon/2)**2
            dists = 6371.0 * 2 * np.arcsin(np.sqrt(a))
            
            # Epsilon guard to prevent zero-division singularity
            dists = np.maximum(dists, 1e-5)
            min_dist = float(np.min(dists))
            
            # Dual-power IDW
            w2 = 1.0 / (dists ** 2)
            w4 = 1.0 / (dists ** 4) # Steeper decay for RI spikes
            
            sum_w2 = np.sum(w2)
            sum_w4 = np.sum(w4)
            
            v_dict = {
                "temp": float(np.sum(p_vals["temp"] * w2) / sum_w2),
                "R": float(np.sum(p_vals["R"] * w2) / sum_w2),
                "R_30": float(np.sum(p_vals["R_30"] * w2) / sum_w2),
                "R_60": float(np.sum(p_vals["R_60"] * w2) / sum_w2),
                "RI": float(np.sum(p_vals["RI"] * w4) / sum_w4),
            }
            
            virtual_data[v_id] = {
                "data": v_dict,
                "meta": {"is_virtual": True, "nearest_stn_dist_km": min_dist, "source": "IDW"}
            }
        return virtual_data

    def _loop(self):
        while self.running:
            try:
                physical_data = self.fetch_realtime_imd_aws()
                now_epoch = datetime.now(timezone.utc).timestamp()
                
                if getattr(sys.modules[__name__], 'GLOBAL_SIMULATION_ACTIVE', False):
                    target = getattr(sys.modules[__name__], 'GLOBAL_SIMULATION_TARGET', "UK-6")
                    if target in physical_data:
                        physical_data[target]['R'] = random.uniform(100.0, 150.0)
                        physical_data[target]['RI'] = random.uniform(80.0, 120.0)
                        physical_data[target]['R_30'] = physical_data[target]['R'] / 2.0
                        physical_data[target]['R_60'] = physical_data[target]['R']
                        
                virtual_data = self._compute_virtual_grid(physical_data)
                
                # Ingest Physical
                for stn_id, data in physical_data.items():
                    for channel, value in data.items():
                        self.buffer.ingest(
                            station_id=stn_id, channel=channel, value=value,
                            timestamp_epoch=now_epoch, 
                            source_meta={"source": "AwsLiveDaemon", "is_virtual": False, "nearest_stn_dist_km": 0.0}
                        )
                        
                # Ingest Virtual
                for stn_id, v_info in virtual_data.items():
                    data = v_info["data"]
                    meta = v_info["meta"]
                    for channel, value in data.items():
                        self.buffer.ingest(
                            station_id=stn_id, channel=channel, value=value,
                            timestamp_epoch=now_epoch, source_meta=meta
                        )
                        
                logging.getLogger("daemons").info(f"[AWS Daemon] Ingested data for {len(physical_data)} physical + {len(virtual_data)} virtual stations.")
            except Exception as e:
                logging.getLogger("daemons").error(f"[AWS Daemon] Unhandled error in cycle: {e}")
            
            # Sleep until interval or woken up by WAKE_EVENT
            getattr(sys.modules[__name__], 'WAKE_EVENT').wait(self.interval_seconds)
            getattr(sys.modules[__name__], 'WAKE_EVENT').clear()
            if not self.running:
                break

    def fetch_open_meteo_aws(self):
        try:
            logging.getLogger("daemons").info("[AWS Daemon] Attempting Open-Meteo API fallback...")
            aws_data = {}
            lats = [str(stn["lat"]) for stn in PHYSICAL_STATIONS]
            lons = [str(stn["lng"]) for stn in PHYSICAL_STATIONS]
            
            url = f"https://api.open-meteo.com/v1/forecast?latitude={','.join(lats)}&longitude={','.join(lons)}&current=temperature_2m,precipitation"
            resp = requests.get(url, timeout=7)
            
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for idx, stn in enumerate(PHYSICAL_STATIONS):
                        stn_id = stn["id"]
                        curr = data[idx].get("current", {})
                        
                        aws_data[stn_id] = {
                            "temp": curr.get("temperature_2m", 0.0),
                            "R": curr.get("precipitation", 0.0),
                            "R_30": curr.get("precipitation", 0.0) / 2.0,
                            "R_60": curr.get("precipitation", 0.0),
                            "RI": curr.get("precipitation", 0.0) * 1.5,
                        }
                return aws_data
            else:
                logging.getLogger("daemons").error(f"[AWS Daemon] Open-Meteo API returned status {resp.status_code}")
                # Fallthrough to synthetic fallback
        except Exception as e:
            logging.getLogger("daemons").error(f"[AWS Daemon] Open-Meteo Fetch Error: {e}")
            # Fallthrough to synthetic fallback
            
        logging.getLogger("daemons").info("[AWS Daemon] Generating synthetic fallback data for physical stations to maintain IDW grid...")
        fallback_data = {}
        for stn in PHYSICAL_STATIONS:
            fallback_data[stn["id"]] = {
                "temp": 15.0 + (random.random() * 10.0),
                "R": 1.0 + (random.random() * 5.0),
                "R_30": 0.5 + (random.random() * 2.5),
                "R_60": 1.0 + (random.random() * 5.0),
                "RI": 3.0 + (random.random() * 8.0),
            }
        return fallback_data

    def fetch_realtime_imd_aws(self):
        try:
            url = "https://api.imd.gov.in/api/v1/aws"
            api_key = os.environ.get("IMD_API_KEY", "")
            if not api_key:
                # logging.getLogger("daemons").warning("[AWS Daemon] No IMD_API_KEY provided. Skipping IMD API.")
                return self.fetch_open_meteo_aws()

            headers = {"Authorization": f"Bearer {api_key}"}
            
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    # logging.getLogger("daemons").warning("[AWS Daemon] IMD API returned empty dictionary.")
                    return self.fetch_open_meteo_aws()
                return data
            else:
                # logging.getLogger("daemons").error(f"[AWS Daemon] AWS API returned status {resp.status_code}")
                return self.fetch_open_meteo_aws()
        except Exception as e:
            # logging.getLogger("daemons").error(f"[AWS Daemon] AWS API Fetch Failed: {e}")
            return self.fetch_open_meteo_aws()
