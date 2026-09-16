import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.satellite_reader import SatelliteReader
from mosdac_downloader import load_credentials, get_auth_token, search_date_files, download_record, get_era_dataset_id

LIVE_CACHE_DIR = ROOT / "data" / "live_cache"
LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = [
    {"id": "UK-1", "lat": 30.7346, "lng": 79.0669},
    {"id": "UK-2", "lat": 30.2844, "lng": 78.9811},
    {"id": "UK-3", "lat": 30.6963, "lng": 79.0227},
    {"id": "UK-4", "lat": 30.6377, "lng": 79.0417},
    {"id": "UK-5", "lat": 30.5326, "lng": 79.0795},
    {"id": "UK-6", "lat": 30.4000, "lng": 79.3200},
    {"id": "UK-7", "lat": 30.5620, "lng": 79.5641},
    {"id": "AS-1", "lat": 26.1445, "lng": 91.7362},
    {"id": "AS-2", "lat": 26.6528, "lng": 92.7926},
    {"id": "AS-3", "lat": 26.7509, "lng": 94.2037},
    {"id": "AS-4", "lat": 27.4728, "lng": 94.9120},
    {"id": "AS-5", "lat": 27.2360, "lng": 94.1050},
    {"id": "AS-6", "lat": 26.9500, "lng": 94.1700},
    {"id": "AS-7", "lat": 27.4833, "lng": 94.5833},
]

class MosdacLiveDaemon:
    def __init__(self, buffer):
        self.buffer = buffer
        self.running = False
        self.thread = None
        self.creds = None
        
        # We only really need UTH for now based on previous model training, but we can pull HEM too
        self.products = [("UTH", "read_uth", "uth", "uth_kalpana", 4, 250.0),
                         ("HEM", "read_hem", "hem", "hem_kalpana", 1, 15.0)]

    def start(self):
        if self.running: return
        try:
            self.creds = load_credentials()
        except Exception as e:
            logging.error(f"[MOSDAC Daemon] Failed to load credentials: {e}")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="MosdacLiveDaemon")
        self.thread.start()
        logging.info("[MOSDAC Daemon] Started background thread.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _loop(self):
        while self.running:
            try:
                self._fetch_and_ingest()
            except Exception as e:
                logging.error(f"[MOSDAC Daemon] Unhandled error in cycle: {e}")
            
            # Sleep in small increments to allow clean shutdown
            for _ in range(60 * 30): # 30 minute polling interval
                if not self.running:
                    break
                time.sleep(1)

    def _fetch_and_ingest(self):
        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        
        # 1. Ensure Auth works (get_auth_token handles caching and retries internally)
        try:
            token = get_auth_token(self.creds)
        except Exception as e:
            logging.error(f"[MOSDAC Daemon] Auth Failure: {e}")
            return
            
        for prod_suffix, reader_method, h5_key, channel_name, k_neighbors, max_radius in self.products:
            era_dataset = get_era_dataset_id(f"AUTO_{prod_suffix}", date_str)
            entries = search_date_files(era_dataset, date_str, self.creds, count=5)
            
            if not entries:
                logging.warning(f"[MOSDAC Daemon] No {prod_suffix} files found for {date_str}.")
                continue
                
            # Sort by dcDate descending to get the absolute latest
            entries = sorted(entries, key=lambda x: x.get("dcDate", ""), reverse=True)
            latest = entries[0]
            rec_id = str(latest.get("id") or latest.get("identifier"))
            dc_date_str = latest.get("dcDate", "") # Format: "01/08/2026 12:30:00"
            
            try:
                # Parse "DD/MM/YYYY HH:MM:SS" or fallback
                pass_time = datetime.strptime(dc_date_str, "%d/%m/%Y %H:%M:%S")
                pass_time = pass_time.replace(tzinfo=timezone.utc)
                pass_epoch = pass_time.timestamp()
            except Exception:
                pass_epoch = now_utc.timestamp()
                
            time_tag = dc_date_str.split("/")[0].replace(":", "").replace("-", "") if dc_date_str else rec_id
            filename = f"{era_dataset}_{time_tag}_{rec_id}.h5"
            dest_path = LIVE_CACHE_DIR / filename
            
            ok = download_record(rec_id, dest_path, self.creds)
            if not ok:
                logging.error(f"[MOSDAC Daemon] Failed to download {filename}")
                continue
                
            # 2. Extract features using identical logic to historical (KNN + IDW)
            reader_fn = getattr(SatelliteReader, reader_method)
            try:
                sat_data = reader_fn(dest_path)
            except Exception as e:
                logging.error(f"[MOSDAC Daemon] H5 Read Error for {filename}: {e}")
                continue
                
            if h5_key not in sat_data:
                # E.g., for HEM fallback to qpe
                h5_key = 'qpe' if 'qpe' in sat_data else h5_key
                if h5_key not in sat_data:
                    logging.warning(f"[MOSDAC Daemon] Key {h5_key} missing in {filename}")
                    continue

            lats = sat_data['lats']
            lons = sat_data['lons']
            vals = sat_data[h5_key]
            
            target_lats = np.array([s["lat"] for s in STATIONS])
            target_lons = np.array([s["lng"] for s in STATIONS])
            
            res = SatelliteReader.extract_points_knn(
                lats, lons, vals, target_lats, target_lons, k=k_neighbors, max_radius_km=max_radius
            )
            
            extracted_vals = res['val']
            dists = res['dist_km']
            valids = res['valid']
            
            # 3. Ingest into DataFusionBuffer
            for i, stn in enumerate(STATIONS):
                if valids[i] == 1.0:
                    meta = {"nearest_px_km": dists[i]}
                    self.buffer.ingest(
                        station_id=stn["id"],
                        channel=channel_name,
                        value=extracted_vals[i],
                        timestamp_epoch=pass_epoch,
                        source_meta=meta
                    )
            logging.info(f"[MOSDAC Daemon] Ingested {channel_name} into buffer. (Pass time: {dc_date_str})")
