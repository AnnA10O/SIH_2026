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

from src.grid_utils import generate_virtual_grid

PHYSICAL_STATIONS = [
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

VIRTUAL_GRID = generate_virtual_grid(physical_stations=PHYSICAL_STATIONS, max_dist_km=30.0)
ALL_STATIONS = PHYSICAL_STATIONS + VIRTUAL_GRID

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
            logging.getLogger("daemons").error(f"[MOSDAC Daemon] Failed to load credentials: {e}")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="MosdacLiveDaemon")
        self.thread.start()
        logging.getLogger("daemons").info("[MOSDAC Daemon] Started background thread.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _loop(self):
        while self.running:
            try:
                self._fetch_and_ingest()
            except Exception as e:
                logging.getLogger("daemons").error(f"[MOSDAC Daemon] Unhandled error in cycle: {e}")
            
            # Sleep in small increments to allow clean shutdown
            for _ in range(60 * 30): # 30 minute polling interval
                if not self.running:
                    break
                time.sleep(1)

    def _fetch_and_ingest(self):
        now_utc = datetime.now(timezone.utc)
        
        # --- HACKATHON MOCK OVERRIDE ---
        # Since the live ISRO MOSDAC API is unreachable/returning empty, we simulate 
        # live ingest by randomly sampling historical 2013 monsoon satellite files.
        import glob
        import random
        
        for prod_suffix, reader_method, h5_key, channel_name, k_neighbors, max_radius in self.products:
            hist_dir = ROOT / "data" / "raw" / "satellite" / f"K1VHR_L2B_{prod_suffix}" / "2013-09-05"
            h5_files = glob.glob(str(hist_dir / "*.h5"))
            
            if not h5_files:
                logging.getLogger("daemons").warning(f"[MOSDAC Daemon] Mock fallback failed: No files found in {hist_dir}")
                continue
                
            dest_path = Path(random.choice(h5_files))
            filename = dest_path.name
            
            # Use current time as the "pass time" for live simulation
            pass_epoch = now_utc.timestamp()
            dc_date_str = now_utc.strftime("%d/%m/%Y %H:%M:%S")
            
            # 2. Extract features using identical logic to historical (KNN + IDW)
            reader_fn = getattr(SatelliteReader, reader_method)
            try:
                sat_data = reader_fn(dest_path)
            except Exception as e:
                logging.getLogger("daemons").error(f"[MOSDAC Daemon] H5 Read Error for {filename}: {e}")
                continue
                
            if h5_key not in sat_data:
                # E.g., for HEM fallback to qpe
                h5_key = 'qpe' if 'qpe' in sat_data else h5_key
                if h5_key not in sat_data:
                    logging.getLogger("daemons").warning(f"[MOSDAC Daemon] Key {h5_key} missing in {filename}")
                    continue

            lats = sat_data['lats']
            lons = sat_data['lons']
            vals = sat_data[h5_key]
            
            target_lats = np.array([s["lat"] for s in ALL_STATIONS])
            target_lons = np.array([s["lng"] for s in ALL_STATIONS])
            
            res = SatelliteReader.extract_points_knn(
                lats, lons, vals, target_lats, target_lons, k=k_neighbors, max_radius_km=max_radius
            )
            
            extracted_vals = res['val']
            dists = res['dist_km']
            valids = res['valid']
            
            # 3. Ingest into DataFusionBuffer
            for i, stn in enumerate(ALL_STATIONS):
                if valids[i] == 1.0:
                    meta = {"nearest_px_km": dists[i]}
                    self.buffer.ingest(
                        station_id=stn["id"],
                        channel=channel_name,
                        value=extracted_vals[i],
                        timestamp_epoch=pass_epoch,
                        source_meta=meta
                    )
            logging.getLogger("daemons").info(f"[MOSDAC Daemon] Ingested {channel_name} into buffer. (Pass time: {dc_date_str})")
