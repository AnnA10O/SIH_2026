"""MOSDAC Satellite Downloader — Synchronized with Ground Truth Temporal Windows.

Ensures strict synchronization between ground observations (GAGAN IWV + Assam AWS)
and satellite products.

Supported Modes:
  1. --events     : Downloads satellite scenes ONLY for the exact dates of flagged
                    cloudburst candidates in outputs/cloudburst_events.csv.
  2. --monsoon    : Downloads for the primary monsoon window (2013-06-01 to 2013-09-30).
  3. --dates      : User-specified range: --start YYYY-MM-DD --end YYYY-MM-DD.

Supported Datasets:
  - Historical 2013-2014 archive (co-registered with GAGAN ground truth):
      K1VHR_L2B_QPE  : Kalpana-1 Quantitative Precipitation Estimate (30-min)
      K1VHR_L2B_HEM  : Kalpana-1 Hydro-Estimator Convective Rain (30-min)
      K1VHR_L1B_STD  : Kalpana-1 VHRR Thermal IR + Water Vapor radiance
  - Operational 2024+ (for live deployment):
      3RIMG_L2B_SST  : INSAT-3DR Sea Surface Temperature
      3RIMG_L2B_CTT  : INSAT-3DR Cloud Top Temperature
      3RIMG_L3B_QPE  : INSAT-3DR Quantitative Precipitation

Auth: Reads credentials from d:/SIH/config.json (in .gitignore).
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

# Configure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
OUTPUTS_DIR = ROOT / "outputs"
EVENTS_CSV = OUTPUTS_DIR / "cloudburst_events.csv"
DOWNLOAD_BASE = ROOT / "data" / "raw" / "satellite"

TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"

# Guwahati cluster bounding box: min_lon, min_lat, max_lon, max_lat
GUWAHATI_BBOX = "88.5,24.5,94.5,28.5"


def load_credentials() -> dict:
    """Load credentials from config.json."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config.json at {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)
    creds = cfg.get("user_credentials", {})
    username = creds.get("username") or creds.get("username/email")
    password = creds.get("password")
    if not username or not password:
        raise ValueError("config.json must contain username and password")
    return {"username": username, "password": password}


TOKEN_CACHE_FILE = ROOT / ".mosdac_token.json"

def get_auth_token(creds: dict) -> str:
    """Authenticate and return JWT token, reusing cached token if valid."""
    if TOKEN_CACHE_FILE.exists():
        try:
            with open(TOKEN_CACHE_FILE, "r") as f:
                cache = json.load(f)
            if cache.get("expires_at", 0) > time.time():
                print("[AUTH] Using cached token.")
                return cache["token"]
        except Exception:
            pass

    for attempt in range(15):
        try:
            resp = requests.post(TOKEN_URL, json=creds, headers={"Content-Type": "application/json"}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("token") or data.get("access_token")
                if token:
                    try:
                        with open(TOKEN_CACHE_FILE, "w") as f:
                            json.dump({"token": token, "expires_at": time.time() + 3600}, f)
                    except Exception:
                        pass
                    return token
                else:
                    raise KeyError(f"Token not found in response: {data}")
            elif resp.status_code == 429:
                print(f"[AUTH] Rate limit (429). Sleeping 2 minutes before retry {attempt+1}/15...")
                time.sleep(120)
            else:
                raise PermissionError(f"MOSDAC Authentication failed (HTTP {resp.status_code}): {resp.text[:150]}")
        except requests.RequestException as e:
            print(f"[AUTH] Connection error: {e}. Retrying in 30s...")
            time.sleep(30)
            
    raise PermissionError("MOSDAC Authentication failed after 15 retries.")


def search_date_files(dataset_id: str, date_str: str, creds: dict, count: int = 48) -> list:
    """Query MOSDAC catalog for specific date."""
    print(f"\r\033[K  [SEARCH] Querying catalog for {date_str}...", end="", flush=True)
    for attempt in range(5):
        token = get_auth_token(creds)
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "datasetId": dataset_id,
            "startTime": date_str,
            "endTime": date_str,
            "count": str(count),
            "boundingBox": GUWAHATI_BBOX
        }
        try:
            r = requests.get(SEARCH_URL, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                entries = r.json().get("entries", [])
                print(f"\r\033[K  [SEARCH] Found {len(entries)} scenes for {date_str}") # completes line
                return entries
            elif r.status_code == 429:
                err_text = r.text.strip()[:100]
                for sec in range(60, 0, -1):
                    print(f"\r\033[K  [WAIT] 429 Rate Limit (Response: {err_text}). Retrying in {sec}s...", end="", flush=True)
                    time.sleep(1)
            elif r.status_code == 401:
                print(f"\r\033[K  [WARN] Token expired (401) on search. Forcing refresh...", end="", flush=True)
                if TOKEN_CACHE_FILE.exists():
                    TOKEN_CACHE_FILE.unlink()
                time.sleep(2)
            else:
                print(f"\r\033[K  [FAIL] Search HTTP {r.status_code} for {date_str}")
                return []
        except requests.RequestException as e:
            for sec in range(10, 0, -1):
                print(f"\r\033[K  [WARN] Search exception: {e}. Retrying in {sec}s...", end="", flush=True)
                time.sleep(1)
    print(f"\r\033[K  [FAIL] Search exhausted retries for {date_str}")
    return []


def download_record(record_id: str, dest_path: Path, creds: dict) -> bool:
    """Download single HDF5 product."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"\r\033[K  [SKIP] Already exists: {dest_path.name}", end="", flush=True)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\r\033[K  [DOWNLOADING] {dest_path.name} ...", end="", flush=True)

    for attempt in range(5):
        token = get_auth_token(creds)
        headers = {"Authorization": f"Bearer {token}"}
        params = {"id": record_id}
        
        try:
            r = requests.get(DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=30)
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"\r\033[K  [OK] Saved {dest_path.name} ({dest_path.stat().st_size / 1024:.1f} KB)", end="", flush=True)
                return True
            elif r.status_code == 429:
                err_text = r.text.strip()[:100]
                for sec in range(60, 0, -1):
                    print(f"\r\033[K  [WAIT] 429 Rate Limit (Response: {err_text}) for {dest_path.name}. Retrying in {sec}s...", end="", flush=True)
                    time.sleep(1)
            elif r.status_code == 401:
                print(f"\r\033[K  [WARN] Token expired (401) on download. Forcing refresh...", end="", flush=True)
                if TOKEN_CACHE_FILE.exists():
                    TOKEN_CACHE_FILE.unlink()
                time.sleep(2)
            else:
                print(f"\r\033[K  [FAIL] HTTP {r.status_code} for record {record_id}", end="", flush=True)
                return False
        except requests.RequestException as e:
            for sec in range(10, 0, -1):
                print(f"\r\033[K  [WARN] Download exception: {e}. Retrying in {sec}s...", end="", flush=True)
                time.sleep(1)
    if dest_path.exists():
        dest_path.unlink()
    print(f"\r\033[K  [FAIL] Exhausted retries for {dest_path.name}", end="", flush=True)
    return False



def get_era_dataset_id(dataset_id: str, date_str: str) -> str:
    """Dynamically select the product ID based on operational eras if a generic prefix is used."""
    if not dataset_id.startswith("AUTO_"):
        return dataset_id
        
    product = dataset_id.replace("AUTO_", "")
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year = d.year
    month = d.month
    
    if year < 2014:
        # Kalpana-1 era
        if product == "HEM": return "K1VHR_L2B_QPE"
        if product == "QPE": return "K1VHR_L2B_QPE"
        return f"K1VHR_L2B_{product}"
    elif year < 2016 or (year == 2016 and month <= 9):
        # INSAT-3D era
        return f"3DIMG_L2B_{product}"
    else:
        # INSAT-3DR era
        return f"3RIMG_L2B_{product}"

def run_downloader(mode: str = "events", dataset_id: str = "K1VHR_L2B_QPE", start_date: str = None, end_date: str = None, dry_run: bool = False, max_files_per_day: int = 5):
    """Execute date-aware downloader."""
    print("=" * 65)
    print("  MOSDAC SATELLITE DOWNLOADER — SYNCHRONIZED TEMPORAL PIPELINE")
    print("=" * 65)

    creds = load_credentials()
    print(f"User: {creds['username']}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Bounding Box: {GUWAHATI_BBOX} (Guwahati / Assam)")

    # 1. Determine target dates
    target_dates = []
    if mode == "events":
        if not EVENTS_CSV.exists():
            print(f"ERROR: {EVENTS_CSV} does not exist. Run pipeline.py first.")
            return
        df = pd.read_csv(EVENTS_CSV)
        
        pos_mask = df.final_label != "NORMAL"
        pos_dates = pd.to_datetime(df[pos_mask]["timestamp"]).dt.strftime("%Y-%m-%d").unique()
        neg_dates = pd.to_datetime(df[~pos_mask]["timestamp"]).dt.strftime("%Y-%m-%d").unique()
        
        # Sample negative dates to maintain roughly 10x dates relative to positive events
        # Since downloading 104x dates is too expensive, we sample a comparable large volume 
        # of negative dates (e.g., 10x) to ensure feature availability doesn't correlate with positive events
        # while keeping footprint to ~55 GB, fitting within the 85 GB available disk space.
        np.random.seed(42)
        n_neg = min(len(neg_dates), len(pos_dates) * 10) # 10x dates fits within 55 GB footprint at 1 file/day
        sampled_neg = np.random.choice(neg_dates, size=n_neg, replace=False)
        
        target_dates = sorted(list(set(pos_dates) | set(sampled_neg)))
        print(f"Mode: TARGETED EVENTS ({len(pos_dates)} pos dates, {len(sampled_neg)} neg dates sampled for unbiased availability)")
    elif mode == "auto-uth":
        # Targeted UTH download for 2013-2014 Assam AWS window:
        # 1. Event days +- 5 days
        # 2. Systematic 5th-day sampling
        from src.phase_d_training import load_imd_parquets, build_feature_matrix
        events_df = load_imd_parquets()
        df = events_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
        _, y_ser, _, _, _ = build_feature_matrix(events_df)
        df["is_positive"] = y_ser.values[df.index]
        
        df_jjas = df[(df["timestamp"].dt.year >= 2013) & (df["timestamp"].dt.year <= 2014)]
        pos_dates = df_jjas[df_jjas["is_positive"] == 1]["timestamp"].dt.floor('D').unique()
        
        target_dates_set = set()
        for pd_date in pos_dates:
            for delta in range(-5, 6):
                target_dates_set.add(pd_date + pd.Timedelta(days=delta))
                
        for year in [2013, 2014]:
            start = pd.Timestamp(year=year, month=6, day=1)
            end = pd.Timestamp(year=year, month=9, day=30)
            target_dates_set.update(pd.date_range(start, end, freq='5D'))
            
        target_dates = sorted([d.strftime("%Y-%m-%d") for d in target_dates_set])
        print(f"Mode: AUTO-UTH ({len(target_dates)} dates, encompassing events +-5 days & 5th-day negatives)")
    elif mode == "monsoon":
        # Sample days from the 2013 monsoon season
        target_dates = ["2013-06-15", "2013-07-01", "2013-07-15", "2013-08-01", "2013-08-15", "2013-09-01"]
        print(f"Mode: MONSOON WINDOW ({len(target_dates)} key synoptic dates)")
    elif mode == "dates":
        if not start_date or not end_date:
            print("ERROR: --start and --end are required in 'dates' mode.")
            return
        dt_range = pd.date_range(start_date, end_date)
        target_dates = [d.strftime("%Y-%m-%d") for d in dt_range]
        print(f"Mode: CUSTOM DATE RANGE ({start_date} to {end_date}, {len(target_dates)} days)")

    if not target_dates:
        print("No target dates found.")
        return

    # 2. Authenticate
    token = get_auth_token(creds)
    print("[AUTH] Successfully authenticated with MOSDAC.\n")

    # 3. Process each date (Fast Resume Logic)
    dataset_dir = DOWNLOAD_BASE / get_era_dataset_id(dataset_id, target_dates[0]) if target_dates else None
    if dataset_dir and dataset_dir.exists():
        existing_dates = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
        if existing_dates:
            latest_date = sorted(existing_dates)[-1]
            original_len = len(target_dates)
            target_dates = [d for d in target_dates if d >= latest_date]
            print(f"[RESUME] Found existing downloads up to {latest_date}. Skipping {original_len - len(target_dates)} dates.", flush=True)

    total_downloaded = 0
    for d in target_dates:
        print(f"--- Processing Date: {d} ---", flush=True)
        era_dataset = get_era_dataset_id(dataset_id, d)
        
        date_dir = DOWNLOAD_BASE / era_dataset / d
        if date_dir.exists():
            valid_files = [f for f in date_dir.glob("*.h5") if f.stat().st_size > 0]
            if len(valid_files) >= max_files_per_day:
                print(f"  [SKIP] Date {d} already satisfied ({len(valid_files)} files).")
                continue

        time.sleep(0.25) # Rate limit protection for search API
        entries = search_date_files(era_dataset, d, creds, count=min(max_files_per_day*3, 100))
        print(f"  Files available in MOSDAC archive for {era_dataset}: {len(entries)}")

        if not entries:
            print(f"  No scenes found for {era_dataset} on {d}")
            continue

        entries_to_process = entries[:max_files_per_day]
        print(f"  Downloading up to {len(entries_to_process)} scenes for {d}...")

        for entry in entries_to_process:
            rec_id = str(entry.get("id") or entry.get("identifier"))
            dc_date = entry.get("dcDate", "")
            # Extract timestamp for clean filename
            time_tag = dc_date.split("/")[0].replace(":", "").replace("-", "") if dc_date else rec_id
            filename = f"{era_dataset}_{time_tag}_{rec_id}.h5"
            dest = DOWNLOAD_BASE / era_dataset / d / filename

            print(f"  Scene: {dc_date} (Record ID: {rec_id})")
            if dry_run:
                print(f"    [DRY RUN] Would download -> {dest}")
                total_downloaded += 1
            else:
                ok = download_record(rec_id, dest, creds)
                if ok:
                    total_downloaded += 1
            time.sleep(0.5)  # Respect server rate limits (always sleep, even on skip or fail)

    print("\n" + "=" * 65)
    print(f"Download complete: {total_downloaded} files processed.")
    print(f"Location: {DOWNLOAD_BASE}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Date-Synchronized MOSDAC Satellite Downloader")
    parser.add_argument("--mode", choices=["events", "monsoon", "dates", "auto-uth"], default="events",
                        help="Target mode: 'events' (cloudburst dates), 'auto-uth' (event +-5 days + 5th day), 'monsoon' (2013 peak monsoon), or 'dates'")
    parser.add_argument("--dataset", default="K1VHR_L2B_QPE",
                        help="MOSDAC dataset ID (e.g., K1VHR_L2B_QPE, K1VHR_L2B_HEM, K1VHR_L1B_STD)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD) for 'dates' mode")
    parser.add_argument("--end", help="End date (YYYY-MM-DD) for 'dates' mode")
    parser.add_argument("--limit", type=int, default=3, help="Max files to download per day (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Search and report without downloading")
    args = parser.parse_args()

    run_downloader(
        mode=args.mode,
        dataset_id=args.dataset,
        start_date=args.start,
        end_date=args.end,
        dry_run=args.dry_run,
        max_files_per_day=args.limit
    )
