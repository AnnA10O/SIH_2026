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


def get_auth_token(creds: dict) -> str:
    """Fetch bearer access token from MOSDAC."""
    resp = requests.post(TOKEN_URL, json=creds, timeout=15)
    if resp.status_code == 200:
        token = resp.json().get("access_token")
        if token:
            return token
    raise PermissionError(f"MOSDAC Authentication failed (HTTP {resp.status_code}): {resp.text[:150]}")


def search_date_files(dataset_id: str, date_str: str, token: str, bbox: str = GUWAHATI_BBOX, count: int = 50) -> list:
    """Search for files on a specific date in bounding box."""
    params = {
        "datasetId": dataset_id,
        "startTime": date_str,
        "endTime": date_str,
        "count": str(count),
        "boundingBox": bbox
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(SEARCH_URL, params=params, headers=headers, timeout=20)
        if r.status_code == 200:
            entries = r.json().get("entries", [])
            return entries
    except Exception as e:
        print(f"  [WARN] Search failed for {date_str}: {e}")
    return []


def download_record(record_id: str, dest_path: Path, token: str) -> bool:
    """Download single satellite record by ID."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [SKIP] Already exists: {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    params = {"id": record_id}

    try:
        r = requests.get(DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=30)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_kb = dest_path.stat().st_size / 1024
            print(f"  [OK] Saved {dest_path.name} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  [FAIL] HTTP {r.status_code} for record {record_id}")
    except Exception as e:
        print(f"  [ERROR] {record_id}: {e}")
        if dest_path.exists():
            dest_path.unlink()
    return False


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
        non_normal = df[df.final_label != "NORMAL"]
        event_dates = sorted(pd.to_datetime(non_normal["timestamp"]).dt.strftime("%Y-%m-%d").unique())
        target_dates = event_dates
        print(f"Mode: TARGETED EVENTS ({len(target_dates)} dates identified from ground truth)")
        for d in target_dates:
            print(f"  - Event Date: {d}")
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

    # 3. Process each date
    total_downloaded = 0
    for d in target_dates:
        print(f"--- Processing Date: {d} ---")
        entries = search_date_files(dataset_id, d, token)
        print(f"  Files available in MOSDAC archive: {len(entries)}")

        if not entries:
            print(f"  No scenes found for {dataset_id} on {d}")
            continue

        entries_to_process = entries[:max_files_per_day]
        print(f"  Downloading up to {len(entries_to_process)} scenes for {d}...")

        for entry in entries_to_process:
            rec_id = str(entry.get("id") or entry.get("identifier"))
            dc_date = entry.get("dcDate", "")
            # Extract timestamp for clean filename
            time_tag = dc_date.split("/")[0].replace(":", "").replace("-", "") if dc_date else rec_id
            filename = f"{dataset_id}_{time_tag}_{rec_id}.h5"
            dest = DOWNLOAD_BASE / dataset_id / d / filename

            print(f"  Scene: {dc_date} (Record ID: {rec_id})")
            if dry_run:
                print(f"    [DRY RUN] Would download -> {dest}")
                total_downloaded += 1
            else:
                ok = download_record(rec_id, dest, token)
                if ok:
                    total_downloaded += 1
                time.sleep(0.5)  # Respect server rate limits

def get_era_dataset_id(dataset_id: str, date_str: str) -> str:
    """Dynamically select the product ID based on operational eras if a generic prefix is used."""
    if not dataset_id.startswith("AUTO_"):
        return dataset_id
        
    product = dataset_id.replace("AUTO_", "")
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year = d.year
    month = d.month
    
    if year < 2014:
        if product == "HEM": return "K1VHR_L2B_HEM"
        if product == "QPE": return "K1VHR_L2B_QPE"
        return f"K1VHR_L2B_{product}"
    elif year < 2016 or (year == 2016 and month <= 9):
        if product == "HEM": return "3D_HEM"
        if product == "QPE": return "3D_QPE"
        return f"3D_{product}"
    else:
        if product == "HEM": return "3DR_HEM"
        if product == "QPE": return "3DR_QPE"
        return f"3DR_{product}"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Date-Synchronized MOSDAC Satellite Downloader")
    parser.add_argument("--mode", choices=["events", "monsoon", "dates"], default="events",
                        help="Target mode: 'events' (cloudburst dates), 'monsoon' (2013 peak monsoon), or 'dates'")
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
