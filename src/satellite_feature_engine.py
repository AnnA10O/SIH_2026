"""
Satellite Feature Engine for PS-26077 Cloudburst Nowcasting.

Scans all downloaded INSAT-3D/3DR HDF5 files for the disaster event windows
and produces a flat daily parquet:
    data/processed/satellite_features.parquet

One row per (date, event_window) with columns:
    ctt_mean, ctt_min, ctt_cold_frac, hem_mean, hem_max, olr_mean, uth_mean

This parquet is left-joined to the IMD rain parquets in phase_d_training.py
so the model sees satellite features wherever they exist and NaN elsewhere
(handled by median imputation in build_feature_matrix).

Usage:
    python src/satellite_feature_engine.py
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.satellite_reader import SatelliteReader
from src.config import SATELLITE_DIR, DATA_PROCESSED, DISASTER_WINDOWS


# ── Constants ──────────────────────────────────────────────────────────────────

DEEP_CONV_K = 233.15  # 233.15 K = -40°C — threshold for deep convective cloud tops

# Map product directory prefixes (both INSAT-3D and 3DR) to variable + reader
# Format: (reader_method_name, output_key)
PRODUCT_MAP = {
    "3DIMG_L2B_CTP": ("read_ctt",  "ctt"),
    "3RIMG_L2B_CTP": ("read_ctt",  "ctt"),
    "3DIMG_L2B_HEM": ("read_hem",  "hem"),
    "3RIMG_L2B_HEM": ("read_hem",  "hem"),
    "3DIMG_L2B_OLR": ("read_olr",  "olr"),
    "3RIMG_L2B_OLR": ("read_olr",  "olr"),
    "3DIMG_L2B_UTH": ("read_uth",  "uth"),
    "3RIMG_L2B_UTH": ("read_uth",  "uth"),
}

# Regex to extract UTC date from INSAT-3D filename
# e.g. 3DIMG_21AUG2019_1200_L2B_CTP_V01R00.h5 → 2019-08-21
_DATE_RE = re.compile(
    r"(?:3[DR]IMG|3SIMG)_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_L2B",
    re.IGNORECASE
)
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_date(filename: str) -> Optional[datetime]:
    """Extract UTC datetime from INSAT-3D filename (e.g. 3DIMG_21AUG2019_1200_...)."""
    m = _DATE_RE.search(filename)
    if not m:
        return None
    day, mon_str, year, hhmm = m.groups()
    month = _MONTH_MAP.get(mon_str.upper())
    if month is None:
        return None
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    return datetime(int(year), month, int(day), hour, minute)


# ── Per-file feature extraction ────────────────────────────────────────────────

def _extract_file_features(
    h5_path: Path,
    product_dir: str,
    bbox: tuple,
) -> Optional[Dict]:
    """
    Extract scalar summary features from a single HDF5 file.
    Returns dict or None if file is unreadable / empty after clipping.
    """
    reader_method, var_key = PRODUCT_MAP[product_dir]
    reader_fn = getattr(SatelliteReader, reader_method)

    try:
        result = reader_fn(h5_path, bbox=bbox)
    except Exception as e:
        print(f"    WARN: {h5_path.name}: {e}")
        return None

    values = result.get(var_key, np.array([]))
    if len(values) == 0:
        return None

    row: Dict = {}

    if var_key == "ctt":
        row["ctt_mean"] = float(np.nanmean(values))
        row["ctt_min"]  = float(np.nanmin(values))
        row["ctt_cold_frac"] = float(np.sum(values < DEEP_CONV_K) / len(values))

    elif var_key == "hem":
        row["hem_mean"] = float(np.nanmean(values))
        row["hem_max"]  = float(np.nanmax(values))

    elif var_key == "olr":
        row["olr_mean"] = float(np.nanmean(values))

    elif var_key == "uth":
        row["uth_mean"] = float(np.nanmean(values))

    return row if row else None


# ── Event window assignment ────────────────────────────────────────────────────

def _assign_event_window(date: datetime) -> Optional[str]:
    """
    Map a UTC datetime to its disaster event window name (matches DISASTER_WINDOWS keys).
    Returns None if date doesn't fall within any known window.
    """
    y, m, d = date.year, date.month, date.day

    if y == 2016 and m == 7 and d <= 9:
        return "Pithoragarh_Jul2016"
    if y == 2016 and m == 7 and d >= 14:
        return "Chamoli_Jul2016"
    if y == 2019 and m == 8 and (16 <= d <= 21):
        return "Uttarkashi_Aug2019"
    if y == 2021 and m == 10 and (16 <= d <= 21):
        return "Chamoli_Oct2021"
    return None


# ── Main build function ────────────────────────────────────────────────────────

def build_satellite_features(
    satellite_dir: Path = SATELLITE_DIR,
    output_path: Path = DATA_PROCESSED / "satellite_features.parquet",
) -> pd.DataFrame:
    """
    Scan all satellite HDF5 files, aggregate to daily features per event window,
    and write satellite_features.parquet.

    Output columns:
        date (str YYYY-MM-DD), event_window, ctt_mean, ctt_min, ctt_cold_frac,
        hem_mean, hem_max, olr_mean, uth_mean
    """
    print("=" * 65)
    print("  SATELLITE FEATURE ENGINE  —  PS-26077")
    print("=" * 65)

    # Accumulate per-(date, window) measurements across all files
    # Structure: { (date_str, window): { feature: [values...] } }
    from collections import defaultdict
    daily_acc: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    total_files = 0
    total_usable = 0

    for prod_dir_name, (reader_method, var_key) in PRODUCT_MAP.items():
        prod_path = satellite_dir / prod_dir_name
        if not prod_path.exists():
            print(f"\n  [{prod_dir_name}] directory not found — skipping")
            continue

        h5_files = sorted(prod_path.rglob("*.h5"))
        print(f"\n  [{prod_dir_name}] {len(h5_files)} files")
        total_files += len(h5_files)

        for h5_path in h5_files:
            dt = _parse_date(h5_path.name)
            if dt is None:
                print(f"    SKIP (unparseable date): {h5_path.name}")
                continue

            window = _assign_event_window(dt)
            if window is None:
                continue  # outside known disaster windows

            bbox = DISASTER_WINDOWS[window]
            features = _extract_file_features(h5_path, prod_dir_name, bbox)
            if features is None:
                continue

            date_str = dt.strftime("%Y-%m-%d")
            key = (date_str, window)
            for feat_name, feat_val in features.items():
                daily_acc[key][feat_name].append(feat_val)

            total_usable += 1

    if not daily_acc:
        print("\nERROR: No satellite features extracted. Check satellite_dir and file names.")
        return pd.DataFrame()

    # Aggregate: for each (date, window), take the mean across all half-hourly observations
    print(f"\n  Aggregating {len(daily_acc)} (date, window) combinations...")
    rows = []
    for (date_str, window), feat_dict in sorted(daily_acc.items()):
        row = {"date": date_str, "event_window": window}
        # Daily mean across all half-hourly snapshots
        for col in ["ctt_mean", "ctt_min", "ctt_cold_frac", "hem_mean", "hem_max", "olr_mean", "uth_mean"]:
            vals = feat_dict.get(col, [])
            if vals:
                # For min-type stats, take the daily minimum across all observations
                if col == "ctt_min":
                    row[col] = float(np.min(vals))
                elif col == "hem_max":
                    row[col] = float(np.max(vals))
                elif col == "ctt_cold_frac":
                    row[col] = float(np.max(vals))  # peak deep conv fraction in the day
                else:
                    row[col] = float(np.mean(vals))
            else:
                row[col] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Total HDF5 files scanned : {total_files}")
    print(f"  Files yielding features  : {total_usable}")
    print(f"  Daily feature rows       : {len(df)}")
    print(f"\n  Feature coverage:")
    for col in ["ctt_mean", "ctt_min", "ctt_cold_frac", "hem_mean", "hem_max", "olr_mean", "uth_mean"]:
        non_null = df[col].notna().sum() if col in df.columns else 0
        print(f"    {col:<18}: {non_null}/{len(df)} days non-null")

    print(f"\n  By event window:")
    for window, grp in df.groupby("event_window"):
        print(f"    {window}: {len(grp)} days")

    print(f"\n  Sample rows:")
    print(df.head(8).to_string(index=False))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n  Saved → {output_path}")
    print("=" * 65)
    return df


if __name__ == "__main__":
    build_satellite_features()
