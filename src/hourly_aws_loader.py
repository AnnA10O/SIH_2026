"""
Hourly AWS Data Loader — Layer 2 of the FAR Mitigation Plan (Document 11).

Sources:
  - Assam AWS:          ASSAM_ALL_2013-02-01_2014-03-15_Sep2026_177056.csv  (70,551 rows, verified)
  - Uttarakhand (IMD):  data/raw/imd_uttarakhand_hourly_aws.csv             (pending acquisition)

Problem fixed:
  phase_d_training.py previously set rain_mm_hr = rain_mm_day, making R, R_30, R_60
  computed from a daily total rather than an hourly rate. A 100 mm/day gentle soak and
  a 100 mm/hr cloudburst were identical in the feature matrix. This module provides true
  time-aware accumulations to replace that proxy.

Output columns (joined back into the training DataFrame on [timestamp, station_id]):
  R     : Instantaneous rain rate (mm/hr), = rain_mm / elapsed_hours
  R_30  : 30-min accumulation (mm) — time-windowed sum, NOT row-count rolling
  R_60  : 60-min accumulation (mm) — time-windowed sum, NOT row-count rolling
  RI    : Rain acceleration dR/dt (mm/hr^2) — (R_current - R_prev) / elapsed_hours

Design mirrors StationFeatureBuffer.add_reading() in src/station_feature_engine.py,
which already uses pd.Timedelta cutoffs. This module replicates that pattern for
batch (historical) processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("d:/SIH")
ASSAM_AWS_PATH  = ROOT / "ASSAM_ALL_2013-02-01_2014-03-15_Sep2026_177056.csv"
IMD_UK_AWS_PATH = ROOT / "data" / "raw" / "imd_uttarakhand_hourly_aws.csv"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_assam_aws() -> pd.DataFrame:
    """
    Load the verified Assam AWS series (70,551 rows).

    Actual CSV schema (confirmed from file inspection):
      @STATION_ID, LATITUDE, LONGITUDE, ALTITUDE(m),
      TIME(GMT), DATE(GMT), TIME(IST), DATE(IST),
      AIR_TEMP, WIND_SPEED, WIND_DIRECTION, ATMO_PRESSURE,
      HUMIDITY(%), RAIN_FALL(mm), SUN_SHINE, BATTERY_VOLTAGE

    We combine DATE(IST) + TIME(IST) into a single 'timestamp' column,
    use @STATION_ID as station_id, and RAIN_FALL(mm) as rain_mm.
    """
    df = pd.read_csv(ASSAM_AWS_PATH)  # do NOT pass parse_dates — columns don't exist yet

    # Build timestamp from IST date + time columns
    # DATE(IST) format: DD/MM/YYYY  e.g. '03/02/2013'
    # TIME(IST) format: HH:MM       e.g. '13:30'
    df["timestamp"] = pd.to_datetime(
        df["DATE(IST)"].astype(str) + " " + df["TIME(IST)"].astype(str),
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )
    df = df.rename(columns={
        "@STATION_ID":  "station_id",
        "RAIN_FALL(mm)": "rain_mm",
    })
    df["source"] = "assam_aws"
    # Drop rows where timestamp parsing failed (bad/missing data)
    df = df.dropna(subset=["timestamp", "station_id", "rain_mm"])
    return df[["timestamp", "station_id", "rain_mm", "source"]].copy()


def _load_uttarakhand_aws() -> pd.DataFrame:
    """
    Load IMD Uttarakhand hourly AWS.
    Returns an empty DataFrame if the file has not yet been acquired.
    """
    if not IMD_UK_AWS_PATH.exists():
        print(f"[hourly_aws_loader] Uttarakhand AWS not found: {IMD_UK_AWS_PATH}")
        print("  Uttarakhand basins will fall back to daily proxy.")
        print("  Acquire IMD hourly records (Document 10, Task 2) to unlock Layer 2 for UK basins.")
        return pd.DataFrame(columns=["timestamp", "station_id", "rain_mm", "source"])
    df = pd.read_csv(IMD_UK_AWS_PATH, parse_dates=["timestamp"])
    rain_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ("rain", "precip", "rf"))),
        None
    )
    if rain_col is None:
        raise ValueError(
            f"No rainfall column found in {IMD_UK_AWS_PATH.name}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns={rain_col: "rain_mm"})
    df["source"] = "imd_uttarakhand"
    return df[["timestamp", "station_id", "rain_mm", "source"]].copy()


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_intensity_features(station_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-aware intensity features for one station's chronological series.
    Vectorised implementation — O(N log N) via rolling on a DatetimeIndex.

    Mirrors StationFeatureBuffer.add_reading() semantics:
      - R_30 / R_60: sum of rain_mm within the preceding 30 / 60 minutes
        (time-aware window, not positional row count)
      - RI: (R_current - R_prev) / elapsed_hours → units mm/hr²

    Args:
        station_df: must have [timestamp, rain_mm], any index.
    Returns:
        station_df with additional columns [R, R_30, R_60, RI].
    """
    df = station_df.sort_values("timestamp").copy()
    df["rain_mm"] = df["rain_mm"].fillna(0.0)

    # Set a monotonic DatetimeIndex — required for time-based rolling windows
    df = df.set_index("timestamp")
    df = df.sort_index()   # pandas enforces monotonicity for offset-string rolling

    # Elapsed hours since previous reading (per row)
    elapsed_s  = df.index.to_series().diff().dt.total_seconds().fillna(3600.0)
    elapsed_hr = elapsed_s.clip(lower=60.0) / 3600.0   # floor at 1 min

    # Instantaneous rain rate R (mm/hr)
    R = df["rain_mm"] / elapsed_hr

    # Time-window accumulations — offset-string rolling requires monotonic DatetimeIndex
    R_30 = df["rain_mm"].rolling("30min", closed="right").sum()
    R_60 = df["rain_mm"].rolling("60min", closed="right").sum()

    # Rain acceleration RI = dR/dt  (mm/hr²)
    RI = R.diff().fillna(0.0) / elapsed_hr

    df["R"]    = R.values
    df["R_30"] = R_30.values
    df["R_60"] = R_60.values
    df["RI"]   = RI.values

    # Restore timestamp as a column
    df = df.reset_index()
    return df



# ── Public API ────────────────────────────────────────────────────────────────

def build_hourly_aws_features() -> pd.DataFrame:
    """
    Merge both AWS sources and compute time-aware intensity features per station.

    Returns:
        DataFrame with columns [timestamp, station_id, R, R_30, R_60, RI, source].
        Empty DataFrame if no source data is available (caller handles fallback).
    """
    assam = _load_assam_aws()
    uk    = _load_uttarakhand_aws()
    combined = pd.concat([assam, uk], ignore_index=True)
    combined  = combined.dropna(subset=["timestamp", "station_id"])

    if len(combined) == 0:
        print("[hourly_aws_loader] No AWS data available from any source.")
        return pd.DataFrame(columns=["timestamp", "station_id", "R", "R_30", "R_60", "RI", "source"])

    n_stations = combined["station_id"].nunique()
    print(f"[hourly_aws_loader] Processing {len(combined):,} readings "
          f"across {n_stations} stations...")

    parts = []
    for sid, grp in combined.groupby("station_id"):
        parts.append(_compute_intensity_features(grp))

    result = pd.concat(parts, ignore_index=True)
    n_assam = int((result["source"] == "assam_aws").sum())
    n_uk    = int((result["source"] == "imd_uttarakhand").sum())
    print(f"  Done. Assam: {n_assam:,}  Uttarakhand: {n_uk:,}")

    return result[["timestamp", "station_id", "R", "R_30", "R_60", "RI", "source"]]
