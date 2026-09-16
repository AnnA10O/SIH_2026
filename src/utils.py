"""Shared utilities — Haversine, psychrometrics, counter detection, IWV seasonal check."""

import numpy as np
import pandas as pd
from typing import Tuple, List


# ─── Spatial ───────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def pairwise_distances(coords: List[Tuple[float, float]]) -> np.ndarray:
    """Return NxN symmetric matrix of haversine distances (km) for a list of (lat, lon)."""
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(*coords[i], *coords[j])
            mat[i, j] = mat[j, i] = d
    return mat


# ─── Psychrometrics ────────────────────────────────────────────────────────────

def dewpoint_from_rh(temp_c: float, rh_pct: float) -> float:
    """Magnus formula: dewpoint temperature (°C) from temperature and relative humidity."""
    a, b = 17.625, 243.04
    alpha = np.log(rh_pct / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * alpha) / (a - alpha)


def dewpoint_depression(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """T - Td: positive values indicate drier air / more lift potential."""
    td = dewpoint_from_rh(temp_c, rh_pct)
    return temp_c - td


# ─── Rainfall Counter Detection & Diffing ──────────────────────────────────────

def is_cumulative_counter(series: pd.Series, threshold: float = 0.90) -> bool:
    """
    Return True if the series looks like a cumulative (running-total) rain gauge.
    Heuristic: >threshold fraction of consecutive differences are >= 0 (non-decreasing).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        return False
    diffs = s.diff().dropna()
    non_decreasing_frac = (diffs >= 0).mean()
    return non_decreasing_frac >= threshold


def diff_cumulative_counter(series: pd.Series) -> pd.Series:
    """
    Convert a cumulative rain gauge counter to per-interval increments.
    Counter resets (negative diff) are treated as new baseline — the drop is NOT
    flagged as negative rainfall but as a gauge rollover.
    """
    s = pd.to_numeric(series, errors="coerce")
    diffs = s.diff()
    # Reset events: diff < 0 means gauge rolled over or was reset
    diffs[diffs < 0] = 0.0
    return diffs


def to_hourly_rain(df: pd.DataFrame, ts_col: str, rain_col: str) -> pd.Series:
    """
    Given a dataframe with a datetime index column, resample rain increments to
    1-hourly accumulated totals (mm/hr).
    Returns a pd.Series indexed by hourly datetime.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col)

    # Get numeric rain, handle sentinels
    from src.config import SENTINEL_VALUES
    rain = pd.to_numeric(df[rain_col], errors="coerce")
    rain = rain.replace(SENTINEL_VALUES, np.nan)

    # Detect cumulative counter and diff if needed
    if is_cumulative_counter(rain.dropna()):
        rain = diff_cumulative_counter(rain)

    # Resample to hourly sum
    hourly = rain.resample("1h").sum(min_count=1)
    return hourly


# ─── IWV Seasonal Sanity ───────────────────────────────────────────────────────

def iwv_seasonal_ratio(iwv_series: pd.Series, timestamps: pd.Series) -> float:
    """
    Monsoon (Jul-Aug) mean IWV / Winter (Jan-Feb) mean IWV.
    Returns NaN if either season has no data.
    """
    ts = pd.to_datetime(timestamps, errors="coerce")
    monsoon_mean = iwv_series[ts.dt.month.isin([7, 8])].mean()
    winter_mean  = iwv_series[ts.dt.month.isin([1, 2])].mean()
    if winter_mean == 0 or np.isnan(winter_mean) or np.isnan(monsoon_mean):
        return np.nan
    return float(monsoon_mean / winter_mean)


def iwv_shape_sanity(iwv_series: pd.Series, timestamps: pd.Series) -> Tuple[bool, str]:
    """
    Returns (True, 'ok') if IWV shows expected South Asian monsoon seasonal arc.
    Returns (False, reason) otherwise.
    """
    ratio = iwv_seasonal_ratio(iwv_series, timestamps)
    if np.isnan(ratio):
        return False, "insufficient_season_data"
    if ratio < 1.5:
        return False, f"weak_seasonal_signal (monsoon/winter ratio={ratio:.2f} < 1.5)"
    # Check that winter months have plausibly low values
    ts = pd.to_datetime(timestamps, errors="coerce")
    winter_mean = iwv_series[ts.dt.month.isin([1, 2])].mean()
    if winter_mean > 40.0:
        return False, f"winter_IWV_too_high ({winter_mean:.1f} mm > 40 expected)"
    return True, "ok"


# ─── Rolling-Window Feature Engineering ────────────────────────────────────────

def compute_features(hourly_rain: pd.Series,
                     hourly_rh: pd.Series = None,
                     hourly_temp: pd.Series = None) -> pd.DataFrame:
    """
    Build the per-hour feature matrix from hourly rain, RH, and temperature series.
    All series must share the same hourly DatetimeIndex.
    """
    feat = pd.DataFrame(index=hourly_rain.index)
    feat["R"]   = hourly_rain
    feat["R_30"] = hourly_rain.rolling(2, min_periods=1, closed="left").sum()    # 2 × 30-min = 1hr → use 2hr window
    feat["R_60"] = hourly_rain.rolling(2, min_periods=1, closed="left").sum()
    feat["R_30"] = hourly_rain.shift(0) + hourly_rain.shift(1).fillna(0)   # last 2 hrs
    feat["R_60"] = hourly_rain.rolling(2, min_periods=1, closed="left").sum()
    feat["RI"]   = hourly_rain.diff()   # Rain Intensity change rate (mm/hr per hr)

    if hourly_rh is not None:
        feat["RH"]       = hourly_rh
        feat["RH_trend"] = hourly_rh.diff()

    if hourly_rh is not None and hourly_temp is not None:
        feat["dewpoint_depression"] = dewpoint_depression(hourly_temp, hourly_rh)

    return feat
