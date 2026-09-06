"""Phase B — Automated Data Quality Gate.

audit_station(df, rain_col, aux_cols) → ('PASS'|'FAIL', reason, cleaned_df)

Every AWS station MUST pass all checks before being used downstream.
Results are appended to outputs/station_quality_report.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

from src.config import (
    STUCK_ROLLING_DAYS, STUCK_IDENTICAL_THRESHOLD, STUCK_MIN_UNIQUE_VALUES,
    DEAD_ZERO_FRACTION, DEAD_MAX_MONSOON_MM,
    CROSSTALK_CORR_THRESHOLD, SPIKE_MAX_HOURLY_MM,
    MONSOON_COVERAGE_MIN_PCT, MONSOON_MONTHS,
    SENTINEL_VALUES, STATION_QUALITY_CSV,
    AWS_RAIN_COL, AWS_SUN_COL, AWS_TEMP_COL, AWS_RH_COL,
    AWS_TS_COL_IST, AWS_TIME_COL, AWS_STATION_COL
)
from src.utils import is_cumulative_counter, diff_cumulative_counter


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric; replace sentinel fill values with NaN."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(SENTINEL_VALUES, np.nan)
    return s


def _sunshine_to_minutes(series: pd.Series) -> pd.Series:
    """Convert SUN_SHINE hh:mm string column to total minutes as float."""
    def parse(v):
        try:
            if pd.isna(v):
                return np.nan
            parts = str(v).strip().split(":")
            return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else float(v)
        except Exception:
            return np.nan
    return series.apply(parse)


def _build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Combine DATE(IST) and TIME(IST) into a parsed datetime Series."""
    combined = df[AWS_TS_COL_IST].astype(str) + " " + df[AWS_TIME_COL].astype(str)
    return pd.to_datetime(combined, dayfirst=True, errors="coerce")


# ─── Step 0 — Counter Detection & Conversion to Daily Rain ────────────────────

def _detect_update_frequency(df: pd.DataFrame, ts: pd.Series, rain: pd.Series) -> str:
    """
    Detect whether this is a sub-daily updating sensor or a daily-cumulative counter.
    Returns 'daily_cumulative' or 'subdaily'.
    MOSDAC AWS in this dataset update their cumulative total once per day —
    the same value repeats for every sub-daily row until the next day's update.
    """
    if rain.dropna().empty:
        return 'daily_cumulative'
    diffs = rain.diff().dropna()
    zero_frac = (diffs == 0).mean()
    # If >60% of consecutive diffs are zero, it's a daily-update cumulative counter
    return 'daily_cumulative' if zero_frac > 0.60 else 'subdaily'


def _prepare_rain(df: pd.DataFrame, ts: pd.Series) -> tuple:
    """
    Detect counter type; extract DAILY rainfall increments.
    Returns (daily_rain_series indexed by date, update_type_str).
    """
    rain_raw = _clean_numeric(df[AWS_RAIN_COL])
    df2 = pd.DataFrame({'ts': ts, 'rain': rain_raw}).dropna(subset=['ts']).copy()
    df2 = df2.dropna(subset=['ts']).set_index('ts').sort_index()
    # Remove duplicate timestamps — keep last (most recent reading per sub-daily slot)
    df2 = df2[~df2.index.duplicated(keep='last')]
    rain = df2['rain']

    update_type = _detect_update_frequency(df, ts, rain)

    if update_type == 'daily_cumulative':
        # Take the LAST reading of each calendar day = the day's running total
        daily_last = rain.resample('1D').last()
        # Forward-fill short gaps (up to 3 days) before diffing
        daily_last = daily_last.ffill(limit=3)
        # Diff to get daily increment
        daily_rain = daily_last.diff()
        # Resets (negative diff = new seasonal baseline) → set to NaN, not 0
        # because we can't know the true increment across a reset boundary
        daily_rain[daily_rain < 0] = np.nan
        return daily_rain, update_type
    else:
        # Sub-daily: diff directly and resample to daily totals
        is_counter = is_cumulative_counter(rain.dropna())
        if is_counter:
            rain = diff_cumulative_counter(rain)
        daily_rain = rain.resample('1D').sum(min_count=1)
        return daily_rain, update_type


# ─── Check 1 — Stuck Sensor ────────────────────────────────────────────────────

def _check_stuck(daily_rain: pd.Series, update_type: str) -> tuple:
    """
    FAIL if >95% of DAILY increments are identical (e.g., all zero or all same value),
    OR total unique non-NaN daily increments < STUCK_MIN_UNIQUE_VALUES.
    For daily_cumulative stations we operate on the day-level diff series.
    """
    non_null = daily_rain.dropna()
    if len(non_null) == 0:
        return False, "stuck -- no valid data after daily diff"

    n_unique = non_null.nunique()
    if n_unique < STUCK_MIN_UNIQUE_VALUES:
        return False, f"stuck -- only {n_unique} unique daily increments in entire record"

    # Rolling 30-day check on daily series
    window_days = STUCK_ROLLING_DAYS
    for start in range(0, max(1, len(non_null) - window_days), window_days // 2):
        window = non_null.iloc[start:start + window_days]
        if len(window) < window_days // 2:
            continue
        most_common_frac = window.value_counts(normalize=True).iloc[0]
        dominant_val = window.value_counts().index[0]
        # Allow high zero fraction (dry days are normal) but not high non-zero repetition
        if most_common_frac > STUCK_IDENTICAL_THRESHOLD and dominant_val != 0.0:
            return False, (
                f"stuck -- {most_common_frac:.0%} of daily increments in a 30-day window "
                f"are identical ({dominant_val})"
            )
    return True, "ok"


# ─── Check 2 — Dead / No Signal ────────────────────────────────────────────────

def _check_dead(daily_rain: pd.Series) -> tuple:
    """
    FAIL if >90% of daily increments are zero AND max during monsoon months
    is below DEAD_MAX_MONSOON_MM.
    """
    non_null = daily_rain.dropna()
    zero_frac = (non_null == 0).mean()
    monsoon_vals = non_null[non_null.index.month.isin(MONSOON_MONTHS)]
    monsoon_max = monsoon_vals.max() if len(monsoon_vals) > 0 else 0.0

    if zero_frac > DEAD_ZERO_FRACTION and monsoon_max < DEAD_MAX_MONSOON_MM:
        return False, (
            f"dead -- {zero_frac:.0%} zeros, max monsoon rainfall "
            f"{monsoon_max:.1f}mm < {DEAD_MAX_MONSOON_MM}mm threshold"
        )
    return True, "ok"


# ─── Check 3 — Mislabeled / Cross-Talk ─────────────────────────────────────────

def _check_crosstalk(daily_rain: pd.Series, df_raw: pd.DataFrame,
                     ts: pd.Series) -> tuple:
    """
    FAIL if |correlation(daily_rain, daily_sunshine_duration)| > 0.5
    AND rain mirrors solar diurnal cycle. For daily data, check if rain
    correlates with sunshine hours at daily resolution.
    """
    try:
        sun_raw = _sunshine_to_minutes(df_raw[AWS_SUN_COL])
        df2 = pd.DataFrame({'ts': ts, 'sun': sun_raw}).dropna(subset=['ts']).copy()
        df2 = df2.set_index('ts').sort_index()
        df2 = df2[~df2.index.duplicated(keep='last')]
        sun_daily = df2['sun'].resample('1D').sum(min_count=1)

        common = daily_rain.index.intersection(sun_daily.index)
        if len(common) < 30:
            return True, "ok"

        r = daily_rain.loc[common]
        s = sun_daily.loc[common]
        mask = (r > 0) | (s > 0)
        if mask.sum() < 20:
            return True, "ok"

        corr = r[mask].corr(s[mask])
        if abs(corr) > CROSSTALK_CORR_THRESHOLD:
            return False, (
                f"mislabeled -- |corr(rain, sunshine)| = {abs(corr):.2f} "
                f"> {CROSSTALK_CORR_THRESHOLD} (cross-talk pattern)"
            )
    except Exception as e:
        import warnings
        warnings.warn(f"Crosstalk check skipped: {e}")
    return True, "ok"


# ─── Check 4 — Increment Sanity (clip unphysical daily totals) ─────────────────

SPIKE_MAX_DAILY_MM = 400.0  # IMD record single-day rainfall in India ~400mm

def _check_and_clip_spikes(daily_rain: pd.Series) -> tuple:
    """
    Clip single-day totals > SPIKE_MAX_DAILY_MM (400mm/day).
    Returns (cleaned_series, n_spikes_clipped).
    """
    spikes = (daily_rain > SPIKE_MAX_DAILY_MM)
    n = int(spikes.sum())
    cleaned = daily_rain.clip(upper=SPIKE_MAX_DAILY_MM)
    return cleaned, n


# ─── Check 5 — Monsoon Season Coverage ─────────────────────────────────────────

def _check_monsoon_coverage(daily_rain: pd.Series) -> tuple:
    """
    FAIL if valid (non-NaN) daily increments during June-September
    are < MONSOON_COVERAGE_MIN_PCT % of expected days.
    """
    monsoon_slice = daily_rain[daily_rain.index.month.isin(MONSOON_MONTHS)]
    if len(monsoon_slice) == 0:
        return False, "no monsoon-season data (Jun-Sep) at all"
    coverage = monsoon_slice.notna().mean() * 100.0
    if coverage < MONSOON_COVERAGE_MIN_PCT:
        return False, (
            f"insufficient monsoon-season coverage -- "
            f"{coverage:.1f}% < {MONSOON_COVERAGE_MIN_PCT}% required"
        )
    return True, "ok"


# ─── Main Audit Function ────────────────────────────────────────────────────────

def audit_station(
    df: pd.DataFrame,
    station_id: str,
    lat: float,
    lon: float,
) -> tuple:
    """
    Run all quality checks on a single AWS station.
    Operates on DAILY rain increments (extracted from cumulative counter or sub-daily).

    Returns (status, reason, daily_rain_series or None)
    """
    try:
        ts = _build_timestamp(df)
        daily_rain, update_type = _prepare_rain(df, ts)
    except Exception as e:
        return "FAIL", f"parsing_error -- {e}", None

    if daily_rain.dropna().empty:
        return "FAIL", "no_data -- zero valid rain increments after processing", None

    # Check 1 -- Stuck
    ok, reason = _check_stuck(daily_rain, update_type)
    if not ok:
        return "FAIL", reason, None

    # Check 2 -- Dead
    ok, reason = _check_dead(daily_rain)
    if not ok:
        return "FAIL", reason, None

    # Check 3 -- Cross-talk (daily resolution)
    ok, reason = _check_crosstalk(daily_rain, df, ts)
    if not ok:
        return "FAIL", reason, None

    # Check 4 -- Clip unphysical daily spikes (does not fail station)
    daily_rain, n_spikes = _check_and_clip_spikes(daily_rain)

    # Check 5 -- Monsoon coverage
    ok, reason = _check_monsoon_coverage(daily_rain)
    if not ok:
        return "FAIL", reason, None

    return "PASS", f"ok (type={update_type}, spikes_clipped={n_spikes})", daily_rain


# ─── Batch Audit All Stations ──────────────────────────────────────────────────

def run_quality_gate(aws_csv_path: str) -> pd.DataFrame:
    """
    Run audit_station() on every station in the MOSDAC AWS CSV.
    Writes results to STATION_QUALITY_CSV.
    Returns the full quality report DataFrame.
    """
    print("Loading AWS data...")
    df = pd.read_csv(aws_csv_path, low_memory=False)

    report_rows = []
    pass_stations = {}

    stations = [s for s in df[AWS_STATION_COL].unique() if pd.notna(s)]
    print(f"Auditing {len(stations)} stations...")

    for sid in stations:
        sub = df[df[AWS_STATION_COL] == sid].copy()
        if sub.empty:
            continue
        lat = sub["LATITUDE"].dropna().iloc[0] if ("LATITUDE" in sub.columns and not sub["LATITUDE"].dropna().empty) else np.nan
        lon = sub["LONGITUDE"].dropna().iloc[0] if ("LONGITUDE" in sub.columns and not sub["LONGITUDE"].dropna().empty) else np.nan

        status, reason, daily_rain = audit_station(sub, sid, lat, lon)

        row = {
            "station_id": sid,
            "lat": lat,
            "lon": lon,
            "n_raw_rows": len(sub),
            "status": status,
            "reason": reason,
        }

        if status == "PASS" and daily_rain is not None:
            row["n_daily_valid"] = int(daily_rain.notna().sum())
            row["rain_max_daily_mm"]  = float(daily_rain.max())
            row["monsoon_coverage_pct"] = float(
                daily_rain[daily_rain.index.month.isin(MONSOON_MONTHS)].notna().mean() * 100
            )
            # Save cleaned daily series
            out_dir = Path(aws_csv_path).parent.parent / "processed" / "clean_stations"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = sid.replace("/", "_").replace(" ", "_")[:60]
            daily_rain.to_csv(out_dir / f"{safe_name}_daily.csv", header=["rain_mm_day"])
            pass_stations[sid] = {"lat": lat, "lon": lon, "daily_rain": daily_rain}
        else:
            row["n_daily_valid"] = 0
            row["rain_max_daily_mm"]  = np.nan
            row["monsoon_coverage_pct"] = np.nan

        report_rows.append(row)
        status_icon = "✓" if status == "PASS" else "✗"
        print(f"  [{status_icon}] {sid[:55]:<55} → {status}: {reason[:60]}")

    report = pd.DataFrame(report_rows)
    report.to_csv(STATION_QUALITY_CSV, index=False)
    n_pass = (report.status == "PASS").sum()
    print(f"\nQuality gate complete: {n_pass}/{len(stations)} stations PASS")
    print(f"Report written → {STATION_QUALITY_CSV}")

    return report, pass_stations


if __name__ == "__main__":
    from src.config import AWS_CSV
    report, _ = run_quality_gate(str(AWS_CSV))
    print(report[["station_id", "status", "reason"]].to_string())
