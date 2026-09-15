"""Phase A — Region Discovery via GAGAN IWV.

Algorithm:
  1. Parse gagan_iwv_v1.txt for all stations in the March 2013–Feb 2014 window.
  2. Filter by coverage >= 85% AND seasonal shape sanity (monsoon/winter ratio >= 1.5).
  3. For each surviving GAGAN station, find AWS cluster within 50km.
  4. Run AWS candidates through Phase B quality gate.
  5. Score and rank candidate regions.
  6. Write region_scores.csv; return top-ranked region.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import (
    GAGAN_IWV_FILE, GAGAN_STATIONS_FILE, AWS_CSV,
    WINDOW_START, WINDOW_END,
    GAGAN_MIN_COVERAGE_PCT, GAGAN_SEASONAL_RATIO_MIN,
    AWS_CLUSTER_RADIUS_KM, AWS_MIN_STATIONS,
    AWS_MIN_SPACING_KM, AWS_MAX_SPACING_KM,
    SCORE_WEIGHT_IWV_COVERAGE, SCORE_WEIGHT_N_AWS_PASS,
    SCORE_WEIGHT_MONSOON_COV, SCORE_WEIGHT_SPACING,
    MONSOON_MONTHS, REGION_SCORES_CSV,
    AWS_STATION_COL, AWS_LAT_COL, AWS_LON_COL
)
from src.utils import haversine_km, pairwise_distances, iwv_shape_sanity
from src.phase_b_quality_gate import run_quality_gate


# ─── Station name lookup ────────────────────────────────────────────────────────

GAGAN_STATION_NAMES = {
    1:"Madurai", 2:"Delhi", 3:"Bangalore", 4:"Hyderabad", 5:"Bhopal",
    6:"Lucknow", 7:"Shimla", 8:"Agatti", 9:"Port Blair", 10:"Mumbai",
    11:"Bagdogra", 12:"Lengpui", 13:"Guwahati", 14:"Kolkata", 15:"Raipur",
    16:"Vizag", 17:"Gaya", 18:"Trivandrum", 19:"Hubli", 20:"Ahmedabad1",
    21:"Ahmedabad2", 22:"Ahmedabad3", 23:"Nagpur", 24:"Aurangabad",
    25:"Agra", 26:"Bhubaneswar", 27:"Khajuraho", 28:"Jodhpur"
}


# ─── Step 1: Parse GAGAN IWV ───────────────────────────────────────────────────

def load_gagan_iwv() -> pd.DataFrame:
    """Load and parse gagan_iwv_v1.txt into a tidy DataFrame."""
    print("Loading GAGAN IWV data...")
    gagan = pd.read_csv(
        GAGAN_IWV_FILE, sep=r"\s+", header=None,
        names=["station_id", "lat", "lon", "year", "month", "day",
               "hour", "minute", "IWV", "ZTD"]
    )
    gagan["dt"] = pd.to_datetime({
        "year": gagan.year, "month": gagan.month, "day": gagan.day,
        "hour": gagan.hour, "minute": gagan.minute
    }, errors="coerce")
    # Remove tiny negative IWV (GPS retrieval artifact) — replace with NaN
    gagan.loc[gagan.IWV < 0, "IWV"] = np.nan
    gagan["name"] = gagan.station_id.map(GAGAN_STATION_NAMES)
    print(f"  {len(gagan):,} records, {gagan.station_id.nunique()} stations")
    return gagan


# ─── Step 2: Coverage & Seasonal Filter ────────────────────────────────────────

def evaluate_gagan_stations(gagan: pd.DataFrame) -> pd.DataFrame:
    """
    For each GAGAN station compute coverage_pct and seasonal sanity.
    Returns DataFrame of stations that pass both filters.
    """
    window_start = pd.Timestamp(WINDOW_START)
    window_end   = pd.Timestamp(WINDOW_END + " 23:30")
    expected = int((window_end - window_start).total_seconds() / (30 * 60)) + 1

    gagan_win = gagan[(gagan.dt >= window_start) & (gagan.dt <= window_end)]

    rows = []
    for sid, grp in gagan_win.groupby("station_id"):
        grp = grp.sort_values("dt")
        lat = grp.lat.iloc[0]
        lon = grp.lon.iloc[0]
        name = GAGAN_STATION_NAMES.get(int(sid), f"Station_{sid}")
        count = grp.IWV.notna().sum()
        coverage = count / expected * 100.0

        shape_ok, shape_reason = iwv_shape_sanity(grp.IWV, grp.dt)

        passes = coverage >= GAGAN_MIN_COVERAGE_PCT and shape_ok
        rows.append({
            "station_id": sid, "name": name, "lat": lat, "lon": lon,
            "coverage_pct": round(coverage, 2),
            "iwv_mean": round(grp.IWV.mean(), 2),
            "iwv_max":  round(grp.IWV.max(), 2),
            "iwv_min":  round(grp.IWV.min(), 2),
            "shape_ok": shape_ok,
            "shape_reason": shape_reason,
            "gagan_passes": passes
        })

    df = pd.DataFrame(rows)
    passing = df[df.gagan_passes]
    print(f"  GAGAN stations passing (cov≥{GAGAN_MIN_COVERAGE_PCT}%, seasonal OK): "
          f"{len(passing)} of {len(df)}")
    for _, r in passing.iterrows():
        print(f"    ID {int(r.station_id):>2} {r['name']:<14} lat={r.lat:.3f} lon={r.lon:.3f} "
              f"cov={r.coverage_pct:.1f}% IWV={r.iwv_mean:.1f}mm")
    return df


# ─── Step 3 & 4: AWS Cluster Matching + Phase B ────────────────────────────────

def find_aws_cluster(
    gagan_lat: float, gagan_lon: float,
    pass_stations: Dict,
    quality_report: pd.DataFrame
) -> Tuple[List[str], float, float]:
    """
    Find AWS stations that PASS Phase B within AWS_CLUSTER_RADIUS_KM of the
    GAGAN station, have acceptable inter-station spacing, and cover monsoon months.

    Returns (list_of_station_ids, monsoon_coverage_mean, spacing_score)
    """
    candidates = quality_report[quality_report.status == "PASS"].copy()
    if candidates.empty:
        return [], 0.0, 0.0

    # Distance from GAGAN station
    candidates = candidates.copy()
    candidates["dist_km"] = candidates.apply(
        lambda row: haversine_km(gagan_lat, gagan_lon, row.lat, row.lon), axis=1
    )
    nearby = candidates[candidates.dist_km <= AWS_CLUSTER_RADIUS_KM].copy()

    if len(nearby) < AWS_MIN_STATIONS:
        return [], 0.0, 0.0

    # Pairwise inter-station distances — check spacing constraint
    coords = list(zip(nearby.lat, nearby.lon))
    dist_matrix = pairwise_distances(coords)
    np.fill_diagonal(dist_matrix, np.inf)

    # Greedy spatial thinning — keep one station if multiple are within AWS_MIN_SPACING_KM
    keep_idx = []
    for i in range(len(coords)):
        if not any(dist_matrix[i, k] < AWS_MIN_SPACING_KM for k in keep_idx):
            keep_idx.append(i)

    nearby_valid = nearby.iloc[keep_idx].copy()

    # Accept whatever passes spacing filter down to AWS_MIN_STATIONS
    if len(nearby_valid) < AWS_MIN_STATIONS:
        return [], 0.0, 0.0

    # Spacing score: reward spread up to AWS_MAX_SPACING_KM
    valid_coords = list(zip(nearby_valid.lat, nearby_valid.lon))
    d = pairwise_distances(valid_coords)
    np.fill_diagonal(d, 0)
    if len(valid_coords) > 1:
        mean_spacing = d[d > 0].mean()
        spacing_score = min(mean_spacing / AWS_MAX_SPACING_KM, 1.0)
    else:
        spacing_score = 0.0

    # Monsoon coverage across cluster
    monsoon_cov_mean = nearby_valid.monsoon_coverage_pct.mean() / 100.0

    station_ids = nearby_valid[AWS_STATION_COL].tolist() if AWS_STATION_COL in nearby_valid else nearby_valid.station_id.tolist()
    return station_ids, monsoon_cov_mean, spacing_score


# ─── Step 5: Scoring ──────────────────────────────────────────────────────────

def score_region(
    coverage_pct: float,
    n_pass_aws: int,
    monsoon_cov: float,
    spacing_score: float
) -> float:
    """Composite 0–1 score for a candidate region."""
    aws_score = min(n_pass_aws / 5.0, 1.0)   # Normalise to 5 stations target
    return (
        SCORE_WEIGHT_IWV_COVERAGE  * (coverage_pct / 100.0) +
        SCORE_WEIGHT_N_AWS_PASS    * aws_score +
        SCORE_WEIGHT_MONSOON_COV   * monsoon_cov +
        SCORE_WEIGHT_SPACING       * spacing_score
    )


# ─── Master Phase A Runner ────────────────────────────────────────────────────

def run_region_discovery() -> Tuple[pd.DataFrame, Dict]:
    """
    Execute Phase A end-to-end.
    Returns (region_scores_df, best_region_info_dict).
    """
    print("\n" + "=" * 60)
    print("PHASE A — Region Discovery via GAGAN IWV")
    print("=" * 60)

    # 1. Load GAGAN IWV
    gagan = load_gagan_iwv()

    # 2. GAGAN station filter
    print("\nStep 2: Evaluating GAGAN station coverage & seasonal shape...")
    gagan_eval = evaluate_gagan_stations(gagan)
    passing_gagan = gagan_eval[gagan_eval.gagan_passes]

    if passing_gagan.empty:
        print("ERROR: No GAGAN stations pass quality filter. Cannot proceed.")
        return pd.DataFrame(), {}

    # 3 & 4. Run Phase B on AWS, then match clusters
    print("\nStep 3-4: Running Phase B quality gate on all AWS stations...")
    quality_report, pass_stations = run_quality_gate(str(AWS_CSV))

    # Rename station_id in quality_report to match AWS station col
    if "station_id" in quality_report.columns:
        quality_report = quality_report.rename(columns={"station_id": AWS_STATION_COL})

    # 5. Score each GAGAN candidate region
    print("\nStep 5: Scoring candidate regions...")
    score_rows = []
    for _, grow in passing_gagan.iterrows():
        station_ids, monsoon_cov, spacing_score = find_aws_cluster(
            grow.lat, grow.lon, pass_stations, quality_report
        )
        n_aws = len(station_ids)
        composite = score_region(
            grow.coverage_pct, n_aws, monsoon_cov, spacing_score
        )
        score_rows.append({
            "gagan_station_id": int(grow.station_id),
            "region_name": grow["name"],
            "gagan_lat": grow.lat,
            "gagan_lon": grow.lon,
            "gagan_coverage_pct": grow.coverage_pct,
            "gagan_iwv_mean": grow.iwv_mean,
            "n_aws_pass": n_aws,
            "aws_monsoon_coverage_pct": round(monsoon_cov * 100, 1),
            "aws_spacing_score": round(spacing_score, 3),
            "composite_score": round(composite, 4),
            "aws_station_ids": "|".join(station_ids)
        })

    scores_df = pd.DataFrame(score_rows).sort_values("composite_score", ascending=False)
    scores_df.to_csv(REGION_SCORES_CSV, index=False)

    print("\n── Region Scores (ranked) ──")
    for _, r in scores_df.iterrows():
        print(f"  [{r.composite_score:.3f}] {r.region_name:<14} "
              f"gagan_cov={r.gagan_coverage_pct:.1f}%  "
              f"n_aws={r.n_aws_pass}  "
              f"monsoon_cov={r.aws_monsoon_coverage_pct:.0f}%")

    print(f"\nRegion scores written → {REGION_SCORES_CSV}")

    if scores_df.empty:
        print("WARNING: No region meets minimum AWS cluster requirements.")
        return scores_df, {}

    best = scores_df.iloc[0]
    print(f"\n>>> Selected region: {best.region_name} "
          f"(score={best.composite_score:.3f}, "
          f"n_aws={best.n_aws_pass}, "
          f"gagan_cov={best.gagan_coverage_pct:.1f}%)")

    # Return best region info and all active regions for multi-region downstream phases
    active_regions = []
    for _, r in scores_df[scores_df.n_aws_pass > 0].iterrows():
        active_regions.append({
            "region_name": r.region_name,
            "gagan_lat": r.gagan_lat,
            "gagan_lon": r.gagan_lon,
            "gagan_station_id": int(r.gagan_station_id),
            "aws_station_ids": r.aws_station_ids.split("|") if r.aws_station_ids else [],
            "composite_score": r.composite_score
        })

    best_info = {
        "region_name": best.region_name,
        "gagan_lat": best.gagan_lat,
        "gagan_lon": best.gagan_lon,
        "gagan_station_id": int(best.gagan_station_id),
        "aws_station_ids": best.aws_station_ids.split("|") if best.aws_station_ids else [],
        "pass_stations": pass_stations,
        "gagan_df": gagan,
        "all_active_regions": active_regions
    }
    return scores_df, best_info


if __name__ == "__main__":
    scores, best = run_region_discovery()
    print("\nTop region:", best.get("region_name"))
    print("AWS stations selected:", best.get("aws_station_ids"))
