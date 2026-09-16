import numpy as np
import pandas as pd
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import (
    CLOUDBURST_EVENTS_CSV, TRAINING_REPORT_MD,
    WEIGHT_CONFIRMED_CLOUDBURST, WEIGHT_CANDIDATE_CLOUDBURST,
    WEIGHT_WIDESPREAD_HEAVY_RAIN, WEIGHT_HEAVY_RAIN,
    WEIGHT_MODERATE_RAIN, WEIGHT_NORMAL,
    LOEO_EVENT_BUFFER_HOURS, L2_C_VALUES,
    FEATURES_AWS, FEATURES_IWV, FEATURES_SAT, FEATURES_STALENESS,
    POSITIVE_LABELS, HARD_NEGATIVE, DATA_RAW, DATA_PROCESSED
)

# ─── Staleness & Dropout Simulator ─────────────────────────────────────────────

def simulate_sensor_outage_blocks(df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    """
    Inject block-structured synthetic dropouts to simulate real-world infra failures.
    - AWS Rain (R): Empirical 46.5% missingness. Simulate contiguous ~4 hour blocks.
    - UTH/HEM: Empirical 16.8% normal missing, 27.6% on storm days. Simulate ~3 hour blocks.
    """
    rng = np.random.RandomState(seed)
    out = df.copy().reset_index(drop=True)
    
    stations = out["station_id"].unique()
    is_positive = out[label_col].isin(POSITIVE_LABELS)
    
    for stn in stations:
        mask = (out["station_id"] == stn)
        stn_idx = np.where(mask)[0]
        n_rows = len(stn_idx)
        if n_rows == 0: continue
        
        # AWS R dropout (~46.5% target). Assume 1 hr rows -> blocks of 4.
        aws_starts = rng.binomial(1, 0.116, size=n_rows).astype(bool)
        aws_drop = np.zeros(n_rows, dtype=bool)
        for i in np.where(aws_starts)[0]:
            block_len = rng.poisson(4) + 1
            aws_drop[i : i + block_len] = True
            
        out.loc[stn_idx[aws_drop], "rain_mm_hr"] = np.nan
        if "rain_mm_day" in out.columns:
            out.loc[stn_idx[aws_drop], "rain_mm_day"] = np.nan
        
        # UTH / HEM dropout (27.6% storm, 16.8% normal). Blocks of ~3.
        pos_mask = is_positive.iloc[stn_idx].values
        start_prob = np.where(pos_mask, 0.092, 0.056)
        sat_starts = (rng.rand(n_rows) < start_prob)
        sat_drop = np.zeros(n_rows, dtype=bool)
        for i in np.where(sat_starts)[0]:
            block_len = rng.poisson(3) + 1
            sat_drop[i : i + block_len] = True
            
        if "uth_mean" in out.columns: out.loc[stn_idx[sat_drop], "uth_mean"] = np.nan
        if "hem_mean" in out.columns: out.loc[stn_idx[sat_drop], "hem_mean"] = np.nan

    return out

def generate_staleness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute `{channel}_staleness_s` (continuous) and `{channel}_valid` (binary) 
    BEFORE imputation occurs, passing missingness explicitly to the model.
    """
    df = df.copy()
    
    for base_col, prefix in [("R", "R"), ("uth_mean", "uth"), ("hem_mean", "hem")]:
        valid_col = f"{prefix}_valid"
        stale_col = f"{prefix}_staleness_s"
        
        if base_col in df.columns:
            if valid_col not in df.columns:
                df[valid_col] = df[base_col].notna().astype(float)
            else:
                df[valid_col] = df[valid_col].fillna(0.0)
                
            # Time since last valid reading
            valid_times = df["timestamp"].where(df[base_col].notna())
            last_valid_time = valid_times.groupby(df["station_id"]).ffill()
            
            staleness = (df["timestamp"] - last_valid_time).dt.total_seconds()
            df[stale_col] = staleness.fillna(86400.0)  # Default to 24h if never seen
        else:
            df[valid_col] = 0.0
            df[stale_col] = 86400.0
            
    return df

# ─── Feature Engineering ───────────────────────────────────────────────────────

def build_feature_matrix(events_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, List[str]]:
    """
    Build X (feature matrix), y (binary label), weights, and event group metadata.
    """
    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values(["station_id", "timestamp"])

    def _rolling_features(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.set_index("timestamp").sort_index()
        rain = g["rain_mm_hr"].fillna(0.0)
        elapsed_s  = g.index.to_series().diff().dt.total_seconds()
        
        if len(elapsed_s.dropna()) > 0:
            assert (elapsed_s.dropna() > 0).all(), f"Timestamps are not strictly sequential for station {grp['station_id'].iloc[0]}"
            
        elapsed_s = elapsed_s.fillna(3600.0)
        elapsed_hr = (elapsed_s.clip(lower=60.0) / 3600.0)
        R    = rain / elapsed_hr
        R_30 = rain.rolling(pd.Timedelta("30min"), closed="right", min_periods=1).sum()
        R_60 = rain.rolling(pd.Timedelta("60min"), closed="right", min_periods=1).sum()
        RI   = R.diff().fillna(0.0) / elapsed_hr

        rain_3d = rain.rolling(pd.Timedelta("3D"), closed="left", min_periods=1).sum()
        rain_7d = rain.rolling(pd.Timedelta("7D"), closed="left", min_periods=1).sum()

        def _slope(x):
            if len(x) < 2: return 0.0
            t = np.arange(len(x), dtype=float)
            return float(np.polyfit(t, x, 1)[0])
        rain_trend_7d = rain.rolling(pd.Timedelta("7D"), closed="left", min_periods=2).apply(
            _slope, raw=True
        ).fillna(0.0)

        g["R"]               = R.values
        g["R_30"]            = R_30.values
        g["R_60"]            = R_60.values
        g["RI"]              = RI.values
        g["rain_3day_accum"] = rain_3d.values
        g["rain_7day_accum"] = rain_7d.values
        g["rain_trend_7day"] = rain_trend_7d.values
        return g.reset_index()

    parts = [_rolling_features(grp) for _, grp in df.groupby("station_id", sort=False)]
    df = pd.concat(parts, ignore_index=True)
    df["rain_mm_hr"] = df["R"]

    for col in ["doy", "month", "lat", "lon"]:
        if col not in df.columns and col in events_df.columns:
            df[col] = events_df[col].values

    if "L_score" in df.columns:
        df["spatial_contrast"] = df["R"] * df["L_score"].fillna(0.0)
    else:
        df["spatial_contrast"] = 0.0

    df = df.dropna(subset=["timestamp", "rain_mm_hr", "final_label"])

    for col in FEATURES_SAT + FEATURES_IWV:
        if col not in df.columns:
            df[col] = np.nan

    label_col = "rain_label" if "rain_label" in df.columns else "final_label"
    df = generate_staleness_features(df)
    
    available = [f for f in (FEATURES_AWS + FEATURES_IWV + FEATURES_SAT + FEATURES_STALENESS)
                 if f in df.columns and df[f].notna().mean() > 0.05]

    X = df[available].copy()
    
    if "uth_nearest_px_km" in X.columns:
        X["uth_nearest_px_km"] = X["uth_nearest_px_km"].fillna(250.0)

    y = df[label_col].map({
        "CONFIRMED_CLOUDBURST":  1,
        "CANDIDATE_CLOUDBURST":  1,
        "WIDESPREAD_HEAVY_RAIN": 0,
        "HEAVY_RAIN":            0,
        "MODERATE_RAIN":         0,
        "NORMAL":                0,
    }).fillna(0).astype(int)

    weight_map = {
        "CONFIRMED_CLOUDBURST":  WEIGHT_CONFIRMED_CLOUDBURST,
        "CANDIDATE_CLOUDBURST":  WEIGHT_CANDIDATE_CLOUDBURST,
        "WIDESPREAD_HEAVY_RAIN": WEIGHT_WIDESPREAD_HEAVY_RAIN,
        "HEAVY_RAIN":            WEIGHT_HEAVY_RAIN,
        "MODERATE_RAIN":         WEIGHT_MODERATE_RAIN,
        "NORMAL":                WEIGHT_NORMAL,
    }
    weights = df[label_col].map(weight_map).fillna(0.05)

    event_groups = _assign_event_groups(df)

    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Features used: {available}")
    print(f"Positive (cloudburst) samples: {y.sum()} / {len(y)}")

    return X, y, weights, event_groups, available


def _assign_event_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster positive events with a >=7 day gap.
    Returns a DataFrame containing cluster_id, timestamp, and region_name
    to support the 7-day universal boundary purge logic.
    """
    df = df.copy()
    label_col = "rain_label" if "rain_label" in df.columns else "final_label"
    region_id = df["region_name"].astype(str) if "region_name" in df.columns else "all"
    
    pos_mask = df[label_col].isin(POSITIVE_LABELS)
    pos_df = df[pos_mask].copy()
    
    cluster_series = pd.Series(-1, index=df.index, dtype=int)
    cluster_id_counter = 0
    
    # 1. Cluster positive events by 7-day gap
    for region, group in pos_df.groupby(region_id):
        group = group.sort_values("timestamp")
        gap = group["timestamp"].diff() > pd.Timedelta("7 days")
        block_id = gap.cumsum()
        
        for b_id, b_group in group.groupby(block_id):
            cluster_id_counter += 1
            cluster_series.loc[b_group.index] = cluster_id_counter
            
    meta_df = pd.DataFrame({
        "cluster_id": cluster_series,
        "timestamp": df["timestamp"],
        "region_name": region_id,
        "is_positive": pos_mask
    }, index=df.index)
    
    print(f"  Event grouping: {cluster_id_counter} independent positive storm clusters (>7 day gap).")
    return meta_df


# ─── Event-Grouped Train / Validation / Test Holdout Split ────────────────────

def greedy_split(cluster_sizes, train_frac=0.7, val_frac=0.15):
    total_events = sum(s[1] for s in cluster_sizes)
    target_train = int(total_events * train_frac)
    target_val = int(total_events * val_frac)
    target_test = total_events - target_train - target_val
    
    train_ids, val_ids, test_ids = [], [], []
    curr_train, curr_val, curr_test = 0, 0, 0
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)
    
    for cid, size in sorted_clusters:
        def_train = target_train - curr_train
        def_val = target_val - curr_val
        def_test = target_test - curr_test
        max_def = max(def_train, def_val, def_test)
        if max_def == def_train:
            train_ids.append(cid)
            curr_train += size
        elif max_def == def_val:
            val_ids.append(cid)
            curr_val += size
        else:
            test_ids.append(cid)
            curr_test += size
    return train_ids, val_ids, test_ids


def event_grouped_split(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    event_groups: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Leak-proof train / validation / test split by storm event cluster.
    Implements a strict 7-day universal purge buffer at every train/test boundary.
    """
    pos_clusters = event_groups[event_groups["is_positive"] == True]
    cluster_counts = pos_clusters.groupby("cluster_id").size().to_dict()
    
    # 1. Distribute clusters via size-aware stratification
    train_ids, val_ids, test_ids = greedy_split(list(cluster_counts.items()), train_frac, val_frac)
    
    # Map back to string sets for quick lookup
    train_gids = set(train_ids)
    val_gids   = set(val_ids)
    test_gids  = set(test_ids)
    
    # 2. Provisional assignment of every row (positive and negative)
    # We assign each row to the closest cluster in time within its region.
    provisional_assignment = np.zeros(len(X), dtype=int)  # 0=train, 1=val, 2=test
    
    cluster_bounds = []
    for cid in pos_clusters["cluster_id"].unique():
        c_df = pos_clusters[pos_clusters["cluster_id"] == cid]
        cluster_bounds.append({
            "cluster_id": cid,
            "region_name": c_df["region_name"].iloc[0],
            "start": c_df["timestamp"].min(),
            "end": c_df["timestamp"].max(),
            "assign": 0 if cid in train_gids else (1 if cid in val_gids else 2)
        })
    c_bounds_df = pd.DataFrame(cluster_bounds)
    
    # Map all rows
    for region, group in event_groups.groupby("region_name"):
        r_clusters = c_bounds_df[c_bounds_df["region_name"] == region]
        if len(r_clusters) == 0:
            # No clusters in this region, assign arbitrarily to train
            provisional_assignment[group.index] = 0
            continue
            
        t_int = group["timestamp"].astype(np.int64).values[:, None]
        starts_int = r_clusters["start"].astype(np.int64).values[None, :]
        ends_int = r_clusters["end"].astype(np.int64).values[None, :]
        
        d_start = np.maximum(0, starts_int - t_int)
        d_end = np.maximum(0, t_int - ends_int)
        dist = np.maximum(d_start, d_end)
        
        nearest_idx = np.argmin(dist, axis=1)
        provisional_assignment[group.index] = r_clusters["assign"].values[nearest_idx]
        
    # 3. Universal 7-Day Boundary Purge
    # Any row within 7 days of a positive event in a DIFFERENT bucket is dropped.
    purge_mask = np.zeros(len(X), dtype=bool)
    
    for region, group in event_groups.groupby("region_name"):
        r_clusters = c_bounds_df[c_bounds_df["region_name"] == region]
        if len(r_clusters) == 0: continue
        
        t_int = group["timestamp"].astype(np.int64).values
        row_assignments = provisional_assignment[group.index]
        
        for target_assign in [0, 1, 2]:
            opposing_clusters = r_clusters[r_clusters["assign"] != target_assign]
            if len(opposing_clusters) == 0: continue
            
            starts_int = opposing_clusters["start"].astype(np.int64).values[None, :]
            ends_int = opposing_clusters["end"].astype(np.int64).values[None, :]
            
            d_start = np.maximum(0, starts_int - t_int[:, None])
            d_end = np.maximum(0, t_int[:, None] - ends_int)
            dist_to_opposing = np.maximum(d_start, d_end).min(axis=1)
            
            # If distance < 7 days AND the row is in the target_assign bucket -> purge
            # 7 days = 7 * 24 * 3600 * 10^9 nanoseconds
            seven_days_ns = 7 * 24 * 3600 * 1000000000
            
            bad_rows = (row_assignments == target_assign) & (dist_to_opposing < seven_days_ns)
            purge_mask[group.index[bad_rows]] = True

    print(f"  Purge Buffer: Dropped {purge_mask.sum()} cross-boundary leaked rows.")
    
    train_mask = (provisional_assignment == 0) & (~purge_mask)
    val_mask   = (provisional_assignment == 1) & (~purge_mask)
    test_mask  = (provisional_assignment == 2) & (~purge_mask)

    def _pack(mask):
        return {
            "X": X.values[mask],
            "y": y.values[mask],
            "w": weights.values[mask],
            "count": int(mask.sum()),
            "pos_count": int(y.values[mask].sum())
        }

    train_data = _pack(train_mask)
    val_data   = _pack(val_mask)
    test_data  = _pack(test_mask)

    return train_data, val_data, test_data
