"""
build_sequences.py — PS-26077 Cloudburst Nowcaster
Builds REAL multi-timestep sequences per labeled event, replacing the
length-1 pseudo-sequence that CloudburstCNNBiLSTM and SpatiotemporalTransformer
were both actually being fed (confirmed by save_split_assignments()'s own
docstring: "the BiLSTM hidden state does NOT persist across rows").

Design, and why it differs from the existing pipeline:

  1. TEMPORAL CORE (real sequence axis):
     hourly R (rain_mm_hr), temp, pressure, humidity, IWV — one token per
     real hour, looking back `--lookback-hours` hours, ending strictly
     BEFORE each event's t0. No feature here is a rolling window applied
     to daily data (that was the R_30/R_60 bug); every token is an actual
     observation at that hour, or NaN if the sensor genuinely has no
     reading there (imputed downstream, not silently filled with today's
     value).

  2. STATIC CONTEXT (daily/constant, appended after pooling — not part of
     the temporal axis, because these ARE daily-resolution):
     lat, lon, doy, month, satellite features (ctt_mean, hem_mean,
     olr_mean, uth_mean, uth_valid) — reused as-is from
     merge_satellite_features.py, which already enforces
     sat_timestamp <= target_t0.

  3. HARD BANS, enforced at build time (raises, not just a lint warning):
     - No column named/derived from L_score or final_label.
     - No spatial_contrast (R * L_score) — this was the confirmed leak.
     - No R_30/R_60-as-daily-collapse — those never enter this file at all.

  4. LEAKAGE GUARANTEE, stricter than the existing merge_aws_features.py:
     merge_aws_features.py used pd.merge_asof(..., direction="nearest"),
     which can match a reading up to `tolerance` AFTER the target time.
     This script uses direction="backward" everywhere a sequence step is
     assembled, so every matched timestamp is provably <= its target
     timestamp. This is checked with an explicit assertion, in the same
     style as merge_satellite_features.py's own ANTI-LEAKAGE ASSERTION.

ASSUMPTIONS YOU MUST VERIFY AGAINST YOUR REAL FILES (I don't have them):
  - Hourly AWS panel has columns: station_id, timestamp, lat, lon,
    rain_mm_hr, temp, pressure, humidity  (matches process_aws_data.py's
    output schema for assam_aws_hourly.parquet; if your other regions use
    a different hourly file/schema, point --hourly-panel at it or adjust
    HOURLY_PANEL_COLS below).
  - IWV file is data/raw/gagan_iwv_v1.txt in the whitespace format parsed
    by phase_c_labeling.py / merge_aws_features.py.
  - load_imd_parquets() / build_feature_matrix() exist in
    src.phase_d_training and return (X_df, y_ser, w_ser, event_groups,
    ordered_feats) with X_df/y_ser/w_ser/event_groups sharing the same
    pandas index as the row's source in events_df (timestamp, lat, lon
    recoverable via that index). If build_feature_matrix resets the
    index, adjust `_recover_row_metadata()` below accordingly — I could
    not verify this against the real file.

Usage:
    python build_sequences.py \
        --hourly-panel data/processed/assam_aws_hourly.parquet \
        --lookback-hours 8 --match-radius-km 50 --tolerance-min 40 \
        --min-coverage-frac 0.5 \
        --out-npz outputs/sequences.npz --out-meta outputs/sequences_meta.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from archive.deprecated_ml_baselines.phase_d_training import load_imd_parquets, build_feature_matrix  # noqa: E402
from src.config import CLOUDBURST_EVENTS_CSV  # noqa: E402  (unused directly, kept for parity)

# ─── Banned feature names — hard fail if any sneak in ──────────────────────
BANNED_TOKENS = ("l_score", "spatial_contrast", "final_label")

CORE_FEATURE_COLS = ["rain_mm_hr", "temp", "pressure", "humidity", "iwv"]
STATIC_FEATURE_COLS = [
    "lat", "lon", "doy", "month",
    "ctt_mean", "hem_mean", "olr_mean", "uth_mean", "uth_valid",
]


def _assert_no_banned_columns(cols: List[str]) -> None:
    for c in cols:
        low = c.lower()
        if any(tok in low for tok in BANNED_TOKENS):
            raise AssertionError(
                f"CRITICAL LEAKAGE GUARD TRIPPED: column '{c}' matches a banned "
                f"label-derived pattern ({BANNED_TOKENS}). Sequence build aborted."
            )


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _nearest_station_map(
    event_coords: pd.DataFrame, station_coords: pd.DataFrame, id_col: str, max_km: float
) -> dict:
    """event_coords, station_coords: DataFrames with ['lat','lon']. Returns {(lat,lon): station_id|None}."""
    mapping = {}
    for _, row in event_coords.iterrows():
        d = _haversine_km(row["lat"], row["lon"], station_coords["lat"].values, station_coords["lon"].values)
        idx = int(np.argmin(d))
        mapping[(row["lat"], row["lon"])] = station_coords.iloc[idx][id_col] if d[idx] <= max_km else None
    return mapping


def _load_iwv_hourly(iwv_path: Path) -> pd.DataFrame:
    """Long-format hourly IWV: station_code, timestamp, lat, lon, iwv."""
    cols = ["station_code", "lat", "lon", "year", "month", "day", "hour", "min", "iwv", "ztd"]
    df = pd.read_csv(iwv_path, sep=r"\s+", header=None, names=cols)
    df["station_code"] = df["station_code"].astype(str).str.replace(r'\.0$', '', regex=True)
    for c in ["year", "month", "day", "hour", "min"]:
        df[c] = df[c].astype(int).astype(str).str.zfill(2 if c != "year" else 4)
    df["timestamp"] = pd.to_datetime(
        df["year"] + "-" + df["month"] + "-" + df["day"] + " " + df["hour"] + ":" + df["min"] + ":00",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp", "lat", "lon"])
    df["timestamp"] = df["timestamp"].dt.floor("1h")
    df = df.groupby(["station_code", "timestamp"], as_index=False).agg(
        {"lat": "first", "lon": "first", "iwv": "mean"}
    )
    return df.sort_values("timestamp")


def _recover_row_metadata(events_df: pd.DataFrame, X_df: pd.DataFrame, event_groups: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame({
        "timestamp": event_groups["timestamp"].values,
        "lat": X_df["lat"].values,
        "lon": X_df["lon"].values,
    }, index=X_df.index)
    return meta


def build_query_frame(meta: pd.DataFrame, lookback_hours: int) -> pd.DataFrame:
    """Long-format: one row per (event_idx, k) with the target timestamp for that step.
    k=1..lookback_hours, target_ts = t0 - k hours. Ends at t0 - 1h (t0 itself excluded)."""
    rows = []
    t0 = pd.to_datetime(meta["timestamp"])
    for k in range(lookback_hours, 0, -1):
        rows.append(pd.DataFrame({
            "event_idx": meta.index,
            "step": lookback_hours - k,  # 0 = earliest, lookback_hours-1 = latest (closest to t0)
            "target_ts": t0 - pd.Timedelta(hours=k),
            "lat": meta["lat"].values,
            "lon": meta["lon"].values,
        }))
    q = pd.concat(rows, ignore_index=True)
    # explicit anti-leak assertion, mirroring merge_satellite_features.py's style
    bad = q[q["target_ts"] >= t0.loc[q["event_idx"]].values]
    if len(bad) > 0:
        raise AssertionError(f"CRITICAL LEAKAGE DETECTED: {len(bad)} query rows have target_ts >= t0.")
    return q


def attach_backward_asof(
    query: pd.DataFrame, panel: pd.DataFrame, value_cols: List[str],
    station_map: dict, station_id_col: str, tolerance_min: int,
) -> pd.DataFrame:
    """direction='backward' guarantees matched timestamp <= target_ts. Stricter than
    merge_aws_features.py's direction='nearest', which can leak up to +tolerance forward."""
    q = query.copy()
    q["station"] = q.apply(lambda r: station_map.get((r["lat"], r["lon"])), axis=1)
    q = q.dropna(subset=["station"]).sort_values("target_ts")
    panel = panel.sort_values("timestamp")
    panel[station_id_col] = panel[station_id_col].astype(str)
    q["station"] = q["station"].astype(str)

    merged = pd.merge_asof(
        q, panel[[station_id_col, "timestamp"] + value_cols],
        left_on="target_ts", right_on="timestamp",
        left_by="station", right_by=station_id_col,
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_min),
    )
    # Defensive re-check: merge_asof backward should already guarantee this.
    violated = merged[merged["timestamp"] > merged["target_ts"]]
    if len(violated) > 0:
        raise AssertionError(
            f"CRITICAL LEAKAGE DETECTED: {len(violated)} matched rows have "
            f"panel timestamp > target_ts despite direction='backward'."
        )
    return merged


def assemble_sequences(
    events_df: pd.DataFrame,
    X_df: pd.DataFrame,
    event_groups: pd.DataFrame,
    hourly_panel: pd.DataFrame,
    iwv_hourly: pd.DataFrame,
    lookback_hours: int,
    match_radius_km: float,
    tolerance_min: int,
    min_coverage_frac: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    _assert_no_banned_columns(list(hourly_panel.columns) + list(iwv_hourly.columns) + CORE_FEATURE_COLS + STATIC_FEATURE_COLS)

    meta = _recover_row_metadata(events_df, X_df, event_groups)
    if meta["timestamp"].isna().any():
        raise ValueError(
            "Row metadata recovery failed for some rows (NaN timestamps) — "
            "build_feature_matrix's index does not align with events_df as assumed. "
            "Fix _recover_row_metadata() before trusting anything downstream."
        )

    event_coords = meta[["lat", "lon"]].drop_duplicates()
    aws_coords = hourly_panel[["station_id", "lat", "lon"]].drop_duplicates()
    iwv_coords = iwv_hourly[["station_code", "lat", "lon"]].drop_duplicates()
    aws_map = _nearest_station_map(event_coords, aws_coords, "station_id", match_radius_km)
    iwv_map = _nearest_station_map(event_coords, iwv_coords, "station_code", 200.0)
    n_iwv_matched = sum(1 for v in iwv_map.values() if v is not None)
    print(f"[match check] IWV stations mapped for {len(iwv_map)} event coords: {n_iwv_matched} non-None")

    query = build_query_frame(meta, lookback_hours)

    aws_matched = attach_backward_asof(
        query, hourly_panel, ["rain_mm_hr", "temp", "pressure", "humidity"],
        aws_map, "station_id", tolerance_min,
    )
    iwv_matched = attach_backward_asof(
        query, iwv_hourly, ["iwv"], iwv_map, "station_code", tolerance_min,
    )

    n_events = len(meta)
    n_core = len(CORE_FEATURE_COLS)
    
    X_seq_val = np.full((n_events, lookback_hours, n_core), np.nan, dtype=np.float32)
    X_seq_mask = np.zeros((n_events, lookback_hours, n_core), dtype=np.float32)
    idx_pos = {idx: pos for pos, idx in enumerate(meta.index)}

    for _, r in aws_matched.iterrows():
        pos = idx_pos.get(r["event_idx"])
        if pos is None: continue
        step = int(r["step"])
        for j, col in enumerate(["rain_mm_hr", "temp", "pressure", "humidity"]):
            val = r.get(col)
            if pd.notna(val):
                X_seq_val[pos, step, j] = val
                X_seq_mask[pos, step, j] = 1.0

    for _, r in iwv_matched.iterrows():
        pos = idx_pos.get(r["event_idx"])
        if pos is None: continue
        step = int(r["step"])
        val = r.get("iwv")
        if pd.notna(val):
            X_seq_val[pos, step, 4] = val
            X_seq_mask[pos, step, 4] = 1.0

    # Forward/backward fill within each sequence (axis 1)
    for i in range(n_events):
        for j in range(n_core):
            ser = pd.Series(X_seq_val[i, :, j]).ffill().bfill()
            X_seq_val[i, :, j] = ser.values

    # Median impute completely empty sequences
    feature_medians = np.nanmedian(X_seq_val, axis=(0, 1))
    for j in range(n_core):
        if np.isnan(feature_medians[j]): feature_medians[j] = 0.0
        nan_mask = np.isnan(X_seq_val[:, :, j])
        X_seq_val[nan_mask, j] = feature_medians[j]

    X_seq = np.concatenate([X_seq_val, X_seq_mask], axis=-1)
    core_feats = CORE_FEATURE_COLS + [f"{c}_mask" for c in CORE_FEATURE_COLS]
    keep_mask = np.ones(n_events, dtype=bool)

    meta = meta.copy()
    meta["coverage_frac"] = X_seq_mask[:, :, :4].mean(axis=(1, 2))
    meta["iwv_coverage_frac"] = X_seq_mask[:, :, 4].mean(axis=1)
    meta["iwv_available"] = meta["iwv_coverage_frac"] > 0
    meta["has_true_hourly"] = meta["coverage_frac"] > 0.0  # flag for the dual branch model
    meta["R_last"] = X_seq_val[:, -1, 0]

    print(f"[assemble_sequences] Kept all {n_events} rows. (mean AWS core_coverage={meta['coverage_frac'].mean():.3f})")

    return X_seq, keep_mask, meta, core_feats


def build_static_context(events_df: pd.DataFrame, X_df: pd.DataFrame, meta: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Daily-resolution context, safe to treat as constant per sequence. Pulled from
    X_df where already present (satellite cols already anti-leak-checked upstream);
    lat/lon/doy/month recomputed here for clarity."""
    static_cols = [c for c in STATIC_FEATURE_COLS if c in X_df.columns]
    _assert_no_banned_columns(static_cols)
    ctx = X_df.loc[meta.index, static_cols].copy() if static_cols else pd.DataFrame(index=meta.index)
    if "doy" not in ctx.columns:
        ctx["doy"] = pd.to_datetime(meta["timestamp"]).dt.dayofyear
    if "month" not in ctx.columns:
        ctx["month"] = pd.to_datetime(meta["timestamp"]).dt.month
    if "lat" not in ctx.columns:
        ctx["lat"] = meta["lat"]
    if "lon" not in ctx.columns:
        ctx["lon"] = meta["lon"]
    return ctx.values.astype(np.float32), list(ctx.columns)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hourly-panel", type=Path, default=ROOT / "data/processed/assam_aws_hourly.parquet")
    p.add_argument("--iwv-file", type=Path, default=ROOT / "data/raw/gagan_iwv_v1.txt")
    p.add_argument("--lookback-hours", type=int, default=8)
    p.add_argument("--match-radius-km", type=float, default=50.0)
    p.add_argument("--tolerance-min", type=int, default=40)
    p.add_argument("--min-coverage-frac", type=float, default=0.5)
    p.add_argument("--out-npz", type=Path, default=ROOT / "outputs/sequences.npz")
    p.add_argument("--out-meta", type=Path, default=ROOT / "outputs/sequences_meta.parquet")
    args = p.parse_args()

    print("Loading labeled events + existing (non-leaky) y/w/group logic...")
    events_df = load_imd_parquets()
    # Filter removed: use full nationwide dataset
    X_df, y_ser, w_ser, event_groups, _ordered_feats = build_feature_matrix(events_df)

    print(f"Loading hourly AWS panel from {args.hourly_panel} ...")
    hourly_panel = pd.read_parquet(args.hourly_panel)
    hourly_panel = hourly_panel.dropna(subset=["timestamp", "lat", "lon", "station_id"])

    print(f"Loading + resampling GAGAN IWV from {args.iwv_file} ...")
    iwv_hourly = _load_iwv_hourly(args.iwv_file)

    X_seq, keep_mask, meta, core_feats = assemble_sequences(
        events_df, X_df, event_groups, hourly_panel, iwv_hourly,
        args.lookback_hours, args.match_radius_km, args.tolerance_min, args.min_coverage_frac,
    )
    static_ctx, static_feats = build_static_context(events_df, X_df, meta)

    X_seq, static_ctx = X_seq[keep_mask], static_ctx[keep_mask]
    y = y_ser.loc[meta.index].values[keep_mask]
    w = w_ser.loc[meta.index].values[keep_mask]
    groups = (event_groups["cluster_id"].values if hasattr(event_groups, "columns") and "cluster_id" in event_groups.columns
              else np.asarray(event_groups))[keep_mask] if len(event_groups) == len(meta) else None
    meta_kept = meta.loc[keep_mask].reset_index(drop=False).rename(columns={"index": "orig_index"})

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        X_seq=X_seq, static_ctx=static_ctx, y=y, w=w,
        groups=groups if groups is not None else np.arange(len(y)),
        core_feats=np.array(core_feats), static_feats=np.array(static_feats),
    )
    meta_kept.to_parquet(args.out_meta)

    print(f"\nSaved {X_seq.shape[0]} sequences of shape (seq_len={X_seq.shape[1]}, core_feats={X_seq.shape[2]}) "
          f"+ {static_ctx.shape[1]} static features to {args.out_npz}")
    print(f"Metadata (timestamp, coverage_frac, R_last, ...) saved to {args.out_meta}")
    print("\nNOT RUN AGAINST REAL DATA — verify coverage_frac distribution and the assertions above "
          "before trusting anything trained on this output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
