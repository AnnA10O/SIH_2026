"""
Two-Stage Precision Cascade — Layer 3 of the FAR Mitigation Plan (Document 11).

Stage 1: CNN+BiLSTM at tau=0.90 (unchanged, high-recall filter)
Stage 2: LightGBM precision refiner trained on Stage 1's validation-set FPs vs TPs

Data discipline (strictly enforced):
  Stage 2 trains on:      outputs/val_predictions.npz     (validation split rows)
  Final evaluation on:    the 230,604-row test split       (ONE pass only)
  Split indices from:     outputs/split_assignments.npz   (frozen before Layer 1)

The compute_metrics() signature is:
    compute_metrics(y_true, y_pred_ignored, y_proba, threshold) -> dict
  y_pred is accepted but always ignored; y_proba is always used.
  All calls below pass probabilities, never hard predictions.

CAUTION: Do NOT re-run this script after seeing the test-split CSI and then
adjust Stage 2. That converts the test CSI into an optimistic estimate.
If further tuning is needed, go back to the internal 80/20 pool split in
train_stage2() — the test split is a one-shot.
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from src.train_neural_nowcaster_v2 import CloudburstCNNBiLSTM, evaluate_loader
from src.phase_d_training import (
    load_imd_parquets, build_feature_matrix, compute_metrics
)

MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LightGBM with sklearn fallback ────────────────────────────────────────────

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("[WARNING] LightGBM not installed. Falling back to LogisticRegression.")
    print("  Install with: pip install lightgbm")
    LGB_AVAILABLE = False
    from sklearn.linear_model import LogisticRegression

# Stage 2 feature set — confirmed against station_feature_engine.py exports:
#   StationFeatureBuffer only exports lat/lon (no per-station slope or terrain data).
#   slope is excluded until a DEM station-join is built.
#   region_id (basin name) is present in the training DataFrame as a string column.
STAGE2_FEATURES = ["stage1_prob", "R", "R_30", "R_60", "RI", "region_id"]


# ── Internal utility: score a model on a specific set of rows ─────────────────

def _score_rows(
    idx: np.ndarray,
    feature_names: list,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
    w_all: np.ndarray,
    model: nn.Module,
    batch_size: int = 4096,
) -> tuple:
    """
    Score model on the rows selected by integer index array.
    Returns (probabilities, targets) both as np.ndarray.
    """
    X_s = ((X_all[idx] - scaler_mean) / scaler_scale).astype(np.float32)
    y_s = y_all[idx].astype(np.float32)
    w_s = w_all[idx].astype(np.float32)

    ds     = TensorDataset(torch.from_numpy(X_s), torch.from_numpy(y_s), torch.from_numpy(w_s))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        pin_memory=torch.cuda.is_available())
    _, probs, targets = evaluate_loader(
        model, loader, nn.BCEWithLogitsLoss(reduction="none")
    )
    return probs, targets


def _load_stage1() -> tuple:
    """
    Load the canonical Stage 1 checkpoint.
    Returns (model, feature_names, scaler_mean, scaler_scale).
    """
    ckpt = torch.load(
        MODELS_DIR / "cloudburst_cnn_bilstm_best.pt", weights_only=False
    )
    feature_names = list(ckpt["features"])
    scaler_mean   = np.array(ckpt["scaler_mean"])
    scaler_scale  = np.array(ckpt["scaler_scale"])

    model = CloudburstCNNBiLSTM(in_features=len(feature_names)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, feature_names, scaler_mean, scaler_scale


# ── Step 1: Score Stage 1 on validation split → val_predictions.npz ──────────

def generate_val_predictions(stage1_tau: float = 0.90) -> tuple:
    """
    Score the canonical Stage 1 checkpoint on the persisted validation split.
    Saves outputs/val_predictions.npz.

    Returns:
        val_probs    : np.ndarray of Stage 1 output probabilities (val rows)
        val_targets  : np.ndarray of binary labels (val rows)
        X_val_raw    : np.ndarray of un-scaled features (val rows) for Stage 2
        feature_names: list of feature column names
    """
    split   = np.load(OUTPUTS_DIR / "split_assignments.npz", allow_pickle=True)
    val_idx = split["val_idx"]

    model, feature_names, scaler_mean, scaler_scale = _load_stage1()

    events_df = load_imd_parquets()
    X_df, y_ser, w_ser, _, _ = build_feature_matrix(events_df)
    X_df = X_df[feature_names]

    val_probs, val_targets = _score_rows(
        val_idx, feature_names, scaler_mean, scaler_scale,
        X_df.values, y_ser.values, w_ser.values, model
    )
    X_val_raw = X_df.values[val_idx]   # unscaled — Stage 2 re-scales internally

    np.savez_compressed(
        OUTPUTS_DIR / "val_predictions.npz",
        y_true=val_targets, y_prob=val_probs,
        tau=stage1_tau, features=np.array(feature_names)
    )
    m = compute_metrics(val_targets, None, val_probs, threshold=stage1_tau)
    print(f"[val_predictions] Saved ({len(val_targets):,} val rows)")
    print(f"  Stage 1 @ tau={stage1_tau}: "
          f"CSI={m['CSI']:.4f}  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")
    return val_probs, val_targets, X_val_raw, feature_names


# ── Step 2: Build Stage 2 training pool from val FPs vs TPs ──────────────────

def build_stage2_pool(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    X_val_raw: np.ndarray,
    feature_names: list,
    stage1_tau: float = 0.90,
) -> tuple:
    """
    Restrict Stage 2's training data to the rows Stage 1 flagged as positive.
    Stage 2 labels: True Positives → 1, False Positives → 0.

    This is the correct pool: Stage 2 only sees cases that passed Stage 1's filter
    and learns to distinguish real events from spurious detections within that group.

    Returns:
        X_pool : pd.DataFrame with feature columns + 'stage1_prob'
        y_pool : np.ndarray of binary Stage 2 labels
    """
    flagged  = val_probs >= stage1_tau
    X_pool   = pd.DataFrame(X_val_raw[flagged], columns=feature_names)
    X_pool["stage1_prob"] = val_probs[flagged]
    y_pool   = val_targets[flagged]   # 1 = TP (real event), 0 = FP (false alarm)

    n_pos = int(y_pool.sum())
    n_neg = int((1 - y_pool).sum())
    print(f"\n[Stage 2 Pool] {len(y_pool):,} rows flagged by Stage 1 at tau={stage1_tau}")
    print(f"  True Positives (keep): {n_pos}  "
          f"False Positives (hard negatives): {n_neg}  "
          f"Ratio: {n_neg/max(1, n_pos):.1f}:1")
    print(f"  (Original training imbalance: ~104:1 — Stage 2 pool is far milder)")
    return X_pool, y_pool


# ── Step 3: Train Stage 2 with internal tau sweep ────────────────────────────

def train_stage2(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
) -> tuple:
    """
    Train a LightGBM Stage 2 refiner.

    Uses an 80/20 internal split of the pool to find the optimal Stage 2 tau.
    The tau is swept on this internal val set only — the main test split is
    not involved here.

    Returns:
        s2_model       : fitted classifier
        best_s2_tau    : float, optimal Stage 2 threshold
        available_feats: list of feature names actually used (warns on missing)
    """
    # Check each expected feature and warn explicitly on missing ones
    available = []
    for f in STAGE2_FEATURES:
        if f in X_pool.columns:
            available.append(f)
        else:
            print(f"[Stage 2 WARNING] Feature '{f}' not in pool — excluded. "
                  f"Check whether hourly_aws_loader and build_feature_matrix export it.")
    if len(available) < 2:
        raise ValueError(
            f"Stage 2 has only {len(available)} feature(s) available ({available}). "
            "Too few to train a meaningful classifier. "
            "Ensure at least stage1_prob and one meteorological feature are present."
        )
    print(f"[Stage 2] Using {len(available)} features: {available}")

    X = X_pool[available].fillna(0.0).values
    y = y_pool

    # 80/20 internal split for tau sweep — stratified to preserve class balance
    X_tr, X_sv, y_tr, y_sv = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    if LGB_AVAILABLE:
        s2_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            class_weight="balanced",  # handles residual imbalance in pool
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        s2_model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )

    s2_model.fit(X_tr, y_tr)

    # Sweep Stage 2 tau on internal val set
    s2_probs_sv  = s2_model.predict_proba(X_sv)[:, 1]
    best_s2_tau  = 0.50
    best_s2_csi  = -1.0
    for tau in np.arange(0.30, 0.80, 0.05):
        m = compute_metrics(y_sv, None, s2_probs_sv, threshold=float(tau))
        if m["CSI"] > best_s2_csi:
            best_s2_csi = m["CSI"]
            best_s2_tau = float(tau)

    print(f"[Stage 2] Trained {type(s2_model).__name__} on {len(X_tr):,} samples")
    print(f"  Internal val sweep: best tau={best_s2_tau:.2f}  "
          f"Stage-2-only CSI={best_s2_csi:.4f}")
    return s2_model, best_s2_tau, available


# ── Step 4: Evaluate full cascade on test split (ONCE) ───────────────────────

def evaluate_cascade(
    s2_model,
    s2_features: list,
    stage1_tau: float = 0.90,
    stage2_tau: float = 0.50,
) -> tuple:
    """
    Evaluate the two-stage cascade on the test split.

    Called EXACTLY ONCE. Any further tuning based on the printed CSI number
    converts it into an optimistic estimate — go back to train_stage2()'s
    internal split instead.

    Cascade probability:
        cascade_prob[i] = stage1_prob[i] * stage2_prob[i]  if Stage 1 flags row i
        cascade_prob[i] = 0.0                               otherwise
    Threshold applied: stage1_tau * stage2_tau (so both stages must be confident).

    Returns:
        cascade_metrics : dict from compute_metrics() for the full cascade
        s1_metrics      : dict from compute_metrics() for Stage 1 alone (reference)
    """
    split    = np.load(OUTPUTS_DIR / "split_assignments.npz", allow_pickle=True)
    test_idx = split["test_idx"]

    model, feature_names, scaler_mean, scaler_scale = _load_stage1()

    events_df = load_imd_parquets()
    X_df, y_ser, w_ser, _, _ = build_feature_matrix(events_df)
    X_df = X_df[feature_names]

    # Score Stage 1 on the test split
    s1_probs, test_targets = _score_rows(
        test_idx, feature_names, scaler_mean, scaler_scale,
        X_df.values, y_ser.values, w_ser.values, model
    )
    X_test_raw = X_df.values[test_idx]

    # Stage 1 baseline metrics
    s1_metrics = compute_metrics(test_targets, None, s1_probs, threshold=stage1_tau)

    # Stage 2: apply only to Stage 1 positives
    flagged = s1_probs >= stage1_tau
    X_s2 = pd.DataFrame(X_test_raw[flagged], columns=feature_names)
    X_s2["stage1_prob"] = s1_probs[flagged]
    avail = [f for f in s2_features if f in X_s2.columns]
    s2_probs = s2_model.predict_proba(X_s2[avail].fillna(0.0).values)[:, 1]

    # Cascade probability array: Stage1_prob * Stage2_prob for flagged rows, 0 elsewhere
    cascade_probs = np.zeros(len(test_targets), dtype=np.float32)
    cascade_probs[flagged] = s1_probs[flagged] * s2_probs

    # Combined threshold: both stages must independently be confident
    combined_tau = stage1_tau * stage2_tau
    cascade_metrics = compute_metrics(
        test_targets, None, cascade_probs, threshold=combined_tau
    )

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CASCADE EVALUATION — {len(test_targets):,}-row Test Split (single pass)")
    print(f"Stage 1 tau={stage1_tau}  Stage 2 tau={stage2_tau:.2f}  "
          f"Combined threshold={combined_tau:.4f}")
    print(f"{'='*60}")
    print(f"{'Metric':<8} {'Stage 1 only':>15} {'S1 + S2 Cascade':>18}")
    print("-" * 45)
    for k in ["CSI", "POD", "FAR"]:
        s1_v  = s1_metrics[k]
        cas_v = cascade_metrics[k]
        delta = cas_v - s1_v
        sign  = "+" if delta >= 0 else ""
        print(f"{k:<8} {s1_v:>15.4f} {cas_v:>18.4f}  ({sign}{delta:.4f})")
    print("-" * 45)
    for k in ["TP", "FP", "FN"]:
        print(f"{k:<8} {s1_metrics[k]:>15} {cascade_metrics[k]:>18}")

    gate = 0.4132
    gate_passed = cascade_metrics["CSI"] > gate
    print(f"\n  Decision gate (CSI > {gate}): "
          f"{'PASSED ✓' if gate_passed else 'NOT PASSED'}")
    if not gate_passed:
        print("  → Review Stage 2 tau or feature set via train_stage2()'s internal split.")
        print("    Do NOT re-run evaluate_cascade() with adjusted parameters —")
        print("    that would convert the test CSI into an optimistic estimate.")

    return cascade_metrics, s1_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    STAGE1_TAU = 0.90

    # Step 1: Score Stage 1 on validation split → val_predictions.npz
    val_probs, val_targets, X_val_raw, feat_names = generate_val_predictions(
        stage1_tau=STAGE1_TAU
    )

    # Step 2: Build Stage 2 training pool (val-set FPs vs TPs)
    X_pool, y_pool = build_stage2_pool(
        val_probs, val_targets, X_val_raw, feat_names, stage1_tau=STAGE1_TAU
    )

    # Step 3: Train Stage 2, sweep Stage 2 tau on the internal pool split
    s2_model, s2_tau, s2_features = train_stage2(X_pool, y_pool)

    # Step 4: Evaluate the full cascade ONCE on the test split
    cascade_m, s1_m = evaluate_cascade(
        s2_model, s2_features,
        stage1_tau=STAGE1_TAU,
        stage2_tau=s2_tau,
    )
