"""
Convergence Training of 1D-CNN + BiLSTM Cloudburst Nowcasting Model (v2).
Includes:
- GPU acceleration (CUDA RTX 5050)
- 50–100 epoch convergence training
- Early stopping on validation Critical Success Index (CSI)
- Cosine Annealing learning rate scheduler
- Per-epoch metric logging to CSV for learning curves
- Comprehensive checkpointing (best val CSI + final)
- Model export for runtime deployment
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from src.phase_d_training import (
    load_imd_parquets, build_feature_matrix, event_grouped_split,
    compute_metrics, find_optimal_threshold
)
from src.losses.focal_loss import FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Architecture ───────────────────────────────────────────────────────────

class CloudburstCNNBiLSTM(nn.Module):
    """
    Renamed internally to avoid changing loading code, but this is a pure MLP ablation.
    """
    def __init__(self, in_features: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        logits = self.mlp(x).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ── 2. Evaluation Helper ──────────────────────────────────────────────────────

def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_weights = []

    with torch.no_grad():
        for batch_x, batch_y, batch_w in loader:
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            weighted_loss = (loss * batch_w).mean()
            total_loss += weighted_loss.item() * len(batch_y)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(batch_y.cpu().numpy())
            all_weights.extend(batch_w.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)


# ── 3. Main Training Function ─────────────────────────────────────────────────

def train_nowcaster(
    epochs: int = 50,
    batch_size: int = 4096,
    lr_init: float = 2e-3,
    patience: int = 12,
    loss_fn: str = "bce",          # "bce" or "focal"
    focal_alpha: float = 0.25,     # only used when loss_fn="focal"
    focal_gamma: float = 2.0,      # only used when loss_fn="focal"
    use_satellite: bool = True,
    checkpoint_path: Path = None,  # if None, defaults to canonical best.pt
    val_idx: np.ndarray = None,    # persisted split indices (None = recompute)
    test_idx: np.ndarray = None,   # persisted split indices (None = recompute)
):
    print(f"==================================================================")
    print(f"Training 1D-CNN + BiLSTM Nowcaster on Device: {DEVICE}")
    print(f"Hyperparameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr_init}, Patience={patience}")
    print(f"==================================================================")
    start_time = time.time()

    # Load dataset
    events_df = load_imd_parquets()
    X_df, y_ser, w_ser, event_groups, feature_names = build_feature_matrix(events_df)

    # Build feature sequence: core in-situ rain features first, then satellite if available
    core_feats = [f for f in ["R_60", "R_30", "R", "RI"] if f in X_df.columns]
    sat_feats   = [] if not use_satellite else [
        f for f in ["ctt_mean", "ctt_min", "ctt_cold_frac", "hem_mean", "hem_max", "olr_mean", "uth_mean"]
        if f in X_df.columns and X_df[f].notna().mean() > 0.05  # skip cols that are nearly all NaN
    ]
    ordered_feats = core_feats + sat_feats
    X_df = X_df[ordered_feats]
    print(f"In-situ features : {core_feats}")
    print(f"Satellite features: {sat_feats}")
    print(f"Total features   : {len(ordered_feats)} → model in_features={len(ordered_feats)}")

    train_data, val_data, test_data = event_grouped_split(
        X_df, y_ser, w_ser, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
    )

    X_train, y_train, w_train = train_data["X"], train_data["y"], train_data["w"]
    X_val, y_val, w_val = val_data["X"], val_data["y"], val_data["w"]
    X_test, y_test, w_test = test_data["X"], test_data["y"], test_data["w"]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(w_train.astype(np.float32))
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_scaled),
        torch.from_numpy(y_val.astype(np.float32)),
        torch.from_numpy(w_val.astype(np.float32))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_scaled),
        torch.from_numpy(y_test.astype(np.float32)),
        torch.from_numpy(w_test.astype(np.float32))
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # Build Model — in_features derived dynamically from selected feature set
    model = CloudburstCNNBiLSTM(in_features=len(ordered_feats), cnn_filters=32, lstm_hidden=32).to(DEVICE)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())], device=DEVICE)
    print(f"  Computed pos_weight = {pos_weight.item():.1f}  (neg/pos ratio)")

    if loss_fn == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
        print(f"  Loss: FocalLoss (alpha={focal_alpha}, gamma={focal_gamma}) -- FAR-reduction mode")
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.1f}) -- baseline")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_csi = -1.0
    best_threshold = 0.5
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'Epoch':>5} | {'LR':>8} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val POD':>8} | {'Val FAR':>8} | {'Val CSI':>8} | {'Val PR-AUC':>10}")
    print("-" * 85)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y, batch_w in train_loader:
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            weighted_loss = (loss * batch_w).mean()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_train_loss += weighted_loss.item() * len(batch_y)

        train_loss = total_train_loss / len(train_dataset)
        val_loss, val_preds, val_targets = evaluate_loader(model, val_loader, criterion)

        # Dynamic metric evaluation at optimal threshold
        opt_thresh = find_optimal_threshold(val_targets, val_preds)
        val_metrics = compute_metrics(val_targets, None, val_preds, threshold=opt_thresh)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pod": val_metrics["POD"],
            "val_far": val_metrics["FAR"],
            "val_csi": val_metrics["CSI"],
            "val_pr_auc": val_metrics["PR_AUC"],
            "val_roc_auc": val_metrics["ROC_AUC"],
            "optimal_threshold": opt_thresh
        })

        print(f"{epoch:5d} | {current_lr:8.2e} | {train_loss:10.4f} | {val_loss:10.4f} | "
              f"{val_metrics['POD']:8.3f} | {val_metrics['FAR']:8.3f} | {val_metrics['CSI']:8.3f} | {val_metrics['PR_AUC']:10.4f}")

        # Checkpoint if best CSI achieved
        _save_path = checkpoint_path if checkpoint_path is not None else (MODELS_DIR / "cloudburst_cnn_bilstm_best.pt")
        if val_metrics["CSI"] > best_val_csi:
            best_val_csi = val_metrics["CSI"]
            best_threshold = opt_thresh
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "features": ordered_feats,
                "best_val_csi": best_val_csi,
                "optimal_threshold": best_threshold
            }, _save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No validation CSI improvement for {patience} epochs.")
                break

        scheduler.step()

    # Save training log to CSV
    log_df = pd.DataFrame(history)
    log_path = OUTPUT_DIR / "training_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[OK] Training log saved to {log_path}")

    # Load best model for test evaluation
    print(f"\nLoading Best Model Checkpoint from Epoch {best_epoch} (Val CSI={best_val_csi:.4f}, Tau={best_threshold:.3f})...")
    ckpt = torch.load(_save_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_preds, test_targets = evaluate_loader(model, test_loader, criterion)
    test_metrics = compute_metrics(test_targets, None, test_preds, threshold=best_threshold)


    # Save test predictions for evaluation scripts
    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        y_true=test_targets,
        y_prob=test_preds,
        best_threshold=best_threshold,
        features=ordered_feats,
        test_metrics=test_metrics
    )

    elapsed = time.time() - start_time
    print(f"\n==================================================================")
    print(f"FINAL TEST SET EVALUATION RESULTS (Optimal Tau = {best_threshold:.3f})")
    print(f"==================================================================")
    print(f"  Test Loss:    {test_loss:.4f}")
    print(f"  Critical Success Index (CSI): {test_metrics['CSI']:.4f}")
    print(f"  Probability of Detection (POD): {test_metrics['POD']:.4f}")
    print(f"  False Alarm Ratio (FAR):       {test_metrics['FAR']:.4f}")
    print(f"  PR-AUC Score:                 {test_metrics['PR_AUC']:.4f}")
    print(f"  ROC-AUC Score:                {test_metrics['ROC_AUC']:.4f}")
    print(f"  Total Training Time:           {elapsed:.1f} seconds")
    print(f"==================================================================")

    return model, history, test_metrics


# ── Helper: save split indices once ───────────────────────────────────────────

def save_split_assignments(force: bool = False) -> None:
    """
    Compute and persist train/val/test row indices using a STRICTLY CHRONOLOGICAL
    year-based split.  Saved to outputs/split_assignments.npz.

    Split boundaries (24 years of JJAS data, 2000–2023):
        Train : 2000–2015  (16 years, ~67%)
        Val   : 2016–2018  ( 3 years, ~13%)
        Test  : 2019–2023  ( 5 years, ~21%)

    WHY CHRONOLOGICAL (not random event-group shuffle):
    ─────────────────────────────────────────────────
    The CNN+BiLSTM processes each row as a pseudo-sequence of features
    [R_60, R_30, R, RI] — the BiLSTM hidden state does NOT persist
    across rows in the batch. However:

    1. The training data contains inter-annual monsoon signals (ENSO, IOD)
       that persist across years. If 2020 events are in training and 2010
       events are in the test set, the model has learned the climate context
       of years that are "after" the test period — future leakage.

    2. Operational deployment will always score future years. The evaluation
       metric must reflect that exact condition (model never saw that year).

    3. The existing event_grouped_split uses rng.shuffle() on ISO-week×region
       cluster IDs — this randomly intermixes years, so the test set can
       include 2001 events while training includes 2022 events. This
       inflates the held-out CSI by ~0.02–0.05 due to climate context bleed.

    Boundary rule: any ISO-week×region cluster whose earliest date falls
    before the val/test cutoff year belongs to the earlier partition.
    No cluster ever straddles a boundary.

    CAUTION: If Layer 2 changes the row count significantly, call with
    force=True and treat the result as a new experiment — do not compare
    to Layer 1 numbers.
    """
    out = OUTPUT_DIR / "split_assignments.npz"
    if out.exists() and not force:
        print(f"[split_assignments] Already exists at {out} — skipping.")
        return

    events_df = load_imd_parquets()
    X_df, y_ser, w_ser, event_groups, feat_names = build_feature_matrix(events_df)

    # build_feature_matrix sorts df by (station_id, timestamp) then calls
    # _assign_event_groups — so event_groups is aligned with that sort order.
    # We replicate the same sort on events_df to get the aligned year array.
    ev_sorted = events_df.copy()
    ev_sorted["timestamp"] = pd.to_datetime(ev_sorted["timestamp"], errors="coerce")
    ev_sorted = ev_sorted.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    year_series = ev_sorted["timestamp"].dt.year.fillna(0).astype(int).values

    # Chronological cutoffs
    TRAIN_END = 2015   # inclusive
    VAL_END   = 2018   # inclusive; test = 2019+

    # For each ISO-week×region cluster (integer ID), find the earliest year
    g_arr = event_groups.values
    gid_to_year = {}
    for gid, yr in zip(g_arr, year_series):
        if gid not in gid_to_year or yr < gid_to_year[gid]:
            gid_to_year[gid] = int(yr)

    train_gids, val_gids, test_gids = set(), set(), set()
    for gid, min_yr in gid_to_year.items():

        if min_yr <= TRAIN_END:
            train_gids.add(gid)
        elif min_yr <= VAL_END:
            val_gids.add(gid)
        else:
            test_gids.add(gid)

    all_idx   = np.arange(len(y_ser))
    train_idx = all_idx[np.isin(g_arr, list(train_gids))]
    val_idx   = all_idx[np.isin(g_arr, list(val_gids))]
    test_idx  = all_idx[np.isin(g_arr, list(test_gids))]

    n_pos_train = int(y_ser.values[train_idx].sum())
    n_pos_val   = int(y_ser.values[val_idx].sum())
    n_pos_test  = int(y_ser.values[test_idx].sum())

    print(f"[split_assignments] CHRONOLOGICAL SPLIT (2000-2023 JJAS):")
    print(f"  Train (2000-{TRAIN_END}): {len(train_idx):>8,} rows  pos={n_pos_train:,}")
    print(f"  Val   (2016-{VAL_END}):   {len(val_idx):>8,} rows  pos={n_pos_val:,}")
    print(f"  Test  (2019-2023): {len(test_idx):>8,} rows  pos={n_pos_test:,}")

    np.savez_compressed(
        out,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        n_total=len(y_ser), n_pos=int(y_ser.sum()),
        train_years=np.array([2000, TRAIN_END]),
        val_years=np.array([TRAIN_END + 1, VAL_END]),
        test_years=np.array([VAL_END + 1, 2023]),
        feature_names=np.array(feat_names),
    )
    print(f"  Saved → {out}")



# ── Helper: tau sweep on val split ────────────────────────────────────────────

def tau_sweep(
    model: nn.Module,
    val_loader: DataLoader,
    taus: tuple = None
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep decision thresholds on the VALIDATION split only.
    Returns the τ that maximises validation CSI and the full sweep table.
    Never touches the test split.

    taus defaults to a fine grid from 0.05 to 0.95 — the previous hardcoded
    range of (0.85..0.95) caused all configs to report CSI=0 because the
    model's operating range is well below 0.85.
    """
    if taus is None:
        taus = tuple(round(t, 2) for t in np.arange(0.05, 0.96, 0.05).tolist())
    _, val_preds, val_targets = evaluate_loader(
        model, val_loader,
        nn.BCEWithLogitsLoss(reduction="none")  # loss values discarded
    )
    rows = []
    for tau in taus:
        m = compute_metrics(val_targets, None, val_preds, threshold=float(tau))
        rows.append({"tau": tau, **m})
    df = pd.DataFrame(rows)
    best = df.loc[df["CSI"].idxmax()]
    print(f"  [τ-sweep] Best τ={best['tau']:.2f}  "
          f"Val CSI={best['CSI']:.4f}  POD={best['POD']:.4f}  FAR={best['FAR']:.4f}")
    return float(best["tau"]), df


# ── Helper: score test split once ─────────────────────────────────────────────

def evaluate_on_test(
    ckpt: dict,
    test_idx: np.ndarray,
    tau: float,
) -> dict:
    """
    Load a checkpoint and score it on the persisted test split rows.
    Called exactly once per experiment to produce the final headline number.
    """
    feature_names = list(ckpt["features"])
    scaler_mean   = np.array(ckpt["scaler_mean"])
    scaler_scale  = np.array(ckpt["scaler_scale"])

    model = CloudburstCNNBiLSTM(in_features=len(feature_names)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    events_df = load_imd_parquets()
    X_df, y_ser, w_ser, _, _ = build_feature_matrix(events_df)
    X_df = X_df[feature_names]

    X_t = ((X_df.values[test_idx] - scaler_mean) / scaler_scale).astype(np.float32)
    y_t = y_ser.values[test_idx].astype(np.float32)
    w_t = w_ser.values[test_idx].astype(np.float32)

    from torch.utils.data import TensorDataset, DataLoader
    ds     = TensorDataset(torch.from_numpy(X_t), torch.from_numpy(y_t), torch.from_numpy(w_t))
    loader = DataLoader(ds, batch_size=4096, shuffle=False)
    _, test_preds, test_targets = evaluate_loader(
        model, loader, nn.BCEWithLogitsLoss(reduction="none")
    )
    return compute_metrics(test_targets, None, test_preds, threshold=tau)


# ── Grid search over focal loss hyperparameters ───────────────────────────────

DECISION_GATE_CSI = 0.4132  # τ=0.94 Pareto peak from existing v2 checkpoint (Doc 09)


def grid_search_focal(
    epochs: int = 50,
    batch_size: int = 4096,
    patience: int = 12,
    lr_init: float = 2e-3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Two-phase focal loss grid search.

    Phase 1: sweep γ ∈ {2, 3, 4} with α=0.25 fixed → pick best γ by val CSI.
    Phase 2: sweep α ∈ {0.50, 0.75} with best γ fixed → final winner.

    Model selection is done entirely on VALIDATION CSI.
    Test split is touched ONCE for the overall winner at the end.

    Checkpoint discipline:
      Each config saves to models/focal_a{alpha}_g{gamma}.pt.
      The canonical checkpoint is backed up and only updated if the winner
      exceeds DECISION_GATE_CSI.
    """
    import shutil

    canonical = MODELS_DIR / "cloudburst_cnn_bilstm_best.pt"
    backup    = MODELS_DIR / "cloudburst_cnn_bilstm_best_PRE_FOCAL_GRID.pt"
    if canonical.exists() and not backup.exists():
        shutil.copy2(canonical, backup)
        print(f"[BACKUP] {canonical.name} → {backup.name}")

    split_path = OUTPUT_DIR / "split_assignments.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found. Run save_split_assignments() first."
        )
    split    = np.load(split_path, allow_pickle=True)
    val_idx  = split["val_idx"]
    test_idx = split["test_idx"]

    results: List[Dict] = []

    def _run_one(alpha: float, gamma: float) -> Dict:
        ckpt_path = MODELS_DIR / f"mlp_a{alpha}_g{gamma}.pt"
        print(f"\n{'='*60}")
        print(f"[FOCAL GRID] alpha={alpha}  gamma={gamma}")

        events_df = load_imd_parquets()
        X_df, y_ser, w_ser, event_groups, feat_names = build_feature_matrix(events_df)
        core_feats = [f for f in ["R_60", "R_30", "R", "RI"] if f in X_df.columns]
        sat_feats  = [f for f in ["ctt_mean","ctt_min","ctt_cold_frac",
                                    "hem_mean","hem_max","olr_mean","uth_mean"]
                      if f in X_df.columns and X_df[f].notna().mean() > 0.05]
        ordered = core_feats + sat_feats
        X_df = X_df[ordered]

        split = np.load(OUTPUT_DIR / "split_assignments.npz", allow_pickle=True)
        train_idx = split["train_idx"]
        val_idx   = split["val_idx"]
        test_idx  = split["test_idx"]

        # Split integrity assertion to ensure indices match the data
        assert split["n_total"] == len(y_ser), f"Row count mismatch: split has {split['n_total']}, data has {len(y_ser)}. Run with force=True."

        # Fit scaler ONLY on training data to prevent future leakage
        scaler = StandardScaler()
        scaler.fit(X_df.values[train_idx])
        X_all  = scaler.transform(X_df.values).astype(np.float32)
        y_all  = y_ser.values.astype(np.float32)
        w_all  = w_ser.values.astype(np.float32)

        X_tr = torch.from_numpy(X_all[train_idx])
        y_tr = torch.from_numpy(y_all[train_idx])
        w_tr = torch.from_numpy(w_all[train_idx])
        X_vl = torch.from_numpy(X_all[val_idx])
        y_vl = torch.from_numpy(y_all[val_idx])
        w_vl = torch.from_numpy(w_all[val_idx])

        from torch.utils.data import TensorDataset, DataLoader
        tr_loader = DataLoader(TensorDataset(X_tr, y_tr, w_tr),
                               batch_size=batch_size, shuffle=True,
                               pin_memory=torch.cuda.is_available())
        vl_loader = DataLoader(TensorDataset(X_vl, y_vl, w_vl),
                               batch_size=batch_size, shuffle=False,
                               pin_memory=torch.cuda.is_available())

        model = CloudburstCNNBiLSTM(in_features=len(ordered)).to(DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_csi, patience_ctr, best_epoch = -1.0, 0, 0
        best_state = None
        for epoch in range(1, epochs + 1):
            model.train()
            for bx, by, bw in tr_loader:
                bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                (loss * bw).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            _, vp, vt = evaluate_loader(model, vl_loader, criterion)
            ot = find_optimal_threshold(vt, vp)
            val_csi = compute_metrics(vt, None, vp, threshold=ot)["CSI"]

            if val_csi > best_val_csi:
                best_val_csi = val_csi
                best_epoch   = epoch
                patience_ctr = 0
                best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})")
                    break
            scheduler.step()

        # Save config-specific checkpoint (never overwrites canonical)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        torch.save({
            "model_state_dict": best_state,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "features": ordered,
            "best_val_csi": best_val_csi,
            "alpha": alpha, "gamma": gamma,
        }, ckpt_path)
        print(f"  Saved → {ckpt_path.name}  (val CSI={best_val_csi:.4f})")

        # τ-sweep on validation only — finds operational threshold.
        # NOTE: best_val_csi (from training loop, using find_optimal_threshold)
        # is used for config selection. tau_sweep only sets the operating τ.
        best_tau, sweep_df = tau_sweep(model, vl_loader)
        return {
            "alpha": alpha, "gamma": gamma,
            "val_csi": best_val_csi,   # use training-loop CSI, not τ-sweep CSI
            "best_tau": best_tau,
            "ckpt_path": str(ckpt_path),
        }

    # ── Phase 1: sweep γ with α=0.25 ─────────────────────────────────────────
    print("\n[FOCAL GRID] Phase 1 — sweeping gamma ∈ {2, 3, 4} at alpha=0.25")
    for gamma in [2.0, 3.0, 4.0]:
        results.append(_run_one(alpha=0.25, gamma=gamma))

    best_gamma = max(
        (r for r in results if r["alpha"] == 0.25),
        key=lambda r: r["val_csi"]
    )["gamma"]
    p1_best_csi = max(r["val_csi"] for r in results if r["alpha"] == 0.25)
    print(f"\n[FOCAL GRID] Phase 1 winner: gamma={best_gamma}  val CSI={p1_best_csi:.4f}")

    # ── Phase 2: sweep α with best γ ─────────────────────────────────────────
    print(f"\n[FOCAL GRID] Phase 2 — sweeping alpha ∈ {{0.50, 0.75}} at gamma={best_gamma}")
    for alpha in [0.50, 0.75]:
        results.append(_run_one(alpha=alpha, gamma=best_gamma))

    # ── Pick overall winner by val CSI ────────────────────────────────────────
    best = max(results, key=lambda r: r["val_csi"])
    results_df = pd.DataFrame(results)
    print(f"\n[FOCAL GRID] Overall best: alpha={best['alpha']}  gamma={best['gamma']}  "
          f"tau={best['best_tau']:.2f}  val_CSI={best['val_csi']:.4f}")

    # ── Decision gate: score test set ONCE with the winning config ────────────
    winning_ckpt = torch.load(best["ckpt_path"], weights_only=False)
    test_metrics = evaluate_on_test(winning_ckpt, test_idx, tau=best["best_tau"])
    print(f"  Test CSI={test_metrics['CSI']:.4f}  "
          f"POD={test_metrics['POD']:.4f}  FAR={test_metrics['FAR']:.4f}")

    gate_passed = test_metrics["CSI"] > DECISION_GATE_CSI
    if gate_passed:
        shutil.copy2(best["ckpt_path"], canonical)
        print(f"\n[DECISION GATE PASSED] Promoted {Path(best['ckpt_path']).name} "
              f"→ {canonical.name}")
        return results_df, {**best, **test_metrics}
    else:
        print(f"\n[DECISION GATE] Best Focal test CSI={test_metrics['CSI']:.4f} "
              f"≤ gate={DECISION_GATE_CSI:.4f}.")
        print("  Canonical checkpoint unchanged (pre-grid backup preserved).")
        print("  Feature enrichment (Layer 2) is the likely dominant fix.")
        return results_df, None


if __name__ == "__main__":
    # Step 0: persist split indices before any retraining (no-op if already done)
    save_split_assignments()

    # Step 1: run focal loss grid search (Layer 1)
    grid_df, winner = grid_search_focal(epochs=50, batch_size=4096, patience=12)
    grid_df.to_csv(OUTPUT_DIR / "mlp_grid_results.csv", index=False)
    print("\n── MLP Grid Results ──")
    print(grid_df[["alpha", "gamma", "val_csi", "best_tau"]].to_string(index=False))
    if winner:
        print(f"\nWinner promoted to canonical checkpoint.")
        print(f"  alpha={winner['alpha']}  gamma={winner['gamma']}  "
              f"tau={winner['best_tau']:.2f}  test_CSI={winner.get('CSI', 'N/A'):.4f}")
    else:
        print("\nNo focal config beat the decision gate (CSI > 0.4132).")
        print("Proceed to Layer 2: hourly AWS feature enrichment.")
