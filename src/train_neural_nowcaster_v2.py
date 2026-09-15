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
    compute_metrics, find_optimal_threshold, simulate_sensor_outage_blocks
)
from src.config import FEATURES_SAT, FEATURES_AWS, FEATURES_STALENESS
from src.losses.focal_loss import FocalLoss
from sklearn.impute import SimpleImputer

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
    1D-CNN + BiLSTM In-Situ Convective Precursor Nowcaster.
    Temporal input sequence: [R_60, R_30, R, RI]
    """
    def __init__(self, in_features: int = 4, cnn_filters: int = 32, lstm_hidden: int = 32):
        super().__init__()
        self.in_features = in_features

        # 1D Convolution over sequential temporal features
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=2, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=2, padding=0),
            nn.BatchNorm1d(cnn_filters),
            nn.SiLU()
        )

        # Bidirectional LSTM to capture bidirectional temporal precursor dynamics
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15
        )

        # Non-linear classification head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 16),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        x_seq = x.unsqueeze(1)                        # (batch, 1, seq_len)
        conv = self.conv1d(x_seq)                     # (batch, filters, seq_out)
        lstm_in = conv.permute(0, 2, 1)               # (batch, seq_out, filters)
        lstm_out, _ = self.bilstm(lstm_in)            # (batch, seq_out, 2 * hidden)
        pooled, _ = torch.max(lstm_out, dim=1)        # (batch, 2 * hidden)
        logits = self.head(pooled).squeeze(-1)        # (batch,)
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

    # Load dataset and build BOTH clean and degraded feature matrices
    events_df_clean = load_imd_parquets()
    X_df_clean, y_ser, w_ser, event_groups, _ = build_feature_matrix(events_df_clean)
    
    events_df_deg = simulate_sensor_outage_blocks(events_df_clean, "final_label")
    X_df_deg, _, _, _, feature_names = build_feature_matrix(events_df_deg)

    # Build feature sequence
    core_feats = [f for f in FEATURES_AWS if f in X_df_deg.columns]
    sat_feats   = [] if not use_satellite else [
        f for f in FEATURES_SAT
        if f in X_df_deg.columns and X_df_deg[f].notna().mean() > 0.05
    ]
    # Include staleness features explicitly
    stale_feats = [f for f in FEATURES_STALENESS if f in X_df_deg.columns]
    
    ordered_feats = core_feats + sat_feats + stale_feats
    X_df_clean = X_df_clean[ordered_feats]
    X_df_deg = X_df_deg[ordered_feats]
    
    print(f"In-situ features : {core_feats}")
    print(f"Satellite features: {sat_feats}")
    print(f"Staleness features: {stale_feats}")
    print(f"Total features   : {len(ordered_feats)} → model in_features={len(ordered_feats)}")

    # We use the degraded matrix for training to force robustness
    train_data, val_data, test_data_deg = event_grouped_split(
        X_df_deg, y_ser, w_ser, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
    )
    
    # And we get a clean test split to maintain apples-to-apples CSI benchmarks
    _, _, test_data_clean = event_grouped_split(
        X_df_clean, y_ser, w_ser, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
    )

    X_train, y_train, w_train = train_data["X"], train_data["y"], train_data["w"]
    X_val, y_val, w_val = val_data["X"], val_data["y"], val_data["w"]
    X_test_deg, y_test_deg, w_test_deg = test_data_deg["X"], test_data_deg["y"], test_data_deg["w"]
    X_test_clean, y_test_clean, w_test_clean = test_data_clean["X"], test_data_clean["y"], test_data_clean["w"]

    # Impute missing features and standardize (fit on TRAIN only)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_deg_imputed = imputer.transform(X_test_deg)
    X_test_clean_imputed = imputer.transform(X_test_clean)
    
    X_train_scaled = scaler.fit_transform(X_train_imputed).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_imputed).astype(np.float32)
    X_test_deg_scaled = scaler.transform(X_test_deg_imputed).astype(np.float32)
    X_test_clean_scaled = scaler.transform(X_test_clean_imputed).astype(np.float32)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train.astype(np.float32)), torch.from_numpy(w_train.astype(np.float32)))
    val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val.astype(np.float32)), torch.from_numpy(w_val.astype(np.float32)))
    test_dataset_deg = TensorDataset(torch.from_numpy(X_test_deg_scaled), torch.from_numpy(y_test_deg.astype(np.float32)), torch.from_numpy(w_test_deg.astype(np.float32)))
    test_dataset_clean = TensorDataset(torch.from_numpy(X_test_clean_scaled), torch.from_numpy(y_test_clean.astype(np.float32)), torch.from_numpy(w_test_clean.astype(np.float32)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader_deg = DataLoader(test_dataset_deg, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader_clean = DataLoader(test_dataset_clean, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

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

    test_loss_clean, test_preds_clean, test_targets_clean = evaluate_loader(model, test_loader_clean, criterion)
    test_metrics_clean = compute_metrics(test_targets_clean, None, test_preds_clean, threshold=best_threshold)

    test_loss_deg, test_preds_deg, test_targets_deg = evaluate_loader(model, test_loader_deg, criterion)
    test_metrics_deg = compute_metrics(test_targets_deg, None, test_preds_deg, threshold=best_threshold)

    # Save test predictions for evaluation scripts
    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        y_true=test_targets_deg,
        y_prob=test_preds_deg,
        best_threshold=best_threshold,
        features=ordered_feats,
        test_metrics=test_metrics_deg
    )

    elapsed = time.time() - start_time
    print(f"==================================================================")
    print(f"FINAL STRATIFIED TEST SET EVALUATION (Optimal Tau = {best_threshold:.3f})")
    print(f"==================================================================")
    print(f"  [CLEAN SPLIT] Apples-to-apples benchmark against historical gate")
    print(f"  Critical Success Index (CSI): {test_metrics_clean['CSI']:.4f}")
    print(f"  Probability of Detection (POD): {test_metrics_clean['POD']:.4f}")
    print(f"  False Alarm Ratio (FAR):       {test_metrics_clean['FAR']:.4f}")
    print(f"  Confusion Matrix: TP={test_metrics_clean['TP']} | FP={test_metrics_clean['FP']} | FN={test_metrics_clean['FN']} | TN={test_metrics_clean['TN']}")
    print(f"")
    print(f"  [DEGRADED SPLIT] Synthetically masked (27.6% UTH dropout on storms)")
    print(f"  Critical Success Index (CSI): {test_metrics_deg['CSI']:.4f}")
    print(f"  Probability of Detection (POD): {test_metrics_deg['POD']:.4f}")
    print(f"  False Alarm Ratio (FAR):       {test_metrics_deg['FAR']:.4f}")
    print(f"  Confusion Matrix: TP={test_metrics_deg['TP']} | FP={test_metrics_deg['FP']} | FN={test_metrics_deg['FN']} | TN={test_metrics_deg['TN']}")
    print(f"==================================================================")

    return model, history, test_metrics_deg


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

DECISION_GATE_CSI = 0.4258  # Validated chronologically-split baseline (Layer 1, CNN-BiLSTM) v2 checkpoint (Doc 09)


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
        ckpt_path = MODELS_DIR / f"focal_a{alpha}_g{gamma}.pt"
        print(f"\n{'='*60}")
        print(f"[FOCAL GRID] alpha={alpha}  gamma={gamma}")

        events_df = load_imd_parquets()
        X_df, y_ser, w_ser, event_groups, feat_names = build_feature_matrix(events_df)
        core_feats = [f for f in FEATURES_AWS if f in X_df.columns]
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

        train_idx = split["train_idx"]
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


# ── Post-dedup baseline re-validation ─────────────────────────────────────────
#
# load_imd_parquets() now drops 46,848 duplicate cross-file rows at
# overlapping basin borders (Chamoli/Tehri/Uttarkashi/Rudraprayag/Pithoragarh).
# Those duplicates had a 0-second time delta, which elapsed_hr's
# .clip(lower=60.0) floors to exactly 60 seconds -- so the proxy rate
# (rain / elapsed_hr) was artificially multiplied by 60x for those rows
# (26,295 rows pushed past 100 mm/hr, max 14,035 mm/hr). No `inf` reached the
# StandardScaler, but those 26,295 inflated values still shifted its fitted
# mean/scale on the pre-dedup 913,536-row set. Every checkpoint trained
# before the dedup fix -- including the Doc 09 baseline the focal-loss grid's
# DECISION_GATE_CSI is pinned to, and the FP/FN rows the Layer 3 cascade in
# Document 11 trains on -- was fit against that shifted scaler.
#
# Run `python train_neural_nowcaster_v2.py --recheck-dedup` BEFORE trusting
# DECISION_GATE_CSI or building anything on top of Doc 09/11's confusion
# matrix. It retrains the exact same BCE-baseline architecture (no focal
# loss) on the deduplicated data, using the identical chronological-split +
# evaluate_on_test() path grid_search_focal uses, so the result is directly
# comparable.

# Doc 09 baseline, computed on the PRE-DEDUP 913,536-row set. Fixed reference
# point -- do not edit these numbers if the rerun below produces new ones;
# a new dated entry should be added instead so the drift stays visible.
DOC09_PRE_DEDUP_BASELINE = {
    "n_rows": 913536,
    "TP": 2208, "FP": 3278, "FN": 121, "TN": 224997,
    "CSI": 0.4258,
}

# Above this absolute CSI shift, Doc 09/11 are considered stale and must be
# updated before the focal-loss grid or Layer 3 cascade proceed.
CSI_DRIFT_TOLERANCE = 0.02


def log_basin_breakdown(
    events_df: pd.DataFrame,
    y_ser: pd.Series,
    event_groups: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """
    Print positive-event counts per basin/region for each split, aligned to
    the same (station_id, timestamp) sort order build_feature_matrix uses.

    The 46,848 dropped duplicates were concentrated at basin borders
    (Rudraprayag 26,352 / Chamoli 20,496 / Tehri 20,496 / Uttarkashi 17,568 /
    Pithoragarh 2,928; Assam 0), so dropping them changes each basin's share
    of the dataset, not just the total row count. Run this after any
    row-count change and confirm no basin's test-set representation
    collapsed -- Rudraprayag/Chamoli/Tehri were already thin (107-154
    positives total per Document 09) before the dedup fix.
    """
    ev_sorted = events_df.copy()
    ev_sorted["timestamp"] = pd.to_datetime(ev_sorted["timestamp"], errors="coerce")
    ev_sorted = ev_sorted.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    region = (
        ev_sorted["region_name"] if "region_name" in ev_sorted.columns
        else pd.Series(["unknown"] * len(ev_sorted))
    )

    y = y_ser.values
    print(f"\n{'Basin':<15} | {'Train pos':>10} | {'Val pos':>8} | {'Test pos':>8} "
          f"| {'Train n':>9} | {'Val n':>7} | {'Test n':>7}")
    print("-" * 82)
    for basin in sorted(region.unique()):
        mask_idx = np.where((region == basin).values)[0]
        tr = np.intersect1d(train_idx, mask_idx)
        vl = np.intersect1d(val_idx, mask_idx)
        te = np.intersect1d(test_idx, mask_idx)
        print(f"{str(basin):<15} | {int(y[tr].sum()):>10} | {int(y[vl].sum()):>8} | {int(y[te].sum()):>8} "
              f"| {len(tr):>9} | {len(vl):>7} | {len(te):>7}")


def rerun_baseline_post_dedup(
    epochs: int = 50,
    batch_size: int = 4096,
    patience: int = 12,
    lr_init: float = 2e-3,
) -> dict:
    """
    Re-validate the Doc 09 baseline (BCE + pos_weight, no focal loss) on the
    deduplicated dataset, using the same chronological split methodology and
    same evaluate_on_test() path as grid_search_focal, so the result is
    directly comparable to DOC09_PRE_DEDUP_BASELINE.

    This must run -- and be inspected -- before trusting DECISION_GATE_CSI,
    the focal-loss grid, or the Layer 3 cascade design in Document 11 (which
    trains directly on the old run's FP/FN rows).
    """
    print("=" * 78)
    print("POST-DEDUP BASELINE RE-VALIDATION")
    print("Comparing against Doc 09 pre-dedup baseline (913,536-row set):")
    print(f"  TP={DOC09_PRE_DEDUP_BASELINE['TP']}  FP={DOC09_PRE_DEDUP_BASELINE['FP']}  "
          f"FN={DOC09_PRE_DEDUP_BASELINE['FN']}  TN={DOC09_PRE_DEDUP_BASELINE['TN']}  "
          f"CSI={DOC09_PRE_DEDUP_BASELINE['CSI']}")
    print("=" * 78)

    # The persisted split_assignments.npz predates the dedup fix -- its
    # indices no longer line up with the current (deduplicated) row order.
    # force=True discards it and recomputes chronological splits fresh.
    save_split_assignments(force=True)

    events_df = load_imd_parquets()
    X_df, y_ser, w_ser, event_groups, feat_names = build_feature_matrix(events_df)
    core_feats = [f for f in ["R_60", "R_30", "R", "RI"] if f in X_df.columns]
    sat_feats = [
        f for f in ["ctt_mean", "ctt_min", "ctt_cold_frac", "hem_mean", "hem_max", "olr_mean", "uth_mean"]
        if f in X_df.columns and X_df[f].notna().mean() > 0.05
    ]
    ordered = core_feats + sat_feats
    X_df = X_df[ordered]

    split = np.load(OUTPUT_DIR / "split_assignments.npz", allow_pickle=True)
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    assert split["n_total"] == len(y_ser), (
        f"Row count mismatch after dedup: split has {split['n_total']}, data has {len(y_ser)}. "
        "save_split_assignments(force=True) should have fixed this -- investigate before proceeding."
    )

    print(f"\nRow count check: {len(y_ser):,} rows post-dedup "
          f"(pre-dedup was 913,536; expect ~866,688 if the 46,848-row fix is the only change since).")

    log_basin_breakdown(events_df, y_ser, event_groups, train_idx, val_idx, test_idx)

    # Fit scaler on training rows only.
    scaler = StandardScaler()
    scaler.fit(X_df.values[train_idx])
    X_all = scaler.transform(X_df.values).astype(np.float32)
    y_all = y_ser.values.astype(np.float32)
    w_all = w_ser.values.astype(np.float32)

    X_tr, y_tr, w_tr = X_all[train_idx], y_all[train_idx], w_all[train_idx]
    X_vl, y_vl, w_vl = X_all[val_idx], y_all[val_idx], w_all[val_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(w_tr)),
        batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(y_vl), torch.from_numpy(w_vl)),
        batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    model = CloudburstCNNBiLSTM(in_features=len(ordered)).to(DEVICE)
    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())], device=DEVICE)
    print(f"\nPost-dedup pos_weight = {pos_weight.item():.1f}  "
          f"(compare to the old 103:1 -- expect a small shift now the 26,295 inflated-R rows are gone)")
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_csi, patience_ctr, best_epoch, best_state = -1.0, 0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            (loss * bw).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        _, vp, vt = evaluate_loader(model, val_loader, criterion)
        ot = find_optimal_threshold(vt, vp)
        val_csi = compute_metrics(vt, None, vp, threshold=ot)["CSI"]

        if val_csi > best_val_csi:
            best_val_csi, best_epoch, patience_ctr = val_csi, epoch, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch} (best epoch {best_epoch}, val CSI={best_val_csi:.4f})")
                break
        scheduler.step()

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    ckpt_path = MODELS_DIR / "cloudburst_cnn_bilstm_POST_DEDUP_baseline.pt"
    torch.save({
        "model_state_dict": best_state,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "features": ordered,
        "best_val_csi": best_val_csi,
    }, ckpt_path)
    print(f"  Saved post-dedup baseline checkpoint -> {ckpt_path.name}")

    best_tau, _ = tau_sweep(model, val_loader)

    ckpt = torch.load(ckpt_path, weights_only=False)
    test_metrics = evaluate_on_test(ckpt, test_idx, tau=best_tau)

    print("\n" + "=" * 78)
    print("POST-DEDUP BASELINE vs. DOC 09 (PRE-DEDUP) -- SIDE BY SIDE")
    print("=" * 78)
    print(f"{'Metric':<8} | {'Pre-dedup (Doc 09)':>20} | {'Post-dedup (this run)':>22} | {'Delta'}")
    print("-" * 78)
    for k in ["TP", "FP", "FN", "TN", "CSI"]:
        old = DOC09_PRE_DEDUP_BASELINE[k]
        new = test_metrics[k]
        delta = new - old
        print(f"{k:<8} | {old:>20} | {new:>22} | {delta:+}")
    print("=" * 78)

    csi_shift = abs(test_metrics["CSI"] - DOC09_PRE_DEDUP_BASELINE["CSI"])
    if csi_shift > CSI_DRIFT_TOLERANCE:
        print(f"\n[GATE] CSI shifted by {csi_shift:.4f} (> {CSI_DRIFT_TOLERANCE} tolerance).")
        print("  Documents 09 and 11 are STALE. Do not run the focal-loss grid against the old")
        print("  DECISION_GATE_CSI, and do not build the Layer 3 cascade on the old FP/FN rows,")
        print("  until both documents are updated to this new baseline.")
    else:
        print(f"\n[GATE] CSI shifted by only {csi_shift:.4f} (<= {CSI_DRIFT_TOLERANCE} tolerance).")
        print("  Doc 09/11 numbers hold within tolerance. Safe to proceed with the focal-loss")
        print("  grid and UTH integration against the existing DECISION_GATE_CSI.")

    return test_metrics


if __name__ == "__main__":
    if "--recheck-dedup" in sys.argv:
        # Run this FIRST, before anything else in this file, whenever the
        # underlying row set has changed (e.g. the load_imd_parquets() dedup
        # fix). See the module-level comment above rerun_baseline_post_dedup.
        rerun_baseline_post_dedup(epochs=50, batch_size=4096, patience=12)
        sys.exit(0)

    # Bypass the old chronological focal grid (which requires 24 years of data) 
    # and directly train the 12-feature model on the available disaster windows
    model, history, metrics = train_nowcaster(
        epochs=30, 
        batch_size=4096, 
        use_satellite=True,
        loss_fn="focal",
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    print("\n--- Final 12-Feature CNN+BiLSTM Results ---")
    print(f"POD: {metrics['POD']:.4f}")
    print(f"FAR: {metrics['FAR']:.4f}")
    print(f"CSI: {metrics['CSI']:.4f}")
