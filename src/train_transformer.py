import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from src.phase_d_training import (
    load_imd_parquets, build_feature_matrix, event_grouped_split,
    compute_metrics, find_optimal_threshold, simulate_sensor_outage_blocks
)
from src.config import FEATURES_SAT, FEATURES_AWS, FEATURES_STALENESS
from src.spatiotemporal_transformer import SpatiotemporalTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_weights = [], [], []

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

def train_transformer(epochs: int = 20, batch_size: int = 4096, lr_init: float = 2e-3):
    print(f"Training Spatiotemporal Transformer on Device: {DEVICE}")
    
    events_df_clean = load_imd_parquets()
    X_df_clean, y_ser, w_ser, event_groups, _ = build_feature_matrix(events_df_clean)
    
    core_feats = [f for f in FEATURES_AWS if f in X_df_clean.columns]
    sat_feats = [f for f in FEATURES_SAT if f in X_df_clean.columns and X_df_clean[f].notna().mean() > 0.05]
    stale_feats = [f for f in FEATURES_STALENESS if f in X_df_clean.columns]
    
    ordered_feats = core_feats + sat_feats + stale_feats
    X_df_clean = X_df_clean[ordered_feats]
    
    print(f"Features: {len(core_feats)} Core, {len(sat_feats)} Sat, {len(stale_feats)} Stale")
    
    print("Using event grouped split...")
    train_data, val_data, test_data = event_grouped_split(
        X_df_clean, y_ser, w_ser, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
    )
    X_train, y_train, w_train = np.array(train_data["X"]), np.array(train_data["y"]), np.array(train_data["w"])
    X_val, y_val, w_val = np.array(val_data["X"]), np.array(val_data["y"]), np.array(val_data["w"])
    X_test, y_test, w_test = np.array(test_data["X"]), np.array(test_data["y"]), np.array(test_data["w"])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_imputed).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_imputed).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_imputed).astype(np.float32)
    
    print(f"Test samples: {len(X_test_scaled)}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train.astype(np.float32)), torch.from_numpy(w_train.astype(np.float32))), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val.astype(np.float32)), torch.from_numpy(w_val.astype(np.float32))), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_scaled), torch.from_numpy(y_test.astype(np.float32)), torch.from_numpy(w_test.astype(np.float32))), batch_size=batch_size, shuffle=False)

    model = SpatiotemporalTransformer(
        num_core_feats=len(core_feats),
        num_sat_feats=len(sat_feats),
        num_stale_feats=len(stale_feats),
        d_model=64, nhead=4, num_layers=2
    ).to(DEVICE)
    
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-4)
    
    best_val_csi = -1.0
    best_threshold = 0.5
    save_path = MODELS_DIR / "spatiotemporal_transformer_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, batch_w in train_loader:
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            (loss * batch_w).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
        val_loss, val_preds, val_targets = evaluate_loader(model, val_loader, criterion)
        opt_thresh = find_optimal_threshold(val_targets, val_preds)
        val_metrics = compute_metrics(val_targets, None, val_preds, threshold=opt_thresh)
        
        print(f"Epoch {epoch:2d} | Val Loss: {val_loss:.4f} | Val CSI: {val_metrics['CSI']:.4f} | Val POD: {val_metrics['POD']:.4f}")
        
        if val_metrics["CSI"] > best_val_csi:
            best_val_csi = val_metrics["CSI"]
            best_threshold = opt_thresh
            torch.save(model.state_dict(), save_path)

    print(f"\nEvaluating Best Model on Test Set (Tau={best_threshold:.3f})...")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    _, test_preds, test_targets = evaluate_loader(model, test_loader, criterion)
    
    test_metrics = compute_metrics(test_targets, None, test_preds, threshold=best_threshold)
    roc_auc = roc_auc_score(test_targets, test_preds)
    
    print("\n" + "="*50)
    print("TRANSFORMER TEST METRICS")
    print("="*50)
    print(f"POD:     {test_metrics['POD']:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"CSI:     {test_metrics['CSI']:.4f}")
    print(f"FAR:     {test_metrics['FAR']:.4f}")
    print("="*50)

if __name__ == "__main__":
    train_transformer()
