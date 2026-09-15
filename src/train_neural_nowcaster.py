"""
Train PyTorch 1D-CNN + BiLSTM Cloudburst Nowcasting Model.
Replaces the logistic regression model with the genuine Neural Network architecture
from the SIH 2026 Technical Approach slide.
"""
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from src.phase_d_training import (
    load_imd_parquets, build_feature_matrix, event_grouped_split,
    compute_metrics, find_optimal_threshold
)


# ── 1. PyTorch 1D-CNN + BiLSTM Model ─────────────────────────────────────────

class CloudburstCNNBiLSTM(nn.Module):
    """
    1D-CNN + BiLSTM In-Situ Convective Precursor Nowcaster.
    
    Temporal ordering of features: [R_60, R_30, R, RI]
    representing the convective escalation curve from 60min to instantaneous burst.
    """
    def __init__(self, in_features: int = 4, cnn_filters: int = 16, lstm_hidden: int = 16):
        super().__init__()
        self.in_features = in_features
        # Conv1d expects (batch, in_channels=1, seq_len=4)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=2, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # BiLSTM over the convolved sequence
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4)
        # Add channel dim: (batch, 1, 4)
        x_seq = x.unsqueeze(1)
        conv = self.conv1d(x_seq)                     # (batch, cnn_filters, conv_len)
        lstm_in = conv.permute(0, 2, 1)               # (batch, conv_len, cnn_filters)
        lstm_out, _ = self.bilstm(lstm_in)            # (batch, conv_len, 2 * lstm_hidden)
        # Global max pooling over sequence dimension
        pooled, _ = torch.max(lstm_out, dim=1)        # (batch, 2 * lstm_hidden)
        logits = self.head(pooled).squeeze(-1)        # (batch,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ── 2. Training Loop ─────────────────────────────────────────────────────────

def train_neural_nowcaster(epochs: int = 5, batch_size: int = 2048, lr: float = 1e-3):
    print("=== Training 1D-CNN + BiLSTM Cloudburst Nowcasting Model ===")
    start_t0 = time.time()

    # Load data
    events_df = load_imd_parquets()
    X_df, y_ser, w_ser, event_groups, feature_names = build_feature_matrix(events_df)

    # Order features chronologically: R_60 -> R_30 -> R -> RI
    ordered_feats = ["R_60", "R_30", "R", "RI"]
    X_df = X_df[ordered_feats]
    print(f"Ordered features for sequential 1D-CNN/BiLSTM: {ordered_feats}")

    train_data, val_data, test_data = event_grouped_split(
        X_df, y_ser, w_ser, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(train_data["X"])
    X_val_norm   = scaler.transform(val_data["X"])
    X_test_norm  = scaler.transform(test_data["X"])

    y_train = train_data["y"].astype(np.float32)
    w_train = train_data["w"].astype(np.float32)

    y_val = val_data["y"].astype(np.float32)
    y_test = test_data["y"].astype(np.float32)

    # Datasets and loaders
    train_ds = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Initialize model
    torch.manual_seed(42)
    model = CloudburstCNNBiLSTM(in_features=4, cnn_filters=16, lstm_hidden=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\nModel Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Training
    model.train()
    for epoch in range(1, epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        t_ep = time.time()
        for bx, by, bw in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            # Weighted Binary Cross Entropy Loss
            bce = nn.functional.binary_cross_entropy_with_logits(logits, by, reduction="none")
            loss = (bce * bw).mean()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        avg_loss = ep_loss / n_batches
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.5f} ({time.time() - t_ep:.1f}s)")

    # Evaluate on Validation
    model.eval()
    with torch.no_grad():
        val_x_t = torch.tensor(X_val_norm, dtype=torch.float32)
        val_p = model.predict_proba(val_x_t).numpy()
        opt_thresh = find_optimal_threshold(y_val, val_p)
        val_m = compute_metrics(y_val, None, val_p, threshold=opt_thresh)

        # Evaluate on Test
        test_x_t = torch.tensor(X_test_norm, dtype=torch.float32)
        test_p = model.predict_proba(test_x_t).numpy()
        test_m = compute_metrics(y_test, None, test_p, threshold=opt_thresh)

    print(f"\n=== Neural Network Validation Performance (threshold = {opt_thresh:.2f}) ===")
    print(f"  Val CSI : {val_m['CSI']:.4f} | POD: {val_m['POD']:.4f} | FAR: {val_m['FAR']:.4f} | PR-AUC: {val_m['PR_AUC']:.4f}")

    print(f"\n=== Neural Network Held-Out Test Split Performance ===")
    print(f"  Test CSI: {test_m['CSI']:.4f} | POD: {test_m['POD']:.4f} | FAR: {test_m['FAR']:.4f} | PR-AUC: {test_m['PR_AUC']:.4f}")
    print(f"  TP: {test_m['TP']} | FP: {test_m['FP']} | FN: {test_m['FN']} | TN: {test_m['TN']}")

    # Save PyTorch Model Checkpoint
    os.makedirs(ROOT / "models", exist_ok=True)
    pt_path = ROOT / "models" / "cloudburst_cnn_bilstm.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"in_features": 4, "cnn_filters": 16, "lstm_hidden": 16},
        "feature_names": ordered_feats,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "optimal_threshold": float(opt_thresh),
        "test_metrics": test_m,
        "val_metrics": val_m,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, pt_path)
    print(f"\nSaved PyTorch model checkpoint -> {pt_path}")

    # Export weights to JSON for direct execution in Vanilla JS Dashboard
    export_dict = {
        "feature_names": ordered_feats,
        "scaler": {
            "mean": dict(zip(ordered_feats, scaler.mean_.tolist())),
            "scale": dict(zip(ordered_feats, scaler.scale_.tolist()))
        },
        "optimal_threshold": float(opt_thresh),
        "test_metrics": test_m,
        "weights": {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
    }
    json_path = ROOT / "models" / "cloudburst_cnn_bilstm_weights.json"
    with open(json_path, "w") as f:
        json.dump(export_dict, f, indent=2)
    print(f"Exported JSON weights for JavaScript browser inference -> {json_path}")
    print(f"Total execution time: {time.time() - start_t0:.1f}s")


if __name__ == "__main__":
    train_neural_nowcaster(epochs=5, batch_size=2048, lr=2e-3)
