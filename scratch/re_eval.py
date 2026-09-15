import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, 'd:/SIH')

from src.train_neural_nowcaster_v2 import (
    CloudburstCNNBiLSTM, evaluate_loader
)
from src.phase_d_training import (
    load_imd_parquets, build_feature_matrix, event_grouped_split,
    compute_metrics, simulate_sensor_outage_blocks
)

print("Loading saved best checkpoint from recent training run...")
ckpt = torch.load("d:/SIH/models/cloudburst_cnn_bilstm_best.pt", weights_only=False)
feats = ckpt["features"]
tau = ckpt["optimal_threshold"]
scaler_mean = np.array(ckpt["scaler_mean"])
scaler_scale = np.array(ckpt["scaler_scale"])

print(f"Features: {feats}")
print(f"Optimal Tau (derived on validation set during training): {tau:.3f}")

model = CloudburstCNNBiLSTM(in_features=len(feats)).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

events_df_clean = load_imd_parquets()
X_df_clean, y_ser, w_ser, event_groups, _ = build_feature_matrix(events_df_clean)

events_df_deg = simulate_sensor_outage_blocks(events_df_clean, "final_label")
X_df_deg, _, _, _, _ = build_feature_matrix(events_df_deg)

X_df_clean = X_df_clean[feats]
X_df_deg = X_df_deg[feats]

print("\nVerifying Split Hashes/Counts (Random State = 42 for both clean and degraded)...")
_, _, test_clean = event_grouped_split(X_df_clean, y_ser, w_ser, event_groups, 0.70, 0.15, 42)
_, _, test_deg = event_grouped_split(X_df_deg, y_ser, w_ser, event_groups, 0.70, 0.15, 42)

print(f"Clean Test Split Rows: {test_clean['count']}, Positives: {test_clean['pos_count']}")
print(f"Degraded Test Split Rows: {test_deg['count']}, Positives: {test_deg['pos_count']}")
assert test_clean['count'] == test_deg['count'], "Splits are not identical!"

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# Fit on the identical train set used during training to prevent leakage
_, _, train_clean = event_grouped_split(X_df_clean, y_ser, w_ser, event_groups, 0.70, 0.15, 42)
imputer.fit(train_clean["X"])

def score(data, name):
    X_imp = imputer.transform(data["X"])
    X = ((X_imp - scaler_mean) / scaler_scale).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X), 
            torch.from_numpy(data["y"].astype(np.float32)), 
            torch.from_numpy(data["w"].astype(np.float32))
        ), 
        batch_size=4096
    )
    _, p, t = evaluate_loader(model, loader, nn.BCEWithLogitsLoss(reduction="none"))
    m = compute_metrics(t, None, p, threshold=tau)
    print(f"\n==================================================================")
    print(f"FINAL STRATIFIED TEST SET EVALUATION ({name})")
    print(f"==================================================================")
    print(f"  Critical Success Index (CSI): {m['CSI']:.4f}")
    print(f"  Probability of Detection (POD): {m['POD']:.4f}")
    print(f"  False Alarm Ratio (FAR):       {m['FAR']:.4f}")
    print(f"  Confusion Matrix: TP={m['TP']} | FP={m['FP']} | FN={m['FN']} | TN={m['TN']}")
    print(f"==================================================================")

score(test_clean, "CLEAN SPLIT")
score(test_deg, "DEGRADED SPLIT")
