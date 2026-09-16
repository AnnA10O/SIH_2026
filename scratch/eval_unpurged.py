import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train_neural_nowcaster_v2 import CloudburstCNNBiLSTM
from src.phase_d_training import load_imd_parquets, build_feature_matrix

def run_eval():
    print("Loading data...")
    events_df = load_imd_parquets()
    events_df = events_df.sample(300000, random_state=42).copy()
    print("Building feature matrix (unpurged)...")
    X, y, weights, event_groups, feature_names = build_feature_matrix(events_df)
    
    # Fix the group logic: give every negative event (-1) a unique group ID so they distribute randomly
    mask_neg = (event_groups['cluster_id'] == -1)
    # Assign unique IDs starting from the max existing cluster ID
    max_id = event_groups['cluster_id'].max()
    unique_ids = np.arange(max_id + 1, max_id + 1 + mask_neg.sum())
    
    event_groups_fixed = event_groups['cluster_id'].copy()
    event_groups_fixed.loc[mask_neg] = unique_ids
    
    print(f"Feature matrix built: {len(X)} rows")
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, y, groups=event_groups_fixed))
    
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
    groups_temp = event_groups_fixed.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))
    
    final_test_idx = temp_idx[test_idx]
    
    X_test_raw = X.iloc[final_test_idx]
    y_test_raw = y.iloc[final_test_idx]
    
    print(f'Unpurged Test Set Size: {len(X_test_raw)} rows')
    
    ckpt = torch.load('models/cloudburst_cnn_bilstm_best.pt', map_location='cpu', weights_only=False)
    ordered_feats = ckpt['features']
    model = CloudburstCNNBiLSTM(in_features=len(ordered_feats))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    scaler_mean = np.array(ckpt['scaler_mean'])
    scaler_scale = np.array(ckpt['scaler_scale'])
    
    X_test_array = X_test_raw[ordered_feats].values
    X_test_scaled = (X_test_array - scaler_mean) / scaler_scale
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        y_prob = model(X_tensor).numpy().ravel()
        
    thresh = 0.360
    y_pred = (y_prob >= thresh).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test_raw.values, y_pred).ravel()
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
    print(f'FAR: {far:.4f}')
    print(f'POD: {pod:.4f}')
    print(f'CSI: {csi:.4f}')

if __name__ == '__main__':
    run_eval()
