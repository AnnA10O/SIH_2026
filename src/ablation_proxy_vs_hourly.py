import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))
from src.phase_d_training import compute_metrics, find_optimal_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_features: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

def evaluate_loader(model, loader, criterion):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(by.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)

def train_arm(X_np, y_np, name=""):
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    
    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test.astype(np.float32)))
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    model = MLP(in_features=X_np.shape[1]).to(DEVICE)
    # Give higher weight to positives
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"Training {name} MLP...")
    for epoch in range(1, 31):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    preds, targets = evaluate_loader(model, test_loader, criterion)
    opt_thresh = find_optimal_threshold(targets, preds)
    metrics = compute_metrics(targets, None, preds, threshold=opt_thresh)
    return metrics

def run_ablation():
    print("Loading merged dataset...")
    df = pd.read_parquet(ROOT / "outputs/assam_ablation_set.parquet")
    
    # 1. Build labels
    df["y"] = df["rain_label"].map({
        "CONFIRMED_CLOUDBURST": 1,
        "CANDIDATE_CLOUDBURST": 1,
        "WIDESPREAD_HEAVY_RAIN": 0,
        "HEAVY_RAIN": 0,
        "MODERATE_RAIN": 0,
        "NORMAL": 0
    }).fillna(0).astype(int)
    
    # 2. Build Proxy Features (as done in build_feature_matrix)
    df["rain_mm_hr"] = df["rain_mm_hr"].fillna(0)
    df["R_proxy"] = df["rain_mm_hr"] / 24.0
    df["R_30_proxy"] = df["rain_mm_hr"] / 48.0
    df["R_60_proxy"] = df["rain_mm_hr"] / 24.0
    df["RI_proxy"] = df["rain_mm_hr"] / 24.0 # Fake change
    
    # 3. Build AWS Features
    df["aws_rain_mm_hr"] = df["aws_rain_mm_hr"].fillna(-1.0)
    df["R_aws"] = df["aws_rain_mm_hr"]
    df["R_30_aws"] = df["aws_rain_mm_hr"] / 2.0
    df["R_60_aws"] = df["aws_rain_mm_hr"]
    df["RI_aws"] = df["aws_rain_mm_hr"] 
    
    # Identify intersection where BOTH are fully valid
    # Since proxy was filled from 0, it's always valid.
    # AWS is valid where aws_rain_mm_hr != -1
    # IWV is valid where iwv is not null
    
    valid_mask = (df["aws_rain_mm_hr"] != -1.0) & (df["iwv"].notna())
    df_valid = df[valid_mask].copy()
    
    print(f"\n========================================================")
    print(f"IDENTICAL ROW SET DISCIPLINE")
    print(f"Total Assam rows (2013-14): {len(df)}")
    print(f"Valid AWS + IWV intersection: {len(df_valid)}")
    print(f"Positive Events in Intersection: {df_valid['y'].sum()}")
    print(f"========================================================\n")
    
    if len(df_valid) < 50:
        print("Not enough data to run a meaningful neural net ablation!")
        print("We need to expand the ±30 min tolerance, or the spatial distance.")
        return
        
    y_np = df_valid["y"].values
    
    X_proxy = df_valid[["R_proxy", "R_30_proxy", "R_60_proxy", "RI_proxy", "iwv"]].values
    X_aws   = df_valid[["R_aws", "R_30_aws", "R_60_aws", "RI_aws", "iwv"]].values
    
    m_proxy = train_arm(X_proxy, y_np, name="Proxy Arm")
    m_aws   = train_arm(X_aws, y_np, name="Real Hourly AWS Arm")
    
    print(f"\n========================================================")
    print(f"ABLATION RESULTS (Hold-out Test Split, N={len(y_np)*0.3:.0f})")
    print(f"========================================================")
    print(f"{'Metric':<10} | {'Proxy Arm':<15} | {'Hourly AWS Arm':<15} | {'Delta'}")
    print("-" * 60)
    for k in ["CSI", "POD", "FAR", "PR_AUC", "ROC_AUC"]:
        delta = m_aws[k] - m_proxy[k]
        print(f"{k:<10} | {m_proxy[k]:<15.4f} | {m_aws[k]:<15.4f} | {delta:+.4f}")
    print(f"========================================================\n")

if __name__ == "__main__":
    run_ablation()
