import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CLOUDBURST_EVENTS_CSV, WEIGHT_CONFIRMED_CLOUDBURST, WEIGHT_CANDIDATE_CLOUDBURST, WEIGHT_WIDESPREAD_HEAVY_RAIN, WEIGHT_NORMAL
from src.phase_d_training import event_grouped_split, find_optimal_threshold, compute_metrics, _assign_event_groups

events_df = pd.read_csv(CLOUDBURST_EVENTS_CSV)

df = events_df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
if "rain_mm_day" in df.columns:
    df["rain_mm_hr"] = df["rain_mm_day"]
df = df.dropna(subset=["timestamp", "rain_mm_hr", "final_label"])

df["R"] = df["rain_mm_hr"]
df = df.sort_values(["station_id", "timestamp"])

# 1. Standard rolling features
df["R_30"] = df.groupby("station_id")["R"].transform(lambda s: s.rolling(2, min_periods=1).sum())
df["R_60"] = df.groupby("station_id")["R"].transform(lambda s: s.rolling(2, min_periods=1).sum())
df["RI"]   = df.groupby("station_id")["R"].transform(lambda s: s.diff().fillna(0))

# 2. Fix rolling lag: if current R is 0, active burst accumulation is 0
df["R_30_gated"] = np.where(df["R"] > 0, df["R_30"], 0.0)
df["R_60_gated"] = np.where(df["R"] > 0, df["R_60"], 0.0)

# 3. Spatial Contrast Feature: R_core - R_background = R * L_score
df["L_score"] = df["L_score"].fillna(0.0)
df["spatial_contrast"] = df["R"] * df["L_score"]

# Binary label
y = df["final_label"].map({
    "CONFIRMED_CLOUDBURST":  1,
    "CANDIDATE_CLOUDBURST":  1,
    "WIDESPREAD_HEAVY_RAIN": 0,
    "NORMAL":                0,
}).fillna(0).astype(int)

# Sample weights
weight_map = {
    "CONFIRMED_CLOUDBURST":  WEIGHT_CONFIRMED_CLOUDBURST,
    "CANDIDATE_CLOUDBURST":  WEIGHT_CANDIDATE_CLOUDBURST,
    "WIDESPREAD_HEAVY_RAIN": WEIGHT_WIDESPREAD_HEAVY_RAIN,
    "NORMAL":                WEIGHT_NORMAL,
}
weights = df["final_label"].map(weight_map).fillna(1.0)
event_groups = _assign_event_groups(df)

# Feature set comparison:
# Set A: Original baseline ['R', 'R_30', 'R_60', 'RI']
# Set B: Refined linear features ['R', 'R_30_gated', 'R_60_gated', 'RI', 'spatial_contrast']

for name, feat_cols, C_val in [
    ("BASELINE (Original 4 features, C=0.001)", ["R", "R_30", "R_60", "RI"], 0.001),
    ("FIX 1 (Gated Rolling, C=0.001)", ["R", "R_30_gated", "R_60_gated", "RI"], 0.001),
    ("FIX 2 (Gated + Spatial Contrast, C=0.001)", ["R", "R_30_gated", "R_60_gated", "RI", "spatial_contrast"], 0.001),
    ("FIX 3 (Gated + Spatial Contrast, C=0.10 calibrated)", ["R", "R_30_gated", "R_60_gated", "RI", "spatial_contrast"], 0.10)
]:
    X = df[feat_cols].copy()
    train_data, val_data, test_data = event_grouped_split(
        X, y, weights, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
    )
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C_val, max_iter=1000, solver="lbfgs", random_state=42))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(train_data["X"], train_data["y"], clf__sample_weight=train_data["w"])
    
    val_proba = pipe.predict_proba(val_data["X"])[:, 1]
    opt_t = find_optimal_threshold(val_data["y"], val_proba)
    
    test_proba = pipe.predict_proba(test_data["X"])[:, 1]
    test_m = compute_metrics(test_data["y"], None, test_proba, threshold=opt_t)
    
    # Also evaluate at neutral 0.50 threshold
    test_m_50 = compute_metrics(test_data["y"], None, test_proba, threshold=0.50)
    
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print(f"Optimal Threshold tau: {opt_t:.3f}")
    print(f"  @ Optimal Threshold: TP={test_m['TP']}, FP={test_m['FP']}, FN={test_m['FN']}, TN={test_m['TN']}")
    print(f"    POD = {test_m['POD']:.4f}, FAR = {test_m['FAR']:.4f}, CSI = {test_m['CSI']:.4f}, PR-AUC = {test_m['PR_AUC']:.4f}")
    print(f"  @ Neutral Threshold (0.50): TP={test_m_50['TP']}, FP={test_m_50['FP']}, FN={test_m_50['FN']}, TN={test_m_50['TN']}")
    print(f"    POD = {test_m_50['POD']:.4f}, FAR = {test_m_50['FAR']:.4f}, CSI = {test_m_50['CSI']:.4f}")
    
    clf = pipe.named_steps["clf"]
    print("Learned Weights:")
    print(f"  Intercept: {clf.intercept_[0]:.4f}")
    for col, coef in zip(feat_cols, clf.coef_[0]):
        print(f"  {col:18s}: {coef:+.4f}")
    print()
