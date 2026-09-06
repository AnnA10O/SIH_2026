import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CLOUDBURST_EVENTS_CSV
from src.phase_d_training import build_feature_matrix, event_grouped_split, find_optimal_threshold

events_df = pd.read_csv(CLOUDBURST_EVENTS_CSV)

df = events_df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
if "rain_mm_day" in df.columns:
    df["rain_mm_hr"] = df["rain_mm_day"]
df = df.dropna(subset=["timestamp", "rain_mm_hr", "final_label"])

df["R"] = df["rain_mm_hr"]
df = df.sort_values(["station_id", "timestamp"])
df["R_30"] = df.groupby("station_id")["R"].transform(lambda s: s.rolling(2, min_periods=1).sum())
df["R_60"] = df.groupby("station_id")["R"].transform(lambda s: s.rolling(2, min_periods=1).sum())
df["RI"]   = df.groupby("station_id")["R"].transform(lambda s: s.diff().fillna(0))

X, y, weights, event_groups, feature_names = build_feature_matrix(events_df)
for fn in feature_names:
    df[fn] = X[fn].values
df["y_true"] = y.values
df["event_group"] = event_groups.values

train_data, val_data, test_data = event_grouped_split(
    X, y, weights, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
)

best_C = 0.001
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs", random_state=42))
])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipe.fit(train_data["X"], train_data["y"], clf__sample_weight=train_data["w"])

val_proba = pipe.predict_proba(val_data["X"])[:, 1]
opt_thresh = find_optimal_threshold(val_data["y"], val_proba)

# Get Test Split
rng = np.random.RandomState(42)
pos_gids = np.unique(event_groups[y == 1])
rng.shuffle(pos_gids)
n_pos = len(pos_gids)
n_train_pos = int(n_pos * 0.70)
n_val_pos = int(n_pos * 0.15)
test_pos_gids = set(pos_gids[n_train_pos + n_val_pos:])

norm_gids = np.unique(event_groups[y == 0])
rng.shuffle(norm_gids)
n_norm = len(norm_gids)
n_train_norm = int(n_norm * 0.70)
n_val_norm = int(n_norm * 0.15)
test_norm_gids = set(norm_gids[n_train_norm + n_val_norm:])

test_gids = test_pos_gids.union(test_norm_gids)
test_mask = df["event_group"].isin(test_gids)
test_df = df[test_mask].copy()

test_df["y_proba"] = pipe.predict_proba(test_df[feature_names].values)[:, 1]
test_df["y_pred"] = (test_df["y_proba"] >= opt_thresh).astype(int)

tps = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 1)]
print("=" * 70)
print(f"AUDIT OF 100 TEST POSITIVES (Caught Events)")
print("=" * 70)
print(f"Optimal Threshold tau: {opt_thresh:.3f}")
print(f"Total True Positives: {len(tps)}")

# Check R distribution of True Positives
print("\n--- Rain Intensity (R) Distribution of Caught Events ---")
print(f"Mean R:   {tps['R'].mean():.2f} mm/hr")
print(f"Median R: {tps['R'].median():.2f} mm/hr")
print(f"Min R:    {tps['R'].min():.2f} mm/hr")
print(f"Max R:    {tps['R'].max():.2f} mm/hr")

# Categorize into Deep-in-threshold vs Borderline
deep = tps[tps["R"] >= 100.0]
mid  = tps[(tps["R"] >= 50.0) & (tps["R"] < 100.0)]
borderline_r = tps[(tps["R"] >= 30.0) & (tps["R"] < 50.0)]
low_r = tps[tps["R"] < 30.0]

print(f"\nBreakdown by Rain Rate:")
print(f"  Extreme Cloudburst (R >= 100 mm/hr): {len(deep)} ({len(deep)/len(tps)*100:.1f}%)")
print(f"  Severe Convective (50 <= R < 100):   {len(mid)} ({len(mid)/len(tps)*100:.1f}%)")
print(f"  Borderline Trigger (30 <= R < 50):   {len(borderline_r)} ({len(borderline_r)/len(tps)*100:.1f}%)")
print(f"  Sub-threshold (R < 30):              {len(low_r)} ({len(low_r)/len(tps)*100:.1f}%)")

print("\n--- Probability P(CB) Distribution of Caught Events ---")
print(f"Mean P:   {tps['y_proba'].mean():.4f}")
print(f"Median P: {tps['y_proba'].median():.4f}")
print(f"Min P:    {tps['y_proba'].min():.4f}")
print(f"Max P:    {tps['y_proba'].max():.4f}")

# Threshold Sensitivity Curve
print("\n--- Threshold Sensitivity / POD Decay Curve ---")
print("Threshold |   TP  |   FN  |   FP  |   TN  |   POD   |   FAR   |   CSI")
print("-" * 65)
for t in [0.40, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.50, 0.55, 0.60]:
    y_b = (test_df["y_proba"] >= t).astype(int)
    tp = int(((test_df["y_true"] == 1) & (y_b == 1)).sum())
    fn = int(((test_df["y_true"] == 1) & (y_b == 0)).sum())
    fp = int(((test_df["y_true"] == 0) & (y_b == 1)).sum())
    tn = int(((test_df["y_true"] == 0) & (y_b == 0)).sum())
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    tag = " <-- OPTIMAL TAU" if abs(t - opt_thresh) < 0.005 else ""
    print(f"  {t:.2f}    |  {tp:3d}  |  {fn:3d}  |  {fp:3d}  |  {tn:3d}  |  {pod:.4f} | {far:.4f}  | {csi:.4f}{tag}")
