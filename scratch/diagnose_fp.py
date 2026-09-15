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

# Add engineered features directly to events_df copy
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

# Attach X features and metadata directly to df
for fn in feature_names:
    df[fn] = X[fn].values
df["y_true"] = y.values
df["weight"] = weights.values
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

# Re-run train/val/test masking on df using random_state 42
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

fps = test_df[(test_df["y_true"] == 0) & (test_df["y_pred"] == 1)]
tps = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 1)]
tns = test_df[(test_df["y_true"] == 0) & (test_df["y_pred"] == 0)]
fns = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 0)]

print(f"Total Test Set Samples: {len(test_df)}")
print(f"Confusion Matrix: TP={len(tps)}, FP={len(fps)}, FN={len(fns)}, TN={len(tns)}")
print(f"Optimal Decision Threshold tau: {opt_thresh:.3f}")

print("\n=== Exact Breakdown of 29 False Positives by Ground Truth Label ===")
print(fps["final_label"].value_counts())

print("\n=== Probability Distribution of False Positives ===")
print(f"Mean y_proba of FPs:   {fps['y_proba'].mean():.4f}")
print(f"Median y_proba of FPs: {fps['y_proba'].median():.4f}")
print(f"Min y_proba of FPs:    {fps['y_proba'].min():.4f}")
print(f"Max y_proba of FPs:    {fps['y_proba'].max():.4f}")
print(f"Threshold tau:         {opt_thresh:.4f}")

# Check how close to threshold they are
borderline = fps[(fps["y_proba"] >= opt_thresh) & (fps["y_proba"] < 0.50)]
print(f"\nHow many FPs are borderline (tau <= p < 0.50)? {len(borderline)} out of {len(fps)} ({len(borderline)/len(fps)*100:.1f}%)")

print("\n=== Feature Comparison (Mean / Median / Min / Max) ===")
for fn in feature_names:
    print(f"\nFeature: {fn}")
    print(f"  True Negatives (TN): mean={tns[fn].mean():.2f}, med={tns[fn].median():.2f}, min={tns[fn].min():.2f}, max={tns[fn].max():.2f}")
    print(f"  False Positives (FP): mean={fps[fn].mean():.2f}, med={fps[fn].median():.2f}, min={fps[fn].min():.2f}, max={fps[fn].max():.2f}")
    print(f"  True Positives (TP): mean={tps[fn].mean():.2f}, med={tps[fn].median():.2f}, min={tps[fn].min():.2f}, max={tps[fn].max():.2f}")

print("\n=== Detailed Look at the False Positive Rows ===")
cols = ["station_id", "timestamp", "final_label", "R", "RI", "R_30", "R_60", "y_proba"]
print(fps[cols].to_string())

print("\n=== Scaled Model Equation ===")
clf = pipe.named_steps["clf"]
scaler = pipe.named_steps["scaler"]
print(f"Log-odds = {clf.intercept_[0]:.4f}")
for fn, c, m, s in zip(feature_names, clf.coef_[0], scaler.mean_, scaler.scale_):
    print(f"  + {c:.4f} * (({fn} - {m:.2f}) / {s:.2f})  --> unscaled slope: {c/s:.5f}")
