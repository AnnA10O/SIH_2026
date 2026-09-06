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
from src.phase_d_training import event_grouped_split, find_optimal_threshold, compute_metrics, _assign_event_groups, build_feature_matrix

events_df = pd.read_csv(CLOUDBURST_EVENTS_CSV)

df = events_df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
if "rain_mm_day" in df.columns:
    df["rain_mm_hr"] = df["rain_mm_day"]
df = df.dropna(subset=["timestamp", "rain_mm_hr", "final_label"])

df["R"]    = df["rain_mm_hr"]
df = df.sort_values(["station_id", "timestamp"])
r_30_raw = df.groupby("station_id")["R"].transform(lambda s: s.rolling(2, min_periods=1).sum())
r_60_raw = df.groupby("station_id")["R"].transform(lambda s: s.rolling(2, min_periods=1).sum())
df["R_30"] = np.where(df["R"] > 0, r_30_raw, 0.0)
df["R_60"] = np.where(df["R"] > 0, r_60_raw, 0.0)
df["RI"]   = df.groupby("station_id")["R"].transform(lambda s: s.diff().fillna(0))

feature_names = ["R", "R_30", "R_60", "RI"]
X = df[feature_names].copy()

# Binary labels
y = df["final_label"].map({
    "CONFIRMED_CLOUDBURST":  1,
    "CANDIDATE_CLOUDBURST":  1,
    "WIDESPREAD_HEAVY_RAIN": 0,
    "NORMAL":                0,
}).fillna(0).astype(int)

weight_map = {
    "CONFIRMED_CLOUDBURST":  WEIGHT_CONFIRMED_CLOUDBURST,
    "CANDIDATE_CLOUDBURST":  WEIGHT_CANDIDATE_CLOUDBURST,
    "WIDESPREAD_HEAVY_RAIN": WEIGHT_WIDESPREAD_HEAVY_RAIN,
    "NORMAL":                WEIGHT_NORMAL,
}
weights = df["final_label"].map(weight_map).fillna(1.0)
event_groups = _assign_event_groups(df)

df["y_true"] = y.values
df["event_group"] = event_groups.values

# Reconstruct split exactly
train_data, val_data, test_data = event_grouped_split(
    X, y, weights, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
)

best_C = 0.10
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

# Extract False Negatives: y_true == 1 AND y_pred == 0
fns = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 0)]
tps = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 1)]

print("=" * 80)
print(f"AUDIT OF THE 5 FALSE NEGATIVES (MISSED EVENTS)")
print("=" * 80)
print(f"Optimal Threshold tau: {opt_thresh:.3f}")
print(f"Total True Positives (Caught): {len(tps)}")
print(f"Total False Negatives (Missed): {len(fns)}")

cols_to_print = [
    "station_id", "timestamp", "final_label", "R", "R_30", "R_60", "RI",
    "L_score", "y_proba"
]
avail_cols = [c for c in cols_to_print if c in fns.columns]

print("\n--- Detailed Records of the 5 Missed Events ---")
print(fns[avail_cols].to_string())

print("\n--- Comparison: Missed Events (FN) vs Caught Events (TP) ---")
for col in ["R", "R_30", "R_60", "RI", "L_score", "y_proba"]:
    if col in fns.columns:
        print(f"Feature: {col}")
        print(f"  Missed (FN, N=5) : mean={fns[col].mean():.2f}, med={fns[col].median():.2f}, min={fns[col].min():.2f}, max={fns[col].max():.2f}")
        print(f"  Caught (TP, N=95): mean={tps[col].mean():.2f}, med={tps[col].median():.2f}, min={tps[col].min():.2f}, max={tps[col].max():.2f}")

print("\n--- Learned Model Weights ---")
clf = pipe.named_steps["clf"]
scaler = pipe.named_steps["scaler"]
print(f"Intercept: {clf.intercept_[0]:.4f}")
for fn, c, m, s in zip(feature_names, clf.coef_[0], scaler.mean_, scaler.scale_):
    print(f"  {fn:10s}: coef={c:+.4f}, mean={m:.2f}, std={s:.2f}, unscaled_slope={c/s:+.5f}")
