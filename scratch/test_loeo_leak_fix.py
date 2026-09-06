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

# Global validation calibration to get opt_threshold
train_data, val_data, test_data = event_grouped_split(
    X, y, weights, event_groups, train_frac=0.70, val_frac=0.15, random_state=42
)
cand_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=0.10, max_iter=1000, solver="lbfgs", random_state=42))
])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cand_pipe.fit(train_data["X"], train_data["y"], clf__sample_weight=train_data["w"])
val_p = cand_pipe.predict_proba(val_data["X"])[:, 1]
opt_thresh = find_optimal_threshold(val_data["y"], val_p)
print(f"Global Val-Calibrated Threshold: tau = {opt_thresh:.3f}")

# Now run LOEO-CV under three regimes:
# Regime 1: OLD Leaky (tuned on y_test)
# Regime 2: Train-tuned (tuned on y_train per fold)
# Regime 3: Fixed Global Threshold (tau = 0.310)

event_ids = event_groups[y == 1].unique()
X_arr = X.values
y_arr = y.values
w_arr = weights.values
g_arr = event_groups.values

print(f"Running LOEO-CV across {len(event_ids)} positive event clusters...")

old_metrics = []
train_tuned_metrics = []
fixed_tau_metrics = []

all_yt = []
all_yp = []

for held_out_gid in event_ids:
    test_mask  = (g_arr == held_out_gid)
    train_mask = ~test_mask

    if train_mask.sum() < 10 or test_mask.sum() == 0:
        continue

    X_train, y_train, w_train = X_arr[train_mask], y_arr[train_mask], w_arr[train_mask]
    X_test,  y_test            = X_arr[test_mask],  y_arr[test_mask]

    if y_train.sum() == 0 or (1 - y_train).sum() == 0:
        continue

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.10, max_iter=1000, solver="lbfgs", random_state=42))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train, clf__sample_weight=w_train)

    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    y_proba_train = pipe.predict_proba(X_train)[:, 1]

    all_yt.extend(y_test.tolist())
    all_yp.extend(y_proba_test.tolist())

    # Regime 1: Leaky (old)
    if len(set(y_test)) > 1:
        t_leaky = find_optimal_threshold(y_test, y_proba_test)
        old_m = compute_metrics(y_test, None, y_proba_test, threshold=t_leaky)
        old_metrics.append(old_m)

    # Regime 2: Train-Tuned per fold
    t_train = find_optimal_threshold(y_train, y_proba_train)
    m_train = compute_metrics(y_test, None, y_proba_test, threshold=t_train)
    train_tuned_metrics.append(m_train)

    # Regime 3: Fixed Global Val-Calibrated tau = 0.310
    m_fixed = compute_metrics(y_test, None, y_proba_test, threshold=opt_thresh)
    fixed_tau_metrics.append(m_fixed)

def summarize(m_list, name):
    pods = [m["POD"] for m in m_list if not np.isnan(m["POD"])]
    fars = [m["FAR"] for m in m_list if not np.isnan(m["FAR"])]
    csis = [m["CSI"] for m in m_list if not np.isnan(m["CSI"])]
    praucs = [m["PR_AUC"] for m in m_list if not np.isnan(m["PR_AUC"])]
    print(f"\n--- {name} (across {len(m_list)} evaluated folds) ---")
    print(f"  Mean POD   : {np.mean(pods):.4f} (median {np.median(pods):.4f})")
    print(f"  Mean FAR   : {np.mean(fars):.4f} (median {np.median(fars):.4f})")
    print(f"  Mean CSI   : {np.mean(csis):.4f} (median {np.median(csis):.4f})")
    print(f"  Mean PR-AUC: {np.mean(praucs):.4f} (median {np.median(praucs):.4f})")

summarize(old_metrics, "REGIME 1: OLD LEAKY (tuned on y_test)")
summarize(train_tuned_metrics, "REGIME 2: HONEST TRAIN-TUNED (per-fold y_train)")
summarize(fixed_tau_metrics, f"REGIME 3: FIXED GLOBAL VAL-CALIBRATED (tau = {opt_thresh:.3f})")

# Also pooled out-of-fold metrics across ALL samples
all_yt = np.array(all_yt)
all_yp = np.array(all_yp)
pooled_m = compute_metrics(all_yt, None, all_yp, threshold=opt_thresh)
print(f"\n--- POOLED OUT-OF-FOLD METRICS (N={len(all_yt)}, tau={opt_thresh:.3f}) ---")
print(f"  Pooled POD   : {pooled_m['POD']:.4f} (TP={pooled_m['TP']}, FN={pooled_m['FN']})")
print(f"  Pooled FAR   : {pooled_m['FAR']:.4f} (FP={pooled_m['FP']}, TN={pooled_m['TN']})")
print(f"  Pooled CSI   : {pooled_m['CSI']:.4f}")
print(f"  Pooled PR-AUC: {pooled_m['PR_AUC']:.4f}")
