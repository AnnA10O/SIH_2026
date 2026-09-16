import sys
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.phase_d_training import load_imd_parquets, build_feature_matrix, compute_metrics, event_grouped_split

def run_assam_ablation():
    print("=" * 60)
    print("MEGA-CLUSTER ABLATION: ASSAM (Doubt 3)")
    print("=" * 60)
    
    events_df = load_imd_parquets()
    
    # Split the raw dataframe FIRST
    assam_df = events_df[events_df['region_name'] == 'Assam_Meghalaya'].copy()
    other_df = events_df[events_df['region_name'] != 'Assam_Meghalaya'].copy()
    
    print(f"Train Regions (Non-Assam): {len(other_df)} rows")
    print(f"Test Region (Assam): {len(assam_df)} rows")
    
    X_train, y_train, w_train, _, _ = build_feature_matrix(other_df)
    X_test, y_test, w_test, _, _ = build_feature_matrix(assam_df)
    
    print(f"Post-Feature Engineering Train: {len(X_train)} rows")
    print(f"Post-Feature Engineering Test: {len(X_test)} rows")
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42))
    ])
    
    print("Fitting model on Non-Assam regions...")
    pipe.fit(X_train, y_train, clf__sample_weight=w_train)
    
    print("Evaluating on Assam mega-cluster...")
    test_proba = pipe.predict_proba(X_test)[:, 1]
    
    test_m = compute_metrics(y_test, None, test_proba, threshold=0.40)
    
    print(f"\n── Assam Ablation Metrics ──")
    print(f"  POD (Detection Rate) : {test_m['POD']:.4f}")
    print(f"  FAR (False Alarm)    : {test_m['FAR']:.4f}")
    print(f"  CSI (Critical Success): {test_m['CSI']:.4f}")
    print(f"  PR-AUC               : {test_m['PR_AUC']:.4f}")
    
    # Save results to a report file
    with open("d:/SIH/scratch/ablation_results.txt", "w") as f:
        f.write("Assam Ablation Metrics:\n")
        f.write(f"CSI: {test_m['CSI']:.4f}\n")
        f.write(f"POD: {test_m['POD']:.4f}\n")
        f.write(f"FAR: {test_m['FAR']:.4f}\n")
    print("\nResults saved to d:/SIH/scratch/ablation_results.txt")

if __name__ == "__main__":
    run_assam_ablation()
