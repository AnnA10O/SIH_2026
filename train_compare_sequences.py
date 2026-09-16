"""
train_compare_sequences.py — PS-26077 Cloudburst Nowcaster
Trains SequenceBiLSTM and TinySeqTransformer on the IDENTICAL real-sequence
data, identical split, identical loss/weighting, then refuses to print a
"final" comparison unless leakage_audit.py's runnable checks all pass.

This is the retrain step that was explicitly missing from the earlier
implementation plan ("Missing step: retrain CNN+BiLSTM on the new
feature_matrix.parquet with the fixed split... Without that retrain, the
audit script has nothing to check against.") — this script IS that step,
generalized to both architectures.

Usage:
    python train_compare_sequences.py \
        --sequences-npz outputs/sequences.npz --sequences-meta outputs/sequences_meta.parquet \
        --epochs 40 --batch-size 128

Exit code is non-zero if the post-hoc leakage audit fails — treat that as
"do not cite these numbers," not as a script bug to silence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for leakage_audit.py alongside this file

from sequence_models import SequenceBiLSTM, TinySeqTransformer  # noqa: E402
import leakage_audit as audit  # noqa: E402  (the file already in this repo)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Metrics: try the project's own implementation first, fall back locally ──
def _get_metric_fns():
    try:
        from archive.deprecated_ml_baselines.phase_d_training import compute_metrics, find_optimal_threshold  # noqa
        print("[metrics] using src.phase_d_training.compute_metrics / find_optimal_threshold")
        return compute_metrics, find_optimal_threshold
    except Exception as e:
        print(f"[metrics] could not import project metrics ({e}); using local fallback")

        def compute_metrics(y_true, _unused, y_prob, threshold=0.5):
            y_pred = (np.asarray(y_prob) >= threshold).astype(int)
            y_true = np.asarray(y_true).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            denom = tp + fp + fn
            csi = tp / denom if denom else 0.0
            pod = tp / (tp + fn) if (tp + fn) else 0.0
            far = fp / (tp + fp) if (tp + fp) else 0.0
            return {"CSI": csi, "POD": pod, "FAR": far, "TP": tp, "FP": fp, "FN": fn, "TN": tn}

        def find_optimal_threshold(y_true, y_prob, grid=np.arange(0.05, 1.0, 0.01)):
            best_t, best_csi = 0.5, -1.0
            for t in grid:
                m = compute_metrics(y_true, None, y_prob, threshold=t)
                if m["CSI"] > best_csi:
                    best_csi, best_t = m["CSI"], t
            return best_t

        return compute_metrics, find_optimal_threshold


compute_metrics, find_optimal_threshold = _get_metric_fns()


# ─── Split: try the project's event_grouped_split, fall back to a simple one ─
def _event_grouped_split(n, groups, train_frac=0.70, val_frac=0.15, random_state=42):
    try:
        from archive.deprecated_ml_baselines.phase_d_training import event_grouped_split
        idx_df = pd.DataFrame({"__pos__": np.arange(n)})
        y_dummy = pd.Series(np.zeros(n), index=idx_df.index)
        w_dummy = pd.Series(np.ones(n), index=idx_df.index)
        train_d, val_d, test_d = event_grouped_split(
            idx_df, y_dummy, w_dummy, groups, train_frac=train_frac, val_frac=val_frac, random_state=random_state
        )
        train_idx = np.asarray(train_d["X"]).reshape(-1).astype(int)
        val_idx = np.asarray(val_d["X"]).reshape(-1).astype(int)
        test_idx = np.asarray(test_d["X"]).reshape(-1).astype(int)
        print("[split] using src.phase_d_training.event_grouped_split")
        return train_idx, val_idx, test_idx
    except Exception as e:
        print(f"[split] could not use project split ({e}); using local group-aware fallback "
              f"(NOTE: less battle-tested than the project's own function — verify no overlap below)")
        rng = np.random.default_rng(random_state)
        bg_idx = np.where(groups == -1)[0]
        rng.shuffle(bg_idx)
        n_bg_train = int(len(bg_idx) * train_frac)
        n_bg_val = int(len(bg_idx) * val_frac)
        train_idx = list(bg_idx[:n_bg_train])
        val_idx = list(bg_idx[n_bg_train:n_bg_train + n_bg_val])
        test_idx = list(bg_idx[n_bg_train + n_bg_val:])
        
        event_groups = groups[groups != -1]
        uniq_groups = np.unique(event_groups)
        rng.shuffle(uniq_groups)
        n_train = int(len(uniq_groups) * train_frac)
        n_val = int(len(uniq_groups) * val_frac)
        train_g = set(uniq_groups[:n_train])
        val_g = set(uniq_groups[n_train:n_train + n_val])
        test_g = set(uniq_groups[n_train + n_val:])
        
        train_idx.extend(np.where(np.isin(groups, list(train_g)))[0])
        val_idx.extend(np.where(np.isin(groups, list(val_g)))[0])
        test_idx.extend(np.where(np.isin(groups, list(test_g)))[0])
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _make_loader(X_seq, X_static, y, w, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X_seq.astype(np.float32)),
        torch.from_numpy(X_static.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(w.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for bx_seq, bx_static, by, bw in loader:
            bx_seq, bx_static, by, bw = bx_seq.to(DEVICE), bx_static.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            logits = model(bx_seq, bx_static)
            loss = criterion(logits, by)
            total_loss += (loss * bw).mean().item() * len(by)
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(by.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(preds), np.array(targets)


def _train_one_model(model, train_loader, val_loader, epochs, lr, weight_decay, y_train, tag):
    model = model.to(DEVICE)
    raw_pw = (len(y_train) - y_train.sum()) / max(1.0, y_train.sum())
    capped_pw = min(raw_pw, 50.0)
    print(f"[{tag}] Training on {len(y_train)} rows | Positives={y_train.sum()} | pos_weight capped: {raw_pw:.1f} -> {capped_pw:.1f}")
    pos_weight = torch.tensor([capped_pw], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_csi, best_thresh, best_state = -1.0, 0.5, None
    for epoch in range(1, epochs + 1):
        model.train()
        for bx_seq, bx_static, by, bw in train_loader:
            bx_seq, bx_static, by, bw = bx_seq.to(DEVICE), bx_static.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx_seq, bx_static), by)
            (loss * bw).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        val_loss, val_preds, val_targets = _evaluate(model, val_loader, criterion)
        thresh = find_optimal_threshold(val_targets, val_preds)
        m = compute_metrics(val_targets, None, val_preds, threshold=thresh)
        print(f"[{tag}] epoch {epoch:2d} | val_loss={val_loss:.4f} | val_CSI={m['CSI']:.4f} "
              f"| val_POD={m['POD']:.4f} | val_FAR={m['FAR']:.4f}")
        if m["CSI"] > best_csi:
            best_csi, best_thresh = m["CSI"], thresh
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_thresh


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences-npz", type=Path, default=ROOT / "outputs/sequences.npz")
    p.add_argument("--sequences-meta", type=Path, default=ROOT / "outputs/sequences_meta.parquet")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--split-npz-out", type=Path, default=ROOT / "outputs/sequence_split_assignments.npz")
    p.add_argument("--far-target", type=float, default=0.40)
    args = p.parse_args()

    data = np.load(args.sequences_npz, allow_pickle=True)
    X_seq, X_static, y, w, groups = data["X_seq"], data["static_ctx"], data["y"], data["w"], data["groups"]
    core_feats, static_feats = list(data["core_feats"]), list(data["static_feats"])
    meta = pd.read_parquet(args.sequences_meta)
    print(f"Loaded {X_seq.shape[0]} sequences | seq_len={X_seq.shape[1]} | core_feats={core_feats} "
          f"| static_feats={static_feats}")

    train_idx, val_idx, test_idx = _event_grouped_split(len(y), groups)
    print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Save split assignments for the audit's split_integrity / chronological_order checks.
    args.split_npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.split_npz_out, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler_seq = StandardScaler()
    scaler_static = StandardScaler()

    n_train, seq_len, n_core = X_seq[train_idx].shape
    X_seq_train_flat = imputer.fit_transform(X_seq[train_idx].reshape(n_train, -1).astype(np.float64))
    X_seq_train_flat = scaler_seq.fit_transform(X_seq_train_flat)
    X_seq_train = X_seq_train_flat.reshape(n_train, seq_len, n_core).astype(np.float32)

    def _transform_seq(idx):
        n = len(idx)
        flat = imputer.transform(X_seq[idx].reshape(n, -1).astype(np.float64))
        flat = scaler_seq.transform(flat)
        return flat.reshape(n, seq_len, n_core).astype(np.float32)

    X_seq_val, X_seq_test = _transform_seq(val_idx), _transform_seq(test_idx)

    static_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_static_train = scaler_static.fit_transform(static_imputer.fit_transform(X_static[train_idx]))
    X_static_val = scaler_static.transform(static_imputer.transform(X_static[val_idx]))
    X_static_test = scaler_static.transform(static_imputer.transform(X_static[test_idx]))

    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    w_train, w_val, w_test = w[train_idx], w[val_idx], w[test_idx]

    train_loader = _make_loader(X_seq_train, X_static_train, y_train, w_train, args.batch_size, True)
    val_loader = _make_loader(X_seq_val, X_static_val, y_val, w_val, args.batch_size, False)
    test_loader = _make_loader(X_seq_test, X_static_test, y_test, w_test, args.batch_size, False)

    results = {}
    for name, model in [
        ("SequenceBiLSTM", SequenceBiLSTM(core_feats=n_core, static_feats=X_static.shape[1])),
        ("TinySeqTransformer", TinySeqTransformer(core_feats=n_core, static_feats=X_static.shape[1], max_len=seq_len)),
    ]:
        print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
        trained, thresh = _train_one_model(
            model, train_loader, val_loader, args.epochs, args.lr, args.weight_decay, y_train, name
        )
        raw_pw = (len(y_train) - y_train.sum()) / max(1.0, y_train.sum())
        pos_weight = torch.tensor([min(raw_pw, 50.0)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        _, test_preds, test_targets = _evaluate(trained, test_loader, criterion)
        m = compute_metrics(test_targets, None, test_preds, threshold=thresh)
        results[name] = m
        print(f"[{name}] TEST  CSI={m['CSI']:.4f}  POD={m['POD']:.4f}  FAR={m['FAR']:.4f}  "
              f"(tau={thresh:.3f})  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")

    print(f"\n{'='*60}\nUNAUDITED COMPARISON (do not cite yet)\n{'='*60}")
    print(f"{'Model':<20} | {'CSI':<8} | {'POD':<8} | {'FAR':<8}")
    for name, m in results.items():
        flag = "  <-- meets FAR target" if m["FAR"] < args.far_target else ""
        print(f"{name:<20} | {m['CSI']:<8.4f} | {m['POD']:<8.4f} | {m['FAR']:<8.4f}{flag}")

    # ─── Gate everything above through leakage_audit.py before calling it final ──
    print(f"\n{'='*60}\nRUNNING LEAKAGE AUDIT (gating the comparison above)\n{'='*60}")
    src_dirs = [ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent]
    py_files = audit._collect_py_files(src_dirs, exclude_dirs=["archive", ".venv", "venv", "site-packages", "__pycache__", "nowcast"])
    audit.check_label_derived_features(py_files)
    audit.check_no_legacy_model_imports(py_files)

    if "R_last" in meta.columns and "final_label" in meta.columns:
        best_model_csi = max(m["CSI"] for m in results.values())
        positive_labels = {"CONFIRMED_CLOUDBURST", "CANDIDATE_CLOUDBURST"}
        audit.check_trivial_baseline(
            meta, "R_last", "final_label", positive_labels,
            threshold_value=100.0, model_csi=best_model_csi,
        )
    else:
        audit._record("trivial_baseline", None, "meta missing 'R_last' or 'final_label' — cannot check")

    if "L_score" in meta.columns:
        feature_cols = [c for c in meta.columns if c not in {"final_label", "L_score", "timestamp"}]
        audit.check_feature_label_correlation(meta, feature_cols, "L_score")
    else:
        audit._record("feature_label_correlation", None, "no L_score in sequence metadata (expected — good sign)")

    meta_reindexed = meta.reset_index(drop=True)
    audit.check_split_integrity(meta_reindexed.index[np.concatenate([train_idx, val_idx, test_idx])], args.split_npz_out)
    if "timestamp" in meta.columns:
        audit.check_chronological_order(meta_reindexed, "timestamp", args.split_npz_out)

    failed = [r for r in audit.RESULTS if r[1] == "FAIL"]
    print("\n--- Audit Summary ---")
    for name, status, _ in audit.RESULTS:
        print(f"  {status:4s}  {name}")

    if failed:
        print(f"\n{len(failed)} AUDIT CHECK(S) FAILED. The comparison above is NOT cleared for reporting.")
        return 1

    print("\nAll runnable audit checks passed. Comparison above may be cited, "
          "with the caveat that this script has not been independently re-run by a reviewer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
