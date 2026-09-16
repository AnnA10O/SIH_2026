"""
leakage_audit.py — PS-26077 Cloudburst Nowcaster
Reusable checks to catch the leak classes already found in this pipeline
(spatial_contrast = R * L_score; R_30/R_60 collapsing to same-day rain_mm_day)
and to stop them recurring as the feature set is rebuilt.

Usage (see __main__ for the CLI):

    python leakage_audit.py \
        --src-dirs src scripts \
        --exclude-dirs archive \
        --feature-parquet outputs/feature_matrix.parquet \
        --split-npz outputs/split_assignments.npz \
        --time-col timestamp --label-col final_label --score-col L_score \
        --raw-signal-col R --raw-signal-threshold 100.0 --model-csi 0.4853

Every check prints PASS/FAIL/SKIP and the script exits non-zero if anything
FAILs, so it can gate a "final" training run or sit in CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

RESULTS: list[tuple[str, str, str]] = []  # (check_name, status, detail)


def _record(name: str, ok: Optional[bool], detail: str) -> None:
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((name, status, detail))
    msg = f"[{status}] {name}: {detail}"
    print(msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))


# ─── 1. Static: feature assignments that reference the label/score ──────────
# Generic version of the exact bug found: `df["x"] = ... L_score ...`
# Catches any future column built from something with "label"/"score"/"target"
# in the name, not just spatial_contrast specifically.

LEAK_NAME_PATTERN = re.compile(r"(?i)\b(l_score|final_label|label|target)\b")
ASSIGN_PATTERN = re.compile(r'df(?:_\w+)?\[["\'][\w]+["\']\]\s*=\s*(.+)')


def check_label_derived_features(py_files: Iterable[Path]) -> None:
    hits = []
    for f in py_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = ASSIGN_PATTERN.search(line)
            if m and LEAK_NAME_PATTERN.search(m.group(1)):
                hits.append(f"{f}:{lineno}: {line.strip()}")
    ok = len(hits) == 0
    detail = "no feature assignment references a label/score column" if ok else (
        f"{len(hits)} suspicious assignment(s):\n    " + "\n    ".join(hits)
    )
    _record("label_derived_features", ok, detail)


# ─── 2. Static: rolling windows without an explicit left-closed / lag ───────
# Heuristic grep for `.rolling(` calls missing `closed="left"` (or `closed='left'`)
# on the same line. Not a full AST check — flags for manual review, doesn't
# prove correctness either way.

ROLLING_PATTERN = re.compile(r"\.rolling\(")
LEFT_CLOSED_PATTERN = re.compile(r"closed\s*=\s*[\"']left[\"']")


def check_rolling_window_lag(py_files: Iterable[Path]) -> None:
    flagged = []
    for f in py_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ROLLING_PATTERN.search(line) and not LEFT_CLOSED_PATTERN.search(line):
                flagged.append(f"{f}:{lineno}: {line.strip()}")
    ok = len(flagged) == 0
    detail = "every .rolling( call is closed='left'" if ok else (
        f"{len(flagged)} rolling window(s) without closed='left' — verify each "
        f"one excludes the current row before trusting it as a precursor feature:\n    "
        + "\n    ".join(flagged)
    )
    _record("rolling_window_lag", ok, detail)


# ─── 3. Static: legacy classical model kept out of the active pipeline ──────

BANNED_PATTERNS = [
    re.compile(r"\bLogisticRegression\b"),
    re.compile(r"\bsklearn\.linear_model\b"),
    re.compile(r"\bfrom\s+archive\b"),
    re.compile(r"\bimport\s+archive\b"),
    re.compile(r"\bcalibrated_nowcast_model\b"),
    re.compile(r"\bMODEL_PATH\b"),  # the classical-model path constant in config.py
]


def check_no_legacy_model_imports(py_files: Iterable[Path]) -> None:
    hits = []
    for f in py_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in BANNED_PATTERNS:
                if pat.search(line):
                    hits.append(f"{f}:{lineno}: {line.strip()}")
    ok = len(hits) == 0
    detail = "no active file imports the legacy classical model or archive/" if ok else (
        f"{len(hits)} reference(s) to the legacy baseline in active code:\n    "
        + "\n    ".join(hits)
    )
    _record("no_legacy_model_imports", ok, detail)


# ─── 4. Data: feature/label correlation audit ───────────────────────────────

def check_feature_label_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
    score_col: Optional[str],
    threshold: float = 0.90,
) -> None:
    if score_col is None or score_col not in df.columns:
        _record("feature_label_correlation", None, f"score column '{score_col}' not in dataframe")
        return
    flagged = {}
    for col in feature_cols:
        if col not in df.columns or col == score_col:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        corr = df[col].corr(df[score_col])
        if pd.notna(corr) and abs(corr) >= threshold:
            flagged[col] = round(float(corr), 4)
    ok = len(flagged) == 0
    detail = f"no feature exceeds |corr|>={threshold} with '{score_col}'" if ok else (
        f"{flagged} exceed |corr|>={threshold} with '{score_col}' — treat as label-derived"
    )
    _record("feature_label_correlation", ok, detail)


# ─── 5. Data: trivial-baseline sanity check ─────────────────────────────────
# If "predict positive iff raw_signal >= threshold" gets close to the model's
# CSI, the model isn't adding skill over restating the labeling rule.

def compute_csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def check_trivial_baseline(
    df: pd.DataFrame,
    raw_signal_col: str,
    label_col: str,
    positive_labels: set[str],
    threshold_value: float,
    model_csi: Optional[float],
    margin: float = 0.05,
) -> None:
    if raw_signal_col not in df.columns or label_col not in df.columns:
        _record("trivial_baseline", None, f"missing '{raw_signal_col}' or '{label_col}'")
        return
    y_true = df[label_col].isin(positive_labels).astype(int).values
    y_pred = (df[raw_signal_col].fillna(-np.inf).values >= threshold_value).astype(int)
    trivial_csi = compute_csi(y_true, y_pred)
    if model_csi is None:
        _record("trivial_baseline", None, f"trivial CSI={trivial_csi:.4f}; pass --model-csi to compare")
        return
    ok = model_csi > trivial_csi + margin
    detail = f"trivial ({raw_signal_col}>={threshold_value}) CSI={trivial_csi:.4f} vs model CSI={model_csi:.4f}"
    _record("trivial_baseline", ok, detail)


# ─── 6. Data: split integrity (index alignment) ─────────────────────────────

def check_split_integrity(current_index: pd.Index, split_npz_path: Path) -> None:
    if not split_npz_path.exists():
        _record("split_integrity", None, f"{split_npz_path} not found")
        return
    data = np.load(split_npz_path, allow_pickle=True)
    train_idx = set(data["train_idx"].tolist())
    val_idx = set(data["val_idx"].tolist())
    test_idx = set(data["test_idx"].tolist())
    current = set(current_index.tolist())

    overlaps = (train_idx & val_idx) | (train_idx & test_idx) | (val_idx & test_idx)
    union = train_idx | val_idx | test_idx
    missing_from_current = union - current
    extra_in_current = current - union

    ok = not overlaps and not missing_from_current and not extra_in_current
    detail = (
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"overlaps={len(overlaps)} missing_from_df={len(missing_from_current)} "
        f"extra_in_df={len(extra_in_current)}"
    )
    _record("split_integrity", ok, detail)


# ─── 7. Data: chronological ordering across the split ───────────────────────

def check_chronological_order(
    df: pd.DataFrame,
    time_col: str,
    split_npz_path: Path,
    buffer_hours: float = 12.0,
) -> None:
    if not split_npz_path.exists() or time_col not in df.columns:
        _record("chronological_order", None, f"{split_npz_path} or '{time_col}' not available")
        return
    data = np.load(split_npz_path, allow_pickle=True)
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    times = pd.to_datetime(df[time_col])
    train_max = times.loc[times.index.intersection(train_idx)].max()
    test_min = times.loc[times.index.intersection(test_idx)].min()
    gap_hours = (test_min - train_max).total_seconds() / 3600.0
    ok = gap_hours >= buffer_hours
    detail = f"train_max={train_max}, test_min={test_min}, gap={gap_hours:.1f}h (need >= {buffer_hours}h)"
    _record("chronological_order", ok, detail)


# ─── 8. Optional: label-shuffle sanity check (needs a trainer callback) ─────
# Wire this to train_neural_nowcaster_v2.py's train/predict entrypoint. Not
# run by default because it needs a real training loop; call it explicitly.

def check_label_shuffle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_and_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_repeats: int = 3,
    tolerance: float = 0.05,
) -> None:
    """train_and_predict(X_train, y_train, X_test) -> predicted probabilities on X_test."""
    prevalence = float(np.mean(y_test))
    rng = np.random.default_rng(42)
    shuffled_csis = []
    for _ in range(n_repeats):
        y_shuf = rng.permutation(y_train)
        probs = train_and_predict(X_train, y_shuf, X_test)
        preds = (probs >= 0.5).astype(int)
        shuffled_csis.append(compute_csi(y_test, preds))
    max_shuffled = max(shuffled_csis)
    ok = max_shuffled <= prevalence + tolerance
    detail = f"shuffled-label CSIs={['%.4f' % c for c in shuffled_csis]}, prevalence={prevalence:.4f}"
    _record("label_shuffle", ok, detail)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _collect_py_files(src_dirs: list[Path], exclude_dirs: list[str]) -> list[Path]:
    files = []
    for d in src_dirs:
        for f in d.rglob("*.py"):
            if any(part in exclude_dirs for part in f.parts):
                continue
            files.append(f)
    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dirs", nargs="+", type=Path, default=[Path("src"), Path("scripts")])
    p.add_argument("--exclude-dirs", nargs="+", default=["archive"])
    p.add_argument("--feature-parquet", type=Path, default=None)
    p.add_argument("--split-npz", type=Path, default=None)
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--label-col", default="final_label")
    p.add_argument("--score-col", default="L_score")
    p.add_argument("--raw-signal-col", default="R")
    p.add_argument("--raw-signal-threshold", type=float, default=100.0)
    p.add_argument("--model-csi", type=float, default=None)
    p.add_argument("--corr-threshold", type=float, default=0.90)
    p.add_argument("--buffer-hours", type=float, default=12.0)
    args = p.parse_args()

    py_files = _collect_py_files(args.src_dirs, args.exclude_dirs)
    print(f"Scanning {len(py_files)} Python file(s) under {args.src_dirs} (excluding {args.exclude_dirs})\n")

    check_label_derived_features(py_files)
    check_rolling_window_lag(py_files)
    check_no_legacy_model_imports(py_files)

    if args.feature_parquet and args.feature_parquet.exists():
        df = pd.read_parquet(args.feature_parquet)
        feature_cols = [c for c in df.columns if c not in {args.label_col, args.score_col, args.time_col}]
        check_feature_label_correlation(df, feature_cols, args.score_col, args.corr_threshold)
        positive_labels = {"CONFIRMED_CLOUDBURST", "CANDIDATE_CLOUDBURST"}
        check_trivial_baseline(
            df, args.raw_signal_col, args.label_col, positive_labels,
            args.raw_signal_threshold, args.model_csi,
        )
        if args.split_npz:
            check_split_integrity(df.index, args.split_npz)
            check_chronological_order(df, args.time_col, args.split_npz, args.buffer_hours)
    else:
        for name in ("feature_label_correlation", "trivial_baseline", "split_integrity", "chronological_order"):
            _record(name, None, "no --feature-parquet provided")

    print("\n--- Summary ---")
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    for name, status, _ in RESULTS:
        print(f"  {status:4s}  {name}")
    if failed:
        print(f"\n{len(failed)} check(s) FAILED.")
        return 1
    
    # Write sentinel file for the inference server
    out_dir = Path("d:/SIH/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".audit_passed").touch()

    print("\nAll runnable checks passed. Sentinel file .audit_passed created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
