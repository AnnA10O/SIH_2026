# 11 — False Alarm Reduction Plan: CNN+BiLSTM Precision Recovery & Multi-Layer FAR Mitigation Architecture

**PS 26077: AI Hyper-Local Cloudburst & Thunderstorm Early Warning System**
**Date:** September 13, 2026
**Companion to:** Document 08 — Action Plan (Ledger #1 & Improvement 1) and Document 09 — Sprint Validation & Audit
**Revision:** v2 — corrected after a targeted technical review and cross-checked against the uploaded Document 09. See Revision Log at the end.

---



---

## Executive Summary

The v2 retrain of the CNN+BiLSTM nowcaster (Ledger #1) is frequently described as having "increased false negatives." The evaluation data does not support this reading: misses fell by more than half. What actually increased is the false alarm rate, and — more importantly — the retrain's headline skill score (CSI) quietly regressed. This document (1) corrects the diagnosis with the exact numbers, now confirmed against Document 09's audited confusion matrix, (2) identifies the specific mechanism responsible and how much of the regression is a threshold artifact versus a genuine training defect, (3) proposes a three-layer active mitigation architecture (Focal Loss redesign → input feature enrichment → two-stage precision cascade) that extends Improvement 1 from Document 08 rather than replacing it, and (4) documents a fourth layer — multi-tier public alert corroboration — as future scope. Public alert delivery is human-supervised and handled outside this system; it is documented here for institutional completeness and handoff planning, not as an active implementation target.

---

## 1. Diagnostic Correction: What Actually Changed

| Metric | Baseline (5-epoch) | Retrained v2 (`pos_weight=103`) | Delta |
|:---|:---:|:---:|:---:|
| Misses (False Negatives) | 262 | 125 | −137 (−52.3%) |
| False Alarms (False Positives) | 2,638 | 2,681 | +43 (+1.6%) |
| POD | 88.75% | 94.11% | +5.36 pts |
| FAR | 56.15% | 57.30% | +1.15 pts |
| CSI | 0.4154 | 0.4159 | **+0.0005 (identical)** |
| PR-AUC | 0.5334 | 0.5142 | −0.0192 (regression) |
| ROC-AUC | 0.9940 | 0.9940 | 0.0000 (ranking capability unchanged) |

Confirmed test-set confusion matrix at τ=0.90 (Document 09):
**TP = 2,208 | FP = 3,278 | FN = 121 | TN = 224,997** (230,604 total, exact)

**Finding:** the retrain moved the model along the same precision/recall curve — it did not improve overall skill. Judged on CSI, the metric this project already treats as authoritative (Document 09), v2 is marginally *worse* than the 5-epoch baseline despite catching more events. The actionable defect is **false-alarm inflation**, not missed detections.

---

## 2. Root Cause: Why `pos_weight = 103` Inflates FAR — and Why τ Alone Can't Fully Fix It

`src/train_neural_nowcaster_v2.py` sets:

```python
pos_weight = torch.tensor([103.0]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This multiplies the loss on every positive example by the same constant regardless of whether the model already classifies it correctly — a blunt instrument at a ~104:1 imbalance ratio.

**Part of the apparent regression is a free threshold artifact, not purely a training defect.** Document 09's own τ-sweep, re-scoring the *same* v2 checkpoint at higher thresholds, recovers most of the baseline's precision characteristics with zero retraining:

| τ | POD | FAR | CSI |
|:---:|:---:|:---:|:---:|
| 0.90 (reported headline) | 94.80% | 59.75% | 0.3938 |
| 0.93 | 87.85% | 56.43% | 0.4109 |
| **0.94 (v2's own peak)** | **83.42%** | **54.98%** | **0.4132** |

At τ=0.93, v2's POD (87.85%) and FAR (56.43%) nearly match the 5-epoch baseline (88.75% / 56.15%). Comparing "v2 at τ=0.90" against "baseline at its default τ" is therefore partly a comparison of two different operating points on two different curves — not purely two models of different quality.

**But it isn't only a threshold artifact.** Even at its own best swept threshold (τ=0.94), v2's peak CSI of 0.4132 still falls short of the baseline's 0.4154 — while ROC-AUC is identical between the two models (0.9940 both). That means the *ranking* of storms by risk score is equally good after retraining, but the *spread of scores* near the decision boundary is slightly worse. That residual gap — small but present at every threshold in the sweep — is what a loss-function change should target, not a threshold change:

- **Changing τ** moves along a fixed curve produced by a fixed model.
- **Changing the loss function** reshapes the curve itself, by changing which examples the model spends gradient on during training, rather than just where the output is cut afterward.

Static weighting like `pos_weight=103` acts as a uniform global penalty that, as the weight grows large, tends to bias a network toward over-predicting the minority class (consistent with general findings on weighted cross-entropy — cf. Akkus et al., arXiv:1902.01977). Focal loss instead reweights *per example* based on how hard that example already is, which is the lever needed to close the residual CSI gap rather than just relocating along the existing Pareto frontier. Independent convective-nowcasting research flags the same underlying imbalance issue and identifies loss-function redesign — not threshold tuning — as the standard next step (Ahmad et al., QJRMS, 2026).

Static class weighting is the wrong tool at this imbalance ratio; a **difficulty-aware loss** is the correct next step, which is exactly what Document 08's Improvement 1 already proposes. This document formalizes why, and extends it.

---

## 3. Four-Layer Mitigation Architecture

The four layers are designed to be additive: each subsequent layer attacks whatever residual FAR the previous layer cannot resolve alone. Layers 1 and 2 fix the model itself; Layer 3 adds a post-classifier filter; Layer 4 moves the false-alarm burden off the classifier entirely.

---

### Layer 1 — Replace Static `pos_weight` with Focal Loss

Focal loss down-weights examples the model already classifies correctly and concentrates gradient on the ones it still gets wrong, which directly targets the "hard negative" rainstorms currently driving FAR up rather than penalizing all positives uniformly.

```python
# src/losses/focal_loss.py
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for extreme class imbalance (104:1 positive rate).
    Alpha balances the rare positive class; gamma focuses on hard examples.

    Args:
        logits:  Raw model output (pre-sigmoid), shape (N,)
        targets: Binary labels {0, 1}, shape (N,)
        alpha:   Positive class prior weighting factor. Sweep {0.25, 0.50, 0.75}
                 at fixed best-gamma to isolate its effect.
        gamma:   Focusing parameter. Sweep {2, 3, 4} first at alpha=0.25
                 to isolate its effect before cross-sweeping alpha.
    Returns:
        Scalar mean focal loss
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss.mean()
```

**Sweep grid:** sweep γ ∈ {2, 3, 4} with α fixed at 0.25 first to isolate γ's effect; then sweep α ∈ {0.25, 0.50, 0.75} at the best γ. 3 + 3 = 6 targeted runs (or 3×3 = 9 if full cross-sweep is feasible on the RTX 5050). Evaluate a joint **Focal + Dice** loss as a second variant (Wang et al., MDPI Atmosphere, 2026). Re-run on the existing 787,974 / 231,911 / 230,604 ISO-week × region split (Ledger #8) so results stay traceable to the v1/v2 headline numbers.

**τ recalibration (mandatory):** after each Focal Loss variant finishes training, repeat Document 09's τ-sweep methodology (τ ∈ {0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95}) on the **validation** split (231,911 rows), select the τ that maximises validation CSI, and score that single (model, τ) pair exactly once on the test split. Comparing a new loss function at τ=0.90 against `pos_weight=103` at τ=0.90 is only valid if 0.90 happens to be the new model's own optimum — it usually won't be, since focal loss produces a different score calibration.

**Decision gate:** if the best configuration across the full grid doesn't exceed CSI = 0.4132 — the τ=0.94 Pareto peak already available from the *current* v2 checkpoint with zero retraining (Document 09) — treat Layer 1 as informative but insufficient on its own, skip further loss-function space search, and move directly to Layer 2.

---

### Layer 2 — Fix the Input Signal, Not Just the Loss (Ledger #6 / Task 2)

**Critical Finding (The Daily Proxy Reality):** The entire historical 913,536-row dataset (including the 0.4258 baseline) relies on a daily proxy (`rain_mm_hr = rain_mm_day`). The ML model has never actually seen sub-hourly precipitation dynamics; it cannot distinguish a genuine 100 mm/hr convective burst from a gentle 100 mm/day soak. This removes the very information the model needs to discriminate classes.

**Data Acquisition Blocker & Strategy:**
We are abandoning the paid IMD Data Supply Portal requisition. A reconnaissance revealed 21 sovereign **ISRO AWS stations** in Uttarakhand (e.g., ISRO0138, ISRO0911) with HOURLY resolution covering our 2013-2019 disaster windows. The human team must manually export this data from the MOSDAC/ISRO web portal.

**Implementation Plan (PARKED):**
1. **Status**: Layer 2 modeling (including Dual-Head architecture) is strictly **PARKED** until the ISRO AWS data is acquired.
2. Once the human team drops the export into `data/raw/isro_uttarakhand_hourly_aws.csv`, we will merge it and compute true intensity features (R, R₃₀, R₆₀, RI).
3. We will then run a clean single-variable MLP ablation (proxy features vs true hourly features) to definitively validate the sub-hourly signal before resuming architecture search.
---

### Layer 3 — Two-Stage Precision Cascade (Train/Test Split Corrected)

Rather than asking one classifier to be simultaneously high-recall and high-precision, split the decision into two stages in series, trained and evaluated on **non-overlapping** data.

**Stage 1 (unchanged):** The current high-recall CNN+BiLSTM at τ=0.90 — keep it exactly as-is, biased toward catching everything. Document 09's audited test-set confusion matrix (TP=2,208, FP=3,278, FN=121, TN=224,997) is the reference point Stage 2 needs to improve on — it is **not** a source of Stage 2 training data.

**Stage 2 (new):**
1. Score the trained Stage 1 checkpoint on the untouched **231,911-row validation split** (already reserved for early stopping under Ledger #1) to produce `outputs/val_predictions.npz`.
2. Build Stage 2's training pool from Stage 1's **validation-set** false positives (hard negatives) against its validation-set true positives. The pool will be in the low thousands at comparable class balance to the test matrix (~40% positive share), though exact counts must be generated from the validation split, not assumed equal to the test-set numbers.
3. **Model family:** with a training pool in the low thousands, a second CNN+BiLSTM would almost certainly overfit. Use **logistic regression** or a **gradient-boosted tree** (XGBoost / LightGBM) over tabular features: Stage-1 probability score, rain intensity, terrain slope, basin ID, and the Layer 2 intensity features (R, R₃₀, R₆₀, RI) if available. Fast to train, interpretable, and well-suited to small imbalanced tabular pools.

**Evaluation:** run the full two-stage cascade end-to-end **exactly once** on the untouched 230,604-row test split. If Stage 2 requires any further adjustment after seeing that result, the test split is spent — hold out a fresh slice for the next evaluation cycle rather than reusing it (which would convert the reported CSI into an optimistic estimate).

```
INPUT: Meteorological Feature Vector
         |
  [Stage 1: CNN+BiLSTM (tau=0.90)]   <- High Recall: catches everything
         |
   Stage 1 positive predictions only
   (validation set: score -> val_predictions.npz)
         |
  [Stage 2: LR / GBM Precision Refiner]  <- High Precision: filters false alarms
   src/cascade_refiner.py
   Train: validation-set FPs (hard negatives) vs validation-set TPs
   Evaluate: ONE pass on the 230,604-row test split
         |
   Cascade-filtered output (target: CSI > 0.4132)
```

This pattern is well established for extreme rare-event problems specifically because reweighting alone tends to fail at this level of imbalance. A recent rare cardiac-event benchmark (ECG-based VT/VF detection, 3.6:1 imbalance — milder than this project's ~104:1) found that standard reweighted classifiers suffer "rare-class collapse" and that a two-stage cascade outperformed inverse-frequency reweighting, cost-sensitive training, and LDAM loss outright, cutting missed events by 37% (medRxiv, 2026).

---

### Layer 4 — Multi-Tier Public Alert Corroboration *(Future Scope — Out of Current Implementation)*

> [!NOTE]
> Public alert delivery is human-supervised and managed outside this system. The architecture below is documented for institutional handoff and future integration planning — it is **not** an active implementation target in the current workstream. Ledger items #18 and #20 remain in Phase 4 pending institutional partnership with SDMA / NDMA.

The most robust long-term fix for a life-safety system is ensuring no single classifier output directly triggers a public alarm without independent corroboration. Document 10's planned CAP alert ladder describes a cascade built from independent physical sensors:

```
LEVEL 1 (Yellow / Watch) — INTERNAL ONLY
  Trigger:  P(CB) >= 0.30  OR  Single SNN gate (Gate A or Gate B)
  Action:   5-minute AWS polling escalation. No external alert.
  FAR cost: Negligible (internal data collection only)

      | Condition met -> escalate

LEVEL 2 (Orange / Warning) — SDMA DASHBOARD
  Trigger:  P(CB) >= 0.60  AND  Dual-station spatial consensus
            (two independent AWS stations within a basin both flagging)
  Fallback: Where a basin cannot structurally satisfy dual-station consensus
            (see station-density audit below), single-station reading
            plus satellite QPE convective-core corroboration.
  Action:   Alert dispatched to State Disaster Management Authority.
  FAR cost: Moderate (SDMA operators review before action)

      | Condition met -> escalate

LEVEL 3 (Red / Emergency) — PUBLIC CAP BROADCAST
  Trigger:  P(CB) >= 0.80  AND  Satellite QPE convective core corroborated
            (Kalpana-1 / INSAT-3DR R >= 100 mm/hr confirmed)
  Action:   CAP v1.2 XML pushed to NDMA SACHET -> public SMS/siren.
  FAR cost: High (public trust erosion; target near-zero)
```

**Station-density audit (required before Level 2 finalisation):** Document 09's verified basin positive counts (Rudraprayag: 52, Chamoli: 107, Tehri: 123 — versus Assam: 4,893) confirm coverage is highly uneven. Some high-risk Uttarakhand catchments may lack two independent stations near the actual risk zone. **Note on data labels:** The current dataset deduplication assigns border stations to basins alphabetically (e.g., Chamoli swallows Rudraprayag border stations), creating label artifacts. Before finalising the dual-station consensus rule, audit the AWS network station count and inter-station distances per basin, and verify basin boundary assignments spatially rather than by file glob order. Where a basin cannot structurally satisfy the dual-station requirement, define an explicit fallback so Level 2 does not silently become unreachable in exactly the basins most likely to need it.

Running the current high-recall model freely at Level 1, and requiring independent physical corroboration before Level 3, decouples "catch everything" from "only alarm the public when confident" — which is what Layers 1–3 are otherwise trying to approximate statistically. This layer must be treated as a **permanent backstop regardless of how much Layers 1–3 improve the raw classifier.**

---

## 4. Prioritised Execution Order

**Active implementation (Layers 1–3):**

| Order | Action | Ledger Link | Effort | Rationale |
|:---:|:---|:---:|:---:|:---|
| **1** | Swap `pos_weight=103` → Focal Loss (α × γ grid); mandatory τ re-sweep on validation per variant; decision-gated against τ=0.94 Pareto peak | Improvement 1 | **Low** | Pure loss-function change; no new data; decision gate prevents wasted runs |
| **2** | Merge hourly AWS, recompute R / R₃₀ / R₆₀ / RI; retrain with best Layer 1 loss; τ-recalibrate before comparing | Ledger #6 / Task 2 | **Medium** | Attacks the root feature ambiguity, not just the symptom |
| **3** | Generate `val_predictions.npz`; build Stage 2 (LR / GBM) from validation-set FPs; evaluate cascade once on test split | New (Layer 3) | **Medium** | Split-safe cascade — test set stays untouched until final single-pass scoring |

**Future scope (Layer 4 — human-supervised, out of current workstream):**

| Order | Action | Ledger Link | Effort | Rationale |
|:---:|:---|:---:|:---:|:---|
| **4** | Station-density audit per basin; formalise Level 1/2/3 CAP corroboration + sparse-basin fallback in `src/cap_alert_generator.py` | Ledger #18/#20 / Task 5 | **High** | Institutional handoff design — public alert chain is human-supervised; implement when SDMA/NDMA partnership is active |

---

## 5. Metric Governance

"Accuracy" is not a meaningful target at a ~0.96% positive rate — a model that always predicts "no cloudburst" scores above 99% accuracy while missing every event. This project already reports against CSI and the POD/FAR Pareto frontier (Document 09); that discipline must extend to this workstream.

**Governing rule for all future retrains:**

> Accept a retrain if and only if CSI improves on a **held-out split that has not been used for tuning.** POD improvements that come at the cost of CSI regression are not progress at this imbalance ratio. Any test split that has been used for model selection must be retired from the headline evaluation.

The v2 retrain audited in Section 1 (POD +6 pts, CSI −0.02) is the standing counterexample. It must not be characterised as a success in any internal or external reporting.

**Secondary guardrail:** FAR must not increase between versions even if CSI stays flat. A model with equivalent CSI but lower FAR is strictly preferable for a life-safety system.

---

## Revision Log

| Version | Date | Changes |
|:---:|:---:|:---|
| v1 | 2026-09-13 | Initial draft — four-layer architecture proposed. |
| v2 | 2026-09-13 | Corrected after targeted technical review, cross-checked against Document 09: confirmed exact confusion matrix (TP=2,208, FP=3,278, FN=121, TN=224,997); added PR-AUC / ROC-AUC to §1. Rewrote §2 root-cause argument around Document 09's own τ-sweep, removing an over-reached claim about a general "10:1 safe range" for weighted cross-entropy. Added α to Focal Loss sweep grid (Layer 1). Added mandatory τ re-optimisation step and null-result decision gate (Layer 1). Fixed train/test leakage in Layer 3 — Stage 2 now trains on the validation split; test split reserved for a single final cascade evaluation. Named model family for Stage 2 (logistic regression / GBM). Added station-density audit and sparse-basin fallback requirement to Layer 4 dual-station consensus rule. |
| v2.1 | 2026-09-14 | **Marked PROVISIONAL.** `load_imd_parquets()`'s 46,848 duplicate border rows were found to inflate the proxy rain rate 60x for 26,295 rows, contaminating the `StandardScaler` fit on the 913,536-row set every number in this document derives from. Added a status banner pending `rerun_baseline_post_dedup()` (`train_neural_nowcaster_v2.py --recheck-dedup`), which retrains the BCE baseline on the deduplicated 866,688-row set and gates on a 0.02 CSI-drift tolerance against the numbers below. Flagged an unreconciled split-size discrepancy (787,974/231,911/230,604 does not sum to either the pre- or post-dedup row count). No numeric claims in §1–§3 changed yet — that happens once the rerun's result is in. |

---

## References

- Akkus, Z. et al. *Technical Considerations for Semantic Segmentation in MRI using Convolutional Neural Networks.* arXiv:1902.01977.
- Ahmad, S. et al. *Nowcasting convective cores using deep learning.* Quarterly Journal of the Royal Meteorological Society, 2026. https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.70284
- Wang et al. *Impact of Optical Flow and Joint Loss on Nowcasting of Severe Convective Weather at Airports.* MDPI Atmosphere, 2026. https://www.mdpi.com/2073-4433/17/5/497
- Zhou, L. et al. *Physically Explainable Deep Learning for Convective Initiation Nowcasting Using GOES-16 Satellite Observations.* arXiv:2310.16015.
- Anonymous. *Rare-Class Collapse in ECG-Based Ventricular Tachycardia and Fibrillation Detection: A Systematic Benchmark of Class-Imbalance Mitigation from Reweighting to Cascade Classification.* medRxiv, 2026.
- Ko, J. et al. *Deep learning for precipitation nowcasting: A survey from the perspective of time series forecasting.* arXiv:2406.04867.
- Internal: [`09_sprint_validation_and_audit.md`](file:///d:/SIH/09_sprint_validation_and_audit.md) — source of the confirmed confusion matrix and τ-sweep table used in §§1–2.
