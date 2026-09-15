# Pitch Deck Slide Draft: SOTA Benchmarking & The Two-Tier Architecture
## Slide Title: Comparative Benchmark: Where We Stand vs. State of the Art

---

### [SLIDE MOCKUP / LAYOUT]

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  AI CLOUDBURST NOWCASTING (PS 26077) — PEER-REVIEWED BENCHMARK COMPARISON                              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                        │
│   SYSTEM / MODEL         VENUE / SOURCE      PRIMARY TASK & INPUTS         CSI (EXTREME)  FAR / MISSED │
│  ────────────────────────────────────────────────────────────────────────────────────────────────────  │
│   HRRR (Operational)     NOAA / NCEP         3km NWP Physics (Gridded)     0.04 (Das 2024) High miss    │
│   ISRO NETRA             MOSDAC (ISRO)       Satellite top-down (district) Architecture  High FAR      │
│                                               — radar-sparse mountain bias  context only               │
│   DGMR                   Nature (2021)       Radar GAN (1km Gridded)       Supplementary  FAR ~ 0.50   │
│                                               — expert preference study     (not main CSI)             │
│   NowcastNet (Eval)      npj Clim Atmos(2024)Radar Physics-AI (Das et al.) 0.30 @ 16mm/h  FAR ~ 0.40   │
│  ────────────────────────────────────────────────────────────────────────────────────────────────────  │
│   THIS WORK (Tier 1)     PS 26077 (MoES)     Task B: 1D-CNN + BiLSTM       0.415 (Held-Out) POD: 88.8% | FAR: 56.1% │
│   [Deep Neural Nowcaster]                    (230,604 samples, 98:1 ratio) 0.932 (LOEO-CV)   POD: 98.6% | FAR: 5.6%  │
│                                               Prior-shift explanation: King & Zeng (2001)              │
│   THIS WORK (Tier 2)     PS 26077 (MoES)     Task A: Spatial Confirmation  Architecture:  Post-trigger │
│   [Network Cross-Check]                      Neighbor L-Score gate          L-Score gate   — pending   │
│                                               (independent signal needed)   evaluation    held-out eval│
│                                                                                                        │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  CRITICAL SCIENTIFIC CAVEAT: POINT-CLUSTER VERIFICATION VS. GRIDDED 2D DISPLACEMENT                    │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ * "Contextualizing in-situ nowcasting alongside published radar benchmarks":                     │  │
│  │   • In radar nowcasting, Das et al. (npj Climate and Atmospheric Science, 2024) report a median  │  │
│  │     CSI of 0.30 at 16 mm/h for NowcastNet on 2D radar fields. A gridded storm displaced by just   │  │
│  │     3 km incurs a DOUBLE PENALTY (1 false alarm + 1 miss), mechanically lowering spatial CSI.     │  │
│  │   • Point-scale in-situ sensing evaluates the local gauge directly, exempt from spatial          │  │
│  │     displacement penalties, but must confront severe class imbalance (98:1 negative ratio).      │  │
│  │   • On our 24-year multi-region held-out test split (230,604 samples across 6 Himalayan basins), │  │
│  │     Tier 1 1D-CNN + BiLSTM achieves CSI = 0.415 (POD = 88.8%) under untouched 98:1 imbalance.    │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                        │
│  MEASURED HARDWARE EDGE BENEFIT:                                                                        │
│  • Grid Deep Models: Require multi-GPU cloud servers (unviable for remote Himalayan hillsides).        │
│  • Our Edge Architecture: Neuromorphic SNN LIF Gate measured across 66,556 real MOSDAC readings:       │
│    -> 91.39% Telemetry Transmission Reduction | 91.60% Modeled Radio Power Savings                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Slide Speaker Notes (Word-for-Word for Pitching)

1. **Opening & Literature Context**:
   > *"Judges, before discussing accuracy, we want to establish how severe weather is evaluated in modern literature. On continuous 2D radar grids, recent independent evaluations—such as Das et al. (npj Climate and Atmospheric Science, 2024)—report a median CSI of 0.30 at 16 mm/h for advanced models like NowcastNet, illustrating how challenging extreme convective forecasting is across different sensing regimes."*

2. **The Structural Distinction Between Sensing Modes**:
   > *"Gridded radar models predict continuous fields where even a slight 3 km spatial displacement incurs a double penalty (one false alarm plus one miss). In mountain valleys where radar beam blockage is widespread, we deliberately formulate our system as two complementary engineering tiers:*
   > 
   > **Tier 1 (1D-CNN + BiLSTM Neural Nowcasting Model — Task B)**: Evaluates in-situ rain acceleration and precursor dynamics directly at the station gauge using temporal 1D convolution and Bidirectional LSTM recurrence.
   > • Under untouched operational conditions (quarantined 24-year test split of 230,604 samples under a natural 98:1 class imbalance), Tier 1 achieves CSI = 0.415 with POD = 88.8% and FAR = 56.1% (capturing 2,067 out of 2,329 unseen cloudburst events).
   > • Under event-balanced Leave-One-Event-Out cross-validation (10:1 ratio across 60 storm folds), it demonstrates 98.6% storm recall with CSI = 0.932 and FAR = 5.6%. This gap reflects the classic rare-events prior-shift (King & Zeng, 2001): a decision boundary calibrated on a 9% prevalence naturally yields higher false alarms when deployed into an uncurated 1% natural prior.
   >
   > **Tier 2 (Spatial Confirmation Gate — Task A)**: Tier 2 queries live neighboring stations to compute the spatial localization score (L-score = (R_core - R_bg) / R_core). NOTE: Because the ground-truth labels were themselves assigned using L-score thresholds (L < 0.40 → negative; L ≥ 0.40 → positive), re-applying the same L-score at inference constitutes circular evaluation. A genuinely independent Tier-2 signal — such as multi-sensor rain-rate consensus, live radar tile cross-check, or temporal rate-of-rise comparison — is required before a defensible held-out filter rate can be reported."*

4. **The SNN Edge Measurement**:
   > *"Finally, unlike heavy cloud models, our edge gate is a Neuromorphic Spiking Neural Network. We benchmarked it on 66,556 real MOSDAC AWS readings: it stays dormant 91.39% of the time, slashing radio energy consumption by 91.60%."*
