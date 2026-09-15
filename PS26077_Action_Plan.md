# PS 26077 — Action Plan: Prototype → Validated Product
*AI Hyper-Local Cloudburst & Thunderstorm Early Warning System*

**Legend:** 🔴 Critical (blocks any real claim of validity) · 🟡 Important (blocks production-grade trust) · 🟢 Nice-to-have (strengthens rigor/completeness)

---

## Phase 0 — Immediate Fixes (days, no new data needed)

These are bugs and integrity issues that cost nothing to fix and currently undercut every other claim in the report.

- [ ] 🔴 **Fix the thunderstorm trigger bug** — `simulator.py`: the `[T]` key calls `sim.trigger_cloudburst()` instead of a real thunderstorm path. Wire it to actually invoke `ThunderstormSNNGate` (Gate B).
- [ ] 🔴 **Remove hardcoded credentials** — `mosdac_api/config.json` currently has a plaintext username/password. Move to environment variables or a secrets manager before this touches any shared repo or demo machine.
- [ ] 🔴 **Fix the MOSDAC dataset ID** — currently set to `3RIMG_L2B_SST` (sea surface temperature, irrelevant). Change to `K1VHR_L2B_QPE` or `3RIMG_L2B_CTT`, whichever the spatial fusion module actually expects.
- [ ] 🔴 **Stop reporting LOEO-CV numbers as headline metrics anywhere public-facing.** The 0.93 CSI is a balanced-subsample artifact; the honest number is the 0.35–0.42 holdout CSI. Every slide, README, and pitch should lead with holdout performance only, with LOEO-CV shown (if at all) explicitly labeled as a diagnostic, not a performance claim.
- [ ] 🟡 **Fix the "Active Ground Cluster" placeholder** in `training_report.md`'s region_scores.csv — replace with actual positive-event counts per region.
- [ ] 🟡 **Add HEAVY_RAIN / MODERATE_RAIN tier counts** to the training report's label table — currently the ~419,000 unlabeled rows (45.9% of data) are lumped together with no breakdown.

---

## Phase 1 — Data Completion (weeks 1–4)

Every synthetic or empty data source is currently a ceiling on validity. This phase replaces placeholders with real inputs.

- [ ] 🔴 **Acquire real DEM data** (SRTM 30m or CartoDEM 30m, both free via Bhoonidhi/ISRO or USGS Earth Explorer) and replace the synthetic V-shaped valley profile (`compute_bed_profile()` in `pinn_swe.py`). This is the cheapest, highest-leverage fix in the whole plan — there's no reason left to use a synthetic terrain.
- [ ] 🔴 **Download real satellite QPE data at scale** using the now-fixed `mosdac_api/mdapi.py` client — target the June 2013 Kedarnath monsoon window at minimum, then expand to all 6 confirmed disaster-registry events plus a random negative sample for balance.
- [ ] 🔴 **Wire real satellite tiles into `spatial_fusion.py`** — `fuse_with_satellite()` currently always receives `satellite_grid=None`. Parse the downloaded HDF5 QPE files and pass them in; measure whether CTT/CTCR features actually move CSI (a null result here is fine and worth reporting honestly).
- [ ] 🟡 **Download ERA5 reanalysis** via `data/download_era5.py` (CDS API — script exists, was never run) for CAPE, pressure, wind, specific humidity. This is what Gate B (`ThunderstormSNNGate`) needs; right now those inputs are randomly simulated in the demo.
- [ ] 🟡 **Replace the daily→hourly rainfall proxy.** `phase_d_training.py` currently sets `rain_mm_hr = rain_mm_day`, which systematically underestimates instantaneous intensity against the 100mm/hr cloudburst threshold. Source real hourly AWS data (MOSDAC in-situ network) for at least the 6 target regions to retrain on true hourly rates.
- [ ] 🟢 **Add the Chamoli 2021 glacier-burst event** to the disaster registry overrides — currently missing from the 6 confirmed events.
- [ ] 🟢 **Use real IMERG/QPE rainfall as PINN forcing** instead of the Gaussian bell function, once satellite data is flowing.

---

## Phase 2 — Model Retraining & Honest Reporting (weeks 2–6, parallel to Phase 1)

- [ ] 🔴 **Retrain the CNN+BiLSTM to convergence** — currently 5 epochs (~1,930 gradient steps) on 787,974 rows. Run 50–100 epochs with proper early stopping on the held-out validation split, and report the real converged holdout metrics (not just the "5-epoch improvement over baseline" framing currently used).
- [ ] 🔴 **Re-run the full 70/15/15 event-grouped evaluation** once satellite, ERA5, and hourly-rain features are integrated. Check whether the previously-dropped IWV/satellite features (dropped for >70% NaN) are now usable, and whether they materially improve POD/FAR/CSI.
- [ ] 🟡 **Get the PINN calibrated against a real event.** Run the SWE solver for Kedarnath 2013 with the new DEM and satellite-derived rainfall, and compare peak simulated flood depth against documented/published flood levels for that event. Until this happens, keep describing the PINN as "physics-consistent demonstration," not "validated hydrodynamic model" (your report already uses this framing — keep it until earned otherwise).
- [ ] 🟢 **Optional: empirically test the hand-calibrated SNN gates against a BPTT-trained version** on labeled spike sequences from the IMD parquet data — not because the current design is wrong (it's the correct neuromorphic paradigm), but as a rigor check on whether domain calibration is leaving performance on the table.

---

## Phase 3 — Real-World Validation (this monsoon season onward)

This is the phase most SIH projects skip entirely, and it's the one that actually separates "prototype" from "product."

- [ ] 🔴 **Run a shadow-mode trial for one full monsoon season.** Let the retrained pipeline make real-time predictions on live data without pushing any public alert. Log every prediction, timestamp, and confidence score, then score against what actually happened. This is the closest thing to operational proof you can generate without institutional backing.
- [ ] 🔴 **Backtest against events that postdate your training window** (2024–2026 monsoons) rather than only the historically-labeled 2000–2023 set. A model that only proves itself on data it was tuned against isn't validated.
- [ ] 🟡 **Get independent domain review of the labeling methodology.** The L-score threshold (≥0.70 confirmed, ≥0.40 candidate) and the disaster-registry overrides are currently self-defined. Have an NCMRWF/IMD nowcasting scientist or an academic hydrometeorologist sanity-check the thresholds before treating them as ground truth.
- [ ] 🟡 **Benchmark against IMD's own nowcast bulletins and a naive persistence baseline**, not only against ISRO NETRA's published district-level stats. "Better resolution than NETRA" and "better than what a forecaster already produces" are different claims — you currently only have evidence for the first.
- [ ] 🟢 **Publish the honest ablation results** (with-satellite vs without, with-ERA5 vs without) even where the answer is "no significant improvement" — this is the kind of rigor that builds credibility with technical reviewers.

---

## Phase 4 — Product, Security & Institutional Path (ongoing)

- [ ] 🔴 **Decide the actual delivery model before building more UI.** India already has a pan-India, CAP-compliant alert delivery system — **SACHET** (NDMA/C-DOT), which integrates IMD, CWC, and INCOIS as Alert Generating Agencies and already reaches citizens via SMS/app/cell broadcast in 12+ languages. The realistic product path is becoming a CAP-compliant data source feeding into SACHET/IMD, not building a second standalone alerting app that has to solve last-mile distribution from scratch.
- [ ] 🟡 **Set up proper secrets management** (not just removing the hardcoded password — a real secret store/rotation policy) before any pilot deployment.
- [ ] 🟡 **Review MOSDAC API terms of use for production-scale/continuous polling** — the current client is built for research-scale batch downloads; a live product needs a sustainable, compliant access pattern.
- [ ] 🟡 **Draft a false-alarm response protocol** — at FAR ≈ 0.55, roughly half of alerts would currently be false positives. Any pilot needs an explicit, agreed-upon threshold and escalation policy with whichever authority receives the alerts, so false alarms don't erode trust before the model improves.
- [ ] 🟢 **Prepare a one-page technical dossier** (architecture, honest metrics, data provenance, known limitations) formatted for a state disaster management authority (Uttarakhand SDMA / Assam SDMA) or MoES/NCMRWF conversation — this is what actually opens the institutional door, more than a polished demo does.

---

## Full Limitations Ledger

*Every limitation named in the original report, mapped to where it's addressed above — so nothing gets dropped.*

| # | Limitation | Severity | Where fixed |
|---|---|---|---|
| 1 | CNN+BiLSTM trained only 5 epochs | 🔴 | Phase 2 |
| 2 | PINN uses synthetic valley, no real DEM | 🔴 | Phase 1 |
| 3 | Only 1 satellite file; CTT/CTCR are NaN | 🔴 | Phase 1 |
| 4 | ERA5 never downloaded (Gate B features simulated) | 🟡 | Phase 1 |
| 5 | PINN never validated against observed floods | 🟡 | Phase 2 |
| 6 | Daily rainfall used as hourly proxy | 🟡 | Phase 1 |
| 7 | Thunderstorm pipeline clones cloudburst (`[T]` bug) | 🔴 | Phase 0 |
| 8 | LOEO-CV inflated vs. real holdout | 🔴 | Phase 0 |
| 9 | "Active Ground Cluster" placeholder in region scores | 🟡 | Phase 0 |
| 10 | Spatial fusion never receives real satellite tiles | 🔴 | Phase 1 |
| 11 | Hardcoded plaintext MOSDAC credentials | 🔴 | Phase 0 |
| 12 | Wrong MOSDAC dataset ID configured (SST vs QPE) | 🔴 | Phase 0 |
| 13 | Chamoli 2021 glacier burst missing from registry | 🟢 | Phase 1 |
| 14 | PINN forced with Gaussian bell, not real rainfall | 🟢 | Phase 1 |
| 15 | Labeling methodology (L-score) not externally reviewed | 🟡 | Phase 3 |
| 16 | No benchmark against IMD nowcasts / persistence baseline | 🟡 | Phase 3 |
| 17 | No real-time / shadow-mode validation performed | 🔴 | Phase 3 |
| 18 | No institutional integration path defined | 🔴 | Phase 4 |
| 19 | No production-grade secrets management | 🟡 | Phase 4 |
| 20 | No false-alarm response protocol | 🟡 | Phase 4 |

---

*Suggested sequencing: run Phase 0 this week regardless of anything else. Phases 1 and 2 can run in parallel over the following month. Phase 3 needs a live monsoon window, so start it as early as data allows. Phase 4 conversations (SDMA/MoES outreach) can start in parallel with Phase 3 — institutions take time to respond, so don't wait for perfect metrics to start that conversation.*
