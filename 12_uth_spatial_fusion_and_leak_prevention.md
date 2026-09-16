# UTH Spatial Fusion and Confound-to-Feature Conversion (Phase 3)

This document outlines the updates to the Kalpana-1 Upper Tropospheric Humidity (UTH) spatial fusion pipeline and the corresponding structural changes to the neural network training pipeline.

## 1. The Cloud Masking Diagnostic (Confound-to-Feature)

A rigorous diagnostic was executed to compare Kalpana-1 UTH retrieval masking severity between known positive cloudburst events (Storm Days) and non-event scenes (Misc/Clear Days).

### Diagnostic Results

| Category | Avg Nearest Px | Avg KNN=4 Dist | % Failed 100km Cap |
|----------|----------------|----------------|--------------------|
| Positive (Storm) | 61.2 km | 84.3 km | **27.6%** |
| Random (Misc) | 46.1 km | 68.4 km | **16.8%** |

*Note: The "Random" category included all non-cloudburst days, meaning it captured standard rainy days. The true disparity between clear skies and active convective storms is likely even wider.*

**Finding:** Masking severity is a legitimate, physically-motivated precursor signal. Thick, tall convective clouds physically block IR water-vapor retrieval. High masking density (and thus longer interpolation distances) is a **genuine observable** of active deep convection.

**Action:** Rather than viewing this as a "label leak" to be hidden, we explicitly converted this confound into a feature. Exposing the retrieval distance to the network allows it to learn the association between heavy localized masking and convective intensity.

## 2. Robust Spatial Fusion: KNN=4 + IDW

To handle the highly intermittent spatial coverage of the Kalpana-1 UTH data without resorting to arbitrary binary drop-offs, the `SatelliteReader` module was updated with a new algorithm: `extract_point_value_knn`.

### Design Decisions:
- **Algorithm:** K-Nearest Neighbors ($k=4$) with Inverse Distance Weighting (IDW).
- **Distance Mapping:** The radius constraint was widened to an absolute upper bound of `250.0 km`. Instead of a hard 75-100km cutoff, we provide the continuous distance to the nearest valid pixel (`uth_nearest_px_km`), allowing the neural network to naturally learn the spatial decay function.
- **Observability:** For every extracted point, the pipeline now yields three specific columns:
  1. `uth_mean`: The IDW-interpolated humidity value.
  2. `uth_nearest_px_km`: The absolute physical distance to the nearest valid reading.
  3. `uth_valid`: A strict `0/1` binary indicator denoting if the extraction fell within the 250km limit.

## 3. Missing Not At Random (MNAR) Imputation Strategy

The intermittent nature of satellite retrievals during storms constitutes a classic **Missing Not At Random (MNAR)** scenario. Because deep moisture-saturated convective clouds are exactly what blocks the IR retrieval, the rows where `uth_mean` goes missing are disproportionately the rows with the most extreme true humidity.

### The Danger of Pre-Scaler Mean Imputation
If we mean-imputed the pre-scaled `uth_mean` column, we would systematically pull the true (likely very high) humidity values on storm days down toward the dataset average. This would manufacture a bias that cuts against the network learning from the strongest signal, on precisely the rows where the signal-to-noise ratio is worst.

### The Value + Mask Solution
To resolve this, the neural network training pipeline (`src/phase_d_training.py`) was restructured to employ a **Value + Mask pair strategy**:
- **The Mask:** The `uth_valid` binary indicator is explicitly fed into the network.
- **The Value:** We injected a `SimpleImputer(strategy="mean")` directly into the Scikit-Learn `Pipeline`, *before* the `StandardScaler`.

By filling missing `uth_mean` values with the exact training set mean immediately before the `StandardScaler` runs, those specific rows resolve to exactly `0.0` post-standardization. 
This ensures the fill carries **zero directional information**, leaving the network to learn entirely from the `uth_valid` mask and the continuous `uth_nearest_px_km` distance metric for those heavily-masked convective cells.
