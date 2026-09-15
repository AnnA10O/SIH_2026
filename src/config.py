"""
Cloudburst Nowcasting Pipeline — Configuration
PS 26077: AI-Driven Hyper-Local Early Warning System (MoES / NCMRWF)
Data source: 100% MOSDAC (ISRO ecosystem)
"""

import os
from pathlib import Path

# ─── Project Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # d:/SIH/
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
SATELLITE_DIR = DATA_RAW / "satellite"
DEM_DIR = DATA_RAW / "dem"

for d in [DATA_RAW, DATA_PROCESSED, OUTPUTS, DATA_PROCESSED / "clean_stations",
          DATA_PROCESSED / "gagan_processed", SATELLITE_DIR, DEM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Source Data Files ──────────────────────────────────────────────────────────
GAGAN_IWV_FILE = DATA_RAW / "gagan_iwv_v1.txt"
GAGAN_STATIONS_FILE = DATA_RAW / "Stations.txt"
AWS_CSV = ROOT / "ASSAM_ALL_2013-02-01_2014-03-15_Sep2026_177056.csv"

# ─── Fixed Temporal Window (GAGAN IWV archive) ─────────────────────────────────
WINDOW_START = "2013-03-01"
WINDOW_END   = "2014-02-28"
MONSOON_MONTHS = [6, 7, 8, 9]          # June – September
MONSOON_LABEL = "June–September"

# ─── Phase A — Region Discovery ─────────────────────────────────────────────────
GAGAN_MIN_COVERAGE_PCT   = 85.0        # min % of 30-min samples present
GAGAN_SEASONAL_RATIO_MIN = 1.5         # monsoon mean IWV / winter mean IWV
AWS_CLUSTER_RADIUS_KM    = 200.0       # max km to search for AWS around GAGAN station (captures regional cluster)
AWS_MIN_STATIONS         = 1           # min passing AWS stations per candidate region
AWS_MIN_SPACING_KM       = 5.0         # min inter-station distance (avoid co-located sensors)
AWS_MAX_SPACING_KM       = 200.0       # max inter-station distance (scale for spacing score)

# Region scoring weights (must sum to 1.0)
SCORE_WEIGHT_IWV_COVERAGE  = 0.25
SCORE_WEIGHT_N_AWS_PASS    = 0.35
SCORE_WEIGHT_MONSOON_COV   = 0.25
SCORE_WEIGHT_SPACING       = 0.15

# ─── Phase B — Data Quality Gate ────────────────────────────────────────────────
STUCK_ROLLING_DAYS        = 30          # window for rolling stuck-sensor check (days)
STUCK_IDENTICAL_THRESHOLD = 0.95        # fraction identical readings to flag stuck
STUCK_MIN_UNIQUE_VALUES   = 5           # min unique non-null values across whole record
DEAD_ZERO_FRACTION        = 0.90        # fraction zeros = dead sensor trigger
DEAD_MAX_MONSOON_MM       = 10.0        # max ever recorded during monsoon for dead sensor
CROSSTALK_CORR_THRESHOLD  = 0.50        # |r| > this with sunshine/temp = cross-talk flag
SPIKE_MAX_HOURLY_MM       = 300.0       # above this → unphysical spike, clip and log
MONSOON_COVERAGE_MIN_PCT  = 20.0        # min valid data % during Jun–Sep (relaxed for sporadic reporting stations)

# Sentinel fill values that indicate missing/dead data
SENTINEL_VALUES = [9999.0, 9999.9, -9999.0, -999.0, 999.0]

# Rain column name in AWS CSV
AWS_RAIN_COL   = "RAIN_FALL(mm)"
AWS_TEMP_COL   = "AIR_TEMP(\xb0C)"
AWS_RH_COL     = "HUMIDITY(%)"
AWS_SUN_COL    = "SUN_SHINE(hh:mm)"
AWS_PRES_COL   = "ATMO_PRESSURE(hpa)"
AWS_WIND_COL   = "WIND_SPEED(m/s)"
AWS_TS_COL_IST = "DATE(IST)"
AWS_TIME_COL   = "TIME(IST)"
AWS_LAT_COL    = "LATITUDE"
AWS_LON_COL    = "LONGITUDE"
AWS_STATION_COL = "@STATION_ID"

# ─── Phase C — Event Labeling ────────────────────────────────────────────────────
CLOUDBURST_THRESHOLD_MM    = 100.0      # IMD official cloudburst: ≥ 100 mm/hr
TIER2_THRESHOLD_MM         = 30.0       # sub-cloudburst intense trigger
NEIGHBOR_FRACTION_MAX      = 0.20       # neighbor must be < 20% of spike station to call localized
LIGHT_RAIN_THRESHOLD_MM    = 10.0       # below this = "not significantly raining"

# IWV precursor check
IWV_PRECURSOR_HOURS        = 6          # hours before event to check IWV accumulation
IWV_DRAWDOWN_HOURS         = 3          # hours after event to confirm drawdown
IWV_MIN_BUILDUP_MM         = 2.0        # minimum IWV rise counted as a precursor signal

# CAPE verification via Open-Meteo (external, verification only)
CAPE_INSTABILITY_THRESHOLD = 500.0      # J/kg — above this = unstable atmosphere
OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ─── Phase D — Model Training ────────────────────────────────────────────────────
# Positive class weights
WEIGHT_CONFIRMED_CLOUDBURST   = 1.0     # ground truth positive
WEIGHT_CANDIDATE_CLOUDBURST   = 0.5     # soft positive (lower certainty)
WEIGHT_WIDESPREAD_HEAVY_RAIN  = 1.0     # critical hard negative
WEIGHT_HEAVY_RAIN             = 0.3     # medium-difficulty negative (sub-cloudburst rain)
WEIGHT_MODERATE_RAIN          = 0.1     # typical monsoon ambient negative
WEIGHT_NORMAL                 = 0.05    # downsampled quiescent background negative

# Leave-One-Event-Out CV
LOEO_EVENT_BUFFER_HOURS = 12            # hrs before/after peak to group into same event fold

# L2 regularization sweep
L2_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

# Feature columns (built from MOSDAC data; satellite cols added when available)
# Feature columns (built from MOSDAC data; satellite cols added when available)
FEATURES_AWS = [
    # Layer 1: sub-daily intensity signals
    "R", "R_30", "R_60", "RI",
    # Layer 2a: antecedent moisture (soil saturation proxy)
    "rain_3day_accum",   # 3-day prior rainfall (mm) — closed="left" excludes today
    "rain_7day_accum",   # 7-day prior rainfall (mm)
    "rain_trend_7day",   # linear slope of 7-day window (mm/day) — +ve = moistening
    # Layer 2b: calendar + terrain (zero-cost, already in parquet)
    "doy",               # day of year (monsoon phase: onset ~155, withdrawal ~270)
    "month",             # integer month (coarser monsoon phase)
    "lat",               # station latitude (terrain type proxy)
    "lon",               # station longitude
    "spatial_contrast",  # R × L_score (convective organisation proxy)
]
FEATURES_IWV = ["IWV_now", "IWV_trend_3hr"]
# Satellite-derived daily features — produced by satellite_feature_engine.py
# Only available for disaster windows with downloaded HDF5 data (2016, 2019, 2021)
FEATURES_SAT = [
    "ctt_mean",       # Mean cloud top temperature (K) — lower = deeper convection
    "ctt_min",        # Minimum CTT (coldest cloud top) — peak convective intensity
    "ctt_cold_frac",  # Fraction of pixels < 233 K (< -40°C) — deep convective area
    "hem_mean",       # Mean satellite hydro-estimator rain rate (mm/hr)
    "hem_max",        # Peak satellite rain rate in window (mm/hr)
    "olr_mean",       # Mean outgoing longwave radiation (W/m²) — low OLR = deep cloud
    "uth_mean",          # Mean upper troposphere humidity (%)
    "uth_nearest_px_km", # Distance to closest valid UTH reading (anti-leak confound feature)
    "uth_valid",         # Binary 0/1 MNAR mask for UTH missing values
]

# Explicit staleness features matching the DataFusionBuffer ChannelState
FEATURES_STALENESS = [
    "R_staleness_s", "R_valid",
    "uth_staleness_s",
    "hem_staleness_s", "hem_valid"
]

FEATURES_TERRAIN = ["elevation"]        # added when DEM available

# ─── Disaster Event Windows ──────────────────────────────────────────────────────
# Bounding boxes (lat_min, lat_max, lon_min, lon_max) used for spatial clipping
# when extracting satellite features for each disaster event.
DISASTER_WINDOWS = {
    "Pithoragarh_Jul2016":  (28.5, 31.0, 79.5, 81.5),
    "Chamoli_Jul2016":      (29.5, 31.5, 78.5, 80.5),
    "Uttarkashi_Aug2019":   (30.0, 32.0, 77.5, 79.5),
    "Chamoli_Oct2021":      (29.5, 31.5, 78.5, 80.5),
    "Chamoli_Feb2021":      (29.5, 31.5, 78.5, 80.5),
}

# Satellite product directory (populated by MOSDAC downloader)
SATELLITE_DIR = DATA_RAW / "satellite"

# Label tiers (numeric for training)
LABEL_MAP = {
    "NORMAL":                  0,
    "WIDESPREAD_HEAVY_RAIN":   1,
    "CANDIDATE_CLOUDBURST":    2,
    "CONFIRMED_CLOUDBURST":    3,
}
POSITIVE_LABELS = {"CONFIRMED_CLOUDBURST", "CANDIDATE_CLOUDBURST"}
HARD_NEGATIVE   = "WIDESPREAD_HEAVY_RAIN"

# Output files & Models
REGION_SCORES_CSV       = OUTPUTS / "region_scores.csv"
STATION_QUALITY_CSV     = OUTPUTS / "station_quality_report.csv"
CLOUDBURST_EVENTS_CSV   = OUTPUTS / "cloudburst_events.csv"
TRAINING_REPORT_MD      = OUTPUTS / "training_report.md"

MODELS_DIR              = ROOT / "models"
MODEL_PATH              = MODELS_DIR / "calibrated_nowcast_model.joblib"  # Baseline linear benchmark
MODEL_LOCKED_SHA256     = "25dc224652ceec8f5d56da83bbdab1cef7e1eec11d81cabc60e71fe48956905c"
NEURAL_MODEL_PT         = MODELS_DIR / "cloudburst_cnn_bilstm.pt"         # Production 1D-CNN + BiLSTM PyTorch model
NEURAL_MODEL_JSON       = MODELS_DIR / "cloudburst_cnn_bilstm_weights.json" # Production browser/edge weights

