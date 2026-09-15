import pandas as pd
import numpy as np
from pathlib import Path
import glob
import time
import sys

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.satellite_reader import SatelliteReader

DATA_RAW = ROOT / "data" / "raw"
DATA_DIR = DATA_RAW / "imd_rain"
SATELLITE_DIR = DATA_RAW / "satellite"
OUTPUT_DIR = ROOT / "data" / "processed" / "satellite_merged"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_h5_file(dataset_prefix: str, date_str: str) -> Path:
    """Find the H5 file for a given dataset and date."""
    # Prefix can be generic like HEM, we check all eras
    for era_prefix in ["K1VHR_L2B_", "3DIMG_L2B_", "3RIMG_L2B_"]:
        # Fix Kalpana-1 HEM -> QPE mapping
        search_dataset = f"{era_prefix}{dataset_prefix}"
        if era_prefix == "K1VHR_L2B_" and dataset_prefix == "HEM":
            search_dataset = "K1VHR_L2B_QPE"
            
        date_dir = SATELLITE_DIR / search_dataset / date_str
        if date_dir.exists():
            files = list(date_dir.glob("*.h5"))
            if files:
                return files[0] # Grab the first (and only) file
    return None

def merge_features_for_region(parquet_path: Path):
    print(f"\n--- Processing {parquet_path.name} ---")
    df = pd.read_parquet(parquet_path)
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print(f"  -> Skipping (missing required spatial columns)")
        return pd.DataFrame()
        
    time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    if time_col not in df.columns:
        print(f"  -> Skipping (missing time column)")
        return pd.DataFrame()
    
    # Initialize new columns
    df['ctt_mean'] = np.nan
    df['hem_mean'] = np.nan
    df['olr_mean'] = np.nan
    df['uth_mean'] = np.nan
    df['uth_nearest_px_km'] = np.nan
    df['uth_valid'] = 0.0

    # Get unique dates in this region's dataframe
    df['date_str'] = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d')
    unique_dates = df['date_str'].unique()
    
    processed_rows = 0
    
    for date_str in unique_dates:
        # Check if we have satellite data for this date
        hem_file = find_h5_file("HEM", date_str)
        olr_file = find_h5_file("OLR", date_str)
        uth_file = find_h5_file("UTH", date_str)
        ctp_file = find_h5_file("CTP", date_str) # CTP contains CTT
        
        if not hem_file and not olr_file and not uth_file and not ctp_file:
            continue
            
        # Get all rows for this date
        date_mask = df['date_str'] == date_str
        rows = df[date_mask]
        
        # Load grids once per date
        sat_data = {}
        try:
            if hem_file: sat_data['HEM'] = SatelliteReader.read_hem(hem_file)
            if olr_file: sat_data['OLR'] = SatelliteReader.read_olr(olr_file)
            if uth_file: sat_data['UTH'] = SatelliteReader.read_uth(uth_file)
            if ctp_file: sat_data['CTP'] = SatelliteReader.read_ctt(ctp_file) # Returns 'ctt'
        except Exception as e:
            print(f"  [WARN] Failed to read H5 for {date_str}: {e}")
            continue
            
        target_lats = rows['lat'].values
        target_lons = rows['lon'].values
        row_indices = rows.index
        
        # 1. CTP
        if 'CTP' in sat_data:
            res = SatelliteReader.extract_points_knn(
                sat_data['CTP']['lats'], sat_data['CTP']['lons'], sat_data['CTP']['ctt'], 
                target_lats, target_lons, k=1, max_radius_km=15.0
            )
            df.loc[row_indices, 'ctt_mean'] = res['val']
            
        # 2. HEM
        if 'HEM' in sat_data:
            val_key = 'hem' if 'hem' in sat_data['HEM'] else 'qpe'
            res = SatelliteReader.extract_points_knn(
                sat_data['HEM']['lats'], sat_data['HEM']['lons'], sat_data['HEM'][val_key], 
                target_lats, target_lons, k=1, max_radius_km=15.0
            )
            df.loc[row_indices, 'hem_mean'] = res['val']
            
        # 3. OLR
        if 'OLR' in sat_data:
            res = SatelliteReader.extract_points_knn(
                sat_data['OLR']['lats'], sat_data['OLR']['lons'], sat_data['OLR']['olr'], 
                target_lats, target_lons, k=1, max_radius_km=15.0
            )
            df.loc[row_indices, 'olr_mean'] = res['val']
            
        # 4. UTH (BallTree Vectorized KNN)
        if 'UTH' in sat_data:
            res = SatelliteReader.extract_points_knn(
                sat_data['UTH']['lats'], sat_data['UTH']['lons'], sat_data['UTH']['uth'], 
                target_lats, target_lons, k=4, max_radius_km=250.0
            )
            # Assign values back using exact indices
            df.loc[row_indices, 'uth_mean'] = res['val']
            df.loc[row_indices, 'uth_nearest_px_km'] = res['dist_km']
            df.loc[row_indices, 'uth_valid'] = res['valid']
            
        processed_rows += len(rows)
            
    # Drop rows without satellite data (keep only dates we downloaded)
    # We require at least one satellite feature to be present to keep the row
    sat_cols = ['ctt_mean', 'hem_mean', 'olr_mean', 'uth_mean']
    # If all of these are missing, it means NO satellite matched. We keep rows where at least one matched.
    df_sat = df.dropna(subset=sat_cols, how='all').copy()
    
    # Impute remaining missing values (e.g. CTP missing pre-2018) with column means
    # NOTE: uth_mean is purposefully NOT imputed with the mean here, because it uses the Value+Mask MNAR strategy post-scaling.
    # The MNAR zero-fill will happen in phase_d_training.py.
    sat_cols_to_impute = ['ctt_mean', 'hem_mean', 'olr_mean']
    for col in sat_cols_to_impute:
        if df_sat[col].isna().any():
            df_sat[col] = df_sat[col].fillna(df_sat[col].mean())
            
    # For UTH nearest distance, we can fill missing with 250km (the cap) so it doesn't break scaler
    if df_sat['uth_nearest_px_km'].isna().any():
        df_sat['uth_nearest_px_km'] = df_sat['uth_nearest_px_km'].fillna(250.0)
            
    df_sat = df_sat.drop(columns=['date_str'])
    
    out_path = OUTPUT_DIR / parquet_path.name
    df_sat.to_parquet(out_path)
    print(f"  -> Saved {len(df_sat)} rows with satellite features to {out_path.name}")
    return df_sat

if __name__ == "__main__":
    print("================================================================")
    print(" SATELLITE FEATURE MERGER ")
    print("================================================================")
    
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    total_rows = 0
    total_pos = 0
    
    for p in parquet_files:
        if p.is_file():
            df_sat = merge_features_for_region(p)
            total_rows += len(df_sat)
            label_col = "rain_label" if "rain_label" in df_sat.columns else "final_label"
            if not df_sat.empty and label_col in df_sat.columns:
                total_pos += len(df_sat[df_sat[label_col] != "NORMAL"])
            
    print("\n================================================================")
    print(f"MERGE COMPLETE.")
    print(f"Total Rows: {total_rows}")
    print(f"Total Positive (Cloudburst) Rows: {total_pos}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("================================================================")
