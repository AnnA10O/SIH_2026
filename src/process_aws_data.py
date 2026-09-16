import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = Path("d:/SIH/data/raw")
OUT_FILE = Path("d:/SIH/data/processed/assam_aws_hourly.parquet")

def process_aws_data():
    csv_files = list(RAW_DIR.glob("ASSAM_ALL_*.csv"))
    if not csv_files:
        print("No AWS CSV files found.")
        return

    print(f"Loading {len(csv_files)} CSV files...")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, skipinitialspace=True)
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.columns = [c.strip() for c in full_df.columns]
    
    # Identify key columns, handling potential encoding issues in column names (like C)
    temp_col = [c for c in full_df.columns if 'AIR_TEMP' in c][0]
    press_col = 'ATMO_PRESSURE(hpa)'
    hum_col = 'HUMIDITY(%)'
    rain_col = 'RAIN_FALL(mm)'
    
    print(f"Initial raw rows: {len(full_df)}")
    
    # 1. Parse Timestamps
    date_str = full_df['DATE(GMT)'].astype(str)
    time_str = full_df['TIME(GMT)'].astype(str).str.zfill(2)
    full_df['timestamp'] = pd.to_datetime(date_str + ' ' + time_str + ':00:00', format='%m/%d/%Y %H:%M:%S', errors='coerce')
    
    # Rename station column for consistency
    full_df.rename(columns={'@STATION_ID': 'station_id', 'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'ALTITUDE(m)': 'alt'}, inplace=True)
    
    # Drop rows without valid timestamps or station_ids
    df_clean = full_df.dropna(subset=['timestamp', 'station_id']).copy()
    
    # 2. Deduplicate
    df_clean = df_clean.drop_duplicates(subset=['station_id', 'timestamp'], keep='first')
    print(f"Rows after deduplication: {len(df_clean)}")
    
    # 3. Universal Missing Value Masking
    # Replace -999.0, -999 with NaN
    cols_to_clean = [temp_col, press_col, hum_col, rain_col]
    for col in cols_to_clean:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean.loc[df_clean[col] <= -999.0, col] = np.nan
        
    # Specifically mask obviously broken temperature sensors (like -39.3 in Assam) without dropping the row
    # Parameterize broadly: < -30 is invalid anywhere in the project scope, > 55 is invalid
    df_clean.loc[(df_clean[temp_col] < -30) | (df_clean[temp_col] > 55), temp_col] = np.nan
    
    # Pressure sanity check (station pressure varies by altitude, but < 300hPa is invalid for troposphere ground stations)
    df_clean.loc[(df_clean[press_col] < 300) | (df_clean[press_col] > 1100), press_col] = np.nan
    
    # Humidity bounds
    df_clean.loc[(df_clean[hum_col] < 0) | (df_clean[hum_col] > 100), hum_col] = np.nan
    
    # Rain bounds
    df_clean.loc[df_clean[rain_col] < 0, rain_col] = np.nan
    
    # 4. Resample to strict 1-hour intervals per station and interpolate short gaps
    # We want to ensure a continuous hourly timeline for each station, interpolating up to 2 hours
    print("Resampling and interpolating (limit=2 hours)...")
    
    resampled_dfs = []
    
    for station, group in df_clean.groupby('station_id'):
        group = group.sort_values('timestamp').set_index('timestamp')
        
        # Keep static columns
        static_data = group[['lat', 'lon', 'alt']].iloc[0]
        
        # Resample to hourly
        # Mean for continuous vars, sum for rain? 
        # Actually AWS usually reports hourly accumulated rain or instantaneous rate. We'll use sum for rain to be safe if there are sub-hourly records, though data is mostly hourly.
        # Since data is mostly hourly snapshots, `.mean()` is safe for temp/press/hum.
        numeric_group = group[[temp_col, press_col, hum_col, rain_col]]
        
        resampled = numeric_group.resample('1h').mean()
        
        # Interpolate gaps up to 2 hours
        resampled[temp_col] = resampled[temp_col].interpolate(method='time', limit=2)
        resampled[press_col] = resampled[press_col].interpolate(method='time', limit=2)
        resampled[hum_col] = resampled[hum_col].interpolate(method='time', limit=2)
        # For rainfall, interpolation doesn't make physical sense (rain is sparse). Fill short gaps with 0? No, leave as NaN so we know it's missing.
        
        resampled['station_id'] = station
        resampled['lat'] = static_data['lat']
        resampled['lon'] = static_data['lon']
        resampled['alt'] = static_data['alt']
        
        resampled_dfs.append(resampled.reset_index())

    final_df = pd.concat(resampled_dfs, ignore_index=True)
    
    # Rename features to clean names
    final_df.rename(columns={
        temp_col: 'temp',
        press_col: 'pressure',
        hum_col: 'humidity',
        rain_col: 'rain_mm_hr'
    }, inplace=True)
    
    print(f"Final rows after resampling: {len(final_df)}")
    print("Missing value counts:")
    print(final_df[['temp', 'pressure', 'humidity', 'rain_mm_hr']].isna().sum())
    
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_FILE, index=False)
    print(f"Saved cleaned AWS data to {OUT_FILE}")

if __name__ == "__main__":
    process_aws_data()
