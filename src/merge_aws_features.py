import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
import sys

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))
from src.phase_d_training import load_imd_parquets

OUT_FILE = ROOT / "outputs" / "assam_ablation_set.parquet"

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def merge_aws_and_iwv():
    print("Loading base IMD parquets...")
    events_df = load_imd_parquets()
    
    # Filter to Assam and 2013-2014 window
    events_df = events_df[events_df['region_name'] == 'Assam_Meghalaya'].copy()
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    events_df = events_df[(events_df['timestamp'].dt.year >= 2012) & (events_df['timestamp'].dt.year <= 2014)]
    print(f"Filtered to Assam 2012-2014: {len(events_df)} rows")
    
    print("Loading hourly AWS data...")
    aws_df = pd.read_parquet(ROOT / "data/processed/assam_aws_hourly.parquet")
    aws_df = aws_df.dropna(subset=['lat', 'lon', 'timestamp'])
    
    print("Loading GAGAN IWV data...")
    # IWV format: station_code lat lon year month day hour min iwv ztd
    iwv_file = ROOT / "data/raw/gagan_iwv_v1.txt"
    iwv_cols = ['station_code', 'lat', 'lon', 'year', 'month', 'day', 'hour', 'min', 'iwv', 'ztd']
    iwv_df = pd.read_csv(iwv_file, sep=r'\s+', header=None, names=iwv_cols)
    
    # Parse IWV timestamp
    # Ensure columns are treated correctly, handling any potential errors
    iwv_df['year'] = iwv_df['year'].astype(str)
    iwv_df['month'] = iwv_df['month'].astype(str).str.zfill(2)
    iwv_df['day'] = iwv_df['day'].astype(str).str.zfill(2)
    iwv_df['hour'] = iwv_df['hour'].astype(str).str.zfill(2)
    iwv_df['min'] = iwv_df['min'].astype(str).str.zfill(2)
    
    iwv_df['timestamp'] = pd.to_datetime(
        iwv_df['year'] + '-' + iwv_df['month'] + '-' + iwv_df['day'] + ' ' + 
        iwv_df['hour'] + ':' + iwv_df['min'] + ':00', 
        errors='coerce'
    )
    iwv_df = iwv_df.dropna(subset=['timestamp', 'lat', 'lon'])
    
    # Merge Logic
    # 1. AWS Matching
    # Maximum distance: 50km. Tolerance: ±30 min.
    
    # To optimize, we'll use a Cartesian join chunking or ball tree, but since data is small, 
    # we can use merge_asof if we sort by timestamp.
    # merge_asof requires exact match on 'by' keys, which we don't have for lat/lon.
    # So we'll pre-compute nearest stations.
    
    print("Computing nearest AWS stations...")
    event_coords = events_df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    aws_coords = aws_df[['station_id', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)
    
    # Map event coords to nearest AWS station
    mapping = {}
    for _, row in event_coords.iterrows():
        dists = haversine_vectorized(row['lat'], row['lon'], aws_coords['lat'].values, aws_coords['lon'].values)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        if min_dist <= 50.0:
            mapping[(row['lat'], row['lon'])] = aws_coords.iloc[min_idx]['station_id']
        else:
            mapping[(row['lat'], row['lon'])] = None
            
    events_df['nearest_aws_station'] = events_df.apply(lambda r: mapping[(r['lat'], r['lon'])], axis=1)
    print(f"Matched AWS stations for {events_df['nearest_aws_station'].notna().sum()} out of {len(events_df)} rows")
    
    # 2. IWV Matching
    print("Computing nearest IWV stations...")
    iwv_coords = iwv_df[['station_code', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)
    
    mapping_iwv = {}
    for _, row in event_coords.iterrows():
        dists = haversine_vectorized(row['lat'], row['lon'], iwv_coords['lat'].values, iwv_coords['lon'].values)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        if min_dist <= 50.0:
            mapping_iwv[(row['lat'], row['lon'])] = iwv_coords.iloc[min_idx]['station_code']
        else:
            mapping_iwv[(row['lat'], row['lon'])] = None
            
    events_df['nearest_iwv_station'] = events_df.apply(lambda r: mapping_iwv[(r['lat'], r['lon'])], axis=1)
    print(f"Matched IWV stations for {events_df['nearest_iwv_station'].notna().sum()} out of {len(events_df)} rows")

    # Now perform temporal matching (±30 min)
    # We will sort all dfs by timestamp and use merge_asof
    
    events_df = events_df.sort_values('timestamp')
    aws_df = aws_df.sort_values('timestamp')
    iwv_df = iwv_df.sort_values('timestamp')
    
    events_df['nearest_aws_station'] = events_df['nearest_aws_station'].astype(str)
    aws_df['station_id'] = aws_df['station_id'].astype(str)
    
    # AWS merge
    merged_aws = pd.merge_asof(
        events_df,
        aws_df[['station_id', 'timestamp', 'temp', 'pressure', 'humidity', 'rain_mm_hr']].rename(columns={'rain_mm_hr': 'aws_rain_mm_hr'}),
        left_on='timestamp',
        right_on='timestamp',
        left_by='nearest_aws_station',
        right_by='station_id',
        direction='nearest',
        tolerance=pd.Timedelta('30min')
    )
    
    merged_aws['nearest_iwv_station'] = merged_aws['nearest_iwv_station'].astype(str).str.replace(r'\.0$', '', regex=True)
    iwv_df['station_code'] = iwv_df['station_code'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # IWV merge
    merged_final = pd.merge_asof(
        merged_aws,
        iwv_df[['station_code', 'timestamp', 'iwv', 'ztd']],
        left_on='timestamp',
        right_on='timestamp',
        left_by='nearest_iwv_station',
        right_by='station_code',
        direction='nearest',
        tolerance=pd.Timedelta('30min')
    )
    
    # Ensure unmatched rows are explicitly logged
    unmatched_aws = merged_final['temp'].isna().sum()
    unmatched_iwv = merged_final['iwv'].isna().sum()
    
    print(f"Unmatched AWS records: {unmatched_aws} / {len(merged_final)}")
    print(f"Unmatched IWV records: {unmatched_iwv} / {len(merged_final)}")
    
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_final.to_parquet(OUT_FILE, index=False)
    print(f"Saved merged ablation dataset to {OUT_FILE}")

if __name__ == "__main__":
    merge_aws_and_iwv()
