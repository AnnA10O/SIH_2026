import sys
from pathlib import Path
import numpy as np
import pandas as pd

gagan_fp = Path("data/raw/gagan_iwv_v1.txt")
events_fp = Path("outputs/cloudburst_events.csv")

print(f"Reading GAGAN IWV records from {gagan_fp}...")
# Format in file: stn_id lat lon year month day hour min iwv ztd
gagan_df = pd.read_csv(
    gagan_fp, sep=r'\s+', header=None,
    names=['stn_id', 'lat', 'lon', 'year', 'month', 'day', 'hour', 'min', 'iwv', 'ztd'],
    low_memory=False
)

# Parse datetimes
gagan_df['dt'] = pd.to_datetime({
    'year': gagan_df['year'],
    'month': gagan_df['month'],
    'day': gagan_df['day'],
    'hour': gagan_df['hour'],
    'minute': gagan_df['min']
})
gagan_df = gagan_df.sort_values(['stn_id', 'dt']).reset_index(drop=True)
print(f"Total GAGAN records: {len(gagan_df):,}")
print(f"Date range: {gagan_df['dt'].min()} to {gagan_df['dt'].max()}")

events_df = pd.read_csv(events_fp)
events_df['ts'] = pd.to_datetime(events_df['timestamp'])
print(f"\nEvents to enrich: {len(events_df):,} events from {events_df['ts'].min()} to {events_df['ts'].max()}")

# Index GAGAN by station and dt for fast lookup
station_series = {}
for sid in events_df['gagan_station_id'].unique():
    sub = gagan_df[gagan_df['stn_id'] == sid].copy()
    sub = sub.set_index('dt')['iwv'].resample('1h').mean().interpolate(method='time', limit=3)
    station_series[sid] = sub
    print(f"Station #{sid}: {len(sub):,} hourly IWV points available")

# Compute IWV_now, IWV_trend_3hr, IWV_trend_6hr
iwv_now_list = []
iwv_trend3_list = []
iwv_trend6_list = []

for idx, row in events_df.iterrows():
    gid = row['gagan_station_id']
    t = row['ts']
    s = station_series.get(gid)
    
    if s is None or t not in s.index:
        # Check closest timestamp within 2 hours
        val_now = np.nan
        val_trend3 = np.nan
        val_trend6 = np.nan
        if s is not None and not s.empty:
            window = s.loc[t - pd.Timedelta(hours=3):t + pd.Timedelta(hours=3)]
            if not window.empty:
                val_now = window.iloc[0]
                t_prev3 = t - pd.Timedelta(hours=3)
                t_prev6 = t - pd.Timedelta(hours=6)
                if t_prev3 in s.index:
                    val_trend3 = val_now - s.loc[t_prev3]
                if t_prev6 in s.index:
                    val_trend6 = val_now - s.loc[t_prev6]
    else:
        val_now = s.loc[t]
        t_prev3 = t - pd.Timedelta(hours=3)
        t_prev6 = t - pd.Timedelta(hours=6)
        val_trend3 = (val_now - s.loc[t_prev3]) if t_prev3 in s.index else np.nan
        val_trend6 = (val_now - s.loc[t_prev6]) if t_prev6 in s.index else np.nan
        
    iwv_now_list.append(val_now)
    iwv_trend3_list.append(val_trend3)
    iwv_trend6_list.append(val_trend6)

events_df['IWV_now'] = iwv_now_list
events_df['IWV_trend_3hr'] = iwv_trend3_list
events_df['IWV_trend_6hr'] = iwv_trend6_list

valid_now = events_df['IWV_now'].notna().mean() * 100
valid_trend = events_df['IWV_trend_3hr'].notna().mean() * 100
print("\n" + "=" * 70)
print(f"GAGAN IWV ENRICHMENT RESULTS")
print("=" * 70)
print(f"Events with IWV_now coverage:      {events_df['IWV_now'].notna().sum()}/{len(events_df)} ({valid_now:.1f}%)")
print(f"Events with IWV_trend_3hr coverage:{events_df['IWV_trend_3hr'].notna().sum()}/{len(events_df)} ({valid_trend:.1f}%)")

print("\nMean IWV by Event Class:")
print(events_df.groupby('final_label')[['IWV_now', 'IWV_trend_3hr', 'IWV_trend_6hr']].agg(['mean', 'std']))
