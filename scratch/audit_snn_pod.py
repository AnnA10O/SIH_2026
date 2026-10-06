import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.snn_gate import SNNNeuromorphicGate

csv_path = Path("ASSAM_ALL_2013-02-01_2014-03-15_Sep2026_177056.csv")
print(f"Loading telemetry from {csv_path.name}...")
df = pd.read_csv(csv_path, low_memory=False)

df['ts'] = pd.to_datetime(
    df['DATE(IST)'].astype(str) + ' ' + df['TIME(IST)'].astype(str),
    dayfirst=True, errors='coerce'
)
df['rain'] = pd.to_numeric(df['RAIN_FALL(mm)'], errors='coerce').fillna(0.0)
df = df.dropna(subset=['ts']).sort_values(['@STATION_ID', 'ts'])

# Identify severe events manually: RI >= 15 mm in 15 minutes (60 mm/hr pace)
cb_events = []
stations = df['@STATION_ID'].unique()
for sid in stations:
    sub = df[df['@STATION_ID'] == sid].copy()
    sub['RI'] = sub['rain'].diff().fillna(0.0)
    # Find events
    events = sub[sub['RI'] >= 15.0]
    for _, row in events.iterrows():
        cb_events.append({"station_id": sid, "ts": row['ts'], "R": row['rain'], "RI": row['RI']})

cb_df = pd.DataFrame(cb_events)
print(f"Found {len(cb_df)} severe rainfall surge events (RI >= 15 mm/15min).")
if len(cb_df) == 0:
    sys.exit(0)

captured = 0
missed = 0

print("Evaluating SNN pre-cursor spikes (15-min baseline polling)...")
for idx, row in cb_df.iterrows():
    sid = row['station_id']
    event_time = row['ts']
    # 60 minute precursor window BEFORE the event time
    start_time = event_time - pd.Timedelta(minutes=60)
    
    # Get 15-minute readings in the precursor window (exclusive of the exact event time)
    history = df[(df['@STATION_ID'] == sid) & (df['ts'] >= start_time) & (df['ts'] < event_time)].copy()
    history['RI'] = history['rain'].diff().fillna(0.0)
    
    gate = SNNNeuromorphicGate(station_id=sid, tau_minutes=15.0, v_thresh=0.45)
    
    spike_fired = False
    for _, h_row in history.iterrows():
        feat = {
            "timestamp": h_row['ts'],
            "R": h_row['rain'],
            "RI": h_row['RI']
        }
        res = gate.step(feat)
        if res['fired_spike']:
            spike_fired = True
            break
            
    if spike_fired:
        captured += 1
    else:
        missed += 1

pod = captured / (captured + missed) if (captured + missed) > 0 else 0
print("=" * 60)
print(f"SNN Gate Precursor POD (Recall): {pod*100:.2f}%")
print(f"Total Severe Events Evaluated: {captured + missed}")
print(f"Captured by SNN Gate (pre-spike): {captured}")
print(f"Missed by SNN Gate (slept through): {missed}")
print("=" * 60)
