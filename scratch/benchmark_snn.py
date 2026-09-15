import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.snn_gate import SNNNeuromorphicGate

# Load real AWS time-series dataset
csv_path = Path("ASSAM_ALL_2013-02-01_2014-03-15_Sep2026_177056.csv")
print(f"Loading real telemetry records from {csv_path.name}...")
df = pd.read_csv(csv_path, low_memory=False)

df['ts'] = pd.to_datetime(
    df['DATE(IST)'].astype(str) + ' ' + df['TIME(IST)'].astype(str),
    dayfirst=True, errors='coerce'
)
df['rain'] = pd.to_numeric(df['RAIN_FALL(mm)'], errors='coerce').fillna(0.0)
df['rh'] = pd.to_numeric(df['HUMIDITY(%)'], errors='coerce').fillna(50.0)
df = df.dropna(subset=['ts']).sort_values(['@STATION_ID', 'ts'])

# Benchmark across all stations
total_readings = 0
total_spikes = 0
total_active_intervals = 0
total_dormant_intervals = 0

station_stats = []

stations = df['@STATION_ID'].unique()
print(f"Benchmarking SNN Neuromorphic LIF Gate across {len(stations)} physical AWS stations...")

for sid in stations:
    sub = df[df['@STATION_ID'] == sid].copy()
    if len(sub) < 50:
        continue
    
    sub['R'] = sub['rain']
    sub['RI'] = sub['R'].diff().fillna(0.0)
    
    gate = SNNNeuromorphicGate(
        station_id=sid,
        tau_minutes=15.0,
        v_thresh=1.0,
        v_reset=0.0
    )
    
    stn_evals = 0
    stn_spikes = 0
    stn_active_steps = 0
    
    for _, row in sub.iterrows():
        feat = {
            "timestamp": row['ts'],
            "R": row['R'],
            "RI": row['RI'],
            "RH": row['rh']
        }
        res = gate.step(feat)
        stn_evals += 1
        if res['fired_spike']:
            stn_spikes += 1
        if res['state'] == "ACTIVE":
            stn_active_steps += 1
            
    stn_dormant_steps = stn_evals - stn_active_steps
    savings_pct = (1.0 - (stn_active_steps / stn_evals)) * 100.0
    
    total_readings += stn_evals
    total_spikes += stn_spikes
    total_active_intervals += stn_active_steps
    total_dormant_intervals += stn_dormant_steps
    
    station_stats.append({
        "station": sid[:30],
        "evals": stn_evals,
        "spikes": stn_spikes,
        "active_steps": stn_active_steps,
        "dormant_steps": stn_dormant_steps,
        "telemetry_savings_pct": round(savings_pct, 2)
    })

df_stats = pd.DataFrame(station_stats)

print("\n" + "=" * 75)
print("EMPIRICAL BENCHMARK RESULTS: SNN LIF GATE ON REAL AWS TELEMETRY")
print("=" * 75)
print(f"Total Telemetry Timestamps Evaluated:  {total_readings:,}")
print(f"Total Neuromorphic Spikes Fired:       {total_spikes:,}")
print(f"Total Intervals in Escalated ACTIVE:   {total_active_intervals:,} ({total_active_intervals/total_readings*100:.2f}%)")
print(f"Total Intervals in Low-Power DORMANT:  {total_dormant_intervals:,} ({total_dormant_intervals/total_readings*100:.2f}%)")

empirical_telemetry_reduction = (1.0 - (total_active_intervals / total_readings)) * 100.0
empirical_spike_reduction = (1.0 - (total_spikes / total_readings)) * 100.0

print(f"\nEmpirical Spike Rate:                  {total_spikes/total_readings*100:.2f}%")
print(f"Empirical Telemetry Transmission Cut:  {empirical_telemetry_reduction:.2f}%")

# Radio Energy Model:
# Standard LoRa / GSM AWS:
# Active Transmit (Tx) = 120 mA @ 3.3V = 396 mW
# Microcontroller Listen/Process = 15 mA @ 3.3V = 49.5 mW
# Deep Sleep = 15 uA @ 3.3V = 0.05 mW
# Continuous Baseline (15-min regular poll): Tx every 15 min
# SNN Gated: Tx only during ACTIVE escalation / spikes
baseline_energy = total_readings * (396.0 * 2.0 + 49.5 * 1.0) # approx mWs
snn_energy = (total_active_intervals * 396.0 * 2.0) + (total_readings * 49.5 * 0.05)
realized_energy_savings = (1.0 - (snn_energy / baseline_energy)) * 100.0

print(f"Modeled Radio Energy Consumption Savings: {realized_energy_savings:.2f}%")
print("=" * 75)
print("\nTop 10 Station Breakdown:")
print(df_stats.head(10).to_string(index=False))
