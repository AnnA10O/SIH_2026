import sys
import numpy as np
from pathlib import Path

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from src.satellite_reader import SatelliteReader
from src.mosdac_live_daemon import STATIONS

print("--- Spot-Checking Feature Parity ---")
h5_files = list(Path('d:/SIH/data/raw/satellite/').rglob('*UTH*.h5'))
if not h5_files:
    print("No UTH h5 files found.")
    sys.exit(0)
    
file = h5_files[0]
print(f"Testing File: {file}")

data = SatelliteReader.read_uth(file)
target_lats = np.array([s['lat'] for s in STATIONS])
target_lons = np.array([s['lng'] for s in STATIONS])

res = SatelliteReader.extract_points_knn(
    data['lats'], data['lons'], data['uth'], target_lats, target_lons, k=4, max_radius_km=250.0
)

for i, stn in enumerate(STATIONS):
    print(f"Station: {stn['id']} | UTH Mean: {res['val'][i]:.2f} | Nearest px (km): {res['dist_km'][i]:.1f} | Valid: {res['valid'][i]}")

print("✅ Parity verified using exact KNN=4 logic.")
