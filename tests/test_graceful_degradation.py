import sys
import time
from pathlib import Path

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from src.fusion_buffer import DataFusionBuffer

print("--- Testing Graceful Degradation ---")

# We mock time.time() to simulate time passing
global_time = 0.0
def mock_now():
    return global_time

buffer = DataFusionBuffer(now_fn=mock_now)

# 1. Daemon successfully ingests satellite data at t=0
buffer.ingest("UK-1", "uth_kalpana", 45.0, timestamp_epoch=0.0)

# 2. Inference reads at t=10s
global_time = 10.0
state = buffer.get_channel_state("UK-1", "uth_kalpana")
print(f"t=10s (Auth Good): valid={state.valid}, degraded={state.degraded}, staleness={state.staleness_s}")
assert state.valid == True

# 3. Simulate daemon auth failure - no new ingest. Inference reads at t=50 mins (3000s)
global_time = 3000.0
state = buffer.get_channel_state("UK-1", "uth_kalpana")
print(f"t=50m (Missed 1 fetch): valid={state.valid}, degraded={state.degraded}, staleness={state.staleness_s}")
# Should be valid but degraded (degraded threshold is 40 min)
assert state.valid == True
assert state.degraded == True

# 4. Simulate sustained auth failure. Inference reads at t=100 mins (6000s)
global_time = 6000.0
state = buffer.get_channel_state("UK-1", "uth_kalpana")
print(f"t=100m (Missed 3 fetches): valid={state.valid}, degraded={state.degraded}, staleness={state.staleness_s}")
# Should be INVALID (max staleness is 90 min)
assert state.valid == False

print("✅ Graceful degradation verified. Buffer correctly ages out data on daemon failure without crashing.")
