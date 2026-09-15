"""
Data Fusion Buffer — Shared Real-Time State Assembler
PS 26077: AI Hyper-Local Cloudburst & Thunderstorm Early Warning System

PURPOSE
-------
Multiple independent sources (AWS ground stations, GAGAN GNSS-IWV, Kalpana-1/
INSAT satellite products) arrive on different, irregular cadences. This module
is the SINGLE shared component that both the SNN edge gates (Gate A/B in
simulator.py) and the CNN regional model read from at inference time.

Having one shared buffer — instead of each model independently reconstructing
"what do we currently know" — prevents the two inference paths from silently
disagreeing about staleness/missingness.

Level 0 Liveness Check:
A separate heartbeat process checks if ingestion is actively happening. If a 
sensor goes completely dark (e.g. poller crashed), the system distinguishes
"the weather is quiet" from "the sensor is dead" via Level 0 alerts.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List
from collections import defaultdict


# ── Per-channel staleness policy ──────────────────────────────────────────────
# max_staleness_s: beyond this, `valid` becomes False regardless of last value.
# degraded_staleness_s: beyond this but within max, value is still used but
#   flagged — mirrors the uth_nearest_px_km "quality, not just presence" pattern.
# liveness_timeout_s: beyond this, the source is considered administratively DEAD (Level 0 failure).
CHANNEL_POLICY = {
    "aws_rain":     {"max_staleness_s": 2 * 3600,  "degraded_staleness_s": 45 * 60, "liveness_timeout_s": 4 * 3600},
    "aws_temp":     {"max_staleness_s": 3 * 3600,  "degraded_staleness_s": 60 * 60, "liveness_timeout_s": 4 * 3600},
    "gagan_iwv":    {"max_staleness_s": 2 * 3600,  "degraded_staleness_s": 30 * 60, "liveness_timeout_s": 4 * 3600},
    "uth_kalpana":  {"max_staleness_s": 90 * 60,   "degraded_staleness_s": 40 * 60, "liveness_timeout_s": 3 * 3600},
    "hem_kalpana":  {"max_staleness_s": 90 * 60,   "degraded_staleness_s": 40 * 60, "liveness_timeout_s": 3 * 3600},
    "pressure":     {"max_staleness_s": 1 * 3600,  "degraded_staleness_s": 20 * 60, "liveness_timeout_s": 4 * 3600},
}


@dataclass
class ChannelReading:
    value: float
    timestamp_epoch: float
    source_meta: dict = field(default_factory=dict)  # e.g. {"nearest_px_km": 34.2}


@dataclass
class ChannelState:
    """What a consumer actually receives when it asks 'what do we know right now'."""
    value: Optional[float]
    staleness_s: float
    valid: bool          # False if beyond max_staleness_s (or never received)
    degraded: bool       # True if beyond degraded_staleness_s but still valid
    is_dead: bool        # True if beyond liveness_timeout_s (Level 0 pipeline failure)
    source_meta: dict    # Kept even if invalid, for debugging!


class DataFusionBuffer:
    """
    Thread-safe latest-value store, keyed by (station_id, channel_name).
    Ingestion (writes) and inference (reads) are fully decoupled — a slow or
    silent source never blocks reads for other channels or other stations.
    """

    def __init__(self, policy: Dict[str, dict] = None, now_fn: Callable[[], float] = time.time):
        self._policy = policy or CHANNEL_POLICY
        self._now = now_fn
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, ChannelReading]] = defaultdict(dict)

    # ── Ingestion API — called by each source's independent listener/poller ──

    def ingest(self, station_id: str, channel: str, value: float,
               timestamp_epoch: float = None, source_meta: dict = None) -> None:
        if channel not in self._policy:
            # Fallback to a default generic policy if not explicitly configured
            self._policy[channel] = {"max_staleness_s": 2 * 3600, "degraded_staleness_s": 60 * 60, "liveness_timeout_s": 4 * 3600}
            
        ts = timestamp_epoch if timestamp_epoch is not None else self._now()
        with self._lock:
            existing = self._store[station_id].get(channel)
            if existing is not None and existing.timestamp_epoch > ts:
                # Out-of-order arrival (do not overwrite new with old)
                return
            self._store[station_id][channel] = ChannelReading(
                value=value, timestamp_epoch=ts, source_meta=source_meta or {}
            )

    # ── Read API — called by both SNN and CNN inference paths ────────────────

    def get_channel_state(self, station_id: str, channel: str) -> ChannelState:
        """The single source of truth both models must use."""
        policy = self._policy.get(channel)
        if policy is None:
            # Add fallback policy instead of crashing if queried before first ingest
            self._policy[channel] = {"max_staleness_s": 2 * 3600, "degraded_staleness_s": 60 * 60, "liveness_timeout_s": 4 * 3600}
            policy = self._policy[channel]

        with self._lock:
            reading = self._store.get(station_id, {}).get(channel)

        if reading is None:
            return ChannelState(value=None, staleness_s=float("inf"),
                                valid=False, degraded=False, is_dead=True, source_meta={})

        staleness = self._now() - reading.timestamp_epoch
        valid = staleness <= policy["max_staleness_s"]
        degraded = valid and staleness > policy["degraded_staleness_s"]
        is_dead = staleness > policy.get("liveness_timeout_s", policy["max_staleness_s"] * 2)
        
        return ChannelState(
            value=reading.value if valid else None,
            staleness_s=staleness, 
            valid=valid, 
            degraded=degraded,
            is_dead=is_dead,
            source_meta=reading.source_meta,
        )

    def get_snapshot(self, station_id: str, channels: list) -> Dict[str, ChannelState]:
        """Convenience: full state for one station across the requested channels."""
        return {ch: self.get_channel_state(station_id, ch) for ch in channels}

    def get_data_confidence(self, station_id: str, channels: list) -> float:
        """Coarse 0-1 confidence score: fraction of requested channels currently valid-and-not-degraded."""
        states = self.get_snapshot(station_id, channels)
        if not states:
            return 0.0
        score = sum(
            1.0 if (s.valid and not s.degraded) else (0.5 if s.valid else 0.0)
            for s in states.values()
        )
        return score / len(states)

    def get_level0_health(self, station_id: str, channels: list) -> List[str]:
        """
        Level 0 Data Pipeline Health check. 
        Returns a list of DEAD channels (crashed pollers/sensors) for the given station.
        """
        states = self.get_snapshot(station_id, channels)
        dead_channels = [ch for ch, s in states.items() if s.is_dead]
        return dead_channels
