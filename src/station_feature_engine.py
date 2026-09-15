"""Module 1 — Per-Station Feature Computation Engine.

Maintains a rolling buffer of station readings and computes derived meteorological
and convective features for each station:
  - R: Instantaneous rainfall rate (mm/hr)
  - R_30, R_60: Accumulated rainfall over 30-min and 60-min windows
  - RI: Rainfall intensity acceleration (rate of change of rain rate)
  - RH, dRH: Relative humidity and humidity trend
  - dewpoint_depression: Temp - Dewpoint (moisture saturation proxy)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from collections import deque


class StationFeatureBuffer:
    """Rolling 60-minute feature buffer for a single in-situ AWS station."""

    def __init__(self, station_id: str, lat: float, lon: float, max_window_minutes: int = 60):
        self.station_id = station_id
        self.lat = lat
        self.lon = lon
        self.max_window_minutes = max_window_minutes
        self.buffer: List[Dict] = []

    def _compute_dewpoint(self, temp_c: float, rh_pct: float) -> float:
        """Magnus-Tetens approximation for dew point temperature."""
        if np.isnan(temp_c) or np.isnan(rh_pct) or rh_pct <= 0:
            return np.nan
        # Simple approximation: Td ~= T - ((100 - RH) / 5)
        return temp_c - ((100.0 - rh_pct) / 5.0)

    def add_reading(
        self,
        timestamp: pd.Timestamp,
        rain_mm: float,
        temp_c: Optional[float] = None,
        rh_pct: Optional[float] = None,
        pressure_hpa: Optional[float] = None,
        interval_minutes: float = 15.0
    ) -> Dict[str, float]:
        """
        Append a new AWS reading, prune readings older than window, and compute derived features.
        
        Returns
        -------
        features : Dict of meteorological indicators ready for SNN gate and classifier.
        """
        ts = pd.to_datetime(timestamp)
        reading = {
            "timestamp": ts,
            "rain_mm": float(rain_mm) if not np.isnan(rain_mm) else 0.0,
            "temp_c": float(temp_c) if temp_c is not None and not np.isnan(temp_c) else np.nan,
            "rh_pct": float(rh_pct) if rh_pct is not None and not np.isnan(rh_pct) else np.nan,
            "pressure_hpa": float(pressure_hpa) if pressure_hpa is not None and not np.isnan(pressure_hpa) else np.nan,
            "interval_minutes": interval_minutes
        }
        self.buffer.append(reading)

        # Prune readings older than window
        cutoff = ts - pd.Timedelta(minutes=self.max_window_minutes)
        self.buffer = [r for r in self.buffer if r["timestamp"] >= cutoff]

        # Sort chronologically
        self.buffer.sort(key=lambda r: r["timestamp"])

        # Compute instant rain rate R (mm/hr)
        interval_hr = max(interval_minutes / 60.0, 0.01)
        r_current = reading["rain_mm"] / interval_hr

        # Previous reading for trends
        if len(self.buffer) >= 2:
            prev = self.buffer[-2]
            prev_interval_hr = max(prev["interval_minutes"] / 60.0, 0.01)
            r_prev = prev["rain_mm"] / prev_interval_hr
            ri = (r_current - r_prev) / interval_hr  # acceleration in mm/hr^2
            rh_prev = prev["rh_pct"]
            drh = reading["rh_pct"] - rh_prev if not np.isnan(rh_prev) and not np.isnan(reading["rh_pct"]) else 0.0
        else:
            ri = 0.0
            drh = 0.0

        # Accumulations across buffer
        r_30_cutoff = ts - pd.Timedelta(minutes=30)
        r_30 = sum(r["rain_mm"] for r in self.buffer if r["timestamp"] >= r_30_cutoff)
        r_60 = sum(r["rain_mm"] for r in self.buffer)

        # Dewpoint depression
        if not np.isnan(reading["temp_c"]) and not np.isnan(reading["rh_pct"]):
            dewpoint = self._compute_dewpoint(reading["temp_c"], reading["rh_pct"])
            dewpoint_depression = max(0.0, reading["temp_c"] - dewpoint)
        else:
            dewpoint_depression = np.nan

        features = {
            "station_id": self.station_id,
            "lat": self.lat,
            "lon": self.lon,
            "timestamp": ts,
            "R": round(r_current, 2),
            "R_30": round(r_30, 2),
            "R_60": round(r_60, 2),
            "RI": round(ri, 2),
            "RH": round(reading["rh_pct"], 1) if not np.isnan(reading["rh_pct"]) else np.nan,
            "dRH": round(drh, 2),
            "dewpoint_depression": round(dewpoint_depression, 2) if not np.isnan(dewpoint_depression) else np.nan,
            "pressure_hpa": round(reading["pressure_hpa"], 1) if not np.isnan(reading["pressure_hpa"]) else np.nan,
        }
        return features
