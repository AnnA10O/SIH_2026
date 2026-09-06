"""Module 5 — Spatial Fusion Grid Engine.

Solves the AWS inter-station gap problem:
  - Cloudburst convective cores are 4.5–5.5 km across.
  - AWS station spacing is typically 15–50 km.
  - A cloudburst can develop and drop 100mm between gauges.

Architecture:
  1. Collects point-level P(CB) scores from all AWS stations in the region.
  2. Spatially interpolates station P(CB) onto a continuous 4km mesh grid using
     Inverse Distance Weighting (IDW) or Gaussian kernel weighting.
  3. Fuses with continuous 2D satellite convective field (CTCR / CTT / QPE)
     to detect ungauged convective cells between stations.
  4. Produces a 2D regional risk map: risk_map(x, y).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.utils import haversine_km


class SpatialFusionGrid:
    """2D spatial interpolator and multi-sensor fusion grid."""

    def __init__(
        self,
        lat_min: float = 24.5,
        lat_max: float = 28.5,
        lon_min: float = 88.5,
        lon_max: float = 95.5,
        grid_res_deg: float = 0.04,   # ~4km resolution matching satellite IR
        idw_power: float = 2.0,
        satellite_weight: float = 0.35
    ):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.grid_res = grid_res_deg
        self.idw_power = idw_power
        self.alpha_sat = satellite_weight

        # Construct regular coordinate mesh
        self.lats = np.arange(lat_min, lat_max + 1e-5, grid_res_deg)
        self.lons = np.arange(lon_min, lon_max + 1e-5, grid_res_deg)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)
        self.shape = self.lat_grid.shape

    def interpolate_station_scores(
        self,
        station_predictions: List[Dict]
    ) -> np.ndarray:
        """
        Interpolate point station P(CB) scores across the 2D spatial grid using IDW.

        station_predictions: List of dicts with 'lat', 'lon', 'p_cb'
        """
        if not station_predictions:
            return np.zeros(self.shape, dtype=float)

        stn_lats = np.array([s["lat"] for s in station_predictions])
        stn_lons = np.array([s["lon"] for s in station_predictions])
        stn_p    = np.array([s["p_cb"] for s in station_predictions])

        grid_p = np.zeros(self.shape, dtype=float)

        # Approximate flat distance in km: 1 deg lat ~= 111 km, 1 deg lon ~= 111 * cos(lat)
        cos_mean_lat = np.cos(np.radians(0.5 * (self.lat_min + self.lat_max)))

        weights_sum = np.zeros(self.shape, dtype=float)
        weighted_val_sum = np.zeros(self.shape, dtype=float)

        for lat_s, lon_s, p_s in zip(stn_lats, stn_lons, stn_p):
            dy = (self.lat_grid - lat_s) * 111.0
            dx = (self.lon_grid - lon_s) * 111.0 * cos_mean_lat
            dist_km = np.sqrt(dx**2 + dy**2)

            # Prevent zero division at gauge location
            dist_km = np.maximum(dist_km, 2.0)
            w = 1.0 / (dist_km ** self.idw_power)

            weighted_val_sum += w * p_s
            weights_sum += w

        grid_p = np.where(weights_sum > 0, weighted_val_sum / weights_sum, 0.0)
        return np.clip(grid_p, 0.0, 1.0)

    def fuse_with_satellite(
        self,
        station_grid: np.ndarray,
        satellite_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Combine interpolated ground station probability with top-down satellite field.
        risk_map = (1 - alpha) * station_grid + alpha * satellite_grid
        """
        if satellite_grid is None or satellite_grid.shape != self.shape:
            return station_grid

        # Satellite field is normalized [0..1] (e.g. normalized CTCR or rain rate)
        sat_norm = np.clip(satellite_grid, 0.0, 1.0)
        fused = ((1.0 - self.alpha_sat) * station_grid) + (self.alpha_sat * sat_norm)
        return np.clip(fused, 0.0, 1.0)

    def generate_risk_map(
        self,
        station_predictions: List[Dict],
        satellite_grid: Optional[np.ndarray] = None
    ) -> Dict:
        """
        End-to-end regional risk map generation.

        Returns
        -------
        Dict with:
          - risk_matrix (2D numpy array)
          - lats, lons (1D coordinate arrays)
          - peak_risk (float)
          - peak_centroid (lat, lon)
          - high_risk_cells (count of cells with risk >= 0.60)
          - alert_level (NORMAL | DEVELOPING | HIGH_RISK | CLOUDBURST_LIKELY)
        """
        stn_grid = self.interpolate_station_scores(station_predictions)
        final_risk = self.fuse_with_satellite(stn_grid, satellite_grid)

        peak_idx = np.unravel_index(np.argmax(final_risk), final_risk.shape)
        peak_risk = float(final_risk[peak_idx])
        peak_lat = float(self.lats[peak_idx[0]])
        peak_lon = float(self.lons[peak_idx[1]])

        if peak_risk >= 0.80:
            alert = "CLOUDBURST_LIKELY"
        elif peak_risk >= 0.60:
            alert = "HIGH_RISK"
        elif peak_risk >= 0.30:
            alert = "DEVELOPING"
        else:
            alert = "NORMAL"

        return {
            "risk_matrix": final_risk,
            "lats": self.lats,
            "lons": self.lons,
            "peak_risk": round(peak_risk, 4),
            "peak_centroid": (round(peak_lat, 4), round(peak_lon, 4)),
            "high_risk_cell_count": int(np.sum(final_risk >= 0.60)),
            "cloudburst_likely_cell_count": int(np.sum(final_risk >= 0.80)),
            "alert_level": alert
        }
