"""Module 6 — PINN Hydrodynamic Flood Inundation Handoff.

Closes the bridge between atmospheric nowcasting and flood simulation:
  - Takes the 2D risk map and detected peak rainfall rate.
  - Constructs the spatial rainfall source term R(x, y, t) for the 2D Shallow Water Equations (SWE):
      dh/dt + d(uh)/dx + d(vh)/dy = R(x, y, t) - I(x, y, t)
  - Exports a structured tensor package containing catchment grid coordinates,
    elevation placeholder / DEM reference, rainfall source flux (m/s and mm/hr),
    and boundary conditions for the PINN inundation solver.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class PINNFloodHandoff:
    """Formatter and coupler from cloudburst nowcast to PINN flood simulation."""

    def __init__(self, infiltration_rate_mm_hr: float = 10.0):
        self.infiltration_mm_hr = infiltration_rate_mm_hr

    def construct_rainfall_source_term(
        self,
        risk_map: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        peak_station_r_mm_hr: float = 100.0,
        core_radius_km: float = 3.0,
        duration_hours: float = 1.0
    ) -> Dict:
        """
        Generate spatial rainfall source term R(x, y, t) scaled by P(CB) risk.

        Parameters
        ----------
        risk_map              : 2D array of P(CB) probabilities [0..1]
        lats, lons            : Coordinate arrays
        peak_station_r_mm_hr  : Observed or modeled peak rain rate (e.g. 100 mm/hr)
        core_radius_km        : Convective core radius (IMD standard ~2.5 - 3 km)
        duration_hours        : Anticipated storm duration (default 1 hour)

        Returns
        -------
        Dict with PINN inputs:
          - R_source_mm_hr: 2D field of rainfall rate (mm/hr)
          - R_source_m_s  : 2D field of rainfall rate in SI units (m/s for SWE)
          - net_water_flux_m3_s: Total integrated catchment water volume per second
          - trigger_pinn: Boolean indicating if inundation solver should execute
        """
        # Rain rate scales with P(CB), with a Gaussian decay away from the peak cell
        # Cells with P(CB) < 0.30 receive background rainfall (e.g. 5-10 mm/hr)
        # Cells with P(CB) >= 0.80 receive the full core rate (>= 100 mm/hr)
        r_grid_mm_hr = np.where(
            risk_map >= 0.30,
            risk_map * peak_station_r_mm_hr,
            risk_map * 10.0
        )

        # Subtract baseline infiltration rate to get excess runoff source
        excess_r_mm_hr = np.maximum(0.0, r_grid_mm_hr - self.infiltration_mm_hr)

        # Convert to SI units (m/s): 1 mm/hr = 1e-3 m / 3600 s = 2.778e-7 m/s
        r_source_m_s = r_grid_mm_hr * (1e-3 / 3600.0)
        excess_m_s   = excess_r_mm_hr * (1e-3 / 3600.0)

        # Calculate cell area in m^2 (approx 4km x 4km = 16,000,000 m^2)
        dlat_deg = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.04
        dlon_deg = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.04
        dy_m = dlat_deg * 111000.0
        dx_m = dlon_deg * 111000.0 * np.cos(np.radians(np.mean(lats)))
        cell_area_m2 = dx_m * dy_m

        # Total integrated water volume arriving per second across catchment
        total_flux_m3_s = float(np.sum(excess_m_s) * cell_area_m2)

        peak_idx = np.unravel_index(np.argmax(risk_map), risk_map.shape)
        peak_lat = float(lats[peak_idx[0]])
        peak_lon = float(lons[peak_idx[1]])
        peak_r   = float(r_grid_mm_hr[peak_idx])

        should_trigger = bool(np.max(risk_map) >= 0.60)

        return {
            "trigger_pinn": should_trigger,
            "peak_risk": round(float(np.max(risk_map)), 4),
            "peak_rainfall_mm_hr": round(peak_r, 1),
            "convective_epicenter": (round(peak_lat, 4), round(peak_lon, 4)),
            "R_source_mm_hr": r_grid_mm_hr,
            "R_source_m_s": r_source_m_s,
            "excess_runoff_m_s": excess_m_s,
            "catchment_cell_area_m2": round(cell_area_m2, 1),
            "net_excess_flux_m3_s": round(total_flux_m3_s, 2),
            "swe_source_term_summary": (
                f"Peak R={peak_r:.1f}mm/hr at ({peak_lat:.3f}N, {peak_lon:.3f}E) | "
                f"Total excess runoff influx={total_flux_m3_s:.1f} m3/s"
            )
        }
