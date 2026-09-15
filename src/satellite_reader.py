"""
Satellite Data Reader & Spatial Interpolator for MOSDAC Products.
Handles HDF5 formats for:
- Kalpana-1 VHRR QPE (K1VHR_L2B_QPE)
- INSAT-3D / 3DR Cloud Top Properties (3DIMG_L2B_CTP / 3RIMG_L2B_CTP)
  → Contains CTT (Cloud Top Temperature, K) and CTP (Cloud Top Pressure, hPa)
- INSAT-3D / 3DR Hydro-Estimator Rain (3DIMG_L2B_HEM / 3RIMG_L2B_HEM)
- INSAT-3D / 3DR Outgoing Longwave Radiation (3DIMG_L2B_OLR / 3RIMG_L2B_OLR)
- INSAT-3D / 3DR Upper Troposphere Humidity (3DIMG_L2B_UTH / 3RIMG_L2B_UTH)

File structure note (confirmed from inspection of real files):
  - CTT/CTP:  datasets shape (1, 313, 312), Latitude/Longitude shape (313, 312)
  - HEM:      dataset 'HEM' shape (varies), lat/lon 2D grids
  - OLR:      dataset 'OLR', lat/lon 2D grids
  - UTH:      dataset 'UTH', lat/lon 2D grids
  - Fill value: -999.0 for all L2B products

Provides high-resolution spatial clipping, quality filtering,
regular grid interpolation, and feature extraction for neural nowcasting.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import h5py
from scipy.interpolate import griddata


# ── Internal helpers ────────────────────────────────────────────────────────────

def _read_l2b_grid(
    h5_path: Path,
    data_key: str,
    lat_key: str = "Latitude",
    lon_key: str = "Longitude",
    scale: Optional[float] = None,
    fill_value: float = -999.0,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Generic reader for INSAT-3D / 3DR L2B HDF5 products.
    Handles:
    - 2D lat/lon grids (shape H×W)
    - Data arrays with leading time dimension (shape 1×H×W → squeezed to H×W)
    - Integer lat/lon with scale_factor (scale_factor=0.01 → divide by 100)
    - Fill value masking

    Returns flat 1D arrays of (lats, lons, values) after mask + bbox filtering.
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if data_key not in f:
            raise KeyError(
                f"Dataset '{data_key}' not found in {h5_path.name}. "
                f"Available: {list(f.keys())}"
            )

        # Handle Kalpana-1 structured datasets (e.g. UTH/UTH_Dataset)
        ds = f[data_key]
        is_structured = False
        if isinstance(ds, h5py.Group):
            sub_key = f"{data_key}_Dataset"
            if sub_key in ds:
                ds = ds[sub_key]
            else:
                for k, v in ds.items():
                    if isinstance(v, h5py.Dataset):
                        ds = v
                        break
        
        raw_data = ds[:]
        if raw_data.dtype.names is not None:
            # Structured array
            is_structured = True
            # Find the actual data column (e.g. 'UTH' or 'HEM' or 'OLR')
            col_name = data_key
            if col_name not in raw_data.dtype.names:
                # Try to guess
                for name in raw_data.dtype.names:
                    if name not in ["Latitude", "Longitude", "Time"]:
                        col_name = name
                        break
            data = raw_data[col_name].astype(np.float32)
            lats = raw_data["Latitude"].astype(np.float32)
            lons = raw_data["Longitude"].astype(np.float32)
            
            # Kalpana-1 scale factors
            ds_attrs = dict(ds.attrs)
            scale = float(ds_attrs.get("scale_factor", [1.0])[0]) if "scale_factor" in ds_attrs else 1.0
            if scale != 1.0:
                data = data * scale
        else:
            # Normal INSAT-3D flat datasets
            data = raw_data
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]  # (1, H, W) → (H, W)
            data = data.astype(np.float32)

            ds_attrs = dict(ds.attrs)
            if scale is None:
                scale = float(ds_attrs.get("scale_factor", [1.0])[0]) if "scale_factor" in ds_attrs else 1.0
            if scale != 1.0:
                data = data * scale

            # Read lat / lon for flat dataset
            if lat_key not in f or lon_key not in f:
                raise KeyError(
                    f"Lat/Lon keys '{lat_key}'/'{lon_key}' not found in {h5_path.name}."
                )
            lats_raw = f[lat_key][:]
            lons_raw = f[lon_key][:]

            lat_attrs = dict(f[lat_key].attrs)
            lon_attrs = dict(f[lon_key].attrs)
            lat_scale = float(lat_attrs.get("scale_factor", [1.0])[0]) if "scale_factor" in lat_attrs else 1.0
            lon_scale = float(lon_attrs.get("scale_factor", [1.0])[0]) if "scale_factor" in lon_attrs else 1.0

            lats = lats_raw.astype(np.float32) * lat_scale
            lons = lons_raw.astype(np.float32) * lon_scale

        # If lat/lon are 1D axis vectors, meshgrid them to match data shape
        if lats.ndim == 1 and lons.ndim == 1 and data.ndim > 1:
            lons, lats = np.meshgrid(lons, lats)  # (H, W)

    # Flatten to 1D
    lats_flat = lats.ravel()
    lons_flat = lons.ravel()
    data_flat = data.ravel()

    # Mask fill values and NaN
    valid = (data_flat != fill_value) & (~np.isnan(data_flat)) & (~np.isinf(data_flat))

    # Mask implausible lat/lon fill values (INSAT uses 31172 × scale = ~311.72 as fill)
    valid &= (lats_flat >= -90.0) & (lats_flat <= 90.0)
    valid &= (lons_flat >= -180.0) & (lons_flat <= 180.0)

    # Physical range clipping
    if valid_min is not None:
        valid &= (data_flat >= valid_min)
    if valid_max is not None:
        valid &= (data_flat <= valid_max)

    lats_flat, lons_flat, data_flat = lats_flat[valid], lons_flat[valid], data_flat[valid]

    # Bounding box clipping
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        box = (
            (lats_flat >= lat_min) & (lats_flat <= lat_max) &
            (lons_flat >= lon_min) & (lons_flat <= lon_max)
        )
        lats_flat, lons_flat, data_flat = lats_flat[box], lons_flat[box], data_flat[box]

    return {"lats": lats_flat, "lons": lons_flat, "values": data_flat, "file": h5_path.name}


# ── Public SatelliteReader class ────────────────────────────────────────────────

class SatelliteReader:
    """
    Robust reader and spatial processor for MOSDAC satellite HDF5 products.
    """

    @staticmethod
    def inspect_file(h5_path: Union[str, Path]) -> Dict:
        """Inspect structure, groups, datasets, and metadata of an HDF5 file."""
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        info = {"path": str(h5_path), "keys": [], "datasets": {}, "attrs": {}}
        with h5py.File(h5_path, "r") as f:
            info["keys"] = list(f.keys())
            for k in f.attrs.keys():
                info["attrs"][k] = str(f.attrs[k])

            def _visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info["datasets"][name] = {
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "attrs": {ak: str(obj.attrs[ak]) for ak in obj.attrs.keys()}
                    }

            f.visititems(_visitor)
        return info

    @staticmethod
    def read_qpe(
        h5_path: Union[str, Path],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        min_valid: float = 0.0,
        max_valid: float = 300.0,
    ) -> Dict[str, np.ndarray]:
        """
        Read Kalpana-1 or INSAT-3D Quantitative Precipitation Estimation (QPE).

        Args:
            h5_path: Path to HDF5 file
            bbox: (lat_min, lat_max, lon_min, lon_max) or None for full domain
            min_valid: Minimum physical rainfall rate (mm/hr)
            max_valid: Maximum physical rainfall rate (mm/hr)

        Returns:
            Dict containing:
                'lats': 1D array of latitudes
                'lons': 1D array of longitudes
                'qpe': 1D array of rainfall rates (mm/hr)
                'file': filename
        """
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"File not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            # Handle structured dataset format (e.g. Kalpana-1 QPE_Dataset)
            if "QPE/QPE_Dataset" in f:
                ds = f["QPE/QPE_Dataset"][:]
                lats = ds["Latitude"].astype(np.float32)
                lons = ds["Longitude"].astype(np.float32)
                qpe = ds["QPE"].astype(np.float32)
            elif "QPE" in f and isinstance(f["QPE"], h5py.Dataset):
                qpe = f["QPE"][:].astype(np.float32)
                lats = f["Latitude"][:].astype(np.float32) if "Latitude" in f else None
                lons = f["Longitude"][:].astype(np.float32) if "Longitude" in f else None
            else:
                # Search for any rainfall-like dataset
                target_key = None
                for k in f.keys():
                    if "QPE" in k or "RAIN" in k or "HEM" in k:
                        target_key = k
                        break
                if target_key and isinstance(f[target_key], h5py.Dataset):
                    qpe = f[target_key][:].astype(np.float32)
                    lats = f.get("Latitude", np.array([]))[:].astype(np.float32)
                    lons = f.get("Longitude", np.array([]))[:].astype(np.float32)
                else:
                    raise KeyError(f"Could not locate QPE dataset in {h5_path}. Available keys: {list(f.keys())}")

        # Quality filter
        valid_mask = (qpe >= min_valid) & (qpe <= max_valid) & (~np.isnan(qpe)) & (~np.isinf(qpe))
        lats, lons, qpe = lats[valid_mask], lons[valid_mask], qpe[valid_mask]

        # Spatial bounding box clipping
        if bbox is not None:
            lat_min, lat_max, lon_min, lon_max = bbox
            box_mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
            lats, lons, qpe = lats[box_mask], lons[box_mask], qpe[box_mask]

        return {
            "lats": lats,
            "lons": lons,
            "qpe": qpe,
            "file": h5_path.name
        }

    @staticmethod
    def read_ctt(
        h5_path: Union[str, Path],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        return_kelvin: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Read Cloud Top Temperature (CTT) from INSAT-3D/3DR L2B_CTP product.

        The real 3DIMG_L2B_CTP files have:
          - Dataset 'CTT'     : shape (1, 313, 312), units K, fill=-999.0
          - Dataset 'CTP'     : shape (1, 313, 312), units hPa
          - Dataset 'Latitude': shape (313, 312), scale_factor=0.01
          - Dataset 'Longitude': shape (313, 312), scale_factor=0.01

        Args:
            h5_path: Path to HDF5 file
            bbox: (lat_min, lat_max, lon_min, lon_max) or None
            return_kelvin: If True, keep in Kelvin. If False, convert to Celsius.

        Returns:
            Dict with 'lats', 'lons', 'ctt' (flat 1D arrays), 'ctp', 'file'
        """
        h5_path = Path(h5_path)

        ctt_data = _read_l2b_grid(
            h5_path, data_key="CTT",
            valid_min=150.0, valid_max=340.0,  # plausible cloud top temp range in K
            bbox=bbox
        )

        # Also read CTP (cloud top pressure) for the same valid pixels
        try:
            ctp_data = _read_l2b_grid(
                h5_path, data_key="CTP",
                valid_min=50.0, valid_max=1050.0,  # hPa
                bbox=bbox
            )
            ctp_values = ctp_data["values"]
        except Exception:
            ctp_values = np.full_like(ctt_data["values"], np.nan)

        ctt_values = ctt_data["values"]
        if not return_kelvin:
            ctt_values = ctt_values - 273.15

        return {
            "lats": ctt_data["lats"],
            "lons": ctt_data["lons"],
            "ctt": ctt_values,
            "ctp": ctp_values,
            "file": h5_path.name
        }

    @staticmethod
    def read_hem(
        h5_path: Union[str, Path],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Read Hydro-Estimator rainfall rate from INSAT-3D/3DR L2B_HEM product.

        Returns:
            Dict with 'lats', 'lons', 'hem' (mm/hr), 'file'
        """
        h5_path = Path(h5_path)

        # Try common HEM dataset names
        for key in ("HEM", "HydroEstimator", "RAIN", "QPE"):
            try:
                result = _read_l2b_grid(
                    h5_path, data_key=key,
                    valid_min=0.0, valid_max=400.0,
                    bbox=bbox
                )
                return {
                    "lats": result["lats"],
                    "lons": result["lons"],
                    "hem": result["values"],
                    "file": h5_path.name
                }
            except KeyError:
                continue

        # Fallback: inspect and pick first plausible dataset
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
        raise KeyError(
            f"Could not find HEM dataset in {h5_path.name}. Available: {keys}"
        )

    @staticmethod
    def read_olr(
        h5_path: Union[str, Path],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Read Outgoing Longwave Radiation from INSAT-3D/3DR L2B_OLR product.
        Units: W/m². Typical range: 100–340 W/m². Low OLR = deep convective cloud.

        Returns:
            Dict with 'lats', 'lons', 'olr' (W/m²), 'file'
        """
        h5_path = Path(h5_path)

        for key in ("OLR", "Olr", "olr"):
            try:
                result = _read_l2b_grid(
                    h5_path, data_key=key,
                    valid_min=50.0, valid_max=400.0,
                    bbox=bbox
                )
                return {
                    "lats": result["lats"],
                    "lons": result["lons"],
                    "olr": result["values"],
                    "file": h5_path.name
                }
            except KeyError:
                continue

        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
        raise KeyError(
            f"Could not find OLR dataset in {h5_path.name}. Available: {keys}"
        )

    @staticmethod
    def read_uth(
        h5_path: Union[str, Path],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Read Upper Troposphere Humidity from INSAT-3D/3DR L2B_UTH product.
        Units: %. Range: 0–100%.

        Returns:
            Dict with 'lats', 'lons', 'uth' (%), 'file'
        """
        h5_path = Path(h5_path)

        for key in ("UTH", "Uth", "uth", "WVR"):
            try:
                result = _read_l2b_grid(
                    h5_path, data_key=key,
                    valid_min=0.0, valid_max=110.0,
                    bbox=bbox
                )
                return {
                    "lats": result["lats"],
                    "lons": result["lons"],
                    "uth": result["values"],
                    "file": h5_path.name
                }
            except KeyError:
                continue

        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
        raise KeyError(
            f"Could not find UTH dataset in {h5_path.name}. Available: {keys}"
        )

    @staticmethod
    def extract_point_value(
        lats: np.ndarray,
        lons: np.ndarray,
        values: np.ndarray,
        target_lat: float,
        target_lon: float,
        radius_km: float = 50.0,
    ) -> Optional[float]:
        """
        Compute the spatial mean of 'values' within radius_km of (target_lat, target_lon).

        Uses Haversine distance. Returns None if no valid points within radius.
        """
        if len(values) == 0:
            return None

        # Haversine distance (fast vectorised, ~flat Earth approximation is OK for 50km)
        dlat = np.radians(lats - target_lat)
        dlon = np.radians(lons - target_lon)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(target_lat)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2)
        dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        within = dist_km <= radius_km
        if not within.any():
            return None
        return float(np.nanmean(values[within]))

    @staticmethod
    def extract_point_value_knn(
        lats: np.ndarray,
        lons: np.ndarray,
        values: np.ndarray,
        target_lat: float,
        target_lon: float,
        k: int = 4,
        max_radius_km: float = 250.0,
    ) -> Dict[str, float]:
        """
        Compute the spatial mean of 'values' using KNN+IDW near (target_lat, target_lon).

        Args:
            k: Number of nearest neighbors to average.
            max_radius_km: Distance cap. If the closest pixel is beyond this, returns NaN (invalid).

        Returns:
            Dict containing:
                - val: IDW mean (or NaN if invalid)
                - dist_km: Distance to nearest pixel in km (or NaN if values empty)
                - valid: 1.0 if valid (within max_radius_km), 0.0 otherwise.
        """
        if len(values) == 0:
            return {"val": np.nan, "dist_km": np.nan, "valid": 0.0}

        # Haversine distance
        dlat = np.radians(lats - target_lat)
        dlon = np.radians(lons - target_lon)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(target_lat)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2)
        dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        min_idx = np.argmin(dist_km)
        min_dist = float(dist_km[min_idx])

        if min_dist > max_radius_km:
            return {"val": np.nan, "dist_km": min_dist, "valid": 0.0}

        # KNN
        if len(values) <= k:
            k_dists = dist_km
            k_vals = values
        else:
            sorted_idx = np.argsort(dist_km)
            k_dists = dist_km[sorted_idx[:k]]
            k_vals = values[sorted_idx[:k]]

        # IDW
        weights = 1.0 / (k_dists + 1.0)**2.0
        idw_val = float(np.sum(weights * k_vals) / np.sum(weights))

        return {"val": idw_val, "dist_km": min_dist, "valid": 1.0}

    @staticmethod
    def extract_points_knn(
        lats: np.ndarray,
        lons: np.ndarray,
        values: np.ndarray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
        k: int = 4,
        max_radius_km: float = 250.0,
    ) -> Dict[str, np.ndarray]:
        """
        Fast, vectorized KNN+IDW using BallTree for many target points simultaneously.
        """
        n_targets = len(target_lats)
        if len(values) == 0:
            return {
                "val": np.full(n_targets, np.nan),
                "dist_km": np.full(n_targets, np.nan),
                "valid": np.zeros(n_targets)
            }

        from sklearn.neighbors import BallTree
        # Convert degrees to radians for BallTree haversine metric
        src_rad = np.column_stack((np.radians(lats), np.radians(lons)))
        target_rad = np.column_stack((np.radians(target_lats), np.radians(target_lons)))

        tree = BallTree(src_rad, metric='haversine')
        
        # Query nearest k neighbors (radius returned in radians)
        k = min(k, len(values))
        dists_rad, indices = tree.query(target_rad, k=k)
        
        # Convert radians to km
        dists_km = dists_rad * 6371.0
        
        # Output arrays
        out_val = np.full(n_targets, np.nan)
        out_dist = np.full(n_targets, np.nan)
        out_valid = np.zeros(n_targets)

        for i in range(n_targets):
            min_dist = dists_km[i, 0]
            out_dist[i] = min_dist
            
            if min_dist <= max_radius_km:
                out_valid[i] = 1.0
                k_d = dists_km[i]
                k_v = values[indices[i]]
                weights = 1.0 / (k_d + 1.0)**2.0
                out_val[i] = np.sum(weights * k_v) / np.sum(weights)

        return {"val": out_val, "dist_km": out_dist, "valid": out_valid}

    @staticmethod
    def interpolate_to_grid(
        lats: np.ndarray,
        lons: np.ndarray,
        values: np.ndarray,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        method: str = "linear",
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Interpolate scattered satellite observations onto a 2D regular grid.

        Args:
            lats: 1D array of observed latitudes
            lons: 1D array of observed longitudes
            values: 1D array of values (e.g. QPE mm/hr)
            grid_lat: 1D or 2D target latitudes
            grid_lon: 1D or 2D target longitudes
            method: 'linear', 'nearest', or 'cubic'
            fill_value: Value to fill outside convex hull

        Returns:
            2D numpy array of shape (len(grid_lat), len(grid_lon))
        """
        if len(values) == 0:
            return np.full((len(grid_lat), len(grid_lon)), fill_value, dtype=np.float32)

        # Build 2D mesh if inputs are 1D
        if grid_lat.ndim == 1 and grid_lon.ndim == 1:
            glon, glat = np.meshgrid(grid_lon, grid_lat)
        else:
            glat, glon = grid_lat, grid_lon

        points = np.column_stack((lons, lats))
        grid_val = griddata(points, values, (glon, glat), method=method, fill_value=fill_value)
        return grid_val.astype(np.float32)

    @classmethod
    def extract_convective_metrics(
        cls,
        qpe_grid: np.ndarray,
        heavy_threshold_mm_hr: float = 30.0,
        extreme_threshold_mm_hr: float = 50.0
    ) -> Dict[str, float]:
        """
        Compute convective signature metrics from satellite precipitation grid.
        Useful for feeding spatial context into Gate A / Gate B.
        """
        valid_qpe = qpe_grid[~np.isnan(qpe_grid)]
        if len(valid_qpe) == 0:
            return {
                "sat_max_rain": 0.0,
                "sat_mean_rain": 0.0,
                "convective_area_frac": 0.0,
                "extreme_core_detected": 0.0
            }

        max_rain = float(np.max(valid_qpe))
        mean_rain = float(np.mean(valid_qpe))
        convective_pixels = np.sum(valid_qpe >= heavy_threshold_mm_hr)
        convective_frac = float(convective_pixels / len(valid_qpe))
        extreme_flag = 1.0 if max_rain >= extreme_threshold_mm_hr else 0.0

        return {
            "sat_max_rain": max_rain,
            "sat_mean_rain": mean_rain,
            "convective_area_frac": convective_frac,
            "extreme_core_detected": extreme_flag
        }
