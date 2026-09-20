import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_virtual_grid(spacing_deg=0.075, physical_stations=None, max_dist_km=30.0):
    """
    Generate an 8km spatial grid (approx 0.075 degrees) masked strictly to 
    the administrative boundaries of Uttarakhand and Assam, and filtered to 
    only include points within a trusted radius of physical AWS stations.
    """
    geojson_path = ROOT / "data" / "indian_states.geojson"
    if not geojson_path.exists():
        print(f"Warning: {geojson_path} not found. Returning empty virtual grid.")
        return []
    
    # Load states
    gdf = gpd.read_file(geojson_path)
    
    # State names might vary in different geojsons, usually "Uttarakhand" or "Uttaranchal"
    # Let's find the correct names.
    # Actually, we can just filter by substring.
    target_states = gdf[gdf['NAME_1'].str.contains('Uttarakhand|Assam|Uttaranchal', case=False, na=False)]
    
    if target_states.empty:
        # Fallback if the key isn't NAME_1
        for col in gdf.columns:
            if gdf[col].dtype == 'object':
                matches = gdf[gdf[col].astype(str).str.contains('Uttarakhand|Assam|Uttaranchal', case=False, na=False)]
                if not matches.empty:
                    target_states = matches
                    break
    
    if target_states.empty:
        print("Warning: Could not find Uttarakhand or Assam in GeoJSON. Returning empty virtual grid.")
        return []

    virtual_grid = []
    vid = 1
    
    for _, state_row in target_states.iterrows():
        geom = state_row.geometry
        minx, miny, maxx, maxy = geom.bounds
        
        # Create grid points over the bounding box
        x_coords = np.arange(minx, maxx, spacing_deg)
        y_coords = np.arange(miny, maxy, spacing_deg)
        
        # This can be vectorized, but for a one-time startup generation, looping is fine
        for y in y_coords:
            for x in x_coords:
                pt = Point(x, y)
                if geom.contains(pt):
                    # Filter by distance to nearest physical station if provided
                    if physical_stations:
                        min_dist = min(haversine(y, x, s["lat"], s["lng"]) for s in physical_stations)
                        if min_dist > max_dist_km:
                            continue

                    virtual_grid.append({
                        "id": f"V-{vid:04d}",
                        "lat": float(y),
                        "lng": float(x)
                    })
                    vid += 1
                    
    print(f"Generated {len(virtual_grid)} virtual coordinate points inside trusted radius.")
    return virtual_grid

if __name__ == "__main__":
    grid = generate_virtual_grid()
    if grid:
        print(f"First 5 virtual stations: {grid[:5]}")
