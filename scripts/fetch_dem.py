import os
import sys
import math
import io
import time
import urllib.request
import numpy as np

# Try importing tifffile for GeoTIFF handling
try:
    import tifffile
except ImportError:
    print("[ERROR] tifffile library is missing. Install with pip install tifffile")
    sys.exit(1)

DEM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dem")

# 8 Uttarakhand Region Anchor Tables (Real Lat/Lon, Target Elevation, y_norm)
REGIONS = {
    "rudraprayag": [
        ("Kedarnath Shrine", 30.7346, 79.0669, 3583.0, 0.05),
        ("Rambara Gorge", 30.6800, 79.0400, 2700.0, 0.32),
        ("Sonprayag Choke", 30.6240, 79.0030, 1829.0, 0.66),
        ("Rudraprayag Confluence", 30.2849, 78.9814, 610.0, 0.95)
    ],
    "chamoli": [
        ("Badrinath Shrine", 30.7433, 79.4938, 3133.0, 0.05),
        ("Joshimath Town", 30.5546, 79.5643, 1875.0, 0.38),
        ("Tapovan Barrage", 30.5667, 79.5333, 1350.0, 0.55),
        ("Chamoli HQ", 30.4024, 79.3323, 950.0, 0.95)
    ],
    "uttarkashi": [
        ("Gangotri", 30.9946, 78.9398, 3048.0, 0.05),
        ("Maneri Dam", 30.8667, 78.7833, 1320.0, 0.60),
        ("Uttarkashi HQ", 30.7268, 78.4354, 1165.0, 0.75),
        ("Tehri Reservoir", 30.3783, 78.4805, 650.0, 0.95)
    ],
    "pithoragarh": [
        ("Munsiari", 30.0668, 80.2374, 2200.0, 0.08),
        ("Dharchula", 29.8452, 80.5423, 915.0, 0.50),
        ("Jauljibi", 29.7167, 80.3667, 600.0, 0.90)
    ],
    "tehri": [
        ("Khatling Glacier", 30.8667, 78.9333, 2500.0, 0.08),
        ("New Tehri Town", 30.3841, 78.4802, 1550.0, 0.55),
        ("Devprayag", 30.1462, 78.5978, 520.0, 0.92)
    ],
    "pauri": [
        ("Pauri HQ", 30.1462, 78.7642, 1800.0, 0.10),
        ("Srinagar Garhwal", 30.2280, 78.7803, 560.0, 0.50),
        ("Devprayag", 30.1462, 78.5978, 472.0, 0.75),
        ("Rishikesh", 30.0869, 78.2676, 350.0, 0.95)
    ],
    "nainital": [
        ("Nainital Town", 29.3803, 79.4636, 2084.0, 0.08),
        ("Bhimtal", 29.3494, 79.5636, 1371.0, 0.30),
        ("Haldwani", 29.2183, 79.5130, 424.0, 0.80),
        ("Rudrapur (Terai)", 28.9875, 79.4304, 280.0, 0.95)
    ],
    "almora": [
        ("Almora HQ", 29.5971, 79.6591, 1650.0, 0.10),
        ("Someshwar Gorge", 29.6667, 79.6833, 750.0, 0.50),
        ("Ramganga East Confluence", 29.4500, 79.7500, 350.0, 0.95)
    ]
}

def lon2tile(lon, zoom):
    return int(math.floor((lon + 180.0) / 360.0 * (1 << zoom)))

def lat2tile(lat, zoom):
    lat_rad = math.radians(lat)
    return int(math.floor((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * (1 << zoom)))

def tile2lon(x, zoom):
    return x / (1 << zoom) * 360.0 - 180.0

def tile2lat(y, zoom):
    n = math.pi - 2.0 * math.pi * y / (1 << zoom)
    return math.degrees(math.atan(math.sinh(n)))

def compute_bbox_for_anchors(anchors, padding=0.15):
    lats = [a[1] for a in anchors]
    lons = [a[2] for a in anchors]
    return {
        "south": min(lats) - padding,
        "north": max(lats) + padding,
        "west": min(lons) - padding,
        "east": max(lons) + padding
    }

def download_region_srtm_dem(region_name, bbox):
    os.makedirs(DEM_DIR, exist_ok=True)
    raw_geotiff_path = os.path.join(DEM_DIR, f"{region_name}_srtm30.tif")
    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")

    if api_key:
        print(f"[{region_name}] Attempting OpenTopography API download...")
        url = (f"https://portal.opentopography.org/API/globaldem?"
               f"demtype=SRTMGL1&south={bbox['south']:.4f}&north={bbox['north']:.4f}&"
               f"west={bbox['west']:.4f}&east={bbox['east']:.4f}&outputFormat=GTiff&API_Key={api_key}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp:
                tif_bytes = resp.read()
                if len(tif_bytes) > 10000:
                    with open(raw_geotiff_path, "wb") as f:
                        f.write(tif_bytes)
                    print(f"[{region_name}] Downloaded OpenTopography GeoTIFF -> {raw_geotiff_path}")
                    raster = tifffile.imread(raw_geotiff_path)
                    return raster, (bbox['south'], bbox['north'], bbox['west'], bbox['east'])
        except Exception as e:
            print(f"[{region_name}] OpenTopography download failed ({e}), falling back to AWS S3 SRTM 30m tiles.")

    # AWS Terrain Tiles fallback (zoom level 11 with parallel thread pool)
    zoom = 11
    tx_min = lon2tile(bbox["west"], zoom)
    tx_max = lon2tile(bbox["east"], zoom)
    ty_min = lat2tile(bbox["north"], zoom)
    ty_max = lat2tile(bbox["south"], zoom)

    tile_coords = [(tx, ty) for ty in range(ty_min, ty_max + 1) for tx in range(tx_min, tx_max + 1)]
    print(f"[{region_name}] Downloading {len(tile_coords)} SRTM 30m tiles in parallel: X=[{tx_min}..{tx_max}], Y=[{ty_min}..{ty_max}]...")
    
    def fetch_single_tile(coord):
        tx, ty = coord
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{tx}/{ty}.tif"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    tdata = resp.read()
                    return coord, tifffile.imread(io.BytesIO(tdata))
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(0.5 * (attempt + 1))

    from concurrent.futures import ThreadPoolExecutor
    tiles = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = executor.map(fetch_single_tile, tile_coords)
        for coord, arr in results:
            tiles[coord] = arr

    tile_h, tile_w = list(tiles.values())[0].shape
    n_tiles_y = (ty_max - ty_min + 1)
    n_tiles_x = (tx_max - tx_min + 1)

    full_raster = np.zeros((n_tiles_y * tile_h, n_tiles_x * tile_w), dtype=np.float32)
    for (tx, ty), arr in tiles.items():
        row_idx = ty - ty_min
        col_idx = tx - tx_min
        full_raster[row_idx*tile_h:(row_idx+1)*tile_h, col_idx*tile_w:(col_idx+1)*tile_w] = arr

    lon_min = tile2lon(tx_min, zoom)
    lon_max = tile2lon(tx_max + 1, zoom)
    lat_max = tile2lat(ty_min, zoom)
    lat_min = tile2lat(ty_max + 1, zoom)

    tifffile.imwrite(raw_geotiff_path, full_raster.astype(np.int16))
    return full_raster, (lat_min, lat_max, lon_min, lon_max)

def sample_elevation(lat, lon, raster, bounds):
    lat_min, lat_max, lon_min, lon_max = bounds
    h, w = raster.shape
    py = int((lat_max - lat) / (lat_max - lat_min) * (h - 1))
    px = int((lon - lon_min) / (lon_max - lon_min) * (w - 1))
    py = max(0, min(h - 1, py))
    px = max(0, min(w - 1, px))
    return float(raster[py, px])

def process_region_grid(region_name, anchors):
    grid_npz_path = os.path.join(DEM_DIR, f"{region_name}_valley_grid.npz")
    alias_npz_path = os.path.join(DEM_DIR, "mandakini_valley_grid.npz") if region_name == "rudraprayag" else None

    # Check if already processed
    if os.path.exists(grid_npz_path):
        print(f"[{region_name}] Found existing DEM grid -> {grid_npz_path} (Skipping fetch)")
        data = np.load(grid_npz_path)
        z_meters = data["z_meters"]
    else:
        bbox = compute_bbox_for_anchors(anchors)
        raster, bounds = download_region_srtm_dem(region_name, bbox)

        nx, ny = 51, 51
        x_norm = np.linspace(0.0, 1.0, nx)
        y_norm = np.linspace(0.0, 1.0, ny)

        anchor_ys = [a[4] for a in anchors]
        anchor_lats = [a[1] for a in anchors]
        anchor_lons = [a[2] for a in anchors]

        z_meters = np.zeros((ny, nx), dtype=np.float32)
        cross_width_deg = 0.035

        for i, y_val in enumerate(y_norm):
            lat_c = np.interp(y_val, anchor_ys, anchor_lats)
            lon_c = np.interp(y_val, anchor_ys, anchor_lons)

            eps = 0.01
            lat_next = np.interp(min(1.0, y_val + eps), anchor_ys, anchor_lats)
            lon_next = np.interp(min(1.0, y_val + eps), anchor_ys, anchor_lons)

            dlat = lat_next - lat_c
            dlon = lon_next - lon_c
            norm = math.hypot(dlat, dlon)
            if norm == 0:
                dlat, dlon = -1.0, 0.0
            else:
                dlat /= norm
                dlon /= norm

            nx_lat = -dlon
            nx_lon = dlat

            for j, x_val in enumerate(x_norm):
                offset = (x_val - 0.5) * cross_width_deg
                pt_lat = lat_c + offset * nx_lat
                pt_lon = lon_c + offset * nx_lon

                elev = sample_elevation(pt_lat, pt_lon, raster, bounds)
                z_meters[i, j] = elev

        np.savez_compressed(
            grid_npz_path,
            x_norm=np.linspace(0.0, 1.0, 51),
            y_norm=np.linspace(0.0, 1.0, 51),
            z_meters=z_meters
        )
        if alias_npz_path:
            np.savez_compressed(alias_npz_path, x_norm=np.linspace(0.0, 1.0, 51), y_norm=np.linspace(0.0, 1.0, 51), z_meters=z_meters)
        print(f"[{region_name}] Exported DEM grid (shape {z_meters.shape}) to: {grid_npz_path}")

    # Validation Report for Region
    print("\n" + "="*75)
    print(f"SRTM 30m DEM VALIDATION REPORT: {region_name.upper()} BASIN")
    print("="*75)
    print(f"{'Anchor Point':<26} | {'Target (m)':<10} | {'SRTM DEM (m)':<12} | {'Diff (m)':<10} | Status")
    print("-" * 75)

    ny, nx = z_meters.shape
    for name, lat, lon, target_z, y_val in anchors:
        y_idx = int(round(y_val * (ny - 1)))
        x_idx = nx // 2
        dem_z = z_meters[y_idx, x_idx]
        diff = dem_z - target_z
        flag = "[FLAG >150m]" if abs(diff) > 150.0 else "[OK]"
        print(f"{name:<26} | {target_z:<10.1f} | {dem_z:<12.1f} | {diff:<+10.1f} | {flag}")
    print("="*75 + "\n")

def fetch_all_regions():
    print(f"=== FETCHING REAL SRTM 30m DEM DATA FOR ALL 8 UTTARAKHAND BASINS ===")
    for region_name, anchors in REGIONS.items():
        process_region_grid(region_name, anchors)
        time.sleep(0.3)  # Respectful pause between region downloads

if __name__ == "__main__":
    fetch_all_regions()
