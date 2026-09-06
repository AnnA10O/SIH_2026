import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.spatial_fusion import SpatialFusionGrid

# 1. Define localized Guwahati AWS cluster with real coordinates and realistic convective episode P(CB)
stations = [
    {"name": "Khanapara (Guwahati Core)", "lat": 26.1206, "lon": 91.8164, "p_cb": 0.88, "tier": "RED"},
    {"name": "North Guwahati College",   "lat": 26.2022, "lon": 91.7158, "p_cb": 0.65, "tier": "ORANGE"},
    {"name": "Rangiya College",          "lat": 26.4269, "lon": 91.6139, "p_cb": 0.35, "tier": "YELLOW"},
    {"name": "Nalbari AWS",              "lat": 26.4358, "lon": 91.3350, "p_cb": 0.12, "tier": "GREEN"},
    {"name": "Bongaigaon AWS",           "lat": 26.5231, "lon": 90.8911, "p_cb": 0.04, "tier": "GREEN"},
    {"name": "Salonah AWS",              "lat": 26.4458, "lon": 92.5750, "p_cb": 0.18, "tier": "GREEN"}
]

# 2. Initialize 4km Grid around Guwahati region
lat_min, lat_max = 25.90, 26.70
lon_min, lon_max = 90.70, 92.70
grid_res = 0.04 # ~4.4 km

grid_engine = SpatialFusionGrid(
    lat_min=lat_min, lat_max=lat_max,
    lon_min=lon_min, lon_max=lon_max,
    grid_res_deg=grid_res, idw_power=2.0
)

# 3. IDW Interpolation (Step 2 in spatial_fusion.py)
idw_grid = grid_engine.interpolate_station_scores(stations)

# 4. Gaussian Kernel Interpolation (Alternative)
stn_lats = np.array([s["lat"] for s in stations])
stn_lons = np.array([s["lon"] for s in stations])
stn_p    = np.array([s["p_cb"] for s in stations])

cos_lat = np.cos(np.radians(0.5 * (lat_min + lat_max)))
sigma_km = 14.0 # convective correlation radius (~14 km)

g_weights_sum = np.zeros(grid_engine.shape, dtype=float)
g_val_sum = np.zeros(grid_engine.shape, dtype=float)

for lat_s, lon_s, p_s in zip(stn_lats, stn_lons, stn_p):
    dy = (grid_engine.lat_grid - lat_s) * 111.0
    dx = (grid_engine.lon_grid - lon_s) * 111.0 * cos_lat
    dist_km = np.sqrt(dx**2 + dy**2)
    
    # Gaussian radial basis: exp(-d^2 / (2 * sigma^2))
    w_gauss = np.exp(-(dist_km**2) / (2.0 * (sigma_km**2)))
    g_val_sum += w_gauss * p_s
    g_weights_sum += w_gauss

gauss_grid = np.where(g_weights_sum > 0, g_val_sum / g_weights_sum, 0.0)

# 5. Work out detailed math for an ungauged intermediate point between Khanapara & Nalbari
# Target test point: Jalukbari / Gauhati University (26.15°N, 91.66°E)
target_lat, target_lon = 26.15, 91.66
print("=== Step-by-Step Math for Ungauged Point (26.15°N, 91.66°E) ===")
total_idw_w, total_idw_val = 0.0, 0.0
total_g_w, total_g_val = 0.0, 0.0

for s in stations:
    dy = (target_lat - s["lat"]) * 111.0
    dx = (target_lon - s["lon"]) * 111.0 * cos_lat
    dist = np.sqrt(dx**2 + dy**2)
    dist_clamped = max(dist, 2.0)
    w_idw = 1.0 / (dist_clamped ** 2.0)
    w_g = np.exp(-(dist**2) / (2.0 * (sigma_km**2)))
    
    total_idw_w += w_idw
    total_idw_val += w_idw * s["p_cb"]
    total_g_w += w_g
    total_g_val += w_g * s["p_cb"]
    print(f"Station: {s['name'][:22]:22s} | P(CB)={s['p_cb']:.2f} | Dist={dist:5.1f}km | IDW_w={w_idw:.6f} | Gauss_w={w_g:.6f}")

interp_idw = total_idw_val / total_idw_w
interp_gauss = total_g_val / total_g_w
print("-" * 75)
print(f"Interpolated P(CB) via IDW (power=2):     {interp_idw:.4f}")
print(f"Interpolated P(CB) via Gaussian (sigma=14km): {interp_gauss:.4f}")

# 6. Generate 4-Panel Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=200)

# Panel 1: Discrete AWS Sensors
ax1 = axes[0, 0]
ax1.set_title("(A) Discrete In-Situ AWS Sensors (Sparse Points)", fontsize=13, fontweight="bold", pad=10)
# draw 4km mesh lines lightly
for lat in grid_engine.lats:
    ax1.axhline(lat, color="#e2e8f0", lw=0.5, zorder=1)
for lon in grid_engine.lons:
    ax1.axvline(lon, color="#e2e8f0", lw=0.5, zorder=1)

tier_colors = {"RED": "#ef4444", "ORANGE": "#f97316", "YELLOW": "#eab308", "GREEN": "#22c55e"}
for s in stations:
    col = tier_colors[s["tier"]]
    ax1.scatter(s["lon"], s["lat"], s=180, color=col, edgecolors="black", lw=1.5, zorder=5)
    ax1.text(s["lon"] + 0.03, s["lat"] + 0.02, f"{s['name'].split()[0]}\nP={s['p_cb']:.2f}",
             fontsize=9, fontweight="bold", color="#0f172a",
             path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=6)

ax1.scatter([target_lon], [target_lat], s=140, marker="*", color="#8b5cf6", edgecolors="black", lw=1.5, zorder=7)
ax1.text(target_lon + 0.03, target_lat - 0.04, "Ungauged Point\n(Jalukbari 26.15, 91.66)",
         fontsize=9, fontweight="bold", color="#6d28d9",
         path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=8)

ax1.set_xlim(lon_min, lon_max)
ax1.set_ylim(lat_min, lat_max)
ax1.set_xlabel("Longitude (°E)", fontsize=11)
ax1.set_ylabel("Latitude (°N)", fontsize=11)
ax1.grid(False)

# Panel 2: Continuous IDW 4km Mesh
ax2 = axes[0, 1]
ax2.set_title("(B) Continuous 4km Mesh: Inverse Distance Weighting (IDW, p=2)", fontsize=13, fontweight="bold", pad=10)
c2 = ax2.contourf(grid_engine.lon_grid, grid_engine.lat_grid, idw_grid, levels=np.linspace(0, 1, 21), cmap="turbo")
plt.colorbar(c2, ax=ax2, label="Interpolated P(Cloudburst)")
ax2.contour(grid_engine.lon_grid, grid_engine.lat_grid, idw_grid, levels=[0.30, 0.60, 0.80], colors=["yellow", "orange", "red"], linewidths=1.8)
for s in stations:
    ax2.scatter(s["lon"], s["lat"], s=70, color="white", edgecolors="black", lw=1.2, zorder=5)
ax2.scatter([target_lon], [target_lat], s=120, marker="*", color="#a855f7", edgecolors="white", lw=1.5, zorder=6)
ax2.set_xlim(lon_min, lon_max)
ax2.set_ylim(lat_min, lat_max)
ax2.set_xlabel("Longitude (°E)", fontsize=11)
ax2.set_ylabel("Latitude (°N)", fontsize=11)

# Panel 3: Continuous Gaussian Kernel 4km Mesh
ax3 = axes[1, 0]
ax3.set_title(f"(C) Continuous 4km Mesh: Gaussian Kernel Weighting (σ = {sigma_km:.0f} km)", fontsize=13, fontweight="bold", pad=10)
c3 = ax3.contourf(grid_engine.lon_grid, grid_engine.lat_grid, gauss_grid, levels=np.linspace(0, 1, 21), cmap="turbo")
plt.colorbar(c3, ax=ax3, label="Interpolated P(Cloudburst)")
ax3.contour(grid_engine.lon_grid, grid_engine.lat_grid, gauss_grid, levels=[0.30, 0.60, 0.80], colors=["yellow", "orange", "red"], linewidths=1.8)
for s in stations:
    ax3.scatter(s["lon"], s["lat"], s=70, color="white", edgecolors="black", lw=1.2, zorder=5)
ax3.scatter([target_lon], [target_lat], s=120, marker="*", color="#a855f7", edgecolors="white", lw=1.5, zorder=6)
ax3.set_xlim(lon_min, lon_max)
ax3.set_ylim(lat_min, lat_max)
ax3.set_xlabel("Longitude (°E)", fontsize=11)
ax3.set_ylabel("Latitude (°N)", fontsize=11)

# Panel 4: 1D Cross-Section Transect
ax4 = axes[1, 1]
ax4.set_title("(D) 1D Spatial Transect: Nalbari → Jalukbari → Khanapara (West to East)", fontsize=13, fontweight="bold", pad=10)
# Sample across a latitude transect close to 26.15 - 26.20
transect_lons = np.linspace(91.20, 92.10, 100)
fixed_lat = 26.15

transect_idw = []
transect_gauss = []

for lon_val in transect_lons:
    # IDW
    w_sum, val_sum = 0.0, 0.0
    gw_sum, gval_sum = 0.0, 0.0
    for s in stations:
        dy = (fixed_lat - s["lat"]) * 111.0
        dx = (lon_val - s["lon"]) * 111.0 * cos_lat
        d = np.sqrt(dx**2 + dy**2)
        d_c = max(d, 2.0)
        w = 1.0 / (d_c ** 2.0)
        gw = np.exp(-(d**2) / (2.0 * (sigma_km**2)))
        w_sum += w
        val_sum += w * s["p_cb"]
        gw_sum += gw
        gval_sum += gw * s["p_cb"]
    transect_idw.append(val_sum / w_sum)
    transect_gauss.append(gval_sum / gw_sum)

dist_axis_km = (transect_lons - 91.20) * 111.0 * cos_lat

ax4.plot(dist_axis_km, transect_idw, label="IDW (p=2.0) - Sharper Convective Core", color="#2563eb", lw=2.5)
ax4.plot(dist_axis_km, transect_gauss, label=f"Gaussian Kernel (σ={sigma_km:.0f}km) - Smooth Diffusion", color="#dc2626", lw=2.5, ls="--")

# Draw alert tiers
ax4.axhline(0.80, color="#ef4444", ls=":", lw=1.2, label="Red Alert (0.80)")
ax4.axhline(0.60, color="#f97316", ls=":", lw=1.2, label="Orange Alert (0.60)")
ax4.axhline(0.30, color="#eab308", ls=":", lw=1.2, label="Yellow Alert (0.30)")

# Mark Jalukbari & Khanapara
jalukbari_dist = (target_lon - 91.20) * 111.0 * cos_lat
khanapara_dist = (91.8164 - 91.20) * 111.0 * cos_lat
ax4.axvline(jalukbari_dist, color="#8b5cf6", ls="-.", lw=1.5)
ax4.text(jalukbari_dist, 0.15, "Jalukbari (Ungauged)\nInterpolated P ~ 0.65", color="#6d28d9", fontweight="bold", ha="right", fontsize=9)
ax4.axvline(khanapara_dist, color="#ef4444", ls="-.", lw=1.5)
ax4.text(khanapara_dist, 0.90, "Khanapara Core\nP = 0.88", color="#b91c1c", fontweight="bold", ha="center", fontsize=9)

ax4.set_xlabel("Distance along Transect (km)", fontsize=11)
ax4.set_ylabel("P(Cloudburst)", fontsize=11)
ax4.set_ylim(0, 1.05)
ax4.grid(True, alpha=0.3)
ax4.legend(loc="upper left", fontsize=9)

plt.tight_layout()
out_png = Path("outputs/spatial_interpolation_demo.png")
plt.savefig(out_png, bbox_inches="tight")
plt.close()
print(f"Visualization saved to: {out_png}")
