import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

# 1. Load Data
aws_csv = Path("outputs/all_india_aws_stations.csv")
quality_csv = Path("outputs/station_quality_report.csv")

df_aws = pd.read_csv(aws_csv)
df_qual = pd.read_csv(quality_csv) if quality_csv.exists() else pd.DataFrame()

# GAGAN stations
gagan_stations = [
    {"id": 1,  "name": "Madurai",     "lat": 9.837,  "lon": 78.091, "state": "Tamil Nadu"},
    {"id": 3,  "name": "Bangalore",   "lat": 13.033, "lon": 77.567, "state": "Karnataka"},
    {"id": 4,  "name": "Hyderabad",   "lat": 17.450, "lon": 78.470, "state": "Telangana"},
    {"id": 5,  "name": "Bhopal",      "lat": 23.284, "lon": 77.404, "state": "Madhya Pradesh"},
    {"id": 6,  "name": "Lucknow",     "lat": 26.761, "lon": 80.883, "state": "Uttar Pradesh"},
    {"id": 8,  "name": "Agatti",      "lat": 10.824, "lon": 72.176, "state": "Lakshadweep"},
    {"id": 10, "name": "Mumbai",      "lat": 19.089, "lon": 72.845, "state": "Maharashtra"},
    {"id": 11, "name": "Bagdogra",    "lat": 26.685, "lon": 88.326, "state": "West Bengal"},
    {"id": 12, "name": "Lengpui",     "lat": 23.840, "lon": 92.624, "state": "Mizoram"},
    {"id": 13, "name": "Guwahati",    "lat": 26.120, "lon": 91.590, "state": "Assam"},
    {"id": 14, "name": "Kolkata",     "lat": 22.645, "lon": 88.448, "state": "West Bengal"},
    {"id": 15, "name": "Raipur",      "lat": 21.242, "lon": 81.633, "state": "Chhattisgarh"},
    {"id": 16, "name": "Vizag",       "lat": 17.728, "lon": 83.226, "state": "Andhra Pradesh"},
    {"id": 17, "name": "Gaya",        "lat": 24.747, "lon": 84.945, "state": "Bihar"},
    {"id": 18, "name": "Trivandrum",  "lat": 8.536,  "lon": 76.905, "state": "Kerala"},
    {"id": 21, "name": "Ahmedabad",   "lat": 23.064, "lon": 72.620, "state": "Gujarat"},
    {"id": 26, "name": "Bhubaneswar", "lat": 20.254, "lon": 85.809, "state": "Odisha"},
    {"id": 28, "name": "Jodhpur",     "lat": 26.264, "lon": 73.051, "state": "Rajasthan"},
]
df_gagan = pd.DataFrame(gagan_stations)

print(f"Plotting {len(df_aws)} AWS stations and {len(df_gagan)} GAGAN stations...")

# -------------------------------------------------------------
# 2. GENERATE HIGH-RESOLUTION STATIC PNG MAP (outputs/india_aws_network_map.png)
# -------------------------------------------------------------
fig = plt.figure(figsize=(16, 12), dpi=200, facecolor="#0b0f19")
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.18)

# Panel 1: Full India Map
ax_main = fig.add_subplot(gs[0], facecolor="#0f172a")
ax_main.set_title("ISRO National AWS & GNSS Network Distribution\n(1,228 Ground In-Situ Weather Stations + 18 GAGAN Receivers)",
                  fontsize=14, fontweight="bold", color="#f8fafc", pad=14)

# Plot background density of all 1,228 AWS stations
scatter_aws = ax_main.scatter(
    df_aws["lon"], df_aws["lat"],
    c="#38bdf8", s=18, alpha=0.55, edgecolors="none", zorder=3,
    label=f"ISRO In-Situ AWS ({len(df_aws):,} stations)"
)

# Highlight Northeast / Assam Focus Cluster
ne_mask = (df_aws["lat"] >= 24.0) & (df_aws["lat"] <= 28.5) & (df_aws["lon"] >= 88.0) & (df_aws["lon"] <= 96.0)
df_ne = df_aws[ne_mask]
ax_main.scatter(
    df_ne["lon"], df_ne["lat"],
    c="#f59e0b", s=32, alpha=0.85, edgecolors="#78350f", lw=0.5, zorder=4,
    label=f"Northeast Cloudburst Testbed ({len(df_ne)} stations)"
)

# Plot GAGAN GNSS Stations
ax_main.scatter(
    df_gagan["lon"], df_gagan["lat"],
    c="#ef4444", s=160, marker="*", edgecolors="#fef08a", lw=1.2, zorder=6,
    label="GAGAN GNSS IWV Receivers (18 anchors)"
)

# Add GAGAN city labels
for _, row in df_gagan.iterrows():
    ax_main.annotate(
        row["name"], (row["lon"], row["lat"]),
        xytext=(5, 3), textcoords="offset points",
        fontsize=7.5, fontweight="bold", color="#fef08a",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="#000")],
        zorder=7
    )

# Draw rectangle around Northeast / Assam focus region
rect = patches.Rectangle(
    (88.0, 24.0), 8.0, 4.5,
    linewidth=2, edgecolor="#f59e0b", facecolor="none", linestyle="--", zorder=5
)
ax_main.add_patch(rect)
ax_main.text(88.2, 28.7, "Northeast Cloudburst Testbed Inset →", color="#f59e0b", fontsize=9, fontweight="bold",
             path_effects=[pe.withStroke(linewidth=2, foreground="#000")])

# Aesthetic Map Bounds & Grid
ax_main.set_xlim(67.0, 98.5)
ax_main.set_ylim(7.5, 37.5)
ax_main.set_xlabel("Longitude (°E)", color="#94a3b8", fontsize=10)
ax_main.set_ylabel("Latitude (°N)", color="#94a3b8", fontsize=10)
ax_main.tick_params(colors="#94a3b8")
ax_main.grid(True, color="#334155", linestyle=":", alpha=0.6)
legend = ax_main.legend(loc="lower left", facecolor="#1e293b", edgecolor="#475569", fontsize=9, labelcolor="#f8fafc")

# Panel 2: Zoom-in on the Northeast Assam / Guwahati Cloudburst Testbed
ax_zoom = fig.add_subplot(gs[1], facecolor="#0f172a")
ax_zoom.set_title("Zoom-In: Northeast Cloudburst Testbed\n(Brahmaputra Valley & Foothills Inter-Station Geometry)",
                  fontsize=13, fontweight="bold", color="#f59e0b", pad=14)

# Plot all stations in the NE window
ax_zoom.scatter(
    df_ne["lon"], df_ne["lat"],
    c="#38bdf8", s=30, alpha=0.6, edgecolors="none", zorder=3, label="Regional AWS"
)

# If quality report stations are available, plot PASS vs FAIL
if not df_qual.empty:
    pass_stns = df_qual[df_qual["status"] == "PASS"]
    fail_stns = df_qual[df_qual["status"] == "FAIL"]
    
    ax_zoom.scatter(
        pass_stns["lon"], pass_stns["lat"],
        c="#22c55e", s=110, edgecolors="#ffffff", lw=1.2, zorder=5,
        label=f"Quality Validated (PASS: {len(pass_stns)})"
    )
    ax_zoom.scatter(
        fail_stns["lon"], fail_stns["lat"],
        c="#ef4444", s=70, marker="x", lw=1.8, zorder=4,
        label=f"Quality Filtered (FAIL/Dead: {len(fail_stns)})"
    )
    
    # Label key validated stations
    for _, row in pass_stns.head(8).iterrows():
        short_name = row["station_id"].split("(")[-1].replace(")", "").strip()[:14]
        ax_zoom.annotate(
            short_name, (row["lon"], row["lat"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#ffffff", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="#000")],
            zorder=6
        )

# Plot GAGAN stations in NE
ne_gagan = df_gagan[df_gagan["name"].isin(["Guwahati", "Bagdogra", "Lengpui"])]
ax_zoom.scatter(
    ne_gagan["lon"], ne_gagan["lat"],
    c="#f59e0b", s=220, marker="*", edgecolors="#ffffff", lw=1.5, zorder=7,
    label="GAGAN IWV Core Anchors"
)
for _, row in ne_gagan.iterrows():
    ax_zoom.annotate(
        f"GAGAN: {row['name']}", (row["lon"], row["lat"]),
        xytext=(-15, -14), textcoords="offset points",
        fontsize=9, fontweight="bold", color="#fef08a",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="#000")],
        zorder=8
    )

ax_zoom.set_xlim(88.0, 96.0)
ax_zoom.set_ylim(24.0, 28.5)
ax_zoom.set_xlabel("Longitude (°E)", color="#94a3b8", fontsize=10)
ax_zoom.set_ylabel("Latitude (°N)", color="#94a3b8", fontsize=10)
ax_zoom.tick_params(colors="#94a3b8")
ax_zoom.grid(True, color="#334155", linestyle=":", alpha=0.6)
ax_zoom.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", fontsize=8.5, labelcolor="#f8fafc")

png_path = Path("outputs/india_aws_network_map.png")
plt.savefig(png_path, bbox_inches="tight")
plt.close()
print(f"High-res map image saved to: {png_path}")

# -------------------------------------------------------------
# 3. GENERATE INTERACTIVE LEAFLET.JS WEB MAP (outputs/india_aws_network_map.html)
# -------------------------------------------------------------
aws_json_list = []
for _, r in df_aws.iterrows():
    aws_json_list.append({
        "id": r["station_id"],
        "lat": round(float(r["lat"]), 4),
        "lon": round(float(r["lon"]), 4),
        "page": int(r["page"]) if "page" in r else 0
    })

qual_json_list = []
if not df_qual.empty:
    for _, r in df_qual.iterrows():
        qual_json_list.append({
            "id": r["station_id"],
            "lat": round(float(r["lat"]), 4),
            "lon": round(float(r["lon"]), 4),
            "status": r["status"],
            "reason": str(r["reason"]) if pd.notna(r["reason"]) else "",
            "max_rain": float(r["rain_max_daily_mm"]) if pd.notna(r["rain_max_daily_mm"]) else 0.0
        })

gagan_json_list = gagan_stations

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>India National AWS & GNSS Sensor Spread | PS 26077</title>
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: #0b0f19; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; }}
        header {{ background: #111827; border-bottom: 1px solid #334155; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; z-index: 1000; }}
        h1 {{ font-size: 18px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
        .badge {{ background: #0369a1; color: #e0f2fe; font-size: 11px; padding: 3px 8px; border-radius: 999px; font-weight: 600; }}
        .stats-bar {{ display: flex; gap: 16px; font-size: 12px; color: #94a3b8; }}
        .stat-val {{ color: #fff; font-weight: 700; }}
        #map {{ flex: 1; width: 100%; height: 100%; background: #0b0f19; }}
        
        .control-panel {{
            position: absolute; top: 75px; right: 20px; z-index: 1000;
            background: rgba(17, 24, 39, 0.92); backdrop-filter: blur(8px);
            border: 1px solid #334155; border-radius: 8px; padding: 14px;
            font-size: 13px; min-width: 250px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        .control-panel h3 {{ font-size: 13px; color: #38bdf8; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .layer-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; cursor: pointer; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
        .dot-aws {{ background: #38bdf8; }}
        .dot-gagan {{ background: #ef4444; border: 2px solid #fef08a; }}
        .dot-pass {{ background: #22c55e; }}
        .dot-fail {{ background: #ef4444; }}

        .btn-focus {{
            display: block; width: 100%; margin-top: 10px; padding: 6px 10px;
            background: #0284c7; color: white; border: none; border-radius: 6px;
            font-size: 12px; font-weight: 600; cursor: pointer; text-align: center;
        }}
        .btn-focus:hover {{ background: #0369a1; }}

        /* Popup styling */
        .leaflet-popup-content-wrapper {{ background: #1e293b; color: #f8fafc; border: 1px solid #475569; border-radius: 8px; }}
        .leaflet-popup-tip {{ background: #1e293b; }}
        .popup-title {{ font-size: 13px; font-weight: 700; color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 4px; margin-bottom: 6px; }}
        .popup-row {{ font-size: 12px; color: #cbd5e1; margin-bottom: 3px; }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>National In-Situ Sensor Distribution <span class="badge">MOSDAC Data Foundation</span></h1>
        </div>
        <div class="stats-bar">
            <div>All-India AWS: <span class="stat-val">{len(df_aws):,}</span></div>
            <div>GAGAN GNSS: <span class="stat-val">{len(df_gagan)}</span></div>
            <div>Northeast Testbed: <span class="stat-val">{len(df_ne)}</span></div>
        </div>
    </header>

    <div id="map"></div>

    <div class="control-panel">
        <h3>Sensor Layers</h3>
        <label class="layer-item">
            <input type="checkbox" id="toggleAws" checked>
            <span class="legend-dot dot-aws"></span>
            All India AWS (1,228)
        </label>
        <label class="layer-item">
            <input type="checkbox" id="toggleGagan" checked>
            <span class="legend-dot dot-gagan"></span>
            GAGAN GNSS (18)
        </label>
        <label class="layer-item">
            <input type="checkbox" id="toggleAssam" checked>
            <span class="legend-dot dot-pass"></span>
            Assam Validated AWS (PASS)
        </label>
        <label class="layer-item">
            <input type="checkbox" id="toggleFail" checked>
            <span class="legend-dot dot-fail"></span>
            Assam Filtered AWS (FAIL)
        </label>

        <button class="btn-focus" onclick="focusNortheast()">Focus: Northeast Testbed</button>
        <button class="btn-focus" style="background: #334155; margin-top: 6px;" onclick="focusAllIndia()">Reset: Pan India</button>
    </div>

    <script>
        // Init Leaflet map centered on India
        const map = L.map('map', {{
            center: [22.5, 82.0],
            zoom: 5,
            minZoom: 4,
            maxZoom: 13
        }});

        // Dark tile layer (CartoDB Dark Matter)
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);

        // Data arrays
        const awsData = {json.dumps(aws_json_list)};
        const gaganData = {json.dumps(gagan_json_list)};
        const qualData = {json.dumps(qual_json_list)};

        // Layer groups
        const awsLayer = L.layerGroup();
        const gaganLayer = L.layerGroup();
        const assamPassLayer = L.layerGroup();
        const assamFailLayer = L.layerGroup();

        // 1. Add All India AWS
        awsData.forEach(s => {{
            const marker = L.circleMarker([s.lat, s.lon], {{
                radius: 3.5,
                fillColor: '#38bdf8',
                color: '#0284c7',
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.6
            }});
            marker.bindPopup(`
                <div class="popup-title">${{s.id}}</div>
                <div class="popup-row"><strong>Latitude:</strong> ${{s.lat}}°N</div>
                <div class="popup-row"><strong>Longitude:</strong> ${{s.lon}}°E</div>
                <div class="popup-row"><strong>Source:</strong> ISRO In-Situ Directory</div>
            `);
            awsLayer.addLayer(marker);
        }});
        awsLayer.addTo(map);

        // 2. Add GAGAN GNSS Stations
        gaganData.forEach(g => {{
            const marker = L.circleMarker([g.lat, g.lon], {{
                radius: 8,
                fillColor: '#ef4444',
                color: '#fef08a',
                weight: 2.5,
                opacity: 1,
                fillOpacity: 0.9
            }});
            marker.bindPopup(`
                <div class="popup-title">GAGAN Station: ${{g.name}} (ID: ${{g.id}})</div>
                <div class="popup-row"><strong>State:</strong> ${{g.state}}</div>
                <div class="popup-row"><strong>Latitude:</strong> ${{g.lat}}°N</div>
                <div class="popup-row"><strong>Longitude:</strong> ${{g.lon}}°E</div>
                <div class="popup-row"><strong>Precursor Role:</strong> 30-min Tropospheric IWV Tracking</div>
            `);
            gaganLayer.addLayer(marker);
        }});
        gaganLayer.addTo(map);

        // 3. Add Assam Quality Assessed
        qualData.forEach(q => {{
            const isPass = (q.status === 'PASS');
            const color = isPass ? '#22c55e' : '#ef4444';
            const marker = L.circleMarker([q.lat, q.lon], {{
                radius: isPass ? 6 : 4,
                fillColor: color,
                color: '#ffffff',
                weight: 1.5,
                opacity: 0.9,
                fillOpacity: 0.85
            }});
            marker.bindPopup(`
                <div class="popup-title">${{q.id}}</div>
                <div class="popup-row"><strong>Status:</strong> <span style="color:${{color}}; font-weight:bold;">${{q.status}}</span></div>
                <div class="popup-row"><strong>Coords:</strong> ${{q.lat}}°N, ${{q.lon}}°E</div>
                <div class="popup-row"><strong>Diagnostic Reason:</strong> ${{q.reason}}</div>
                ${{isPass ? `<div class="popup-row"><strong>Max Monsoon Rain:</strong> ${{q.max_rain}} mm/day</div>` : ''}}
            `);
            if (isPass) {{
                assamPassLayer.addLayer(marker);
            }} else {{
                assamFailLayer.addLayer(marker);
            }}
        }});
        assamPassLayer.addTo(map);
        assamFailLayer.addTo(map);

        // UI Toggles
        document.getElementById('toggleAws').addEventListener('change', e => {{
            if (e.target.checked) map.addLayer(awsLayer); else map.removeLayer(awsLayer);
        }});
        document.getElementById('toggleGagan').addEventListener('change', e => {{
            if (e.target.checked) map.addLayer(gaganLayer); else map.removeLayer(gaganLayer);
        }});
        document.getElementById('toggleAssam').addEventListener('change', e => {{
            if (e.target.checked) map.addLayer(assamPassLayer); else map.removeLayer(assamPassLayer);
        }});
        document.getElementById('toggleFail').addEventListener('change', e => {{
            if (e.target.checked) map.addLayer(assamFailLayer); else map.removeLayer(assamFailLayer);
        }});

        function focusNortheast() {{
            map.flyTo([26.2, 92.5], 7.5);
        }}
        function focusAllIndia() {{
            map.flyTo([22.5, 82.0], 5);
        }}
    </script>
</body>
</html>
"""

html_path = Path("outputs/india_aws_network_map.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Interactive Leaflet HTML map saved to: {html_path}")
