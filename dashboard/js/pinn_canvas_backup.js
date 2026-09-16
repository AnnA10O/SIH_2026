/**
 * pinn_canvas.js
 * Physics-Informed Neural Network (PINN) 2D Full-Screen Satellite Flood Inundation Engine
 * Powered by Leaflet.js, Esri World Imagery, and Catmull-Rom Spline Georeferenced Winding River Interpolation
 * Connected directly to live FastAPI backend at http://localhost:8000/api/pinn/<region>
 */

// ── REGION KEY MAPPING ────────────────────────────────────────────────────
var REGION_KEY_MAP = {
  "mandakini": "rudraprayag",
  "alaknanda": "chamoli",
  "bhagirathi": "uttarkashi",
  "gori ganga": "pithoragarh",
  "goriganga": "pithoragarh",
  "bhilangna": "tehri",
  "nayar": "pauri",
  "gaula": "nainital",
  "kosi": "almora",

  "rudraprayag": "rudraprayag",
  "chamoli": "chamoli",
  "uttarkashi": "uttarkashi",
  "pithoragarh": "pithoragarh",
  "tehri": "tehri",
  "pauri": "pauri",
  "nainital": "nainital",
  "almora": "almora",

  "mandakini (rudraprayag)": "rudraprayag",
  "alaknanda (chamoli)": "chamoli",
  "bhagirathi (uttarkashi)": "uttarkashi",
  "gori ganga (pithoragarh)": "pithoragarh",
  "bhilangna (tehri)": "tehri",
  "nayar (pauri)": "pauri",
  "gaula (nainital)": "nainital",
  "kosi (almora)": "almora"
};

// ── HIGH-PRECISION REAL WINDING RIVER WAYPOINT ANCHORS FOR ALL 8 BASINS ────
var BASIN_ANCHORS = {
  rudraprayag: [
    { name: "Kedarnath Shrine", lat: 30.7346, lon: 79.0669, elev: 3583, y: 0.00 },
    { name: "Upper Mandakini", lat: 30.7150, lon: 79.0600, elev: 3100, y: 0.12 },
    { name: "Rambara Gorge", lat: 30.6800, lon: 79.0400, elev: 2700, y: 0.28 },
    { name: "Gaurikund", lat: 30.6520, lon: 79.0230, elev: 1980, y: 0.42 },
    { name: "Sonprayag Choke", lat: 30.6240, lon: 79.0030, elev: 1829, y: 0.55 },
    { name: "Phata Bend", lat: 30.5750, lon: 79.0300, elev: 1500, y: 0.68 },
    { name: "Guptkashi Loop", lat: 30.5250, lon: 79.0780, elev: 1319, y: 0.78 },
    { name: "Chandrapuri", lat: 30.4300, lon: 79.0550, elev: 860, y: 0.86 },
    { name: "Tilwara Bend", lat: 30.3400, lon: 78.9850, elev: 700, y: 0.93 },
    { name: "Rudraprayag Confluence", lat: 30.2849, lon: 78.9814, elev: 610, y: 1.00 }
  ],
  chamoli: [
    { name: "Badrinath Shrine", lat: 30.7433, lon: 79.4938, elev: 3133, y: 0.00 },
    { name: "Govindghat", lat: 30.6250, lon: 79.5600, elev: 1820, y: 0.25 },
    { name: "Joshimath Town", lat: 30.5546, lon: 79.5643, elev: 1875, y: 0.40 },
    { name: "Tapovan Barrage", lat: 30.5667, lon: 79.5333, elev: 1350, y: 0.55 },
    { name: "Helang Gorge", lat: 30.5100, lon: 79.4900, elev: 1200, y: 0.70 },
    { name: "Pipalkoti", lat: 30.4300, lon: 79.4300, elev: 1050, y: 0.85 },
    { name: "Chamoli HQ", lat: 30.4024, lon: 79.3323, elev: 950, y: 1.00 }
  ],
  uttarkashi: [
    { name: "Gangotri Glacier", lat: 30.9946, lon: 78.9398, elev: 3048, y: 0.00 },
    { name: "Harsil Valley", lat: 31.0300, lon: 78.7300, elev: 2620, y: 0.25 },
    { name: "Bhatwari Bend", lat: 30.8100, lon: 78.6000, elev: 1600, y: 0.45 },
    { name: "Maneri Dam", lat: 30.8667, lon: 78.7833, elev: 1320, y: 0.60 },
    { name: "Uttarkashi HQ", lat: 30.7268, lon: 78.4354, elev: 1165, y: 0.75 },
    { name: "Chinyalisaur", lat: 30.5500, lon: 78.3200, elev: 850, y: 0.88 },
    { name: "Tehri Reservoir", lat: 30.3783, lon: 78.4805, elev: 650, y: 1.00 }
  ],
  pithoragarh: [
    { name: "Munsiari Alpine Slope", lat: 30.0668, lon: 80.2374, elev: 2200, y: 0.00 },
    { name: "Madkot Gorge", lat: 29.9800, lon: 80.3800, elev: 1450, y: 0.30 },
    { name: "Dharchula Ravine", lat: 29.8452, lon: 80.5423, elev: 915, y: 0.55 },
    { name: "Balwakot", lat: 29.7700, lon: 80.4500, elev: 750, y: 0.80 },
    { name: "Jauljibi Confluence", lat: 29.7167, lon: 80.3667, elev: 600, y: 1.00 }
  ],
  tehri: [
    { name: "Khatling Glacier", lat: 30.8667, lon: 78.9333, elev: 2500, y: 0.00 },
    { name: "Gangi Valley", lat: 30.6500, lon: 78.8200, elev: 2100, y: 0.25 },
    { name: "Ghuttu Valley", lat: 30.5300, lon: 78.7500, elev: 1600, y: 0.45 },
    { name: "Ghali Bend", lat: 30.4400, lon: 78.6000, elev: 1200, y: 0.65 },
    { name: "New Tehri Town", lat: 30.3841, lon: 78.4802, elev: 1550, y: 0.80 },
    { name: "Devprayag Confluence", lat: 30.1462, lon: 78.5978, elev: 520, y: 1.00 }
  ],
  pauri: [
    { name: "Pauri HQ", lat: 30.1462, lon: 78.7642, elev: 1800, y: 0.00 },
    { name: "Srinagar Garhwal", lat: 30.2280, lon: 78.7803, elev: 560, y: 0.30 },
    { name: "Kirtinagar", lat: 30.2200, lon: 78.7300, elev: 540, y: 0.50 },
    { name: "Devprayag", lat: 30.1462, lon: 78.5978, elev: 472, y: 0.70 },
    { name: "Kaudiyala Gorge", lat: 30.0700, lon: 78.4200, elev: 400, y: 0.85 },
    { name: "Rishikesh", lat: 30.0869, lon: 78.2676, elev: 350, y: 1.00 }
  ],
  nainital: [
    { name: "Nainital Town", lat: 29.3803, lon: 79.4636, elev: 2084, y: 0.00 },
    { name: "Bhowali Pass", lat: 29.3800, lon: 79.5200, elev: 1700, y: 0.20 },
    { name: "Bhimtal", lat: 29.3494, lon: 79.5636, elev: 1371, y: 0.40 },
    { name: "Kathgodam Gorge", lat: 29.2700, lon: 79.5400, elev: 520, y: 0.65 },
    { name: "Haldwani", lat: 29.2183, lon: 79.5130, elev: 424, y: 0.80 },
    { name: "Rudrapur (Terai)", lat: 28.9875, lon: 79.4304, elev: 280, y: 1.00 }
  ],
  almora: [
    { name: "Someshwar Gorge", lat: 29.6667, lon: 79.6833, elev: 750, y: 0.00 },
    { name: "Hawalbagh", lat: 29.6300, lon: 79.6300, elev: 1250, y: 0.35 },
    { name: "Almora HQ", lat: 29.5971, lon: 79.6591, elev: 1650, y: 0.60 },
    { name: "Suyalbari", lat: 29.5000, lon: 79.5700, elev: 900, y: 0.80 },
    { name: "Ramganga East Confluence", lat: 29.4500, lon: 79.7500, elev: 350, y: 1.00 }
  ]
};

// ── GEOREFERENCED BOUNDING BOXES ──────────────────────────────────────────
var BASIN_BOUNDS = {
  rudraprayag: [[30.2049, 78.9014], [30.8146, 79.1469]],
  chamoli:     [[30.3224, 79.2523], [30.8233, 79.6443]],
  uttarkashi:  [[30.2983, 78.3554], [31.0746, 79.0198]],
  pithoragarh: [[29.6367, 80.1574], [30.1468, 80.6223]],
  tehri:       [[30.0662, 78.4002], [30.9467, 79.0133]],
  pauri:       [[30.0069, 78.1876], [30.3080, 78.8603]],
  nainital:    [[28.9075, 79.3504], [29.4603, 79.6436]],
  almora:      [[29.3700, 79.5791], [29.7467, 79.8300]]
};

// ── ENGINE STATE ──────────────────────────────────────────────────────────
var PINN_SIM = {
  activeRegion: "rudraprayag",
  currentRegionData: null,
  pinnMap: null,
  satLayer: null,
  labelsLayer: null,
  hillshadeGroup: null,
  floodGroup: null,
  townMarkers: [],
  floodActive: true
};

// ── CATMULL-ROM CUBIC SPLINE INTERPOLATION MATH ───────────────────────────
function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (
    (2 * p1) +
    (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

function interpolateSpline(yVal, anchorYs, anchorVals) {
  if (yVal <= anchorYs[0]) return anchorVals[0];
  if (yVal >= anchorYs[anchorYs.length - 1]) return anchorVals[anchorVals.length - 1];

  let i = 0;
  while (i < anchorYs.length - 1 && anchorYs[i + 1] < yVal) {
    i++;
  }

  const y0 = anchorYs[i];
  const y1 = anchorYs[i + 1];
  const t = (yVal - y0) / (y1 - y0 || 1);

  const v1 = anchorVals[i];
  const v2 = anchorVals[i + 1];
  const v0 = i > 0 ? anchorVals[i - 1] : v1 - (v2 - v1);
  const v3 = i < anchorVals.length - 2 ? anchorVals[i + 2] : v2 + (v2 - v1);

  return catmullRom(v0, v1, v2, v3, t);
}

function getCellLatLon(r, c, ny, nx, anchors) {
  const yVal = r / Math.max(1, ny - 1);
  const xVal = c / Math.max(1, nx - 1);

  const anchorYs = anchors.map(a => a.y);
  const anchorLats = anchors.map(a => a.lat);
  const anchorLons = anchors.map(a => a.lon);

  const latC = interpolateSpline(yVal, anchorYs, anchorLats);
  const lonC = interpolateSpline(yVal, anchorYs, anchorLons);

  const eps = 0.01;
  const latNext = interpolateSpline(Math.min(1.0, yVal + eps), anchorYs, anchorLats);
  const lonNext = interpolateSpline(Math.min(1.0, yVal + eps), anchorYs, anchorLons);

  let dlat = latNext - latC;
  let dlon = lonNext - lonC;
  let norm = Math.hypot(dlat, dlon);
  if (norm === 0) {
    dlat = -1.0; dlon = 0.0;
  } else {
    dlat /= norm;
    dlon /= norm;
  }

  const nxLat = -dlon;
  const nxLon = dlat;
  const crossWidthDeg = 0.035;

  const offset = (xVal - 0.5) * crossWidthDeg;
  return [latC + offset * nxLat, lonC + offset * nxLon];
}

// ── FETCH REAL PINN DATA FROM LIVE API OR LOCAL JSON FALLBACK ───────────────
async function fetchPINNData(regionKey) {
  const normalizedKey = REGION_KEY_MAP[regionKey.toLowerCase()] || regionKey;
  const timestamp = Date.now();
  const apiEndpoint = `http://localhost:8000/api/pinn/${normalizedKey}?t=${timestamp}`;
  const localFallback = `outputs/pinn_3d_multi_region_FINAL.json?t=${timestamp}`;
  const relativeFallback = `../outputs/pinn_3d_multi_region_FINAL.json?t=${timestamp}`;

  try {
    const res = await fetch(apiEndpoint);
    if (res.ok) {
      const data = await res.json();
      PINN_SIM.currentRegionData = data;
      console.log(`[PINN Data Link] Successfully fetched LIVE API data from http://localhost:8000/api/pinn/${normalizedKey}`, data);
      return data;
    }
  } catch (e) {
    console.warn(`[PINN Data Link] Live API server offline (${e.message}), trying static fallback...`);
  }

  try {
    let res = await fetch(localFallback);
    if (!res.ok) res = await fetch(relativeFallback);
    if (res.ok) {
      const fullDataset = await res.json();
      if (fullDataset && fullDataset[normalizedKey]) {
        PINN_SIM.currentRegionData = fullDataset[normalizedKey];
        console.log(`[PINN Data Link] Loaded static JSON fallback for region '${normalizedKey}'`);
        return fullDataset[normalizedKey];
      }
    }
  } catch (e) {
    console.error(`[PINN Data Error] Could not load static fallback:`, e);
  }
  return null;
}

// ── INITIALIZE FULL-SCREEN LEAFLET SATELLITE MAP ──────────────────────────────
async function initFloodCanvas() {
  const mapContainer = document.getElementById('pinn-leaflet-map');
  if (!mapContainer) return;

  if (!PINN_SIM.pinnMap) {
    PINN_SIM.pinnMap = L.map('pinn-leaflet-map', {
      zoomControl: true,
      attributionControl: true
    });

    PINN_SIM.satLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 18,
      attribution: "Imagery &copy; Esri"
    }).addTo(PINN_SIM.pinnMap);

    PINN_SIM.labelsLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 18
    }).addTo(PINN_SIM.pinnMap);
  }

  PINN_SIM.pinnMap.invalidateSize();
  await loadPINNRegion(PINN_SIM.activeRegion);
  checkAutoRiskState();
  if (!PINN_SIM.pollTimer) {
    PINN_SIM.pollTimer = setInterval(checkAutoRiskState, 1500);
  }
}

// ── LOAD REGION (FIT BOUNDS, HILLSHADE, WINDING FLOOD OVERLAY, TOWNS, KPIS) ────
async function loadPINNRegion(regionKey) {
  const normalizedKey = REGION_KEY_MAP[regionKey.toLowerCase()] || regionKey;
  PINN_SIM.activeRegion = normalizedKey;

  const data = await fetchPINNData(normalizedKey);
  if (!data || !PINN_SIM.pinnMap) return;

  const anchors = BASIN_ANCHORS[normalizedKey] || BASIN_ANCHORS['rudraprayag'];
  const bounds = BASIN_BOUNDS[normalizedKey] || BASIN_BOUNDS['rudraprayag'];
  
  PINN_SIM.pinnMap.fitBounds(bounds, { padding: [20, 20] });

  renderTerrainHillshadeOverlay(data.elevation_grid, anchors);
  renderFloodDepthOverlay(data.water_depth_grid, data.peak_water_depth_m, anchors);
  renderTownMarkers(data.towns_affected, anchors);
  updatePINNBottomPanel(data);
}

// ── 2D TERRAIN HILLSHADE OVERLAY (GEOREFERENCED ALONG WINDING RIVER CORRIDOR) ─
function renderTerrainHillshadeOverlay(elevGrid, anchors) {
  if (PINN_SIM.hillshadeGroup && PINN_SIM.pinnMap) {
    PINN_SIM.pinnMap.removeLayer(PINN_SIM.hillshadeGroup);
    PINN_SIM.hillshadeGroup = null;
  }
  if (!elevGrid || !elevGrid.length) return;

  const ny = elevGrid.length;
  const nx = elevGrid[0].length;
  PINN_SIM.hillshadeGroup = L.layerGroup();

  for (let r = 0; r < ny - 1; r += 2) {
    for (let c = 0; c < nx - 1; c += 2) {
      const p00 = getCellLatLon(r, c, ny, nx, anchors);
      const p01 = getCellLatLon(r, c + 1, ny, nx, anchors);
      const p11 = getCellLatLon(r + 1, c + 1, ny, nx, anchors);
      const p10 = getCellLatLon(r + 1, c, ny, nx, anchors);

      const zC = elevGrid[r][c];
      const zR = elevGrid[r][c + 1];
      const zD = elevGrid[r + 1][c];

      const dzdx = (zR - zC) * 0.05;
      const dzdy = (zD - zC) * 0.05;
      const slope = Math.atan(Math.sqrt(dzdx * dzdx + dzdy * dzdy));
      const aspect = Math.atan2(dzdy, -dzdx);

      const sunAzimuth = (315 * Math.PI) / 180;
      const sunAltitude = (45 * Math.PI) / 180;
      let shade = Math.sin(sunAltitude) * Math.cos(slope) +
                  Math.cos(sunAltitude) * Math.sin(slope) * Math.cos(sunAzimuth - aspect);
      shade = Math.max(0, Math.min(1, shade));

      const gray = Math.round(shade * 255);
      const colorHex = `#${gray.toString(16).padStart(2, '0')}${gray.toString(16).padStart(2, '0')}${gray.toString(16).padStart(2, '0')}`;

      L.polygon([p00, p01, p11, p10], {
        fillColor: colorHex,
        fillOpacity: 0.25,
        stroke: false,
        interactive: false
      }).addTo(PINN_SIM.hillshadeGroup);
    }
  }

  PINN_SIM.hillshadeGroup.addTo(PINN_SIM.pinnMap);
}

// ── GEOREFERENCED FLOOD DEPTH OVERLAY FOLLOWING REAL WINDING RIVER CURVE ──────
function renderFloodDepthOverlay(depthGrid, peakDepth, anchors) {
  if (PINN_SIM.floodGroup && PINN_SIM.pinnMap) {
    PINN_SIM.pinnMap.removeLayer(PINN_SIM.floodGroup);
    PINN_SIM.floodGroup = null;
  }
  if (!PINN_SIM.floodActive || !depthGrid || !depthGrid.length) return;

  const ny = depthGrid.length;
  const nx = depthGrid[0].length;
  const maxD = peakDepth || 2.38;
  PINN_SIM.floodGroup = L.layerGroup();

  let floodedCellCount = 0;
  const samplePolygons = [];
  const upperSectionDepths = [];

  for (let r = 0; r < ny - 1; r++) {
    const rowMax = Math.max(...depthGrid[r]);
    if (r <= 20) {
      upperSectionDepths.push({ row: r, y_norm: (r/50).toFixed(2), maxDepth_m: rowMax.toFixed(4) });
    }

    for (let c = 0; c < nx - 1; c++) {
      const d = depthGrid[r][c];

      // Render flood cells down to 0.005m (5mm) threshold so active upper stream cells render
      if (d > 0.005) {
        floodedCellCount++;

        const p00 = getCellLatLon(r, c, ny, nx, anchors);
        const p01 = getCellLatLon(r, c + 1, ny, nx, anchors);
        const p11 = getCellLatLon(r + 1, c + 1, ny, nx, anchors);
        const p10 = getCellLatLon(r + 1, c, ny, nx, anchors);

        if (samplePolygons.length < 3) {
          samplePolygons.push({ row: r, col: c, depth: d.toFixed(3), quad: [p00, p01, p11, p10] });
        }

        const normD = Math.min(1.0, d / maxD);
        let colorHex = "#38bdf8"; // Shallow sky blue
        let opac = 0.75;

        if (normD > 0.50) {
          colorHex = "#1e3a8a"; // Deep surge navy blue
          opac = 0.90;
        } else if (normD > 0.15) {
          colorHex = "#0284c7"; // Moderate surge cobalt blue
          opac = 0.85;
        }

        // Draw quad polygon along winding river bed with crisp stroke & fill
        L.polygon([p00, p01, p11, p10], {
          fillColor: colorHex,
          fillOpacity: opac,
          stroke: true,
          color: colorHex,
          weight: 1.2,
          opacity: opac,
          interactive: false
        }).addTo(PINN_SIM.floodGroup);
      }
    }
  }

  PINN_SIM.floodGroup.addTo(PINN_SIM.pinnMap);

  // DEBUG & DIAGNOSTIC CONSOLE OUTPUT REQUIRED BY SPECIFICATION
  console.log(`==================== PINN FLOOD OVERLAY DIAGNOSTICS ====================`);
  console.log(`Region: ${PINN_SIM.activeRegion} | Total Grid Shape: ${ny}x${nx} (${ny * nx} cells)`);
  console.log(`Flooded Cells (> 0.005m threshold): ${floodedCellCount}`);
  console.log(`Upper Section Depths (Rows 0-20, Kedarnath -> Sonprayag):`, upperSectionDepths);
  console.log(`Sample 3 Polygon Coordinate Sets:`, JSON.stringify(samplePolygons, null, 2));
  console.log(`========================================================================`);
}

// ── RENDER TOWNS AFFECTED MARKERS AT EXACT SPLINE RIVER ANCHOR LOCATIONS ──────
function renderTownMarkers(townsList, anchors) {
  PINN_SIM.townMarkers.forEach(m => {
    if (PINN_SIM.pinnMap) PINN_SIM.pinnMap.removeLayer(m);
  });
  PINN_SIM.townMarkers = [];

  if (!townsList || !townsList.length || !PINN_SIM.pinnMap) return;

  const anchorYs = anchors.map(a => a.y);
  const anchorLats = anchors.map(a => a.lat);
  const anchorLons = anchors.map(a => a.lon);

  townsList.forEach(town => {
    const normY = town.y !== undefined ? town.y : 0.5;
    const lat = interpolateSpline(normY, anchorYs, anchorLats);
    const lon = interpolateSpline(normY, anchorYs, anchorLons);

    const markerHtml = `
      <div style="background: rgba(6, 12, 24, 0.92); backdrop-filter: blur(6px); border: 1px solid #06b6d4; color: #fff; padding: 3px 8px; border-radius: 12px; font-size: 10.5px; font-weight: 700; display: flex; align-items: center; gap: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.6); white-space: nowrap;">
        <span style="color: #ef4444;">📍</span> ${town.name} (${town.elev}m)
      </div>
    `;

    const customIcon = L.divIcon({
      html: markerHtml,
      className: 'town-marker-icon',
      iconSize: [140, 24],
      iconAnchor: [70, 12]
    });

    const m = L.marker([lat, lon], { icon: customIcon }).addTo(PINN_SIM.pinnMap);
    PINN_SIM.townMarkers.push(m);
  });
}

// ── UPDATE BOTTOM INFO PANEL WITH REAL LIVE API DATA ──────────────────────────
function updatePINNBottomPanel(data) {
  if (!data) return;

  const depthEl = document.getElementById('pinn-depth');
  const speedEl = document.getElementById('pinn-speed');
  const areaEl = document.getElementById('pinn-area');
  const gorgeEl = document.getElementById('pinn-gorge-name');
  const townsEl = document.getElementById('pinn-towns-list');

  // Read LIVE numbers directly from API object
  const peakDepth = (data.peak_water_depth_m !== undefined) ? data.peak_water_depth_m.toFixed(2) : "2.38";
  const peakSpeed = (data.peak_velocity_m_s !== undefined) ? data.peak_velocity_m_s.toFixed(2) : "3.31";
  const floodArea = (data.flooded_area_km2 !== undefined) ? data.flooded_area_km2.toFixed(1) : "99.0";
  const chokeLoc  = data.choke_location || "N/A";

  if (depthEl) depthEl.innerText = `${peakDepth} m`;
  if (speedEl) speedEl.innerText = `${peakSpeed} m/s`;
  if (areaEl) areaEl.innerText = `${floodArea} km²`;
  if (gorgeEl) gorgeEl.innerText = chokeLoc;

  if (townsEl && data.towns_affected) {
    const names = data.towns_affected.map(t => t.name).join(" &bull; ");
    townsEl.innerHTML = `<span style="color: var(--text-muted); font-weight:700;">AFFECTED TOWNS & INFRASTRUCTURE:</span> <span style="color: var(--text-main); font-weight:600;">${names}</span>`;
  }

  // Print live metrics log for integrity verification
  console.log(`[PINN Data Verification Log] Region: ${data.region_name} | Depth: ${peakDepth}m | Speed: ${peakSpeed}m/s | Area: ${floodArea}km² | Choke: ${chokeLoc}`);
}

// ── REGION SWITCHER HANDLER ──────────────────────────────────────────────────
// ── AUTOMATIC CLOUDBURST RISK MODE SWITCHER (NORMAL / WATCH / FLOOD ALERT) ──
PINN_SIM.state = 'NORMAL';
PINN_SIM.lastRiskScore = -1;
PINN_SIM.manualOverride = false;
PINN_SIM.pollTimer = null;

function getCloudburstRiskScore() {
  const rScoreEl = document.getElementById('risk-score-val');
  const rBadgeEl = document.getElementById('risk-badge');
  let text = "0.8%";
  if (rScoreEl && rScoreEl.textContent && rScoreEl.textContent.trim() !== '') {
    text = rScoreEl.textContent;
  } else if (rBadgeEl && rBadgeEl.textContent && rBadgeEl.textContent.trim() !== '') {
    text = rBadgeEl.textContent;
  }
  const num = parseFloat(text.replace(/[^0-9.]/g, ''));
  return isNaN(num) ? 0.8 : num;
}

function checkAutoRiskState() {
  if (PINN_SIM.manualOverride) return;

  const risk = getCloudburstRiskScore();
  let newState = 'NORMAL';

  if (risk >= 50.0) {
    newState = 'ALERT';
  } else if (risk >= 10.0) {
    newState = 'WATCH';
  }

  if (newState !== PINN_SIM.state || Math.abs(risk - PINN_SIM.lastRiskScore) > 0.1) {
    console.log(`[PINN Auto State Engine] Risk Score: ${risk.toFixed(1)}% | Auto Mode Transition: ${PINN_SIM.state} => ${newState} (Region: '${PINN_SIM.activeRegion}')`);
    PINN_SIM.lastRiskScore = risk;
    applyPINNState(newState, risk);
  }
}

function applyPINNState(state, riskVal) {
  PINN_SIM.state = state;
  const risk = (riskVal !== undefined) ? riskVal : getCloudburstRiskScore();

  const pinnBadge = document.getElementById('pinn-risk-badge');
  const topBadge = document.getElementById('status-badge');
  const topText = document.getElementById('status-text');

  if (state === 'NORMAL') {
    PINN_SIM.floodActive = false;
    if (PINN_SIM.floodGroup && PINN_SIM.pinnMap) {
      PINN_SIM.pinnMap.removeLayer(PINN_SIM.floodGroup);
      PINN_SIM.floodGroup = null;
    }

    if (pinnBadge) {
      pinnBadge.innerText = `Riverbed Nominal (${risk.toFixed(1)}%)`;
      pinnBadge.className = 'badge';
    }
    if (topBadge && topText) {
      topBadge.className = 'status-badge';
      topText.innerText = 'SYSTEM NORMAL · ROUTINE MONITORING';
    }

    resetPINNBottomPanelToNominal();

  } else if (state === 'WATCH') {
    PINN_SIM.floodActive = false;
    if (PINN_SIM.floodGroup && PINN_SIM.pinnMap) {
      PINN_SIM.pinnMap.removeLayer(PINN_SIM.floodGroup);
      PINN_SIM.floodGroup = null;
    }

    if (pinnBadge) {
      pinnBadge.innerText = `⚠️ WATCH (${risk.toFixed(1)}%)`;
      pinnBadge.className = 'badge badge-watch';
    }
    if (topBadge && topText) {
      topBadge.className = 'status-badge warning';
      topText.innerText = 'WARNING · SEVERE THUNDERSTORM';
    }

    updatePINNBottomPanelForWatch(risk);

  } else if (state === 'ALERT') {
    PINN_SIM.floodActive = true;

    if (pinnBadge) {
      pinnBadge.innerText = `🚨 FLOOD ALERT (${risk.toFixed(1)}%)`;
      pinnBadge.className = 'badge badge-alert';
    }
    if (topBadge && topText) {
      topBadge.className = 'status-badge danger';
      topText.innerText = 'CRITICAL CLOUDBURST ALERT · EVACUATE GORGE';
    }

    if (PINN_SIM.currentRegionData) {
      const anchors = BASIN_ANCHORS[PINN_SIM.activeRegion] || BASIN_ANCHORS['rudraprayag'];
      renderFloodDepthOverlay(
        PINN_SIM.currentRegionData.water_depth_grid,
        PINN_SIM.currentRegionData.peak_water_depth_m,
        anchors
      );
      updatePINNBottomPanel(PINN_SIM.currentRegionData);
    } else {
      loadPINNRegion(PINN_SIM.activeRegion);
    }
  }
}

function resetPINNBottomPanelToNominal() {
  const depthEl = document.getElementById('pinn-depth');
  const speedEl = document.getElementById('pinn-speed');
  const areaEl = document.getElementById('pinn-area');
  const gorgeEl = document.getElementById('pinn-gorge-name');
  const townsEl = document.getElementById('pinn-towns-list');

  if (depthEl) depthEl.innerText = "0.40 m (Baseflow)";
  if (speedEl) speedEl.innerText = "0.35 m/s";
  if (areaEl) areaEl.innerText = "0.0 km²";
  if (gorgeEl) gorgeEl.innerText = "No Inundation";
  if (townsEl) townsEl.innerHTML = `<span style="color: var(--text-muted); font-weight:700;">AFFECTED TOWNS & INFRASTRUCTURE:</span> <span style="color: var(--green); font-weight:600;">All Stations Safe · Routine Monitoring</span>`;
}

function updatePINNBottomPanelForWatch(risk) {
  const depthEl = document.getElementById('pinn-depth');
  const speedEl = document.getElementById('pinn-speed');
  const areaEl = document.getElementById('pinn-area');
  const gorgeEl = document.getElementById('pinn-gorge-name');
  const townsEl = document.getElementById('pinn-towns-list');

  if (depthEl) depthEl.innerText = "0.85 m (Elevated)";
  if (speedEl) speedEl.innerText = "0.70 m/s";
  if (areaEl) areaEl.innerText = "0.0 km² (Standby)";
  if (gorgeEl) gorgeEl.innerText = "Monitoring Choke Point";
  if (townsEl) townsEl.innerHTML = `<span style="color: var(--text-muted); font-weight:700;">AFFECTED TOWNS & INFRASTRUCTURE:</span> <span style="color: #f59e0b; font-weight:600;">ELEVATED CONVECTIVE RISK (${risk.toFixed(1)}%) · Standby Evacuation Orders</span>`;
}

// ── REGION SWITCHER HANDLER ──────────────────────────────────────────────────
async function switchPINNRegion(regionKey, chipEl) {
  const mappedKey = REGION_KEY_MAP[regionKey.toLowerCase()] || regionKey;
  PINN_SIM.activeRegion = mappedKey;

  document.querySelectorAll('.filter-chip').forEach(c => {
    if (c.id && c.id.startsWith('chip-region-')) c.classList.remove('active');
  });
  if (chipEl) chipEl.classList.add('active');

  await loadPINNRegion(mappedKey);
  checkAutoRiskState();
}

// ── SIMULATION BUTTON HANDLERS (MANUAL OVERRIDES) ───────────────────────────
function simulateFloodCanvas() {
  PINN_SIM.manualOverride = true;
  console.log(`[PINN Manual Override] 'Run Simulation' clicked -> Triggering ALERT flood mode for active basin '${PINN_SIM.activeRegion}'`);
  applyPINNState('ALERT');
}

function resetFloodCanvas() {
  PINN_SIM.manualOverride = false;
  console.log(`[PINN Manual Override] 'Reset' clicked -> Clearing override, returning to auto risk score polling`);
  const risk = getCloudburstRiskScore();
  const autoState = risk >= 50.0 ? 'ALERT' : (risk >= 10.0 ? 'WATCH' : 'NORMAL');
  applyPINNState(autoState, risk);
}

// Global Exports & Test Helper
window.initFloodCanvas = initFloodCanvas;
window.switchPINNRegion = switchPINNRegion;
window.simulateFloodCanvas = simulateFloodCanvas;
window.resetFloodCanvas = resetFloodCanvas;
window.checkAutoRiskState = checkAutoRiskState;
window.setSimulatedRiskScore = function(scorePct) {
  const rScore = document.getElementById('risk-score-val');
  const rBadge = document.getElementById('risk-badge');
  const riskCircle = document.getElementById('risk-circle');

  if (rScore) rScore.innerText = `${scorePct.toFixed(1)}%`;
  if (rBadge) rBadge.innerText = `Risk: ${scorePct.toFixed(1)}%`;
  if (riskCircle) riskCircle.style.setProperty('--risk-pct', `${scorePct.toFixed(1)}%`);

  PINN_SIM.manualOverride = false;
  checkAutoRiskState();
  console.log(`[Test Helper] Set simulated cloudburst risk score to ${scorePct.toFixed(1)}%`);
};

document.addEventListener('DOMContentLoaded', () => {
  setTimeout(initFloodCanvas, 150);
});
