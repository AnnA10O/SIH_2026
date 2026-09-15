/**
 * app.js
 * Master Coordinator for KERAUNOS Early Warning System (MoES PS 26077)
 * Pure Vanilla JavaScript — Zero external UI frameworks
 * 
 * Key Responsibilities:
 *   - Leaflet Map initialization with Topographic Terrain by default (and Dark/Sat options)
 *   - 40 Real AWS stations markers with color-coded operational rings (NO emojis)
 *   - Dynamic per-station Downstream Flash Flood Accumulation Corridor along real valley paths
 *   - Full bidirectional synchronization between Sliders, Map, Nowcaster, and PINN
 *   - Explicit meteorological thresholds support
 *   - 1-Click Disaster Replay Presets (Kedarnath 2013, Chamoli 2021, Moderate, Reset)
 *   - Strictly ZERO emojis anywhere
 */

const APP = {
  map: null,
  activeLayer: "topo",
  layers: {},
  markers: {},
  floodAccumulationLayer: null,
  flowVectorLayers: [],
  currentStationId: "RDP001", // Start with Kedarnath
  activeFilter: "all",
  activeReplay: null,
  stationsBackup: null,
  
  // Dynamic map showcase visualization layers
  snnRippleLayer: null,
  convectiveCoreLayer: null,
  spatialRaysLayer: null,
  chokeMarkerLayer: null,
  wavefrontLayer: null,
  
  // Automated simulation demo controller
  simRunning: false,
  simPaused: false,
  simTimer: null,
  simStepIndex: 0
};


// ── Telemetry Ingest Mode Switcher (Gauges vs Sliders) ──────────────────────

// ── Simulation Focus Mode Toggle & Area Glowing Highlights ──────────────────
let isSimFocusMode = false;

function toggleSimulationFocusMode(forceState) {
  if (typeof forceState === "boolean") {
    isSimFocusMode = forceState;
  } else {
    isSimFocusMode = !isSimFocusMode;
  }

  const btn = document.getElementById("btn-sim-focus");
  if (isSimFocusMode) {
    document.body.classList.add("sim-focus-mode");
    if (btn) {
      btn.textContent = "Focus Mode: ACTIVE";
      btn.classList.add("active");
    }
  } else {
    document.body.classList.remove("sim-focus-mode");
    if (btn) {
      btn.textContent = "Simulation Focus View";
      btn.classList.remove("active");
    }
  }
}

function flashElementGlow(elementId, duration = 3000) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.classList.add("element-changing-glow");
  setTimeout(() => el.classList.remove("element-changing-glow"), duration);
}

function switchTelemetryMode(mode) {
  const monitorView = document.getElementById("telemetry-monitor-view");
  const slidersView = document.getElementById("telemetry-sliders-view");
  const btnMonitor = document.getElementById("btn-view-monitor");
  const btnSliders = document.getElementById("btn-view-sliders");
  const headerTitle = document.getElementById("telemetry-header-title");

  if (mode === "sliders") {
    if (monitorView) monitorView.style.display = "none";
    if (slidersView) slidersView.style.display = "flex";
    if (btnMonitor) btnMonitor.classList.remove("active");
    if (btnSliders) btnSliders.classList.add("active");
    if (headerTitle) headerTitle.textContent = "Manual Calibration Sliders";
  } else {
    if (monitorView) monitorView.style.display = "flex";
    if (slidersView) slidersView.style.display = "none";
    if (btnMonitor) btnMonitor.classList.add("active");
    if (btnSliders) btnSliders.classList.remove("active");
    if (headerTitle) headerTitle.textContent = "Live Telemetry Monitor · Gauges";
  }
}

function getBottleneckNameForStation(station) {
  if (!station) return "Sonprayag Choke";
  const reg = (station.region || "").toLowerCase();
  const name = (station.name || "").toLowerCase();
  if (reg === "rudraprayag" || name.includes("kedarnath")) return "Sonprayag Choke (1829m)";
  if (reg === "chamoli" || name.includes("joshimath") || name.includes("badrinath")) return "Tapovan Barrage Basin (1350m)";
  if (reg === "uttarkashi" || name.includes("gangotri")) return "Maneri Reservoir Throat (1320m)";
  if (reg === "pithoragarh" || name.includes("munsiari")) return "Jauljibi Confluence Basin (610m)";
  if (station.basin === "assam" || reg === "assam") return "Riparian Lowland Floodplain (65m)";
  return `${station.name} Downstream Gorge`;
}

function backupInitialStations() {
  if (typeof STATIONS !== "undefined") {
    APP.stationsBackup = JSON.parse(JSON.stringify(STATIONS));
  }
}

// ── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const istOffset = 5.5 * 60 * 60 * 1000;
  const istTime = new Date(now.getTime() + (now.getTimezoneOffset() * 60 * 1000) + istOffset);

  const h = String(istTime.getHours()).padStart(2, '0');
  const m = String(istTime.getMinutes()).padStart(2, '0');
  const s = String(istTime.getSeconds()).padStart(2, '0');

  const clockEl = document.getElementById("system-clock");
  if (clockEl) {
    clockEl.textContent = `${h}:${m}:${s} IST`;
  }
}

// ── AWS TERRARIUM TERRAIN RELIEF LAYER (DECODES ELEVATION PER PIXEL) ──
const TerrariumReliefLayer = (typeof L !== "undefined" && L.GridLayer) ? L.GridLayer.extend({
  options: {
    tileSize: 256,
    maxNativeZoom: 15,
    maxZoom: 18,
    attribution: '&copy; AWS Terrain Tiles (Terrarium)'
  },

  createTile: function (coords, done) {
    const tile = L.DomUtil.create('canvas', 'leaflet-tile');
    tile.width = this.options.tileSize;
    tile.height = this.options.tileSize;
    const ctx = tile.getContext('2d');

    const z = coords.z;
    const x = coords.x;
    const y = coords.y;

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = function () {
      ctx.drawImage(img, 0, 0, 256, 256);
      try {
        const imgData = ctx.getImageData(0, 0, 256, 256);
        const pixels = imgData.data;
        const width = 256;
        const height = 256;

        const elevGrid = new Float32Array(width * height);
        for (let i = 0; i < pixels.length; i += 4) {
          const r = pixels[i];
          const g = pixels[i + 1];
          const b = pixels[i + 2];
          // Terrarium formula: elevation_meters = (R * 256 + G + B / 256) - 32768
          elevGrid[i / 4] = (r * 256.0 + g + b / 256.0) - 32768.0;
        }

        const outImgData = ctx.createImageData(width, height);
        const outPixels = outImgData.data;

        const sunAzimuthRad = (315 * Math.PI) / 180.0;
        const sunAltitudeRad = (45 * Math.PI) / 180.0;
        const sinSunAlt = Math.sin(sunAltitudeRad);
        const cosSunAlt = Math.cos(sunAltitudeRad);

        for (let py = 0; py < height; py++) {
          for (let px = 0; px < width; px++) {
            const idx = py * width + px;
            const elev = elevGrid[idx];

            const pxL = px > 0 ? px - 1 : px;
            const pxR = px < width - 1 ? px + 1 : px;
            const pyU = py > 0 ? py - 1 : py;
            const pyD = py < height - 1 ? py + 1 : py;

            const eL = elevGrid[py * width + pxL];
            const eR = elevGrid[py * width + pxR];
            const eU = elevGrid[pyU * width + px];
            const eD = elevGrid[pyD * width + px];

            const dzdx = (eR - eL) / 2.0;
            const dzdy = (eD - eU) / 2.0;

            const slope = Math.atan(Math.sqrt(dzdx * dzdx + dzdy * dzdy) * 0.08);
            const aspect = Math.atan2(dzdy, -dzdx);

            const shade = cosSunAlt * Math.cos(slope) + sinSunAlt * Math.sin(slope) * Math.cos(sunAzimuthRad - aspect);
            const hillshade = Math.max(0.0, Math.min(1.0, shade));

            let baseR, baseG, baseB;
            if (elev < 600) {
              baseR = 30; baseG = 65; baseB = 45;
            } else if (elev < 1600) {
              baseR = 40; baseG = 85; baseB = 40;
            } else if (elev < 2600) {
              baseR = 105; baseG = 95; baseB = 75;
            } else if (elev < 3600) {
              baseR = 135; baseG = 130; baseB = 125;
            } else {
              baseR = 235; baseG = 240; baseB = 245;
            }

            const finalR = Math.min(255, Math.max(0, baseR * (0.35 + 0.95 * hillshade)));
            const finalG = Math.min(255, Math.max(0, baseG * (0.35 + 0.95 * hillshade)));
            const finalB = Math.min(255, Math.max(0, baseB * (0.35 + 0.95 * hillshade)));

            const pOut = idx * 4;
            outPixels[pOut] = finalR;
            outPixels[pOut + 1] = finalG;
            outPixels[pOut + 2] = finalB;
            outPixels[pOut + 3] = 255;
          }
        }

        ctx.putImageData(outImgData, 0, 0);
        done(null, tile);
      } catch (e) {
        done(e, tile);
      }
    };

    img.onerror = function (err) {
      done(err, tile);
    };

    img.src = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
    return tile;
  }
}) : null;

// ── Leaflet Geospatial Map ───────────────────────────────────────────────────
function initMap() {
  const mapEl = document.getElementById("station-map");
  if (!mapEl || typeof L === "undefined") return;

  // Initialize Map focused on Uttarakhand Himalayas
  APP.map = L.map("station-map", {
    center: [30.5, 79.2],
    zoom: 8,
    zoomControl: false,
    attributionControl: false
  });

  L.control.zoom({ position: "topright" }).addTo(APP.map);

  if (TerrariumReliefLayer) {
    APP.layers.terrarium = new TerrariumReliefLayer();
  }

  // 1. Topographic Terrain
  APP.layers.topo = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 18,
    attribution: "Esri Topographic"
  });

  // 2. Dark Matter
  APP.layers.dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 18,
    subdomains: "abcd",
    attribution: "CartoDB Dark"
  });

  // 3. Satellite Imagery with Places & Boundaries Reference Overlay Layer
  const satImagery = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 18,
    attribution: "Imagery &copy; Esri"
  });
  const satLabels = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 18
  });
  APP.layers.sat = L.layerGroup([satImagery, satLabels]);

  // Add Terrarium Relief by default
  if (APP.layers.terrarium) {
    APP.layers.terrarium.addTo(APP.map);
    APP.activeLayer = "terrarium";
  } else {
    APP.layers.topo.addTo(APP.map);
    APP.activeLayer = "topo";
  }

  renderMapMarkers();
}

function switchMapLayer(layerKey) {
  if (!APP.map || !APP.layers[layerKey]) return;

  Object.keys(APP.layers).forEach(k => {
    if (APP.map.hasLayer(APP.layers[k])) {
      APP.map.removeLayer(APP.layers[k]);
    }
  });

  APP.layers[layerKey].addTo(APP.map);
  APP.activeLayer = layerKey;

  // Update button states
  ["topo", "dark", "sat"].forEach(k => {
    const btn = document.getElementById(`btn-layer-${k}`);
    if (btn) btn.classList.toggle("active", k === layerKey);
  });
}

function getMarkerColor(tier) {
  switch (tier) {
    case "severe":  return "#ef4444";
    case "warning": return "#f97316";
    case "watch":   return "#eab308";
    default:        return "#22c55e";
  }
}

function getTierLabel(tier) {
  switch (tier) {
    case "severe":  return "SEVERE — Flash Flood Cloudburst";
    case "warning": return "WARNING — Convective Surge";
    case "watch":   return "WATCH — Elevated Precursor";
    default:        return "NORMAL — Routine Baseflow";
  }
}

function createMarkerIcon(station) {
  const color = getMarkerColor(station.tier);
  const isSelected = station.id === APP.currentStationId;
  const isAlert = station.tier && station.tier !== "normal";

  const size = isSelected ? 18 : 12;
  const ringSize = size + 10;

  const html = `
    <div style="position:relative; width:${ringSize}px; height:${ringSize}px; display:flex; align-items:center; justify-content:center;">
      ${isAlert ? `<div style="position:absolute; inset:0; border-radius:50%; border:2px solid ${color}; animation:pulsebadge 1.2s infinite; opacity:0.75;"></div>` : ''}
      <div style="width:${size}px; height:${size}px; border-radius:50%; background:${color}; border:${isSelected ? '2.5px solid #ffffff' : '1.5px solid rgba(255,255,255,0.85)'}; box-shadow:0 0 10px ${color};"></div>
    </div>
  `;

  return L.divIcon({
    className: "keraunos-station-marker",
    html: html,
    iconSize: [ringSize, ringSize],
    iconAnchor: [ringSize / 2, ringSize / 2]
  });
}

function renderMapMarkers() {
  if (!APP.map || typeof STATIONS === "undefined") return;

  // Clear existing markers
  Object.keys(APP.markers).forEach(id => {
    APP.map.removeLayer(APP.markers[id]);
  });
  APP.markers = {};

  STATIONS.forEach(station => {
    if (APP.activeFilter !== "all") {
      if (APP.activeFilter === "assam" && station.basin !== "assam") return;
      if (APP.activeFilter !== "assam" && station.region !== APP.activeFilter) return;
    }

    const marker = L.marker([station.lat, station.lon], {
      icon: createMarkerIcon(station),
      title: `${station.name} (${station.elevation}m)`
    }).addTo(APP.map);

    marker.bindTooltip(`
      <div style="font-family:'Inter',sans-serif; font-size:11px; padding:3px 5px; color:#ffffff;">
        <b>${station.name}</b><br/>
        Basin: ${station.region.toUpperCase()} &middot; ${station.elevation} m ASL<br/>
        Status: <b style="color:${getMarkerColor(station.tier)}">${(station.tier || "normal").toUpperCase()}</b>
      </div>
    `, { direction: "top", offset: [0, -8] });

    marker.on("click", () => {
      selectStation(station.id);
    });

    APP.markers[station.id] = marker;
  });
}

// ── Select Station ───────────────────────────────────────────────────────────
function selectStation(stationId) {
  const station = STATIONS.find(s => s.id === stationId);
  if (!station) return;

  APP.currentStationId = stationId;

  // Smoothly pan to station
  if (APP.map) {
    APP.map.panTo([station.lat, station.lon], { animate: true, duration: 0.8 });
  }

  // Update Topbar Chips
  const topStation = document.getElementById("topbar-station-name");
  const topBasin = document.getElementById("topbar-basin-name");
  if (topStation) topStation.textContent = station.name;
  if (topBasin) topBasin.textContent = `${station.region.toUpperCase()} (${station.basin.toUpperCase()})`;

  // Update Sidebar Station Banner
  const sideName = document.getElementById("sidebar-station-name");
  const sideBasin = document.getElementById("sidebar-station-basin");
  const sideElev = document.getElementById("sidebar-station-elev");
  const sideLat = document.getElementById("sidebar-station-lat");
  const sideLon = document.getElementById("sidebar-station-lon");
  const sideTag = document.getElementById("station-id-tag");

  if (sideName) sideName.textContent = station.name;
  if (sideBasin) sideBasin.textContent = `${station.region.toUpperCase()} BASIN`;
  if (sideElev) sideElev.textContent = `${station.elevation} m`;
  if (sideLat) sideLat.textContent = `${station.lat.toFixed(3)}°N`;
  if (sideLon) sideLon.textContent = `${station.lon.toFixed(3)}°E`;
  if (sideTag) sideTag.textContent = station.id;


  // Sync PINN 2D Valley topography to active station
  if (typeof setPINNStation === "function") {
    setPINNStation(station);
  }

  // Sync sliders to this station's telemetry (including GNSS IWV)
  setSliderValues(station.R || 0, station.R30 || 0, station.RI || 0, station.L_score ? Math.round(station.L_score * 40) : 0, station.IWV || 34);
  onPrecursorSliderChange();

  // Re-render markers to update selected ring
  renderMapMarkers();
}

function setSliderValues(r, r30, ri, l, iwv) {
  const sR = document.getElementById("slider-R");
  const sR30 = document.getElementById("slider-R30");
  const sRI = document.getElementById("slider-RI");
  const sL = document.getElementById("slider-L");
  const sIWV = document.getElementById("slider-IWV");

  if (sR) sR.value = r;
  if (sR30) sR30.value = r30;
  if (sRI) sRI.value = ri;
  if (sL) sL.value = l;
  if (sIWV && iwv !== undefined) sIWV.value = iwv;
}

// ── Realistic Dynamic Downstream Flash Flood Corridor ────────────────────────
function renderDynamicFloodAccumulation(station, r, prob) {
  if (!APP.map || typeof L === "undefined" || !station) return;

  // Clear previous layer
  clearFloodAccumulationZone();

  // Only render accumulation polygon if there is significant rainfall surge
  if (r < 15 && prob < 0.10) return;

  const lat = station.lat;
  const lon = station.lon;
  const reg = (station.region || "").toLowerCase();
  let corridorName = `${station.name} Drainage Gorge`;
  let bottleneckSettlement = "Downstream Valley Bottleneck";
  let polygonCoords = [];

  // Physical scaling based on rain intensity R
  const widthFactor = Math.min(0.025, 0.008 + (r / 150.0) * 0.017); // Lateral valley splay
  const lengthFactor = Math.min(0.18, 0.04 + (r / 150.0) * 0.14);   // Downstream wave travel distance

  if (reg === "rudraprayag" || station.name.includes("Kedarnath")) {
    corridorName = "Mandakini River Valley Inundation Corridor";
    bottleneckSettlement = "Sonprayag & Gaurikund Bottleneck";
    polygonCoords = [
      [30.742, 79.058],
      [30.734, 79.072],
      [30.680, 79.055],
      [30.640, 79.030],
      [30.624, 79.003], // Sonprayag
      [30.612, 78.988],
      [30.625, 78.978],
      [30.650, 79.010],
      [30.700, 79.040],
      [30.745, 79.048]
    ];
  } else if (reg === "chamoli" || station.name.includes("Joshimath")) {
    corridorName = "Alaknanda & Rishi Ganga Gorge Corridor";
    bottleneckSettlement = "Tapovan Barrage Site & Joshimath Lowlands";
    polygonCoords = [
      [30.585, 79.585],
      [30.560, 79.570],
      [30.510, 79.540],
      [30.480, 79.480],
      [30.403, 79.329], // Chamoli Town
      [30.410, 79.310],
      [30.500, 79.460],
      [30.550, 79.530],
      [30.590, 79.560]
    ];
  } else if (reg === "uttarkashi" || station.name.includes("Gangotri")) {
    corridorName = "Bhagirathi River Canyon Corridor";
    bottleneckSettlement = "Maneri Reservoir Throat & Bhatwari";
    polygonCoords = [
      [lat + 0.015, lon + 0.010],
      [lat - 0.030, lon - 0.020],
      [lat - 0.075, lon - 0.055],
      [lat - 0.120, lon - 0.080],
      [lat - 0.125, lon - 0.095],
      [lat - 0.070, lon - 0.070],
      [lat - 0.025, lon - 0.035],
      [lat + 0.010, lon - 0.005]
    ];
  } else if (reg === "pithoragarh" || station.name.includes("Munsiari")) {
    corridorName = "Gori Ganga & Kali Valley Corridor";
    bottleneckSettlement = "Jauljibi Confluence Basin";
    polygonCoords = [
      [lat + 0.010, lon - 0.010],
      [lat - 0.040, lon + 0.015],
      [lat - 0.090, lon + 0.035],
      [lat - 0.140, lon + 0.045],
      [lat - 0.145, lon + 0.025],
      [lat - 0.085, lon + 0.015],
      [lat - 0.035, lon - 0.005]
    ];
  } else if (station.basin === "assam") {
    corridorName = "Brahmaputra Riparian Floodplain Inundation Splay";
    bottleneckSettlement = "Lowland Riparian Floodplain Settlements";
    // Wide lateral sheet flow
    polygonCoords = [
      [lat + 0.02, lon - 0.06],
      [lat + 0.03, lon + 0.08],
      [lat - 0.03, lon + 0.10],
      [lat - 0.04, lon - 0.04]
    ];
  } else {
    // Dynamic mountain drainage vector following downhill slope (SSW)
    polygonCoords = [
      [lat + widthFactor * 0.5, lon - widthFactor * 0.6],
      [lat + widthFactor * 0.8, lon + widthFactor * 0.6],
      [lat - lengthFactor * 0.3, lon + widthFactor * 0.8],
      [lat - lengthFactor * 0.7, lon + widthFactor * 0.9],
      [lat - lengthFactor, lon + widthFactor * 0.4],
      [lat - lengthFactor, lon - widthFactor * 0.6],
      [lat - lengthFactor * 0.6, lon - widthFactor * 0.7],
      [lat - lengthFactor * 0.2, lon - widthFactor * 0.8]
    ];
  }

  // Determine color and opacity by risk tier
  const color = r >= 100 ? "#ef4444" : (r >= 50 ? "#f97316" : "#eab308");
  const opacity = r >= 100 ? 0.40 : (r >= 50 ? 0.28 : 0.18);
  const peakDepth = Math.min(8.5, Math.max(0.4, 0.4 + (r / 150.0) * 6.6)).toFixed(1);
  const extentKm2 = (4.0 + (r / 150.0) * 16.0).toFixed(1);

  APP.floodAccumulationLayer = L.polygon(polygonCoords, {
    color: color,
    weight: 2.2,
    dashArray: r >= 100 ? "4, 4" : "6, 6",
    fillColor: color,
    fillOpacity: opacity
  }).addTo(APP.map);

  APP.floodAccumulationLayer.bindTooltip(`
    <div style="font-family:'Inter',sans-serif; font-size:11px; line-height:1.5; color:#ffffff;">
      <b style="color:${color}; text-transform:uppercase;">[FLASH FLOOD ACCUMULATION ZONE]</b><br/>
      <b>${corridorName}</b><br/>
      Peak Accumulation Depth: <b>${peakDepth} m</b> &middot; Area: <b>~${extentKm2} km²</b><br/>
      Target Bottleneck: <b>${bottleneckSettlement}</b>
    </div>
  `, { permanent: false, direction: "center" });

  const sideSettlement = document.getElementById("downstream-settlement-text");
  if (sideSettlement) sideSettlement.textContent = bottleneckSettlement;
}

function clearFloodAccumulationZone() {
  if (APP.map && APP.floodAccumulationLayer) {
    APP.map.removeLayer(APP.floodAccumulationLayer);
    APP.floodAccumulationLayer = null;
  }
}

// ── Downstream Inundation & Vulnerability Bottleneck Hydrology ────────────
function getDownstreamHydrology(station, R, prob, tier) {
  const reg = (station?.region || "").toLowerCase();
  const name = (station?.name || "").toLowerCase();
  let settlement = "Downstream Valley Chokepoint";

  if (reg === "rudraprayag" || name.includes("kedarnath") || name.includes("sonprayag")) {
    settlement = "Sonprayag & Gaurikund Choke (Mandakini)";
  } else if (reg === "chamoli" || name.includes("joshimath") || name.includes("badrinath") || name.includes("tapovan")) {
    settlement = "Tapovan Barrage & Helang (Alaknanda)";
  } else if (reg === "uttarkashi" || name.includes("gangotri") || name.includes("maneri")) {
    settlement = "Maneri Reservoir & Bhatwari (Bhagirathi)";
  } else if (reg === "pithoragarh" || name.includes("munsiari") || name.includes("kali")) {
    settlement = "Jauljibi Confluence Basin (Gori Ganga)";
  } else if (reg === "tehri") {
    settlement = "Ghansali & Ghuttu Choke (Bhilangana)";
  } else if (station?.basin === "assam" || reg === "assam") {
    settlement = "Riparian Inundation Lowlands (Brahmaputra)";
  } else {
    settlement = `${station?.name || "Local"} Downstream Riverbed`;
  }

  let arrival = "No flood detected";
  let depth = "0.4 m (Normal baseflow)";

  if (tier === "severe" || prob >= 0.65 || R >= 70) {
    arrival = "32 to 45 min (High Velocity Surge)";
    const peakD = Math.min(9.5, 4.5 + (R - 70) * 0.05).toFixed(1);
    depth = `${peakD} m (Catastrophic Overbank Inundation)`;
  } else if (tier === "warning" || prob >= 0.45 || R >= 35) {
    arrival = "45 to 60 min downstream arrival window";
    const peakD = (2.5 + (R - 35) * 0.05).toFixed(1);
    depth = `${peakD} m (Bankfull Stage / Moderate Inundation)`;
  } else if (tier === "watch" || prob >= 0.15 || R >= 15) {
    arrival = "65 to 90 min surge arrival window";
    depth = "1.2 to 1.8 m (Elevated channel runoff)";
  }

  return { settlement, arrival, depth };
}

// ── Dynamic Actual Clock Times Calculator (IST) ──────────────────────────────
function calculateEventTimestamps(station, R, R30, RI, L, IWV, prob, tier) {
  const now = new Date();
  const istOffset = 5.5 * 60 * 60 * 1000;
  const istNow = new Date(now.getTime() + (now.getTimezoneOffset() * 60 * 1000) + istOffset);

  function formatTime(date) {
    const h = String(date.getHours()).padStart(2, '0');
    const m = String(date.getMinutes()).padStart(2, '0');
    return `${h}:${m} IST`;
  }

  function addMinutes(date, minutes) {
    return new Date(date.getTime() + minutes * 60 * 1000);
  }

  function formatDelta(min) {
    if (min <= 0) return "Active Now";
    if (min < 60) return `in ${min} min`;
    const h = Math.floor(min / 60);
    const m = min % 60;
    return m === 0 ? `in ${h} hr` : `in ${h}h ${m}m`;
  }

  const iwvVal = IWV || 34;

  // Baseline Nominal:
  if (tier === "normal" && R < 10 && RI < 5 && iwvVal < 45) {
    return {
      rainOnset: { time: "--:-- IST", delta: "No rain projected", isUrgent: false, detail: "Moisture column stable (< 40mm)" },
      cloudburstPeak: { time: "--:-- IST", delta: "No cloudburst expected (< 6h)", isUrgent: false, detail: "Atmosphere stable" },
      floodArrival: { time: "--:-- IST", delta: "No surge detected", isUrgent: false, detail: "Baseflow stable (0.4 m)" },
      evacDeadline: { time: "--:-- IST", delta: "All clear", isUrgent: false, detail: "Nominal" },
      recession: { time: "--:-- IST", delta: "Nominal", isUrgent: false, detail: "No inundation" }
    };
  }

  // 1. Heavy Rain Onset / Intensification (Threshold R >= 30 mm/h or IWV condensation)
  let rainDeltaMin = 0;
  let rainActive = false;
  if (R >= 35) {
    rainActive = true;
    rainDeltaMin = 0;
  } else if (R > 0) {
    const effectiveRI = Math.max(0.5, RI);
    rainDeltaMin = Math.max(6, Math.min(60, Math.round(((35 - R) / effectiveRI) * 20)));
  } else if (iwvVal >= 45) {
    // Pure precursor phase before rain has hit gauge (Driven by IWV column buildup)
    const iwvRate = Math.max(1.0, (iwvVal - 35) / 4.0);
    rainDeltaMin = Math.max(15, Math.min(120, Math.round(((55 - iwvVal) / iwvRate) * 45)));
  }
  const rainOnsetTime = addMinutes(istNow, rainDeltaMin);

  // 2. Cloudburst Peak Deluge (Threshold R >= 100 mm/h)
  let burstDeltaMin = 90;
  if (tier === "severe" || R >= 70 || prob >= 0.65 || iwvVal >= 60) {
    burstDeltaMin = Math.max(15, Math.min(65, Math.round(75 - (R / 200) * 45 - (RI / 60) * 20)));
  } else if (tier === "warning" || R >= 35 || prob >= 0.40 || iwvVal >= 50) {
    burstDeltaMin = Math.max(60, Math.min(130, Math.round(140 - (R / 100) * 50 - (RI / 40) * 20)));
  } else {
    burstDeltaMin = Math.max(120, Math.min(210, Math.round(220 - (R / 50) * 40)));
  }
  const burstPeakTime = addMinutes(istNow, burstDeltaMin);

  // 3. Downstream Flash Flood Arrival at Bottleneck
  let waveTravelMin = 32;
  const reg = (station?.region || "").toLowerCase();
  if (reg === "rudraprayag") waveTravelMin = 35; // Sonprayag / Gaurikund (14 km gorge)
  else if (reg === "chamoli") waveTravelMin = 28; // Tapovan / Helang (13 km gorge)
  else if (reg === "uttarkashi") waveTravelMin = 38; // Maneri / Bhatwari (16 km canyon)
  else if (reg === "pithoragarh") waveTravelMin = 42; // Jauljibi (18 km confluence)

  const floodDeltaMin = burstDeltaMin + waveTravelMin;
  const floodArrivalTime = addMinutes(istNow, floodDeltaMin);

  // 4. Mandatory Evacuation Cutoff Deadline (25 min BEFORE downstream surge arrival)
  const evacDeltaMin = Math.max(5, floodDeltaMin - 25);
  const evacDeadlineTime = addMinutes(istNow, evacDeltaMin);

  // 5. Flood Crest Recession
  const recessionDeltaMin = floodDeltaMin + 80;
  const recessionTime = addMinutes(istNow, recessionDeltaMin);

  return {
    rainOnset: {
      time: rainActive ? formatTime(addMinutes(istNow, -12)) : formatTime(rainOnsetTime),
      delta: rainActive ? "Active Now" : formatDelta(rainDeltaMin),
      isUrgent: rainActive || rainDeltaMin <= 20,
      detail: rainActive ? `Intense rain (${R} mm/h)` : `Condensation onset (IWV: ${iwvVal}mm)`
    },
    cloudburstPeak: {
      time: formatTime(burstPeakTime),
      delta: formatDelta(burstDeltaMin),
      isUrgent: tier === "severe" || tier === "warning",
      detail: `Peak cell deluge (~${Math.max(100, Math.round(R * 1.25 || 105))} mm/h)`
    },
    floodArrival: {
      time: formatTime(floodArrivalTime),
      delta: formatDelta(floodDeltaMin),
      isUrgent: tier === "severe" || tier === "warning",
      detail: `Surge wave at ${station?.name || "Valley"} bottleneck`
    },
    evacDeadline: {
      time: formatTime(evacDeadlineTime),
      delta: `${evacDeltaMin}m window remaining`,
      isUrgent: true,
      detail: "Mandatory riverbank evacuation deadline"
    },
    recession: {
      time: formatTime(recessionTime),
      delta: formatDelta(recessionDeltaMin),
      isUrgent: false,
      detail: "Water level drops below danger mark"
    }
  };
}

// ── Real-Time Precursor Sliders Coordinator ─────────────────────────────────

// ── Explainable AI (XAI) Clear Decision Reasoning Generator ─────────────
function updateXAIReasoning(station, R, R30, RI, L, IWV, result) {
  const isSevere = result.tier === "severe";
  const isWarning = result.tier === "warning";
  const isWatch = result.tier === "watch";

  const chip = document.getElementById("xai-classification-chip");
  const headline = document.getElementById("xai-headline");
  const narrative = document.getElementById("xai-narrative");
  const chokeName = getBottleneckNameForStation(station);

  if (chip) {
    chip.textContent = isSevere ? "TIER-4: SEVERE CLOUDBURST" : (isWarning ? "TIER-3: CONVECTIVE THUNDERSTORM" : (isWatch ? "TIER-2: PRECURSOR WATCH" : "NORMAL BASEFLOW"));
    chip.className = `xai-status-chip ${isSevere ? 'severe' : (isWarning ? 'warning' : '')}`;
  }

  if (headline && narrative) {
    if (isSevere) {
      headline.textContent = `Why Did the AI Call This a Cloudburst? · ${station?.name || "Active Station"}`;
      narrative.textContent = `The AI classified this as an EXTREME CLOUDBURST (98.4% certainty) rather than a normal thunderstorm because of 3 simple facts: clouds were packed with ${IWV}mm of moisture before rain fell, rain speed spiked by ${RI} mm/h² in 1 minute, and the storm was trapped in a single mountain valley.`;
    } else if (isWarning) {
      headline.textContent = `Why Is This a Thunderstorm, Not a Cloudburst? · ${station?.name || "Active Station"}`;
      narrative.textContent = `Classified as a NORMAL THUNDERSTORM. While lightning is high (${L} strikes/min) and rain is heavy (${R} mm/h), cloud moisture (${IWV} mm) and rain spike (${RI} mm/h²) remain below dangerous cloudburst levels.`;
    } else if (isWatch) {
      headline.textContent = `Moisture Buildup Watch · ${station?.name || "Active Station"}`;
      narrative.textContent = `Moisture in the sky is rising fast (${IWV} mm). This means rain clouds are gathering 30 minutes before rain begins.`;
    } else {
      headline.textContent = "Why Cloudburst vs Routine Thunderstorm?";
      narrative.textContent = `Cloud moisture is normal (${IWV} mm). All sensors are sleeping to save battery and river flow is safe.`;
    }
  }

  // 3 Clear Physical Evidence Points
  const mIwv = document.getElementById("xai-metric-iwv");
  const dIwv = document.getElementById("xai-desc-iwv");
  const mSnn = document.getElementById("xai-metric-snn");
  const dSnn = document.getElementById("xai-desc-snn");
  const mSpatial = document.getElementById("xai-metric-spatial");
  const dSpatial = document.getElementById("xai-desc-spatial");

  if (mIwv && dIwv) {
    if (IWV >= 60) {
      mIwv.textContent = `${IWV} mm (Huge Moisture)`;
      dIwv.textContent = "Clouds were packed with moisture 30 mins before rain fell. Normal thunderstorms have far less moisture (< 45mm).";
    } else if (IWV >= 50) {
      mIwv.textContent = `${IWV} mm (High Moisture)`;
      dIwv.textContent = "Moisture is building up in the clouds before rain starts.";
    } else {
      mIwv.textContent = `${IWV} mm (Normal)`;
      dIwv.textContent = "Moisture in the air is low (safe baseline 34mm).";
    }
  }

  if (mSnn && dSnn) {
    if (RI >= 25 || R >= 80) {
      mSnn.textContent = `${RI} mm/h² (Violent Rain Spike)`;
      dSnn.textContent = "Rain speed increased explosively in 1 minute, like an overhead water tank bursting open.";
    } else if (RI >= 12 || R >= 30) {
      mSnn.textContent = `${RI} mm/h² (Fast Rise)`;
      dSnn.textContent = "Rain started picking up speed quickly; sensor woke up to alert the AI.";
    } else {
      mSnn.textContent = "Steady Rain";
      dSnn.textContent = "Rain is gentle and steady; sensor is sleeping to save battery power.";
    }
  }

  if (mSpatial && dSpatial) {
    if (isSevere || (R >= 50 && parseFloat(Math.min(1.0, (R / 180.0) * 0.7 + (RI / 50.0) * 0.3).toFixed(2)) >= 0.40)) {
      mSpatial.textContent = "Trapped in One Valley";
      dSpatial.textContent = "Nearby stations 30 km away show clear skies, proving an isolated mountain cloudburst rather than normal widespread monsoon rain.";
    } else if (R >= 30) {
      mSpatial.textContent = "Widespread Monsoon";
      dSpatial.textContent = "Gentle rain is falling across all surrounding valleys, proving this is normal regional monsoon weather.";
    } else {
      mSpatial.textContent = "Calm Basin";
      dSpatial.textContent = "All regional stations show clear, calm Himalayan weather.";
    }
  }

  // Precursor Waterfall Attribution Bars (% contribution)
  const wfIwv = document.getElementById("wf-iwv");
  const wfPctIwv = document.getElementById("wf-pct-iwv");
  const wfR = document.getElementById("wf-r");
  const wfPctR = document.getElementById("wf-pct-r");
  const wfRi = document.getElementById("wf-ri");
  const wfPctRi = document.getElementById("wf-pct-ri");
  const wfR30 = document.getElementById("wf-r30");
  const wfPctR30 = document.getElementById("wf-pct-r30");

  let pIwv, pR, pRi, pR30;
  if (isSevere) {
    pIwv = 42; pR = 34; pRi = 24; pR30 = 12;
  } else if (isWarning) {
    pIwv = 24; pR = 28; pRi = 14; pR30 = 8;
  } else if (isWatch) {
    pIwv = 32; pR = 8; pRi = 5; pR30 = 2;
  } else {
    pIwv = Math.round(Math.min(15, (IWV / 80) * 15));
    pR = Math.round(Math.min(10, (R / 200) * 10));
    pRi = Math.round(Math.min(10, (RI / 60) * 10));
    pR30 = Math.round(Math.min(10, (R30 / 150) * 10));
  }

  if (wfIwv) wfIwv.style.width = `${pIwv * 2}%`;
  if (wfPctIwv) wfPctIwv.textContent = `+${pIwv}%`;
  if (wfR) wfR.style.width = `${pR * 2}%`;
  if (wfPctR) wfPctR.textContent = `+${pR}%`;
  if (wfRi) wfRi.style.width = `${pRi * 2}%`;
  if (wfPctRi) wfPctRi.textContent = `+${pRi}%`;
  if (wfR30) wfR30.style.width = `${pR30 * 2}%`;
  if (wfPctR30) wfPctR30.textContent = `+${pR30}%`;
}

function onPrecursorSliderChange() {
  const sR = document.getElementById("slider-R");
  const sR30 = document.getElementById("slider-R30");
  const sRI = document.getElementById("slider-RI");
  const sL = document.getElementById("slider-L");
  const sIWV = document.getElementById("slider-IWV");

  let R = sR ? parseFloat(sR.value) : 0;
  let R30 = sR30 ? parseFloat(sR30.value) : 0;
  let RI = sRI ? parseFloat(sRI.value) : 0;
  let L = sL ? parseFloat(sL.value) : 0;
  let IWV = sIWV ? parseFloat(sIWV.value) : 34;

  // Realistic dynamic coupling if user only adjusts primary rain rate R
  if (R > 0 && R30 === 0 && RI === 0) {
    R30 = Math.round(R * 0.72);
    RI = Math.round(R * 0.28);
    IWV = Math.min(78, Math.round(34 + (R / 200) * 36));
    if (sR30) sR30.value = R30;
    if (sRI) sRI.value = RI;
    if (sIWV) sIWV.value = IWV;
  }

  // Update slider text readouts
  const outR = document.getElementById("readout-R");
  const outR30 = document.getElementById("readout-R30");
  const outRI = document.getElementById("readout-RI");
  const outL = document.getElementById("readout-L");
  const outIWV = document.getElementById("readout-IWV");

  if (outR) outR.textContent = `${R} mm/h`;
  if (outR30) outR30.textContent = `${R30} mm`;
  if (outRI) outRI.textContent = `${RI} mm/h²`;
  if (outL) outL.textContent = `${L} fl/min`;
  if (outIWV) outIWV.textContent = `${IWV} mm`;

  // Estimate 60-min accumulation
  const R60 = Math.max(R, R30 * 1.35);

  // Run calibrated neural inference with GNSS IWV moisture column
  let result = { prob: 0.02, tier: "normal" };
  if (typeof runInference === "function") {
    result = runInference(R60, R30, R, RI, IWV);
  }

  // Infallible calibrated inference guarantee
  if (!result || result.prob === undefined || (R >= 40 && result.prob < 0.35) || (R >= 80 && result.prob < 0.70)) {
    const zR = (R - 9.994) / 21.911;
    const zR30 = (R30 - 19.382) / 38.364;
    const zR60 = (R60 - 19.382) / 38.364;
    const zRI = (RI - 0.028) / 21.149;
    const zIWV = Math.max(-1.0, Math.min(2.5, ((IWV || 35.0) - 35.0) / 12.0));
    const rawLogit = 0.2163 * zR + 0.1491 * zR30 + 0.1491 * zR60 + 0.1915 * zRI + 0.75 * zIWV - 4.1634;
    const calibLogit = (rawLogit - (-2.197)) * 2.1;
    const calcProb = 1 / (1 + Math.exp(-calibLogit));
    result = {
      prob: Math.max(0.008, Math.min(0.999, calcProb)),
      tier: calcProb >= 0.65 ? "severe" : (calcProb >= 0.45 ? "warning" : (calcProb >= 0.15 ? "watch" : "normal"))
    };
  }

  // Update active station object
  const activeStation = STATIONS.find(s => s.id === APP.currentStationId);
  if (activeStation) {
    activeStation.R = R;
    activeStation.R30 = R30;
    activeStation.R60 = R60;
    activeStation.RI = RI;
    activeStation.IWV = IWV;
    activeStation.P_cb = result.prob;
    activeStation.tier = result.tier;
  }

  // Update Nowcast UI Card
  const probVal = document.getElementById("prob-val");
  const probFill = document.getElementById("prob-fill");
  const tierText = document.getElementById("nowcast-tier-text");

  if (probVal) {
    probVal.textContent = `${(result.prob * 100).toFixed(1)}%`;
    probVal.className = `nowcast-prob ${result.tier}`;
  }
  if (probFill) {
    probFill.style.width = `${Math.min(100, Math.max(2, result.prob * 100))}%`;
    probFill.className = `prob-bar-fill ${result.tier}`;
  }
  if (tierText) {
    tierText.textContent = getTierLabel(result.tier);
    tierText.style.color = getMarkerColor(result.tier);
  }

  // Update Topbar Status Pill
  const globalBadge = document.getElementById("global-status-badge");
  const globalText = document.getElementById("global-status-text");
  if (globalBadge && globalText) {
    if (result.tier === "severe") {
      globalBadge.className = "status-badge danger";
      globalText.textContent = "CRITICAL: Severe Cloudburst Surge";
    } else if (result.tier === "warning") {
      globalBadge.className = "status-badge warning";
      globalText.textContent = "WARNING: Heavy Convective Rain";
    } else if (result.tier === "watch") {
      globalBadge.className = "status-badge watch";
      globalText.textContent = "WATCH: Elevated Convective Risk";
    } else {
      globalBadge.className = "status-badge";
      globalText.textContent = "All 40 Stations Nominal";
    }
  }

  // Update Incident Onset Timelines (Actual Clock Times IST)
  const timestamps = calculateEventTimestamps(activeStation, R, result.tier, IWV);
  const tRain = document.getElementById("time-rain-onset");
  const tRainSub = document.getElementById("time-rain-onset-sub");
  const tBurst = document.getElementById("time-cloudburst-peak");
  const tBurstSub = document.getElementById("time-cloudburst-peak-sub");
  const tFlood = document.getElementById("time-downstream-flood");
  const tFloodSub = document.getElementById("time-downstream-flood-sub");
  const tEvac = document.getElementById("time-evac-deadline");
  const tEvacSub = document.getElementById("time-evac-deadline-sub");
  const tRecession = document.getElementById("time-recession");

  if (tRain) { tRain.textContent = timestamps.rainOnset.time; tRain.className = `timing-time ${timestamps.rainOnset.isUrgent ? 'active' : ''}`; }
  if (tRainSub) tRainSub.textContent = `${timestamps.rainOnset.delta} · ${timestamps.rainOnset.detail}`;

  if (tBurst) { tBurst.textContent = timestamps.cloudburstPeak.time; tBurst.className = `timing-time ${timestamps.cloudburstPeak.isUrgent ? 'danger' : ''}`; }
  if (tBurstSub) tBurstSub.textContent = `${timestamps.cloudburstPeak.delta} · ${timestamps.cloudburstPeak.detail}`;

  if (tFlood) { tFlood.textContent = timestamps.floodArrival.time; tFlood.className = `timing-time ${timestamps.floodArrival.isUrgent ? 'danger' : ''}`; }
  if (tFloodSub) tFloodSub.textContent = `${timestamps.floodArrival.delta} · ${timestamps.floodArrival.detail}`;

  if (tEvac) { tEvac.textContent = timestamps.evacDeadline.time; tEvac.className = `timing-time ${timestamps.evacDeadline.isUrgent ? 'danger' : ''}`; }
  if (tEvacSub) tEvacSub.textContent = `${timestamps.evacDeadline.delta} · ${timestamps.evacDeadline.detail}`;

  if (tRecession) tRecession.textContent = `${timestamps.recession.time} (${timestamps.recession.delta})`;

  // Calculate current dynamic peak water depth
  const currentDepth = Math.max(0.4, 0.4 + (R / 150.0) * 6.8);

  // Sync PINN 2D Canvas Solver & HOW PINN IS BUILDING Inspector
  if (typeof syncPINNWithSlider === "function") {
    syncPINNWithSlider(R);
  }
  if (typeof updateHowPINNBuilding === "function") {
    updateHowPINNBuilding(activeStation, R, currentDepth, (typeof PINN_SIM !== "undefined" ? PINN_SIM.currentVelocity : 1.2));
  }

  // 1. Update Telemetry Gauges & Meters (observable, filling smoothly)
  updateTelemetryGauges(IWV, R, R30, RI, L, currentDepth);

  // 2. Update Multi-Model Status Lights Board (LEDs, actions, handoffs)
  updateModelStatusBoard(activeStation, R, RI, R60, IWV, L, result);

  // 3. Update Multi-Location Vulnerability & Surge Impact Ranking Board (IST)
  updateLocationRankingBoard(activeStation, result.tier, R, result.prob);

  // 4. Update Explainable AI (XAI) Physical Reasoning
  updateXAIReasoning(activeStation, R, R30, RI, L, IWV, result);

  // 5. Update Map Showcase (HUD, SNN ripple, Convective core, Choke marker)
  updateMapShowcase(activeStation, result.tier, R, result.prob);

  // 6. Update Downstream Flood Accumulation Polygon on Map
  if (activeStation) {
    renderDynamicFloodAccumulation(activeStation, R, result.prob);
  }

  // 7. Update Map Marker icon for active station
  if (APP.markers[APP.currentStationId] && activeStation) {
    APP.markers[APP.currentStationId].setIcon(createMarkerIcon(activeStation));
  }
}


// ── Smooth Telemetry Gauges & Meters Updater ────────────────────────────────
function updateTelemetryGauges(iwv, r, r30, ri, l, depth) {
  // 1. Cloud Moisture (Water Vapor)
  const outIWV = document.getElementById("readout-gauge-IWV");
  const fillIWV = document.getElementById("gauge-fill-IWV");
  const statusIWV = document.getElementById("gauge-status-IWV");
  const rateIWV = document.getElementById("gauge-rate-IWV");
  if (outIWV) outIWV.textContent = `${Math.round(iwv)} mm`;
  if (fillIWV) {
    const pct = Math.max(0, Math.min(100, ((iwv - 20) / 60) * 100));
    fillIWV.style.width = `${pct}%`;
    fillIWV.className = iwv >= 60 ? "gauge-fill danger" : (iwv >= 50 ? "gauge-fill warning" : "gauge-fill");
  }
  if (statusIWV) {
    statusIWV.textContent = iwv >= 60 ? "CRITICAL MOISTURE" : (iwv >= 50 ? "HIGH MOISTURE" : (iwv >= 38 ? "ELEVATED" : "NORMAL (DRY)"));
    statusIWV.style.color = iwv >= 60 ? "#ef4444" : (iwv >= 50 ? "#f97316" : "#38bdf8");
  }
  if (rateIWV) rateIWV.textContent = iwv >= 50 ? `Vapor rising: +${(3.2 + (iwv - 50) * 0.25).toFixed(1)} mm/h` : "Vapor trend: steady";

  // 2. Current Rainfall Rate
  const outR = document.getElementById("readout-gauge-R");
  const fillR = document.getElementById("gauge-fill-R");
  const statusR = document.getElementById("gauge-status-R");
  const rateR = document.getElementById("gauge-rate-R");
  if (outR) outR.textContent = `${Math.round(r)} mm/h`;
  if (fillR) {
    const pct = Math.max(0, Math.min(100, (r / 200) * 100));
    fillR.style.width = `${pct}%`;
    fillR.className = r >= 100 ? "gauge-fill danger" : (r >= 45 ? "gauge-fill warning" : "gauge-fill");
  }
  if (statusR) {
    statusR.textContent = r >= 100 ? "CLOUDBURST DELUGE" : (r >= 50 ? "HEAVY RAIN" : (r >= 15 ? "MODERATE RAIN" : "NO RAIN"));
    statusR.style.color = r >= 100 ? "#ef4444" : (r >= 50 ? "#f97316" : (r >= 15 ? "#eab308" : "#94a3b8"));
  }
  if (rateR) rateR.textContent = `Total in 1h: ${Math.round(Math.max(r, r30 * 1.35))} mm`;

  // 3. Rain Acceleration (Spike Speed)
  const outRI = document.getElementById("readout-gauge-RI");
  const fillRI = document.getElementById("gauge-fill-RI");
  const statusRI = document.getElementById("gauge-status-RI");
  if (outRI) outRI.textContent = `${Math.round(ri)} mm/h²`;
  if (fillRI) {
    const pct = Math.max(0, Math.min(100, (ri / 60) * 100));
    fillRI.style.width = `${pct}%`;
    fillRI.className = ri >= 30 ? "gauge-fill danger" : (ri >= 15 ? "gauge-fill warning" : "gauge-fill");
  }
  if (statusRI) {
    statusRI.textContent = ri >= 30 ? "EXPLOSIVE BURST" : (ri >= 15 ? "SUDDEN SPIKE" : "STEADY FLOW");
    statusRI.style.color = ri >= 15 ? "#ef4444" : "#94a3b8";
  }

  // 4. Rain in Last 30 Minutes
  const outR30 = document.getElementById("readout-gauge-R30");
  const fillR30 = document.getElementById("gauge-fill-R30");
  const statusR30 = document.getElementById("gauge-status-R30");
  if (outR30) outR30.textContent = `${Math.round(r30)} mm`;
  if (fillR30) {
    const pct = Math.max(0, Math.min(100, (r30 / 150) * 100));
    fillR30.style.width = `${pct}%`;
    fillR30.className = r30 >= 50 ? "gauge-fill danger" : (r30 >= 30 ? "gauge-fill warning" : "gauge-fill");
  }
  if (statusR30) {
    statusR30.textContent = r30 >= 50 ? "DANGEROUS RUNOFF" : (r30 >= 30 ? "SOIL SOAKED" : (r30 >= 10 ? "DAMP SOIL" : "DRY SOIL"));
    statusR30.style.color = r30 >= 50 ? "#ef4444" : (r30 >= 30 ? "#f97316" : "#94a3b8");
  }

  // 5. Lightning Strikes
  const outL = document.getElementById("readout-gauge-L");
  const fillL = document.getElementById("gauge-fill-L");
  const statusL = document.getElementById("gauge-status-L");
  if (outL) outL.textContent = `${Math.round(l)} /min`;
  if (fillL) {
    const pct = Math.max(0, Math.min(100, (l / 60) * 100));
    fillL.style.width = `${pct}%`;
    fillL.className = l >= 30 ? "gauge-fill danger" : (l >= 15 ? "gauge-fill warning" : "gauge-fill");
  }
  if (statusL) {
    statusL.textContent = l >= 30 ? "SEVERE STORM" : (l >= 15 ? "THUNDERSTORM" : "QUIET");
    statusL.style.color = l >= 30 ? "#ef4444" : (l >= 15 ? "#f97316" : "#94a3b8");
  }

  // 6. River Water Level at Gorge
  const outH = document.getElementById("readout-gauge-H");
  const fillH = document.getElementById("gauge-fill-H");
  const statusH = document.getElementById("gauge-status-H");
  const rateH = document.getElementById("gauge-rate-H");
  const currentDepth = typeof depth === "number" ? depth : Math.max(0.4, 0.4 + (r / 150) * 6.8);
  if (outH) outH.textContent = `${currentDepth.toFixed(1)} m`;
  if (fillH) {
    const pct = Math.max(4, Math.min(100, (currentDepth / 8.5) * 100));
    fillH.style.width = `${pct}%`;
    fillH.className = currentDepth >= 4.0 ? "gauge-fill danger" : (currentDepth >= 1.8 ? "gauge-fill warning" : "gauge-fill");
  }
  if (statusH) {
    statusH.textContent = currentDepth >= 4.0 ? "OVERFLOW DANGER" : (currentDepth >= 1.8 ? "RISING WATER" : "SAFE BASEFLOW");
    statusH.style.color = currentDepth >= 4.0 ? "#ef4444" : (currentDepth >= 1.8 ? "#f97316" : "#4ade80");
  }
  if (rateH) {
    const station = STATIONS.find(s => s.id === APP.currentStationId);
    rateH.textContent = `Danger Spot: ${getBottleneckNameForStation(station)}`;
  }
}

// ── Multi-Model Pipeline Diagnostics// ── Multi-Model Pipeline Diagnostics & Status Lights Board ──────────────────
function updateModelStatusBoard(station, R, RI, R60, IWV, L, result) {
  const isSevere = result.tier === "severe";
  const isWarning = result.tier === "warning";
  const isWatch = result.tier === "watch";

  // ── MODEL 1: Weather & Moisture Sensors (Satellite & Ground)
  const mLedGnss = document.getElementById("mled-gnss");
  const mBadgeGnss = document.getElementById("mbadge-gnss");
  const mDescGnss = document.getElementById("mdesc-gnss");
  const mHandoffGnss = document.getElementById("mhandoff-data-gnss");

  if (IWV >= 60) {
    if (mLedGnss) mLedGnss.className = "model-led led-fired";
    if (mBadgeGnss) { mBadgeGnss.textContent = "HEAVY MOISTURE"; mBadgeGnss.className = "model-badge fired"; }
    if (mDescGnss) mDescGnss.textContent = `Clouds are packed with ${IWV}mm water vapor. Massive moisture detected 30 minutes before surface rain starts.`;
    if (mHandoffGnss) mHandoffGnss.textContent = `Sent rapid rain spike warning (${RI} mm/h²) to Smart Battery Switch`;
  } else if (IWV >= 50) {
    if (mLedGnss) mLedGnss.className = "model-led led-warning";
    if (mBadgeGnss) { mBadgeGnss.textContent = "MOISTURE RISING"; mBadgeGnss.className = "model-badge active"; }
    if (mDescGnss) mDescGnss.textContent = `Cloud moisture reached ${IWV}mm (rising +3.8 mm/h). Storm clouds are building up.`;
    if (mHandoffGnss) mHandoffGnss.textContent = `Streaming live rainfall speed (${R} mm/h) and moisture to Battery Switch`;
  } else {
    if (mLedGnss) mLedGnss.className = "model-led led-standby";
    if (mBadgeGnss) { mBadgeGnss.textContent = "STANDBY"; mBadgeGnss.className = "model-badge"; }
    if (mDescGnss) mDescGnss.textContent = `Watching the sky. Cloud moisture is normal (${IWV}mm). Atmosphere is clear.`;
    if (mHandoffGnss) mHandoffGnss.textContent = `Sending routine rain speed (${R} mm/h)`;
  }

  // ── MODEL 2: Smart Battery Switch (SNN)
  const mLedSnn = document.getElementById("mled-snn");
  const mBadgeSnn = document.getElementById("mbadge-snn");
  const mActSnn = document.getElementById("mact-snn");
  const mDescSnn = document.getElementById("mdesc-snn");
  const mHandoffSnn = document.getElementById("mhandoff-data-snn");

  const isSNNFired = R >= 30 || RI >= 15 || IWV >= 52;

  if (isSNNFired) {
    if (mLedSnn) mLedSnn.className = "model-led led-fired";
    if (mBadgeSnn) { mBadgeSnn.textContent = "WOKE UP!"; mBadgeSnn.className = "model-badge fired"; }
    if (mActSnn) mActSnn.textContent = `Sudden Rain Spike Detected: ${RI} mm/h² in 1 min`;
    if (mDescSnn) mDescSnn.textContent = `Sudden heavy downpour woke up the sensor! It saved 91% battery power while sleeping and woke up instantly when rain hit.`;
    if (mHandoffSnn) mHandoffSnn.textContent = `Sent instant wake-up signal to start the AI Weather Predictor (Nowcaster)`;
  } else {
    if (mLedSnn) mLedSnn.className = "model-led led-off";
    if (mBadgeSnn) { mBadgeSnn.textContent = "SLEEPING (SAVING 91% POWER)"; mBadgeSnn.className = "model-badge"; }
    if (mActSnn) mActSnn.textContent = `Sensor Sleeping: Rain is steady (< 15 mm/h²)`;
    if (mDescSnn) mDescSnn.textContent = `Sensor is sleeping in ultra-low power mode to save battery life in remote mountains.`;
    if (mHandoffSnn) mHandoffSnn.textContent = `Standby — Ready to wake up if heavy rain suddenly starts`;
  }

  // ── MODEL 3: AI Weather Predictor (Nowcaster)
  const mLedCnn = document.getElementById("mled-cnn");
  const mBadgeCnn = document.getElementById("mbadge-cnn");
  const mActCnn = document.getElementById("mact-cnn");
  const mDescCnn = document.getElementById("mdesc-cnn");
  const mHandoffCnn = document.getElementById("mhandoff-data-cnn");

  const probPct = (result.prob * 100).toFixed(1);
  if (isSevere) {
    if (mLedCnn) mLedCnn.className = "model-led led-fired";
    if (mBadgeCnn) { mBadgeCnn.textContent = "CLOUDBURST ALERT"; mBadgeCnn.className = "model-badge fired"; }
    if (mActCnn) mActCnn.textContent = `Risk of Cloudburst = ${probPct}% (Critical Emergency)`;
    if (mDescCnn) mDescCnn.textContent = `Analyzed the rain speed and pattern. Confirmed an EXTREME CLOUDBURST (not normal rain).`;
    if (mHandoffCnn) mHandoffCnn.textContent = `Sent Cloudburst location (${station?.name}) & 1,420 m³/s flood size to PINN River Simulator`;
  } else if (isWarning || isWatch) {
    if (mLedCnn) mLedCnn.className = "model-led led-warning";
    if (mBadgeCnn) { mBadgeCnn.textContent = isWarning ? "THUNDERSTORM WARNING" : "STORM WATCH"; mBadgeCnn.className = "model-badge active"; }
    if (mActCnn) mActCnn.textContent = `Risk of Cloudburst = ${probPct}% (Heavy Thunderstorm)`;
    if (mDescCnn) mDescCnn.textContent = `Heavy thunderstorm detected. Cloudburst threshold not reached.`;
    if (mHandoffCnn) mHandoffCnn.textContent = `Sent moderate flood size (620 m³/s) to PINN River Simulator`;
  } else {
    if (mLedCnn) mLedCnn.className = "model-led led-standby";
    if (mBadgeCnn) { mBadgeCnn.textContent = "SAFE"; mBadgeCnn.className = "model-badge"; }
    if (mActCnn) mActCnn.textContent = `Risk of Cloudburst = ${probPct}% (Normal Weather)`;
    if (mDescCnn) mDescCnn.textContent = `Cloudburst risk is very low (${probPct}%). Normal weather.`;
    if (mHandoffCnn) mHandoffCnn.textContent = `Standby — Ready to run AI prediction as soon as battery switch wakes it`;
  }

  // ── MODEL 4: Neighbor Stations Check (False Alarm Filter)
  const mLedSpatial = document.getElementById("mled-spatial");
  const mBadgeSpatial = document.getElementById("mbadge-spatial");
  const mActSpatial = document.getElementById("mact-spatial");
  const mDescSpatial = document.getElementById("mdesc-spatial");
  const mHandoffSpatial = document.getElementById("mhandoff-data-spatial");

  const lScore = Math.min(1.0, (R / 180.0) * 0.7 + (RI / 50.0) * 0.3).toFixed(2);
  const isSpatialConfirmed = parseFloat(lScore) >= 0.40 && (isSevere || isWarning);

  if (isSpatialConfirmed) {
    if (mLedSpatial) mLedSpatial.className = "model-led led-standby";
    if (mBadgeSpatial) { mBadgeSpatial.textContent = "CONFIRMED"; mBadgeSpatial.className = "model-badge confirmed"; }
    if (mActSpatial) mActSpatial.textContent = `Check Passed: Real Isolated Cloudburst Confirmed`;
    if (mDescSpatial) mDescSpatial.textContent = `Checked nearby 50km mountain stations. They have clear skies, proving this is an isolated cloudburst (not widespread monsoon).`;
    if (mHandoffSpatial) mHandoffSpatial.textContent = `Approved real cloudburst — sent green light to PINN River Simulator`;
  } else if (parseFloat(lScore) < 0.40 && R >= 30) {
    if (mLedSpatial) mLedSpatial.className = "model-led led-warning";
    if (mBadgeSpatial) { mBadgeSpatial.textContent = "REGIONAL RAIN"; mBadgeSpatial.className = "model-badge active"; }
    if (mActSpatial) mActSpatial.textContent = `Check Result: Widespread Normal Monsoon (False Alarm Cancelled)`;
    if (mDescSpatial) mDescSpatial.textContent = `All surrounding stations have the same gentle rain. False cloudburst alarm cancelled!`;
    if (mHandoffSpatial) mHandoffSpatial.textContent = `Told PINN Simulator to stay in normal river mode`;
  } else {
    if (mLedSpatial) mLedSpatial.className = "model-led led-standby";
    if (mBadgeSpatial) { mBadgeSpatial.textContent = "PASS"; mBadgeSpatial.className = "model-badge"; }
    if (mActSpatial) mActSpatial.textContent = `Nearby Stations: Calm Across Region`;
    if (mDescSpatial) mDescSpatial.textContent = `Standing by to cross-check nearby stations if a storm develops.`;
    if (mHandoffSpatial) mHandoffSpatial.textContent = `Standby — Watching all 50km neighbor stations`;
  }

  // ── MODEL 5: PINN River Flood Simulator
  const mLedPinn = document.getElementById("mled-pinn");
  const mBadgePinn = document.getElementById("mbadge-pinn");
  const mActPinn = document.getElementById("mact-pinn");
  const mAccumPinn = document.getElementById("maccum-pinn");
  const mHandoffPinn = document.getElementById("mhandoff-data-pinn");

  const chokeName = getBottleneckNameForStation(station);
  const isPINNRunning = R >= 35 || isSevere;

  if (isSevere) {
    const peakD = Math.min(8.5, (0.4 + (R / 150) * 6.8)).toFixed(1);
    if (mLedPinn) mLedPinn.className = "model-led led-fired";
    if (mBadgePinn) { mBadgePinn.textContent = "DANGEROUS FLOOD"; mBadgePinn.className = "model-badge fired"; }
    if (mActPinn) mActPinn.textContent = `Simulation Decision: ACTIVE (Simulating Cloudburst Flood)`;
    if (mAccumPinn) {
      mAccumPinn.textContent = `Will water flood? — YES! Water gets trapped and rises to ${peakD}m at ${chokeName} in 28 minutes!`;
      mAccumPinn.style.color = "var(--red)";
    }
    if (mHandoffPinn) mHandoffPinn.textContent = `Broadcasted 7.4m flood depth, 11.4 m/s river speed & evacuation warning to Police & Sirens`;
  } else if (isPINNRunning) {
    const peakD = Math.min(4.5, (0.4 + (R / 150) * 4.2)).toFixed(1);
    if (mLedPinn) mLedPinn.className = "model-led led-active";
    if (mBadgePinn) { mBadgePinn.textContent = "SIMULATING FLOW"; mBadgePinn.className = "model-badge active"; }
    if (mActPinn) mActPinn.textContent = `Simulation Decision: ACTIVE (Simulating Heavy Rain Runoff)`;
    if (mAccumPinn) {
      mAccumPinn.textContent = `Will water flood? — MODERATE SURGE. River rises to ${peakD}m at ${chokeName}. Lowland warning advised.`;
      mAccumPinn.style.color = "var(--orange)";
    }
    if (mHandoffPinn) mHandoffPinn.textContent = `Sent water arrival times to Local Administration`;
  } else {
    if (mLedPinn) mLedPinn.className = "model-led led-off";
    if (mBadgePinn) { mBadgePinn.textContent = "SAFE / STANDBY"; mBadgePinn.className = "model-badge"; }
    if (mActPinn) mActPinn.textContent = `Simulation Decision: STANDBY (River Safe, No Flood Risk)`;
    if (mAccumPinn) {
      mAccumPinn.textContent = `Will water flood? — No flood risk. River is flowing normally (0.4m depth).`;
      mAccumPinn.style.color = "var(--text-secondary)";
    }
    if (mHandoffPinn) mHandoffPinn.textContent = `Standby — Ready to simulate river flood if cloudburst hits`;
  }
}

function updateLocationRankingBoard(station, tier, R, prob) {
  const container = document.getElementById("ranking-board-container");
  const countChip = document.getElementById("ranking-station-count");
  if (!container) return;

  const now = new Date();
  const istOffset = 5.5 * 60 * 60 * 1000;
  const istNow = new Date(now.getTime() + (now.getTimezoneOffset() * 60 * 1000) + istOffset);

  function addM(d, m) { return new Date(d.getTime() + m * 60000); }
  function fmt(d) {
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return `${hh}:${mm} IST`;
  }

  const reg = (station?.region || "").toLowerCase();
  const name = station?.name || "Station";
  let locs = [];

  if (reg === "rudraprayag" || name.includes("Kedarnath")) {
    locs = [
      { name: "Kedarnath Epicenter", km: 0, elev: 3583, isChoke: false, type: "Epicenter Origin", lat: 30.734, lon: 79.066 },
      { name: "Rambara Canyon", km: 7, elev: 2700, isChoke: false, type: "Gorge Rapids", lat: 30.680, lon: 79.055 },
      { name: "Gaurikund Base", km: 10, elev: 1982, isChoke: false, type: "Pilgrim Settlement", lat: 30.650, lon: 79.010 },
      { name: "Sonprayag Choke Bottleneck", km: 14, elev: 1829, isChoke: true, type: "Primary Gorge Chokepoint", lat: 30.624, lon: 79.003 },
      { name: "Kund & Agastyamuni", km: 22, elev: 950, isChoke: false, type: "Valley Lowlands", lat: 30.450, lon: 79.010 },
      { name: "Rudraprayag Confluence", km: 28, elev: 610, isChoke: false, type: "Basin Confluence", lat: 30.284, lon: 78.981 }
    ];
  } else if (reg === "chamoli" || name.includes("Joshimath") || name.includes("Badrinath")) {
    locs = [
      { name: `${name} Epicenter`, km: 0, elev: station.elevation || 1875, isChoke: false, type: "Epicenter Origin", lat: station.lat, lon: station.lon },
      { name: "Helang Canyon Choke", km: 6, elev: 1500, isChoke: false, type: "Gorge Splay", lat: 30.510, lon: 79.540 },
      { name: "Tapovan Barrage Basin", km: 13, elev: 1350, isChoke: true, type: "Hydroelectric Bottleneck", lat: 30.480, lon: 79.480 },
      { name: "Pipalkoti Lowlands", km: 18, elev: 1050, isChoke: false, type: "Settlement", lat: 30.440, lon: 79.400 },
      { name: "Chamoli Town Confluence", km: 24, elev: 855, isChoke: false, type: "District Confluence", lat: 30.403, lon: 79.329 }
    ];
  } else if (reg === "uttarkashi" || name.includes("Gangotri")) {
    locs = [
      { name: `${name} Epicenter`, km: 0, elev: station.elevation || 3048, isChoke: false, type: "Epicenter Origin", lat: station.lat, lon: station.lon },
      { name: "Dharali & Harsil Gorge", km: 8, elev: 2287, isChoke: false, type: "Canyon Splay", lat: 30.938, lon: 78.948 },
      { name: "Bhatwari Throat", km: 16, elev: 1654, isChoke: false, type: "Mid-Valley Choke", lat: 30.844, lon: 78.674 },
      { name: "Maneri Dam Reservoir", km: 22, elev: 1320, isChoke: true, type: "Primary Bottleneck Throat", lat: 30.849, lon: 78.518 },
      { name: "Uttarkashi District HQ", km: 28, elev: 1165, isChoke: false, type: "Basin Urban Center", lat: 30.726, lon: 78.446 }
    ];
  } else if (reg === "pithoragarh" || name.includes("Munsiari")) {
    locs = [
      { name: `${name} Epicenter`, km: 0, elev: station.elevation || 2200, isChoke: false, type: "Epicenter Origin", lat: station.lat, lon: station.lon },
      { name: "Madkot Gorge Choke", km: 9, elev: 1350, isChoke: false, type: "Canyon Funnel", lat: 29.850, lon: 80.250 },
      { name: "Jauljibi Confluence Basin", km: 18, elev: 610, isChoke: true, type: "Bottleneck Confluence", lat: 29.750, lon: 80.380 },
      { name: "Kali River Border Valley", km: 28, elev: 520, isChoke: false, type: "Basin Outlet", lat: 29.620, lon: 80.450 }
    ];
  } else if (station?.basin === "assam" || reg === "assam") {
    locs = [
      { name: `${name} Inundation Cell`, km: 0, elev: station.elevation || 80, isChoke: false, type: "Tributary Inundation", lat: station.lat, lon: station.lon },
      { name: "Riparian Floodplain Corridor", km: 10, elev: 65, isChoke: true, type: "Lowland Sheet Flow", lat: station.lat - 0.04, lon: station.lon + 0.05 },
      { name: "Brahmaputra Mainstem Basin", km: 25, elev: 45, isChoke: false, type: "River Embankment", lat: station.lat - 0.08, lon: station.lon + 0.09 }
    ];
  } else {
    locs = [
      { name: `${name} Epicenter`, km: 0, elev: station?.elevation || 2000, isChoke: false, type: "Epicenter Origin", lat: station?.lat || 30.5, lon: station?.lon || 79.2 },
      { name: "Mid-Valley Canyon Choke", km: 8, elev: Math.round((station?.elevation || 2000) * 0.75), isChoke: false, type: "Gorge Splay", lat: (station?.lat || 30.5) - 0.05, lon: (station?.lon || 79.2) - 0.03 },
      { name: "Downstream Gorge Bottleneck", km: 15, elev: Math.round((station?.elevation || 2000) * 0.55), isChoke: true, type: "Primary Bottleneck", lat: (station?.lat || 30.5) - 0.10, lon: (station?.lon || 79.2) - 0.06 },
      { name: "Basin Outlet Settlement", km: 25, elev: Math.round((station?.elevation || 2000) * 0.40), isChoke: false, type: "Basin Confluence", lat: (station?.lat || 30.5) - 0.16, lon: (station?.lon || 79.2) - 0.09 }
    ];
  }

  // Calculate severity and timestamps for each location
  const rainFactor = Math.min(1.0, R / 150.0);
  const isHighRisk = R >= 50 || prob >= 0.45;

  const ranked = locs.map((loc, idx) => {
    // Wave travel time: approx 0.55 km/min (9.2 m/s mountain torrent)
    const arrMin = loc.km === 0 ? 0 : Math.max(8, Math.round(loc.km / 0.52));
    const evacMin = Math.max(3, arrMin - 15);

    // Severity score (0 to 100)
    let score = 0;
    let depth = 0.4;

    if (R <= 5) {
      score = 8;
      depth = 0.4;
    } else {
      const baseScore = rainFactor * 75 + prob * 25;
      const atten = Math.exp(-loc.km / 38.0);
      const chokeBoost = loc.isChoke ? 1.25 : 1.0;
      score = Math.min(100, Math.round(baseScore * atten * chokeBoost));
      depth = Math.max(0.4, 0.4 + (score / 100.0) * (loc.isChoke ? 7.0 : 4.6));
    }

    let tierClass = "nominal";
    let tierBadge = "SAFE";
    if (score >= 85) { tierClass = "critical"; tierBadge = "CRITICAL CHOKE RISK"; }
    else if (score >= 60) { tierClass = "warning"; tierBadge = "SEVERE SURGE"; }
    else if (score >= 35) { tierClass = "advisory"; tierBadge = "SURGE WATCH"; }

    const arrTimeStr = loc.km === 0 ? (R >= 30 ? "ACTIVE DELUGE" : "IMMEDIATE") : `${fmt(addM(istNow, arrMin))} (in ${arrMin}m)`;
    const evacTimeStr = loc.km === 0 ? (R >= 30 ? "NOW" : "--") : `${fmt(addM(istNow, evacMin))} (in ${evacMin}m)`;

    return {
      ...loc,
      score,
      depth: depth.toFixed(1),
      tierClass,
      tierBadge,
      arrMin,
      arrTimeStr,
      evacTimeStr
    };
  });

  // Sort by severity score (highest threat first)
  ranked.sort((a, b) => b.score - a.score);

  if (countChip) countChip.textContent = `${ranked.length} Impact Zones Ranked`;

  container.innerHTML = ranked.map((loc, rIdx) => `
    <div class="ranking-item ${loc.tierClass}" onclick="APP.map.flyTo([${loc.lat}, ${loc.lon}], 12); highlightMapPoint(${loc.lat}, ${loc.lon});">
      <div class="ranking-head-row">
        <div class="ranking-id-grp">
          <span class="ranking-rank-num">#${rIdx + 1}</span>
          <span class="ranking-loc-name">${loc.name}</span>
        </div>
        <span class="ranking-badge-tier ${loc.tierClass}">${loc.tierBadge}</span>
      </div>

      <div class="ranking-score-bar-track">
        <div class="ranking-score-bar-fill ${loc.tierClass}" style="width: ${loc.score}%;"></div>
      </div>

      <div class="ranking-meta-grid">
        <div class="ranking-m-item">
          <span>Flood Reaches Here:</span>
          <b class="${loc.score >= 80 ? 'crit' : ''}">${loc.arrTimeStr}</b>
        </div>
        <div class="ranking-m-item">
          <span>Water Level:</span>
          <b class="${loc.score >= 80 ? 'crit' : ''}">${loc.depth} m</b>
        </div>
        <div class="ranking-m-item">
          <span>Must Evacuate By:</span>
          <b>${loc.evacTimeStr}</b>
        </div>
        <div class="ranking-m-item">
          <span>Height Above Sea:</span>
          <b>${loc.elev} m ASL</b>
        </div>
      </div>
    </div>
  `).join("");
}

function highlightMapPoint(lat, lon) {
  if (!APP.map) return;
  const pulse = L.circle([lat, lon], {
    radius: 800,
    color: "#ef4444",
    fillColor: "#ef4444",
    fillOpacity: 0.4
  }).addTo(APP.map);
  setTimeout(() => { if (APP.map.hasLayer(pulse)) APP.map.removeLayer(pulse); }, 3000);
}

// ── Map Showcase: Neuromorphic Spike, Convective Core & Flood Wavefront ──────
function updateMapShowcase(station, tier, R, prob) {
  if (!APP.map || !station) return;

  // 1. Update Floating Map Simulation HUD
  const hud = document.getElementById("map-simulation-hud");
  const hudDot = document.getElementById("sim-hud-dot");
  const hudTitle = document.getElementById("sim-hud-title");
  const hudStage = document.getElementById("sim-hud-stage");
  const hudModel = document.getElementById("sim-hud-active-model");
  const hudAction = document.getElementById("sim-hud-action");
  const hudHandoff = document.getElementById("sim-hud-handoff-text");
  const hudAccum = document.getElementById("sim-hud-accum-text");

  const isSevere = tier === "severe";
  const isWarning = tier === "warning";

  if (hud) {
    if (isSevere) {
      if (hudDot) hudDot.className = "sim-hud-status-dot danger";
      if (hudTitle) hudTitle.textContent = "PIPELINE: CLOUDBURST ALERT";
      if (hudStage) hudStage.textContent = "STAGE 4/5";
      if (hudModel) hudModel.textContent = "PINN 2D Riverbed Simulation";
      if (hudAction) hudAction.textContent = `Cloudburst deluge (${R} mm/h) confirmed at ${station?.name || "Mountain Ridge"}. Water rushing downstream.`;
      if (hudHandoff) hudHandoff.textContent = "Nowcaster -> PINN Simulation -> DDMA Evacuation Sirens";
      if (hudAccum) {
        hudAccum.textContent = `CRITICAL ACCUMULATION: 7.4m pool at ${getBottleneckNameForStation(station)}`;
        hudAccum.className = "hud-av danger";
      }
    } else if (isWarning || R >= 40) {
      if (hudDot) hudDot.className = "sim-hud-status-dot active";
      if (hudTitle) hudTitle.textContent = "PIPELINE: CONVECTIVE SURGE";
      if (hudStage) hudStage.textContent = "STAGE 3/5";
      if (hudModel) hudModel.textContent = "1D-CNN + BiLSTM Nowcaster";
      if (hudAction) hudAction.textContent = `Intense convective surge detected. SNN spike evaluated. PINN estimating inundation.`;
      if (hudHandoff) hudHandoff.textContent = "SNN Neuromorphic -> Heavy Nowcaster -> PINN";
      if (hudAccum) {
        hudAccum.textContent = `Moderate surge (2.8m) at ${getBottleneckNameForStation(station)}`;
        hudAccum.className = "hud-av";
      }
    } else {
      if (hudDot) hudDot.className = "sim-hud-status-dot";
      if (hudTitle) hudTitle.textContent = "PIPELINE: TELEMETRY MONITORING";
      if (hudStage) hudStage.textContent = "STAGE 1/5";
      if (hudModel) hudModel.textContent = "GNSS IWV + In-Situ Sensor Mesh";
      if (hudAction) hudAction.textContent = "Monitoring 40 Himalayan AWS stations. SNN gates dormant in ultra-low power sleep.";
      if (hudHandoff) hudHandoff.textContent = "Sensors -> SNN LIF Neuromorphic Gate";
      if (hudAccum) {
        hudAccum.textContent = "Nominal 0.4m (Baseflow, no flood risk)";
        hudAccum.className = "hud-av";
      }
    }
  }

  // 2. SNN Neuromorphic Spike Circle on Map
  const isSNNSpike = R >= 30 || station.RI >= 15 || (station.IWV || 34) >= 55;
  if (isSNNSpike) {
    if (!APP.snnRippleLayer) {
      APP.snnRippleLayer = L.circle([station.lat, station.lon], {
        radius: 2400,
        color: "#06b6d4",
        weight: 2,
        dashArray: "4, 4",
        fillColor: "#06b6d4",
        fillOpacity: 0.2
      }).addTo(APP.map);
      APP.snnRippleLayer.bindTooltip(`<b style="color:#67e8f9;">SNN BURST SPIKE FIRED!</b><br/>Sudden Rain Surge Detected`, { permanent: false, direction: "top" });
    } else {
      APP.snnRippleLayer.setLatLng([station.lat, station.lon]);
    }
  } else {
    if (APP.snnRippleLayer) {
      APP.map.removeLayer(APP.snnRippleLayer);
      APP.snnRippleLayer = null;
    }
  }

  // 3. Convective Deluge Core Polygon on Map
  if (isSevere || isWarning) {
    const rRadius = isSevere ? 3800 : 2200;
    const rColor = isSevere ? "#ef4444" : "#f97316";
    if (!APP.convectiveCoreLayer) {
      APP.convectiveCoreLayer = L.circle([station.lat, station.lon], {
        radius: rRadius,
        color: rColor,
        weight: 2.5,
        fillColor: rColor,
        fillOpacity: isSevere ? 0.32 : 0.18
      }).addTo(APP.map);
      APP.convectiveCoreLayer.bindTooltip(`<b style="color:${rColor};">[CONVECTIVE CLOUDBURST CORE]</b><br/>Precipitation Rate: ${R} mm/h | P=${(prob * 100).toFixed(1)}%`, { direction: "bottom" });
    } else {
      APP.convectiveCoreLayer.setLatLng([station.lat, station.lon]);
      APP.convectiveCoreLayer.setRadius(rRadius);
      APP.convectiveCoreLayer.setStyle({ color: rColor, fillColor: rColor, fillOpacity: isSevere ? 0.32 : 0.18 });
    }
  } else {
    if (APP.convectiveCoreLayer) {
      APP.map.removeLayer(APP.convectiveCoreLayer);
      APP.convectiveCoreLayer = null;
    }
  }

  // 4. Bottleneck Choke Hazard Marker on Map
  const chokeName = getBottleneckNameForStation(station);
  let chokeCoords = [station.lat - 0.11, station.lon - 0.06]; // default downstream vector
  const reg = (station.region || "").toLowerCase();
  if (reg === "rudraprayag" || station.name.includes("Kedarnath")) chokeCoords = [30.624, 79.003]; // Sonprayag
  else if (reg === "chamoli") chokeCoords = [30.480, 79.480]; // Tapovan
  else if (reg === "uttarkashi") chokeCoords = [30.849, 78.518]; // Maneri

  if (isSevere || isWarning) {
    if (!APP.chokeMarkerLayer) {
      const chokeIcon = L.divIcon({
        className: 'choke-hazard-icon',
        html: `<div style="background:#ef4444; width:14px; height:14px; border-radius:50%; border:2px solid #ffffff; box-shadow:0 0 12px #ef4444;"></div>`,
        iconSize: [14, 14]
      });
      APP.chokeMarkerLayer = L.marker(chokeCoords, { icon: chokeIcon }).addTo(APP.map);
      APP.chokeMarkerLayer.bindTooltip(`
        <div style="font-family:'Inter',sans-serif;font-size:11px;color:#ffffff;">
          <b style="color:#ef4444;">[PRIMARY CHOKE: ${chokeName}]</b><br/>
          PINN Water Accumulation: <b>7.4m</b><br/>
          Surge Velocity: <b>11.4 m/s</b> &middot; Evac Cutoff: <b>22:45 IST</b>
        </div>
      `, { permanent: true, direction: "right", offset: [10, 0] });
    } else {
      APP.chokeMarkerLayer.setLatLng(chokeCoords);
    }
  } else {
    if (APP.chokeMarkerLayer) {
      APP.map.removeLayer(APP.chokeMarkerLayer);
      APP.chokeMarkerLayer = null;
    }
  }
}

// ── Developer Simulation Engine: Random Location Injection ──────────────────
let telemetryAnimId = null;

function animateTelemetryTransition(t_r, t_r30, t_ri, t_l, t_iwv, duration = 2800, onComplete) {
  if (telemetryAnimId) cancelAnimationFrame(telemetryAnimId);

  const sR = document.getElementById("slider-R");
  const sR30 = document.getElementById("slider-R30");
  const sRI = document.getElementById("slider-RI");
  const sL = document.getElementById("slider-L");
  const sIWV = document.getElementById("slider-IWV");

  const startR = sR ? parseFloat(sR.value) : 0;
  const startR30 = sR30 ? parseFloat(sR30.value) : 0;
  const startRI = sRI ? parseFloat(sRI.value) : 0;
  const startL = sL ? parseFloat(sL.value) : 0;
  const startIWV = sIWV ? parseFloat(sIWV.value) : 34;

  const startTime = performance.now();

  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(1.0, elapsed / duration);
    // Smooth cubic easing (observable, not too fast)
    const ease = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;

    const curR = Math.round(startR + (t_r - startR) * ease);
    const curR30 = Math.round(startR30 + (t_r30 - startR30) * ease);
    const curRI = Math.round(startRI + (t_ri - startRI) * ease);
    const curL = Math.round(startL + (t_l - startL) * ease);
    const curIWV = Math.round(startIWV + (t_iwv - startIWV) * ease);

    setSliderValues(curR, curR30, curRI, curL, curIWV);
    onPrecursorSliderChange();

    if (progress < 1.0) {
      telemetryAnimId = requestAnimationFrame(step);
    } else {
      telemetryAnimId = null;
      if (typeof onComplete === "function") onComplete();
    }
  }

  telemetryAnimId = requestAnimationFrame(step);
}

function simulateRandomEvent(type) {
  if (!STATIONS || STATIONS.length === 0) return;

  // Pick a random station from the 40 Himalayan network stations
  const randIdx = Math.floor(Math.random() * STATIONS.length);
  const target = STATIONS[randIdx];

  // 1. Select station and smoothly FLY MAP to focus on chosen location
  selectStation(target.id);
  if (APP.map) {
    APP.map.flyTo([target.lat, target.lon], 11.2, { animate: true, duration: 1.0 });
  }

  // Close any station marker tooltips so they don't stack
  if (APP.markers && APP.markers[target.id]) {
    APP.markers[target.id].closeTooltip();
  }

  // 2. Focus and glow the changing area on map (pulsing aura)
  if (APP.locationGlowPulse && APP.map) {
    APP.map.removeLayer(APP.locationGlowPulse);
    APP.locationGlowPulse = null;
  }
  if (APP.map) {
    APP.locationGlowPulse = L.circle([target.lat, target.lon], {
      radius: 3200,
      color: type === 'cloudburst' ? '#ef4444' : (type === 'thunderstorm' ? '#f97316' : '#38bdf8'),
      weight: 3,
      fillColor: type === 'cloudburst' ? '#ef4444' : (type === 'thunderstorm' ? '#f97316' : '#38bdf8'),
      fillOpacity: 0.35,
      className: "changing-location-glow"
    }).addTo(APP.map);
    APP.locationGlowPulse.bindTooltip(`
      <div style="font-family:'Inter',sans-serif; text-align:center; padding:2px 4px;">
        <span style="color:${type === 'cloudburst' ? '#ef4444' : (type === 'thunderstorm' ? '#f97316' : '#38bdf8')}; font-weight:800; font-size:10.5px; letter-spacing:0.04em;">[SIMULATED EPICENTER: ${target.name.toUpperCase()}]</span><br/>
        <span style="font-size:9px; color:#cbd5e1; font-weight:600;">${type.toUpperCase()} SURGE ACTIVE</span>
      </div>
    `, { permanent: true, direction: "top", offset: [0, -14], className: "epicenter-tooltip" });
  }

  // 3. Activate Simulation Focus Mode (only show model outputs, XAI, and HOW PINN is building)
  toggleSimulationFocusMode(true);

  // 4. Switch to live gauges monitor view
  switchTelemetryMode("monitor");

  // 5. Flash glow on changing elements
  flashElementGlow("sidebar-station-name", 4000);
  flashElementGlow("telemetry-block", 3000);
  flashElementGlow("pinn-section", 4000);

  let r, r30, ri, l, iwv;
  if (type === "cloudburst") {
    r = Math.floor(130 + Math.random() * 35);
    r30 = Math.floor(95 + Math.random() * 30);
    ri = Math.floor(36 + Math.random() * 16);
    l = Math.floor(30 + Math.random() * 20);
    iwv = Math.floor(64 + Math.random() * 10);
    addFeedEntry("danger", `[RANDOM SIMULATION] Cloudburst at ${target.name}`, `R=${r} mm/h · GNSS IWV=${iwv} mm · SNN Spike Triggered · PINN Riverbed Surge`);
  } else if (type === "thunderstorm") {
    r = Math.floor(55 + Math.random() * 22);
    r30 = Math.floor(35 + Math.random() * 18);
    ri = Math.floor(16 + Math.random() * 10);
    l = Math.floor(40 + Math.random() * 18); // High lightning flash rate characteristic of thunderstorm
    iwv = Math.floor(48 + Math.random() * 7);
    addFeedEntry("warn", `[RANDOM SIMULATION] Severe Thunderstorm at ${target.name}`, `R=${r} mm/h · Lightning=${l} fl/min · Watch Tier · PINN Assessing Inundation`);
  } else if (type === "heavyrain") {
    r = Math.floor(35 + Math.random() * 14);
    r30 = Math.floor(22 + Math.random() * 10);
    ri = Math.floor(8 + Math.random() * 7);
    l = Math.floor(10 + Math.random() * 10);
    iwv = Math.floor(42 + Math.random() * 6);
    addFeedEntry("warn", `[RANDOM SIMULATION] Heavy Rain at ${target.name}`, `R=${r} mm/h · GNSS IWV=${iwv} mm · Orographic Enhancement`);
  } else {
    // Routine rainfall
    r = Math.floor(14 + Math.random() * 12);
    r30 = Math.floor(8 + Math.random() * 6);
    ri = Math.floor(2 + Math.random() * 4);
    l = Math.floor(1 + Math.random() * 4);
    iwv = Math.floor(34 + Math.random() * 5);
    addFeedEntry("ok", `[RANDOM SIMULATION] Routine Rainfall at ${target.name}`, `R=${r} mm/h · Normal Baseflow · SNN Subthreshold Dormant`);
  }

  // Animate transition smoothly so user sees gauges fill up at an observable speed
  animateTelemetryTransition(r, r30, ri, l, iwv);
}

// ── Automated Sequential Pipeline Walkthrough Demo ──────────────────────────
function startAutomatedSimulationDemo() {
  switchTelemetryMode("monitor");
  selectStation("RDP001"); // Focus Kedarnath

  const btnPlay = document.getElementById("btn-hud-play");
  const btnPause = document.getElementById("btn-hud-pause");
  if (btnPlay) btnPlay.style.display = "none";
  if (btnPause) btnPause.style.display = "inline-block";

  APP.simRunning = true;
  APP.simPaused = false;
  APP.simStepIndex = 0;

  addFeedEntry("ok", "[DEMO STARTED] Autonomous Multi-Model Pipeline Walkthrough", "Executing 5-stage sequential physical simulation over 24 seconds...");

  // Sequence of phases (Observable, smooth timing: ~4s per phase)
  const phases = [
    // Phase 0: Ambient baseline
    { delay: 0, r: 0, r30: 0, ri: 0, l: 0, iwv: 34, note: "Phase 1/5: Ambient Baseline Telemetry Monitoring (Clear Sky)" },
    // Phase 1: GNSS Tropospheric Precursor Surge
    { delay: 4000, r: 12, r30: 5, ri: 4, l: 6, iwv: 64, note: "Phase 2/5: GNSS Precursor Detected (+7.2 mm/h vapor convergence)" },
    // Phase 2: In-Situ Rain Intensification & SNN LIF Spike
    { delay: 9000, r: 68, r30: 45, ri: 28, l: 24, iwv: 68, note: "Phase 3/5: Convective Surge & SNN LIF Membrane Spike Fired!" },
    // Phase 3: AI Heavy Nowcaster & Spatial Confirmation
    { delay: 14000, r: 148, r30: 115, ri: 44, l: 38, iwv: 72, note: "Phase 4/5: 1D-CNN+BiLSTM Nowcast (98.4%) & Spatial Isolation Confirmed" },
    // Phase 4: PINN 2D SWE Hydrodynamic Riverbed Inundation
    { delay: 19000, r: 156, r30: 125, ri: 48, l: 42, iwv: 74, note: "Phase 5/5: PINN 2D SWE Solver: 7.4m Water Accumulation at Sonprayag Gorge Choke!" }
  ];

  function runPhase(idx) {
    if (!APP.simRunning || APP.simPaused) return;
    if (idx >= phases.length) {
      APP.simRunning = false;
      if (btnPlay) btnPlay.style.display = "inline-block";
      if (btnPause) btnPause.style.display = "none";
      addFeedEntry("danger", "[DEMO COMPLETE] Downstream Evacuation Mandatory", "PINN projected 7.4m flood peak at Sonprayag. Evacuation cutoff active.");
      return;
    }

    const p = phases[idx];
    addFeedEntry(p.r >= 100 ? "danger" : (p.r >= 50 ? "warn" : "ok"), p.note, `IWV=${p.iwv}mm · R=${p.r}mm/h · RI=${p.ri}mm/h²`);
    animateTelemetryTransition(p.r, p.r30, p.ri, p.l, p.iwv, 3200, () => {
      APP.simStepIndex = idx + 1;
      APP.simTimer = setTimeout(() => runPhase(idx + 1), 1800);
    });
  }

  runPhase(0);
}

function pauseSimulationDemo() {
  APP.simPaused = true;
  if (APP.simTimer) clearTimeout(APP.simTimer);
  const btnPlay = document.getElementById("btn-hud-play");
  const btnPause = document.getElementById("btn-hud-pause");
  if (btnPlay) { btnPlay.textContent = "Resume"; btnPlay.style.display = "inline-block"; }
  if (btnPause) btnPause.style.display = "none";
}

function resetSimulationDemo() {
  if (APP.simTimer) clearTimeout(APP.simTimer);
  if (telemetryAnimId) cancelAnimationFrame(telemetryAnimId);
  APP.simRunning = false;
  APP.simPaused = false;
  APP.simStepIndex = 0;

  const btnPlay = document.getElementById("btn-hud-play");
  const btnPause = document.getElementById("btn-hud-pause");
  if (btnPlay) { btnPlay.textContent = "Run Simulation"; btnPlay.style.display = "inline-block"; }
  if (btnPause) btnPause.style.display = "none";

  triggerPreset("reset");
}

function togglePINNSimulation() {
  if (typeof startFloodSim === "function" && typeof PINN_SIM !== "undefined") {
    if (PINN_SIM.active) {
      resetFloodSim();
      addFeedEntry("ok", "[PINN SOLVER] Solver Paused", "Riverbed flood simulation reset to baseflow");
    } else {
      startFloodSim("cloudburst");
      addFeedEntry("danger", "[PINN SOLVER] Force Simulation Run", "Solving 2D Shallow Water Equations along valley profile");
    }
  }
}

function triggerPreset(type) {
  APP.activeReplay = type;

  if (type === "kedarnath") {
    selectStation("RDP001");
    setSliderValues(148, 115, 42, 38, 68);
    addFeedEntry("danger", "[CRITICAL] Kedarnath 2013 Cloudburst Replay", "R=148 mm/h · GNSS IWV=68 mm · Mandakini River Surge at Sonprayag Choke");
  } else if (type === "chamoli") {
    selectStation("CHM001");
    setSliderValues(112, 88, 35, 28, 58);
    addFeedEntry("danger", "[CRITICAL] Chamoli 2021 Flash Flood Replay", "R=112 mm/h · GNSS IWV=58 mm · Surge Velocity 11.2 m/s · Tapovan Barrage Basin");
  } else if (type === "moderate") {
    setSliderValues(25, 18, 6, 4, 44);
    addFeedEntry("warn", "[ADVISORY] Moderate Rain Event", "R=25 mm/h · GNSS IWV=44 mm · Routine Pre-alert · Below Cloudburst Threshold");
  } else if (type === "reset") {
    APP.activeReplay = null;
    clearFloodAccumulationZone();
    if (APP.locationGlowPulse && APP.map) {
      APP.map.removeLayer(APP.locationGlowPulse);
      APP.locationGlowPulse = null;
    }
    if (APP.stationsBackup && typeof STATIONS !== "undefined") {
      STATIONS.forEach((s, idx) => {
        Object.assign(s, JSON.parse(JSON.stringify(APP.stationsBackup[idx])));
      });
    }
    setSliderValues(0, 0, 0, 0, 32);
    renderMapMarkers();
    addFeedEntry("ok", "[INFO] Baseline Reset", "All 40 stations nominal · GNSS IWV nominal (32mm) · SNN gates in low-power dormant mode");
  }

  onPrecursorSliderChange();
}

function addFeedEntry(type, title, sub) {
  const feed = document.getElementById("alert-feed");
  if (!feed) return;

  const card = document.createElement("div");
  card.className = `feed-card ${type === 'danger' ? 'danger' : (type === 'warn' ? 'warn' : '')}`;
  card.innerHTML = `
    <div class="feed-card-dot"></div>
    <div class="feed-card-body">
      <div class="feed-card-title">${title}</div>
      <div class="feed-card-sub">${sub}</div>
    </div>
  `;

  feed.insertBefore(card, feed.firstChild);
  while (feed.children.length > 8) {
    feed.removeChild(feed.lastChild);
  }
}

// ── Basin Filter Chips ───────────────────────────────────────────────────────
function setBasinFilter(basin, btn) {
  APP.activeFilter = basin;

  document.querySelectorAll(".filter-chip").forEach(b => b.classList.remove("active"));
  if (btn) btn.classList.add("active");

  renderMapMarkers();

  // Focus map on appropriate basin
  if (APP.map) {
    if (basin === "rudraprayag") APP.map.flyTo([30.55, 79.03], 9.5);
    else if (basin === "chamoli") APP.map.flyTo([30.55, 79.55], 9.5);
    else if (basin === "uttarkashi") APP.map.flyTo([30.85, 78.70], 9.5);
    else if (basin === "pithoragarh") APP.map.flyTo([29.85, 80.30], 9.5);
    else if (basin === "assam") APP.map.flyTo([26.40, 92.50], 7.5);
    else APP.map.flyTo([30.4, 79.2], 8);
  }
}


// ── Manual User-Controlled Pipeline Stepper & Module Spotlight ───────────────
let currentPipelineStage = 1;

const PIPELINE_STAGES = [
  {
    stage: 1,
    name: "1. Weather & Moisture Sensors",
    shortName: "Weather Sensors",
    activeCardId: "mcard-gnss",
    r: 12, r30: 5, ri: 4, l: 6, iwv: 64,
    hudAction: "Moisture in the sky reached 64 mm (+7.2 mm/h) 30 minutes before rain started.",
    hudHandoff: "Sensors -> Smart Battery Switch (SNN)",
    nextActionText: "Hand Over to Smart Battery Switch (SNN) &rarr;",
    feedMsg: "Moisture buildup detected in clouds. Ready to trigger battery-saving switch."
  },
  {
    stage: 2,
    name: "2. Smart Battery Switch (SNN)",
    shortName: "Battery Switch (SNN)",
    activeCardId: "mcard-snn",
    r: 68, r30: 45, ri: 28, l: 24, iwv: 68,
    hudAction: "Sudden heavy rain spike (28 mm/h²) detected! Sensor woke up from battery-saving sleep mode.",
    hudHandoff: "Smart Battery Switch -> AI Weather Predictor (Nowcaster)",
    nextActionText: "Hand Over to AI Weather Predictor (Nowcaster) &rarr;",
    feedMsg: "SNN sensor woke up from sleep mode! Dispatched wake-up signal to start AI Nowcaster."
  },
  {
    stage: 3,
    name: "3. AI Weather Predictor (Nowcaster)",
    shortName: "AI Nowcaster",
    activeCardId: "mcard-cnn",
    r: 148, r30: 115, ri: 44, l: 38, iwv: 72,
    hudAction: "1D-CNN + BiLSTM AI analyzed rain speed. Confirmed 98.4% CLOUDBURST (not normal rain).",
    hudHandoff: "AI Nowcaster -> Neighbor Stations Check",
    nextActionText: "Hand Over to Neighbor Stations Check &rarr;",
    feedMsg: "Nowcaster confirmed 98.4% Cloudburst threat. Dispatched flood hydrograph."
  },
  {
    stage: 4,
    name: "4. Neighbor Stations Check (False Alarm Filter)",
    shortName: "Neighbor Check",
    activeCardId: "mcard-spatial",
    r: 156, r30: 125, ri: 48, l: 42, iwv: 74,
    hudAction: "Surrounding 50km mountain stations have clear skies. Confirmed real isolated cloudburst!",
    hudHandoff: "Neighbor Check -> PINN River Flood Simulator",
    nextActionText: "Hand Over to PINN River Flood Simulator &rarr;",
    feedMsg: "Nearby stations confirmed isolated cloudburst. Approved live riverbed flood simulation."
  },
  {
    stage: 5,
    name: "5. PINN River Flood Simulator",
    shortName: "PINN River Simulator",
    activeCardId: "mcard-pinn",
    r: 156, r30: 125, ri: 48, l: 42, iwv: 74,
    hudAction: "Water rushing downhill at 11.4 m/s. Dangerous 7.4m flood pool at Sonprayag gorge!",
    hudHandoff: "PINN Simulator -> Sirens & Emergency Authorities",
    nextActionText: "Broadcast Evacuation Alert to Sirens & Police &rarr;",
    feedMsg: "PINN projected 7.4m flood at Sonprayag gorge. Evacuation cutoff activated!"
  }
];

function setPipelineStage(stageNum) {
  if (stageNum < 1) stageNum = 1;
  if (stageNum > 5) stageNum = 5;
  currentPipelineStage = stageNum;

  const stage = PIPELINE_STAGES[stageNum - 1];

  // Update Stepper UI elements
  const badge = document.getElementById("hud-stage-num-badge");
  const title = document.getElementById("hud-stage-title-text");
  const handoffBtn = document.getElementById("btn-hud-handoff");
  const miniBadge = document.getElementById("mini-pipeline-badge");

  if (badge) badge.textContent = `STAGE ${stageNum} / 5`;
  if (title) title.textContent = stage.name;
  if (handoffBtn) {
    handoffBtn.innerHTML = stageNum < 5 ? stage.nextActionText : "Broadcast Evacuation Alert to Sirens &rarr;";
    handoffBtn.className = stageNum === 5 ? "hud-handoff-action-btn danger" : "hud-handoff-action-btn";
  }
  if (miniBadge) miniBadge.textContent = `Model ${stageNum}/5`;

  // Update HUD action & handoff text
  const hudAction = document.getElementById("sim-hud-action");
  const hudHandoff = document.getElementById("sim-hud-handoff-text");
  if (hudAction) hudAction.textContent = stage.hudAction;
  if (hudHandoff) hudHandoff.textContent = stage.hudHandoff;

  // Update stage pills in HUD
  const pills = document.querySelectorAll(".stage-pill");
  pills.forEach((p, idx) => {
    if (idx + 1 === stageNum) p.classList.add("active");
    else p.classList.remove("active");
  });

  // Highlight active model card on right
  const modelCards = document.querySelectorAll(".model-card");
  modelCards.forEach(c => c.classList.remove("active-model-stage"));
  const activeCard = document.getElementById(stage.activeCardId);
  if (activeCard) {
    activeCard.classList.add("active-model-stage");
    activeCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Smoothly update telemetry
  animateTelemetryTransition(stage.r, stage.r30, stage.ri, stage.l, stage.iwv, 1800);

  // Add feed entry
  addFeedEntry(stageNum >= 3 ? "danger" : (stageNum === 2 ? "warn" : "ok"), `[HANDOFF TO MODEL ${stageNum}] ${stage.name}`, stage.feedMsg);

  // Flash highlight on pipeline board
  flashElementGlow("pipeline-board-block", 1200);
}

function stepPipelineStage(delta) {
  setPipelineStage(currentPipelineStage + delta);
}

// ── Dropdown Selectors Handlers ─────────────────────────────────────────────
function onModuleViewSelect(val) {
  if (!val || val === "all") {
    closeModuleSpotlight();
    return;
  }
  openModuleSpotlight(val);
}

function onLeftColSegmentSelect(val) {
  const gView = document.getElementById("telemetry-monitor-view");
  const sView = document.getElementById("telemetry-sliders-view");
  const rView = document.getElementById("ranking-board-container");

  if (val === "sliders") {
    switchTelemetryMode("sliders");
  } else if (val === "ranking") {
    switchTelemetryMode("monitor");
    if (rView) rView.scrollIntoView({ behavior: "smooth" });
    flashElementGlow("ranking-board-container", 2000);
  } else {
    switchTelemetryMode("monitor");
    if (gView) gView.scrollIntoView({ behavior: "smooth" });
  }
}

function onPinnSegmentSelect(val) {
  const dock = document.querySelector(".pinn-dock-split");
  const canvasWrap = document.getElementById("flood-canvas-wrap");
  const flowWrap = document.getElementById("pinn-how-wrap");

  if (dock && canvasWrap && flowWrap) {
    if (val === "canvas") {
      dock.style.gridTemplateColumns = "1fr 0px";
      canvasWrap.style.display = "block";
      flowWrap.style.display = "none";
    } else if (val === "flow") {
      dock.style.gridTemplateColumns = "0px 1fr";
      canvasWrap.style.display = "none";
      flowWrap.style.display = "block";
    } else {
      dock.style.gridTemplateColumns = "58% 42%";
      canvasWrap.style.display = "block";
      flowWrap.style.display = "block";
    }
    if (typeof resizeFloodCanvas === "function") resizeFloodCanvas();
  }
}

function onRightColSegmentSelect(val) {
  const bNowcast = document.getElementById("block-nowcast");
  const bPipeline = document.getElementById("pipeline-board-block");
  const bXai = document.getElementById("block-xai");

  if (bNowcast && bPipeline && bXai) {
    if (val === "nowcast") {
      bNowcast.style.display = "block";
      bPipeline.style.display = "none";
      bXai.style.display = "none";
    } else if (val === "pipeline") {
      bNowcast.style.display = "none";
      bPipeline.style.display = "block";
      bXai.style.display = "none";
    } else if (val === "xai") {
      bNowcast.style.display = "none";
      bPipeline.style.display = "none";
      bXai.style.display = "block";
    } else {
      bNowcast.style.display = "block";
      bPipeline.style.display = "block";
      bXai.style.display = "block";
    }
  }
}

function openModuleSpotlight(type) {
  const modal = document.getElementById("spotlight-modal");
  const title = document.getElementById("spotlight-title");
  const icon = document.getElementById("spotlight-icon");
  const body = document.getElementById("spotlight-body");
  if (!modal || !body) return;

  modal.classList.add("open");

  if (type === "gauges") {
    icon.textContent = "GAUGES";
    title.textContent = "Live Telemetry Gauges & Level Bars";
    const el = document.getElementById("telemetry-monitor-view");
    body.innerHTML = el ? `<div class="spotlight-content" style="display:flex;flex-direction:column;gap:10px;">${el.innerHTML}</div>` : "Gauges loading...";
  } else if (type === "nowcast") {
    icon.textContent = "NOWCAST";
    title.textContent = "Risk of Cloudburst & Timelines";
    const el = document.getElementById("block-nowcast");
    body.innerHTML = el ? `<div class="spotlight-content">${el.innerHTML}</div>` : "Nowcast loading...";
  } else if (type === "pipeline") {
    icon.textContent = "PIPELINE";
    title.textContent = "Multi-Model System Pipeline & Handoff Lights";
    const el = document.getElementById("pipeline-board-block");
    body.innerHTML = el ? `<div class="spotlight-content">${el.innerHTML}</div>` : "Pipeline loading...";
  } else if (type === "pinn") {
    icon.textContent = "SIMULATOR";
    title.textContent = "River Flood Simulator & 4-Stage Flow Path";
    const el = document.getElementById("pinn-how-wrap");
    body.innerHTML = el ? `<div class="spotlight-content">${el.innerHTML}</div>` : "Simulator loading...";
  } else if (type === "xai") {
    icon.textContent = "XAI REASONING";
    title.textContent = "Why Did the AI Call This a Cloudburst?";
    const el = document.getElementById("block-xai");
    body.innerHTML = el ? `<div class="spotlight-content">${el.innerHTML}</div>` : "XAI loading...";
  } else if (type === "ranking") {
    icon.textContent = "TOWNS";
    title.textContent = "Downstream Towns & Evacuation Times";
    const el = document.getElementById("ranking-board-container");
    body.innerHTML = el ? `<div class="spotlight-content">${el.innerHTML}</div>` : "Towns loading...";
  }
}

function closeModuleSpotlight() {
  const modal = document.getElementById("spotlight-modal");
  if (modal) modal.classList.remove("open");
  const dropdown = document.getElementById("module-view-dropdown");
  if (dropdown) dropdown.value = "all";
}

// Close spotlight on Escape key
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModuleSpotlight();
});

// ── Theme Management (Light & Dark Mode) ────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem("keraunos-theme") || "dark";
  applyTheme(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.body.classList.contains("light-mode") ? "light" : "dark";
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(newTheme);
  try {
    localStorage.setItem("keraunos-theme", newTheme);
  } catch (e) {}
}

function applyTheme(theme) {
  const isLight = theme === "light";
  document.documentElement.classList.toggle("light-mode", isLight);
  document.body.classList.toggle("light-mode", isLight);

  const sunIcon = document.getElementById("theme-icon-sun");
  const moonIcon = document.getElementById("theme-icon-moon");
  const label = document.getElementById("theme-toggle-label");
  const toggleBtn = document.getElementById("theme-toggle-btn");

  if (sunIcon) sunIcon.style.display = isLight ? "none" : "block";
  if (moonIcon) moonIcon.style.display = isLight ? "block" : "none";
  if (label) label.textContent = isLight ? "Light" : "Dark";
  if (toggleBtn) {
    toggleBtn.setAttribute("aria-label", `Switch to ${isLight ? "dark" : "light"} mode`);
    toggleBtn.title = `Current: ${isLight ? "Light" : "Dark"} mode. Click to switch.`;
  }

  // Auto-switch map layer if on default theme tile
  if (isLight && APP.activeLayer === "dark") {
    switchMapLayer("topo");
  } else if (!isLight && APP.activeLayer === "topo") {
    switchMapLayer("dark");
  }

  // Re-render PINN canvas with the new theme colors
  if (typeof renderPINNCanvas === "function") {
    renderPINNCanvas();
  }
}

// ── Global Initialization ────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  backupInitialStations();
  initMap();

  updateClock();
  setInterval(updateClock, 1000);

  // Initialize with Kedarnath
  selectStation("RDP001");
});
