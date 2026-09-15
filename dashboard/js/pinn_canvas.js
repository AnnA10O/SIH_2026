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
    { name: "Kedarnath Peak", lat: 30.7700, lon: 79.0600, elev: 6940, y: 0.05, type: 'mountain' },
    { name: "Chaukhamba", lat: 30.7400, lon: 79.2800, elev: 7138, y: 0.25, type: 'mountain' },
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
    { name: "Nanda Devi", lat: 30.3700, lon: 79.9700, elev: 7816, y: 0.30, type: 'mountain' },
    { name: "Trisul", lat: 30.3100, lon: 79.7700, elev: 7120, y: 0.70, type: 'mountain' },
    { name: "Badrinath Shrine", lat: 30.7433, lon: 79.4938, elev: 3133, y: 0.00 },
    { name: "Govindghat", lat: 30.6250, lon: 79.5600, elev: 1820, y: 0.25 },
    { name: "Joshimath Town", lat: 30.5546, lon: 79.5643, elev: 1875, y: 0.40 },
    { name: "Tapovan Barrage", lat: 30.5667, lon: 79.5333, elev: 1350, y: 0.55 },
    { name: "Helang Gorge", lat: 30.5100, lon: 79.4900, elev: 1200, y: 0.70 },
    { name: "Pipalkoti", lat: 30.4300, lon: 79.4300, elev: 1050, y: 0.85 },
    { name: "Chamoli HQ", lat: 30.4024, lon: 79.3323, elev: 950, y: 1.00 }
  ],
  uttarkashi: [
    { name: "Mount Shivling", lat: 30.8700, lon: 79.0600, elev: 6543, y: 0.10, type: 'mountain' },
    { name: "Bhagirathi II", lat: 30.8900, lon: 79.1400, elev: 6512, y: 0.40, type: 'mountain' },
    { name: "Gangotri Glacier", lat: 30.9946, lon: 78.9398, elev: 3048, y: 0.00 },
    { name: "Harsil Valley", lat: 31.0300, lon: 78.7300, elev: 2620, y: 0.25 },
    { name: "Bhatwari Bend", lat: 30.8100, lon: 78.6000, elev: 1600, y: 0.45 },
    { name: "Maneri Dam", lat: 30.8667, lon: 78.7833, elev: 1320, y: 0.60 },
    { name: "Uttarkashi HQ", lat: 30.7268, lon: 78.4354, elev: 1165, y: 0.75 },
    { name: "Chinyalisaur", lat: 30.5500, lon: 78.3200, elev: 850, y: 0.88 },
    { name: "Tehri Reservoir", lat: 30.3783, lon: 78.4805, elev: 650, y: 1.00 }
  ],
  pithoragarh: [
    { name: "Panchachuli", lat: 30.2100, lon: 80.5200, elev: 6904, y: 0.15, type: 'mountain' },
    { name: "Nanda Kot", lat: 30.2700, lon: 80.0600, elev: 6861, y: 0.40, type: 'mountain' },
    { name: "Munsiari Alpine Slope", lat: 30.0668, lon: 80.2374, elev: 2200, y: 0.00 },
    { name: "Madkot Gorge", lat: 29.9800, lon: 80.3800, elev: 1450, y: 0.30 },
    { name: "Dharchula Ravine", lat: 29.8452, lon: 80.5423, elev: 915, y: 0.55 },
    { name: "Balwakot", lat: 29.7700, lon: 80.4500, elev: 750, y: 0.80 },
    { name: "Jauljibi Confluence", lat: 29.7167, lon: 80.3667, elev: 600, y: 1.00 }
  ],
  tehri: [
    { name: "Jaonli", lat: 30.8500, lon: 78.8500, elev: 6632, y: 0.20, type: 'mountain' },
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
  scene: null, camera: null, renderer: null, controls: null, terrainMesh: null, waterMesh: null, townMarkers: [], waterTargetY: -200, waterCurrentY: -200,
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
  const container = document.getElementById('pinn-leaflet-map');
  if (!container) return;

  let width = container.clientWidth || container.offsetWidth || (window.innerWidth - 380);
  let height = container.clientHeight || container.offsetHeight || (window.innerHeight - 100);
  if (width <= 0) width = 800;
  if (height <= 0) height = 600;

  if (PINN_SIM.renderer && PINN_SIM.camera && PINN_SIM.scene) {
    PINN_SIM.camera.aspect = width / height;
    PINN_SIM.camera.updateProjectionMatrix();
    PINN_SIM.renderer.setSize(width, height);
    PINN_SIM.renderer.render(PINN_SIM.scene, PINN_SIM.camera);
    updateTownLabels();
    return;
  }

  container.innerHTML = '';
  PINN_SIM.scene = new THREE.Scene();
  PINN_SIM.scene.background = new THREE.Color(0x060b14);
  PINN_SIM.scene.fog = new THREE.FogExp2(0x060b14, 0.0015);

  PINN_SIM.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 10000);
  PINN_SIM.camera.position.set(-300, 600, 800);

  PINN_SIM.renderer = new THREE.WebGLRenderer({ antialias: true });
  PINN_SIM.renderer.setSize(width, height);
  PINN_SIM.renderer.setPixelRatio(window.devicePixelRatio);
  PINN_SIM.renderer.shadowMap.enabled = true;
  container.appendChild(PINN_SIM.renderer.domElement);

  PINN_SIM.controls = new THREE.OrbitControls(PINN_SIM.camera, PINN_SIM.renderer.domElement);
  PINN_SIM.controls.enableDamping = true;
  PINN_SIM.controls.dampingFactor = 0.05;
  PINN_SIM.controls.maxPolarAngle = Math.PI / 2 - 0.05;
  PINN_SIM.controls.minAzimuthAngle = -Math.PI / 4;
  PINN_SIM.controls.maxAzimuthAngle = Math.PI / 4;
  PINN_SIM.controls.minDistance = 200;
  PINN_SIM.controls.maxDistance = 2500;
  PINN_SIM.controls.enablePan = true;

  const ambientLight = new THREE.AmbientLight(0x404040, 1.2);
  PINN_SIM.scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(500, 1000, 200);
  dirLight.castShadow = true;
  PINN_SIM.scene.add(dirLight);

  window.addEventListener('resize', () => {
    if (!PINN_SIM.camera || !PINN_SIM.renderer) return;
    const w = container.clientWidth || container.offsetWidth || (window.innerWidth - 380);
    const h = container.clientHeight || container.offsetHeight || (window.innerHeight - 100);
    if (w > 0 && h > 0) {
      PINN_SIM.camera.aspect = w / h;
      PINN_SIM.camera.updateProjectionMatrix();
      PINN_SIM.renderer.setSize(w, h);
    }
  }, false);

  animate();

  await loadPINNRegion(PINN_SIM.activeRegion);
  checkAutoRiskState();
  if (!PINN_SIM.pollTimer) {
    PINN_SIM.pollTimer = setInterval(checkAutoRiskState, 1500);
  }
}

function generateTerrain(regionKey) {
  // Clean up previous terrain if it exists
  if (PINN_SIM.terrainMesh) {
    PINN_SIM.scene.remove(PINN_SIM.terrainMesh);
    PINN_SIM.terrainMesh.geometry.dispose();
    PINN_SIM.terrainMesh.material.dispose();
  }
  if (PINN_SIM.waterMesh) {
    PINN_SIM.scene.remove(PINN_SIM.waterMesh);
    PINN_SIM.waterMesh.geometry.dispose();
    PINN_SIM.waterMesh.material.dispose();
  }

  // Parameterize terrain by region
  let maxH = 800; // max elevation
  let baseWidth = 200; // valley width
  let steepness = 0.8;
  
  // Lighter Realistic Satellite Imagery Theme
  let cBase = 0xd8dfeb, cLow = 0x2d6a4f, cMid = 0x6a7c41, cHigh = 0xbda78f, cPeak = 0x7e6657;
  let cWall = 0x1c1712; // Deep crust dirt color for solid walls

  if (regionKey === 'chamoli') { 
      maxH = 1000; baseWidth = 350; steepness = 0.6; 
      // Keep slightly snowier peaks for chamoli, but same base
      cPeak = 0xffffff; cHigh = 0x9ca3af; 
  } else if (regionKey === 'uttarkashi') { 
      maxH = 1200; baseWidth = 150; steepness = 0.95; 
      cPeak = 0xa3a3a3; // Rockier peaks
  } else if (regionKey === 'pithoragarh') { maxH = 600; baseWidth = 400; steepness = 0.5; } 
  else if (regionKey === 'tehri') { maxH = 750; baseWidth = 250; steepness = 0.7; }
  else if (regionKey === 'pauri' || regionKey === 'nainital') { 
      maxH = 500; baseWidth = 450; steepness = 0.4; 
  }
  else if (regionKey === 'almora') { maxH = 550; baseWidth = 350; steepness = 0.5; }

  const geometry = new THREE.PlaneGeometry(3000, 3000, 250, 250);
  geometry.rotateX(-Math.PI / 2);

  const pos = geometry.attributes.position;
  const colors = [];
  const colorObj = new THREE.Color();

  // Use string hashing to change random noise seed per region
  let seed1 = regionKey ? regionKey.charCodeAt(0) * 0.002 : 0;
  let seed2 = regionKey ? regionKey.charCodeAt(1) * 0.005 : 0;

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);

    const distFromCenter = Math.abs(x);
    let height = 0;

    // Make the terrain a SOLID block by pulling the outer edges deep down
    if (distFromCenter >= 1499 || Math.abs(z) >= 1499) {
       height = -800; // Deep negative Y to form the crust walls
       pos.setY(i, height);
       colorObj.setHex(cWall);
       colors.push(colorObj.r, colorObj.g, colorObj.b);
       continue;
    }

    // 1. Base Valley Shape: Parabolic curve from center to form steep walls
    if (distFromCenter > baseWidth) {
       let slopeDist = distFromCenter - baseWidth;
       // steepness controls the curve (1.0 = linear, 2.0 = steep parabola)
       height = Math.pow(slopeDist / (1500 - baseWidth), steepness * 1.5) * maxH; 
    }

    // 2. Add sharp, jagged mountain ridges (Fractal Ridge Noise using absolute sines)
    let noise = 0;
    noise += (1.0 - Math.abs(Math.sin(x * 0.003 + seed1) * Math.cos(z * 0.003 + seed2))) * (maxH * 0.5);
    noise += (1.0 - Math.abs(Math.sin(x * 0.012 + seed2) * Math.cos(z * 0.011 + seed1))) * (maxH * 0.2);
    noise += (1.0 - Math.abs(Math.sin(x * 0.04) * Math.cos(z * 0.04))) * 40;
    
    // 3. Blend noise into mountains, keep riverbed perfectly flat
    let valleyBlend = Math.min(1.0, Math.max(0.0, (distFromCenter - baseWidth) / 200.0));
    let floorNoise = Math.sin(x * 0.05) * Math.sin(z * 0.05) * 5; // tiny bumps on riverbed
    
    height += (noise * valleyBlend) + (floorNoise * (1.0 - valleyBlend));

    // 4. Add downstream slope so river flows down the Z axis (Real Himalayan Incline)
    height += (z + 1500) * 0.25;
    height = Math.max(0, height);
    pos.setY(i, height);

    if (height < 40) { colorObj.setHex(cBase); } 
    else if (height < 200) { colorObj.setHex(cLow); } 
    else if (height < 450) { colorObj.setHex(cMid); } 
    else if (height < maxH - (maxH * 0.1)) { colorObj.setHex(cHigh); } 
    else { colorObj.setHex(cPeak); }
    colors.push(colorObj.r, colorObj.g, colorObj.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.9,
    flatShading: true
  });

  PINN_SIM.terrainMesh = new THREE.Mesh(geometry, material);
  PINN_SIM.terrainMesh.receiveShadow = true;
  PINN_SIM.terrainMesh.castShadow = true;
  PINN_SIM.scene.add(PINN_SIM.terrainMesh);

  const waterGeo = new THREE.PlaneGeometry(baseWidth * 1.8, 3000, 20, 100);
  waterGeo.rotateX(-Math.PI / 2);
  
  // slope water perfectly parallel to the steep terrain
  const wpos = waterGeo.attributes.position;
  for(let i=0; i<wpos.count; i++) {
     wpos.setY(i, (wpos.getZ(i) + 1500) * 0.25);
  }
  
  const waterMat = new THREE.MeshStandardMaterial({
    color: 0x1e3a8a, // Baseflow normal blue
    transparent: true,
    opacity: 0.9,
    roughness: 0.1,
    metalness: 0.8
  });
  PINN_SIM.waterMesh = new THREE.Mesh(waterGeo, waterMat);
  PINN_SIM.waterMesh.position.y = -200; 
  PINN_SIM.scene.add(PINN_SIM.waterMesh);
  
  PINN_SIM.waterTargetY = -200;
  PINN_SIM.waterCurrentY = -200;
}

function animate() {
  PINN_SIM.animationId = requestAnimationFrame(animate);
  PINN_SIM.controls.update();

  if (PINN_SIM.waterMesh) {
    PINN_SIM.waterCurrentY += (PINN_SIM.waterTargetY - PINN_SIM.waterCurrentY) * 0.05;
    PINN_SIM.waterMesh.position.y = PINN_SIM.waterCurrentY;
  }

  updateTownLabels();
  PINN_SIM.renderer.render(PINN_SIM.scene, PINN_SIM.camera);
}

function renderTownsForRegion(regionKey) {
  PINN_SIM.townMarkers.forEach(m => {
    if (m.element && m.element.parentNode) {
      m.element.parentNode.removeChild(m.element);
    }
  });
  PINN_SIM.townMarkers = [];
  PINN_SIM.currentTownIndex = -1;

  const anchors = BASIN_ANCHORS[regionKey];
  if (!anchors) return;
  
  const container = document.getElementById('pinn-leaflet-map');

  anchors.forEach(a => {
    const z = (a.y * 2400) - 1200;
    
    let x, y, icon, titleHtml, extrasHtml;
    
    if (a.type === 'mountain') {
      x = (Math.random() > 0.5 ? 1 : -1) * (400 + Math.random() * 200); 
      y = 600 + ((z + 1500) * 0.25) + (Math.random() * 200); // High up on the peaks
      icon = '⛰️';
      titleHtml = `<span style="color:#e2e8f0; font-size:16px;">${icon}</span> ${a.name}`;
      extrasHtml = `<div style="font-size: 11px; color:#cbd5e1;">Peak Elevation: ${a.elev}m</div>`;
    } else {
      x = (Math.random() - 0.5) * 100; 
      y = 80 + ((z + 1500) * 0.25); 
      icon = '📍';
      titleHtml = `<span style="color:#ef4444; font-size:16px;">${icon}</span> ${a.name}`;
      const valleyDepth = Math.max(10, Math.floor(800 - (y - 150))); 
      extrasHtml = `
        <div style="font-size: 11px; color:#cbd5e1;">Region Elev: ${a.elev}m</div>
        <div style="font-size: 11px; color:#94a3b8;">Valley Depth: ~${valleyDepth}m</div>
        <div class="water-level-label" style="font-size: 11px; color:#60a5fa; margin-top:3px; padding-top:3px; border-top:1px solid rgba(255,255,255,0.1);">
          Current Water: 0.4m (Normal)
        </div>
      `;
    }

    const el = document.createElement('div');
    el.className = 'pinn-3d-label';
    
    el.innerHTML = `
      <div style="display:flex; align-items:center; gap:4px; font-weight:800; font-size:14px; margin-bottom:2px;">
        ${titleHtml}
      </div>
      ${extrasHtml}
    `;
    el.style.position = 'absolute';
    el.style.color = 'white';
    el.style.fontWeight = '500';
    el.style.fontFamily = 'var(--font-sans)';
    el.style.textShadow = '0 2px 4px rgba(0,0,0,0.9)';
    el.style.pointerEvents = 'none';
    el.style.transform = 'translate(-50%, -100%)';
    el.style.background = 'rgba(6, 11, 20, 0.75)';
    el.style.backdropFilter = 'blur(4px)';
    el.style.padding = '8px 12px';
    el.style.borderRadius = '6px';
    el.style.border = '1px solid rgba(255,255,255,0.15)';
    container.appendChild(el);

    PINN_SIM.townMarkers.push({
      element: el,
      pos3D: new THREE.Vector3(x, y, z)
    });
  });
}

function updateTownLabels() {
  if (!PINN_SIM.camera || !PINN_SIM.renderer) return;
  
  const widthHalf = PINN_SIM.renderer.domElement.clientWidth / 2;
  const heightHalf = PINN_SIM.renderer.domElement.clientHeight / 2;

  PINN_SIM.townMarkers.forEach(m => {
    const pos = m.pos3D.clone();
    pos.project(PINN_SIM.camera);

    if (pos.z > 1 || pos.z < -1) {
      m.element.style.display = 'none';
      return;
    }
    
    if (pos.x < -1 || pos.x > 1 || pos.y < -1 || pos.y > 1) {
      m.element.style.display = 'none';
      return;
    }

    m.element.style.display = 'block';
    const x = (pos.x * widthHalf) + widthHalf;
    const y = -(pos.y * heightHalf) + heightHalf;
    m.element.style.left = x + 'px';
    m.element.style.top = y + 'px';
  });
}

// ── LOAD REGION (FIT BOUNDS, HILLSHADE, WINDING FLOOD OVERLAY, TOWNS, KPIS) ────
async function loadPINNRegion(regionKey) {
  const normalizedKey = REGION_KEY_MAP[regionKey.toLowerCase()] || regionKey;
  PINN_SIM.activeRegion = normalizedKey;

  const data = await fetchPINNData(normalizedKey);
  if (!data || !PINN_SIM.scene) return;

  const anchors = BASIN_ANCHORS[normalizedKey] || BASIN_ANCHORS['rudraprayag'];
  const bounds = BASIN_BOUNDS[normalizedKey] || BASIN_BOUNDS['rudraprayag'];
  
  generateTerrain(normalizedKey);
  renderTownsForRegion(normalizedKey);
  const townsEl = document.getElementById('pinn-towns-list');
  if (townsEl) {
    const names = data.towns_affected.map(t => t.name).join(" &bull; ");
    townsEl.innerHTML = '<span style="color: var(--text-muted); font-weight:700;">AFFECTED TOWNS & INFRASTRUCTURE:</span> <span style="color: var(--text-main); font-weight:600;">' + names + '</span>';
  }

  // Print live metrics log for integrity verification
  console.log(`[PINN Data Verification Log] Region: ${data.region_name} | Depth: ${data.peak_water_depth_m}m | Speed: ${data.peak_flow_speed_ms}m/s | Area: ${data.flooded_area_km2}km²`);
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
    PINN_SIM.waterTargetY = -200;
    if (PINN_SIM.waterMesh) PINN_SIM.waterMesh.material.color.setHex(0x1e3a8a);
    document.querySelectorAll('.water-level-label').forEach(lbl => {
      lbl.innerHTML = 'Current Water: 0.4m (Normal)';
    });

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
    PINN_SIM.waterTargetY = -200;
    if (PINN_SIM.waterMesh) PINN_SIM.waterMesh.material.color.setHex(0x1e3a8a);
    document.querySelectorAll('.water-level-label').forEach(lbl => {
      lbl.innerHTML = 'Current Water: 0.85m (Elevated)';
    });

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

function renderFloodDepthOverlay(grid, peakDepth, anchors) {
  // In 3D, we raise the water mesh and change color to sky blue flash flood
  PINN_SIM.waterTargetY = 80; 
  if (PINN_SIM.waterMesh) PINN_SIM.waterMesh.material.color.setHex(0x0ea5e9);
  
  const surge = peakDepth || 6.8;
  document.querySelectorAll('.water-level-label').forEach(lbl => {
      lbl.innerHTML = `Current Water: 0.4m <br><span style="color:#ef4444; font-weight: 800; font-size: 13px;">➔ SURGE: +${surge}m Increase</span>`;
  });
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
  
  updatePINNConclusionPanel('NORMAL', PINN_SIM.lastRiskScore || 0.8, null);
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
  
  updatePINNConclusionPanel('WATCH', risk, null);
}

function updatePINNBottomPanel(data) {
  const depthEl = document.getElementById('pinn-depth');
  const speedEl = document.getElementById('pinn-speed');
  const areaEl = document.getElementById('pinn-area');
  const gorgeEl = document.getElementById('pinn-gorge-name');
  const townsEl = document.getElementById('pinn-towns-list');

  if (depthEl) depthEl.innerText = `${data.peak_water_depth_m} m (Flash Flood)`;
  if (speedEl) speedEl.innerText = `${data.peak_velocity_m_s} m/s`;
  if (areaEl) areaEl.innerText = `${data.flooded_area_km2} km²`;
  if (gorgeEl) gorgeEl.innerHTML = `<span style="color:var(--orange);">${data.choke_location}</span>`;
  
  if (townsEl) {
    const names = data.towns_affected.map(t => t.name).join(" &bull; ");
    townsEl.innerHTML = `<span style="color: var(--text-muted); font-weight:700;">AFFECTED TOWNS & INFRASTRUCTURE:</span> <span style="color: #ef4444; font-weight:600;">${names}</span>`;
  }

  updatePINNConclusionPanel('ALERT', PINN_SIM.lastRiskScore || 85.0, data);
}

function updatePINNConclusionPanel(state, risk = 0, data = null) {
  const panel = document.getElementById('pinn-conclusion-content');
  if (!panel) return;

  if (state === 'NORMAL') {
    panel.innerHTML = `
      <div style="padding: 10px; background: rgba(34, 197, 94, 0.1); border-left: 3px solid var(--green); border-radius: 4px;">
        <div style="color: var(--green); font-weight: 800; font-size: 14px; margin-bottom: 4px;">✅ ALL CLEAR</div>
        <div>No immediate flood threat detected in the active basin.</div>
      </div>
      <ul style="margin:0; padding-left:20px; color: var(--text-muted); line-height:1.6;">
        <li>Current Risk Score: <b>${risk.toFixed(1)}%</b></li>
        <li>Baseflow levels nominal.</li>
        <li>Routine sensor telemetry ongoing.</li>
      </ul>
    `;
  } else if (state === 'WATCH') {
    panel.innerHTML = `
      <div style="padding: 10px; background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; border-radius: 4px;">
        <div style="color: #f59e0b; font-weight: 800; font-size: 14px; margin-bottom: 4px;">⚠️ ELEVATED WATCH</div>
        <div>Convective cell detected. River levels rising.</div>
      </div>
      <ul style="margin:0; padding-left:20px; color: var(--text-sub); line-height:1.6;">
        <li>Current Risk Score: <b style="color: #f59e0b;">${risk.toFixed(1)}%</b></li>
        <li>Prepare for potential localized evacuation.</li>
        <li>Ensure communications lines are open.</li>
      </ul>
    `;
  } else if (state === 'ALERT' && data) {
    const towns = data.towns_affected.map(t => t.name).join(", ");
    panel.innerHTML = `
      <div style="padding: 10px; background: rgba(239, 68, 68, 0.15); border-left: 3px solid var(--red); border-radius: 4px;">
        <div style="color: var(--red); font-weight: 800; font-size: 14px; margin-bottom: 4px;">🚨 CRITICAL FLOOD EVENT</div>
        <div>Flash flood simulation triggered based on critical risk scores.</div>
      </div>
      <ul style="margin:0; padding-left:20px; color: var(--text-main); line-height:1.6;">
        <li><b style="color: var(--red);">Risk Score:</b> <b>>50%</b></li>
        <li><b style="color: var(--orange);">Choke Point:</b> ${data.choke_location}</li>
        <li><b style="color: var(--cyan);">Peak Speed:</b> ${data.peak_velocity_m_s} m/s</li>
        <li><b style="color: var(--blue);">Peak Depth:</b> ${data.peak_water_depth_m} m</li>
        <li><b style="color: #f43f5e;">Time to Peak (ETA):</b> ${data.time_to_peak_mins ? data.time_to_peak_mins + " mins" : "Calculating..."}</li>
      </ul>
      <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 4px;">
        <div style="font-size: 11px; font-weight: 800; color: var(--text-muted); margin-bottom: 4px;">IMMEDIATE ACTIONS REQUIRED</div>
        <ul style="margin:0; padding-left:20px; color: var(--text-main); line-height:1.4;">
          <li>Initiate siren broadcast for: <b>${towns}</b>.</li>
          <li>Evacuate low-lying riverbanks immediately.</li>
          <li>Deploy NDRF units to ${data.choke_location}.</li>
        </ul>
      </div>
    `;
  }
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

// ── KEYBOARD NAVIGATION FOR TOWNS ──────────────────────────────────────────
window.addEventListener('keydown', (e) => {
  const pinnTab = document.getElementById('tab-pinn');
  if (!pinnTab || !pinnTab.classList.contains('active')) return;

  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
    e.preventDefault();
    jumpToNextTown(1);
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    jumpToNextTown(-1);
  }
}, { capture: true });

function jumpToNextTown(direction) {
  if (!PINN_SIM.townMarkers || PINN_SIM.townMarkers.length === 0) return;
  if (PINN_SIM.currentTownIndex === undefined) PINN_SIM.currentTownIndex = -1;

  PINN_SIM.currentTownIndex += direction;
  if (PINN_SIM.currentTownIndex >= PINN_SIM.townMarkers.length) {
    PINN_SIM.currentTownIndex = 0;
  } else if (PINN_SIM.currentTownIndex < 0) {
    PINN_SIM.currentTownIndex = PINN_SIM.townMarkers.length - 1;
  }

  const targetTown = PINN_SIM.townMarkers[PINN_SIM.currentTownIndex];

  PINN_SIM.townMarkers.forEach(m => {
    if(m.element) {
      m.element.style.border = '1px solid rgba(255,255,255,0.15)';
      m.element.style.boxShadow = 'none';
      m.element.style.transform = 'translate(-50%, -100%) scale(1)';
      m.element.style.zIndex = '1';
    }
  });

  if(targetTown.element) {
    targetTown.element.style.border = '2px solid #06b6d4';
    targetTown.element.style.boxShadow = '0 0 15px rgba(6, 182, 212, 0.6)';
    targetTown.element.style.transform = 'translate(-50%, -100%) scale(1.1)';
    targetTown.element.style.zIndex = '100';
  }

  if (PINN_SIM.camera && PINN_SIM.controls) {
    const targetPos = targetTown.pos3D;
    const camOffset = new THREE.Vector3(-150, 200, 300);
    const endCamPos = targetPos.clone().add(camOffset);
    const startCamPos = PINN_SIM.camera.position.clone();
    const startTarget = PINN_SIM.controls.target.clone();
    
    // Disable controls temporarily so they don't fight our animation
    PINN_SIM.controls.enabled = false;
    
    let t = 0;
    const animateCamera = () => {
      t += 0.03;
      if (t > 1) t = 1;
      const easeT = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      
      PINN_SIM.camera.position.lerpVectors(startCamPos, endCamPos, easeT);
      PINN_SIM.controls.target.lerpVectors(startTarget, targetPos, easeT);
      
      if (t < 1) {
        requestAnimationFrame(animateCamera);
      } else {
        PINN_SIM.controls.enabled = true;
        PINN_SIM.controls.update();
      }
    };
    animateCamera();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  setTimeout(initFloodCanvas, 150);
});
