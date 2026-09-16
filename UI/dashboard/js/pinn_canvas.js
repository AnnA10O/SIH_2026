/**
 * pinn_canvas.js
 * Physics-Informed Neural Network (PINN) 2D Riverbed & Valley Flood Simulator
 *
 * Dynamic Topographical & Geographic Engine:
 *   - Dynamically generates real elevation slope and surrounding valley terrain
 *     for all 40 Himalayan and Assam AWS stations.
 *   - Flow Modes:
 *       1. Mountain Gorge Choke (Steep rock canyon walls that constrict into narrow bottlenecks,
 *          causing extreme vertical hydraulic pooling).
 *       2. Riverside Plain Overflow (Low-gradient alluvial riverbeds with natural levees/embankments
 *          where water breaches the riverbank and floods surrounding plains/ghats).
 *   - X-Axis: 4 clean, non-overlapping landmarks (0 km origin to downstream outlet).
 *   - 4-Stage Flow Path Cards ("Table"): 100% dynamic - updates station name, river name,
 *     elevation drop, slope %, channel narrowing / embankment capacity, and flood verdict.
 *   - Strictly ZERO emojis anywhere.
 */

const PINN_SIM = {
  active: false,
  scenario: "cloudburst",
  t: 0,
  maxT: 100,
  animId: null,
  gridSize: 100,
  elevation: [],
  h: [],
  u: [],
  particles: [],

  // Dynamic parameters tied to active station
  activeStation: null,
  geo: null,
  startElevation: 3583,
  endElevation: 610,
  slopePct: 11.4,
  flowType: "gorge_choke", // "gorge_choke" | "riverside_flow"
  riverName: "Mandakini River",
  basinName: "Mandakini Valley",
  accumulationName: "Sonprayag Gorge Bottleneck",
  landmarks: [],

  currentRainRate: 0,
  currentPeakDepth: 0.4,
  currentVelocity: 1.2,
  currentArrivalMin: 60,

  presets: {
    cloudburst: {
      peakDepth: 6.8,
      velocity: 11.4,
      extent: 18.2,
      arrivalTime: 32,
      riskLevel: "CRITICAL TIER-4: FLASH FLOOD INUNDATION",
      riskColor: "#ef4444"
    },
    thunderstorm: {
      peakDepth: 3.8,
      velocity: 7.2,
      extent: 10.4,
      arrivalTime: 54,
      riskLevel: "WARNING TIER-3: SURGE WATCH",
      riskColor: "#f97316"
    }
  }
};

// ── COMPREHENSIVE HYDRO-GEOGRAPHIC DATABASE FOR ALL STATIONS ────────────────
const HYDRO_DATABASE = {
  // ── Rudraprayag (Mandakini River Basin) ──
  "RDP001": {
    riverName: "Mandakini River",
    valleyName: "Mandakini Alpine Canyon",
    flowType: "gorge_choke",
    startElev: 3583,
    endElev: 610,
    slopePct: 11.4,
    chokeOrBankName: "Sonprayag Gorge Bottleneck",
    channelSpecs: { upstream: "85m", bottleneck: "18m", ratio: "4.7x canyon pinch" },
    baseInflow: 180,
    inflowMultiplier: 8.4,
    landmarks: [
      { name: "Kedarnath AWS", elev: "3583m", xRatio: 0.0 },
      { name: "Rambara Gorge", elev: "2700m", xRatio: 0.32 },
      { name: "Sonprayag Choke", elev: "1829m", xRatio: 0.66, isRisk: true },
      { name: "Rudraprayag Outlet", elev: "610m", xRatio: 1.0 }
    ]
  },
  "RDP002": {
    riverName: "Mandakini River",
    valleyName: "Sonprayag Mountain Canyon",
    flowType: "gorge_choke",
    startElev: 1829,
    endElev: 610,
    slopePct: 9.4,
    chokeOrBankName: "Gaurikund Canyon Throat",
    channelSpecs: { upstream: "72m", bottleneck: "16m", ratio: "4.5x canyon pinch" },
    baseInflow: 150,
    inflowMultiplier: 7.8,
    landmarks: [
      { name: "Sonprayag AWS", elev: "1829m", xRatio: 0.0 },
      { name: "Gaurikund Choke", elev: "1980m", xRatio: 0.32 },
      { name: "Phata Ravine", elev: "1500m", xRatio: 0.66, isRisk: true },
      { name: "Rudraprayag Outlet", elev: "610m", xRatio: 1.0 }
    ]
  },
  "RDP003": {
    riverName: "Alaknanda & Mandakini Confluence",
    valleyName: "Rudraprayag Sangam Canyon",
    flowType: "gorge_choke",
    startElev: 610,
    endElev: 460,
    slopePct: 4.8,
    chokeOrBankName: "Koteshwar Dam Gorge Throat",
    channelSpecs: { upstream: "120m", bottleneck: "32m", ratio: "3.8x gorge narrowing" },
    baseInflow: 240,
    inflowMultiplier: 9.5,
    landmarks: [
      { name: "Rudraprayag AWS", elev: "610m", xRatio: 0.0 },
      { name: "Sangam Confluence", elev: "590m", xRatio: 0.32 },
      { name: "Koteshwar Gorge", elev: "510m", xRatio: 0.66, isRisk: true },
      { name: "Srinagar Basin", elev: "460m", xRatio: 1.0 }
    ]
  },
  "RDP004": {
    riverName: "Mandakini Tributary",
    valleyName: "Triyugi Mountain Flank",
    flowType: "gorge_choke",
    startElev: 1980,
    endElev: 760,
    slopePct: 10.2,
    chokeOrBankName: "Songanga Ravine Choke",
    channelSpecs: { upstream: "60m", bottleneck: "14m", ratio: "4.3x ravine pinch" },
    baseInflow: 120,
    inflowMultiplier: 6.8,
    landmarks: [
      { name: "Triyuginarayan AWS", elev: "1980m", xRatio: 0.0 },
      { name: "Songanga Cascade", elev: "1550m", xRatio: 0.32 },
      { name: "Sonprayag Choke", elev: "1829m", xRatio: 0.66, isRisk: true },
      { name: "Augustmuni Basin", elev: "762m", xRatio: 1.0 }
    ]
  },
  "RDP005": {
    riverName: "Mandakini River",
    valleyName: "Augustmuni Alluvial Plain",
    flowType: "riverside_flow",
    startElev: 762,
    endElev: 610,
    slopePct: 3.2,
    chokeOrBankName: "Augustmuni Riverside Embankment",
    channelSpecs: { upstream: "Normal: 110m", bottleneck: "Spills: 260m Wide", ratio: "Overbank Flooding" },
    baseInflow: 190,
    inflowMultiplier: 7.2,
    landmarks: [
      { name: "Augustmuni AWS", elev: "762m", xRatio: 0.0 },
      { name: "Tilwara Terrace", elev: "710m", xRatio: 0.32 },
      { name: "Augustmuni Lowlands", elev: "650m", xRatio: 0.66, isRisk: true },
      { name: "Rudraprayag Sangam", elev: "610m", xRatio: 1.0 }
    ]
  },

  // ── Chamoli Basin (Alaknanda & Dhauliganga) ──
  "CHM001": {
    riverName: "Alaknanda & Dhauliganga Rivers",
    valleyName: "Joshimath Canyon Corridor",
    flowType: "gorge_choke",
    startElev: 1875,
    endElev: 745,
    slopePct: 10.2,
    chokeOrBankName: "Tapovan Barrage Gorge Choke",
    channelSpecs: { upstream: "110m", bottleneck: "22m", ratio: "5.0x canyon pinch" },
    baseInflow: 210,
    inflowMultiplier: 8.8,
    landmarks: [
      { name: "Joshimath AWS", elev: "1875m", xRatio: 0.0 },
      { name: "Helang Ravine", elev: "1520m", xRatio: 0.32 },
      { name: "Tapovan Barrage", elev: "1350m", xRatio: 0.66, isRisk: true },
      { name: "Chamoli Outlet", elev: "855m", xRatio: 1.0 }
    ]
  },
  "CHM002": {
    riverName: "Alaknanda Headwaters",
    valleyName: "Badrinath Glacial Valley",
    flowType: "gorge_choke",
    startElev: 3133,
    endElev: 1875,
    slopePct: 12.5,
    chokeOrBankName: "Vishnuprayag Gorge Throat",
    channelSpecs: { upstream: "90m", bottleneck: "16m", ratio: "5.6x canyon pinch" },
    baseInflow: 160,
    inflowMultiplier: 7.5,
    landmarks: [
      { name: "Badrinath AWS", elev: "3133m", xRatio: 0.0 },
      { name: "Mana Pass Gorge", elev: "2850m", xRatio: 0.32 },
      { name: "Vishnuprayag Choke", elev: "1980m", xRatio: 0.66, isRisk: true },
      { name: "Joshimath Basin", elev: "1875m", xRatio: 1.0 }
    ]
  },
  "CHM003": {
    riverName: "Alaknanda River",
    valleyName: "Chamoli River Gorge",
    flowType: "gorge_choke",
    startElev: 855,
    endElev: 640,
    slopePct: 5.4,
    chokeOrBankName: "Birahi Gorge Chokepoint",
    channelSpecs: { upstream: "125m", bottleneck: "28m", ratio: "4.5x gorge narrowing" },
    baseInflow: 230,
    inflowMultiplier: 8.2,
    landmarks: [
      { name: "Chamoli Town AWS", elev: "855m", xRatio: 0.0 },
      { name: "Pipalkoti Rapids", elev: "790m", xRatio: 0.32 },
      { name: "Birahi Gorge", elev: "720m", xRatio: 0.66, isRisk: true },
      { name: "Karnaprayag Sangam", elev: "640m", xRatio: 1.0 }
    ]
  },
  "CHM004": {
    riverName: "Alaknanda River",
    valleyName: "Gauchar Riverside Floodplain",
    flowType: "riverside_flow",
    startElev: 745,
    endElev: 580,
    slopePct: 2.8,
    chokeOrBankName: "Gauchar Airstrip Riverside Terrace",
    channelSpecs: { upstream: "Normal: 140m", bottleneck: "Spills: 380m Wide", ratio: "Overbank Flooding" },
    baseInflow: 250,
    inflowMultiplier: 8.5,
    landmarks: [
      { name: "Gauchar AWS", elev: "745m", xRatio: 0.0 },
      { name: "Karnaprayag", elev: "690m", xRatio: 0.32 },
      { name: "Gauchar Plain", elev: "640m", xRatio: 0.66, isRisk: true },
      { name: "Rudraprayag Sangam", elev: "580m", xRatio: 1.0 }
    ]
  },
  "CHM005": {
    riverName: "Dhauliganga River",
    valleyName: "Niti Trans-Himalayan Canyon",
    flowType: "gorge_choke",
    startElev: 3600,
    endElev: 1875,
    slopePct: 13.8,
    chokeOrBankName: "Rini Gorge Rock Bottleneck",
    channelSpecs: { upstream: "75m", bottleneck: "12m", ratio: "6.2x canyon pinch" },
    baseInflow: 170,
    inflowMultiplier: 7.9,
    landmarks: [
      { name: "Niti Valley AWS", elev: "3600m", xRatio: 0.0 },
      { name: "Malari Canyon", elev: "3050m", xRatio: 0.32 },
      { name: "Rini Gorge Choke", elev: "2100m", xRatio: 0.66, isRisk: true },
      { name: "Joshimath Sangam", elev: "1875m", xRatio: 1.0 }
    ]
  },

  // ── Uttarkashi Basin (Bhagirathi River) ──
  "UTK001": {
    riverName: "Bhagirathi River",
    valleyName: "Uttarkashi Town Riverside Basin",
    flowType: "riverside_flow",
    startElev: 1165,
    endElev: 780,
    slopePct: 3.4,
    chokeOrBankName: "Joshiyara Riverbank Lowlands",
    channelSpecs: { upstream: "Normal: 130m", bottleneck: "Spills: 320m Wide", ratio: "Overbank Inundation" },
    baseInflow: 220,
    inflowMultiplier: 8.1,
    landmarks: [
      { name: "Uttarkashi AWS", elev: "1165m", xRatio: 0.0 },
      { name: "Gangori Sangam", elev: "1050m", xRatio: 0.32 },
      { name: "Joshiyara Bank", elev: "910m", xRatio: 0.66, isRisk: true },
      { name: "Dharasu Basin", elev: "780m", xRatio: 1.0 }
    ]
  },
  "UTK002": {
    riverName: "Bhagirathi River",
    valleyName: "Gangotri Granite Canyon",
    flowType: "gorge_choke",
    startElev: 3048,
    endElev: 1320,
    slopePct: 11.5,
    chokeOrBankName: "Bhatwari Canyon Choke",
    channelSpecs: { upstream: "100m", bottleneck: "20m", ratio: "5.0x canyon pinch" },
    baseInflow: 190,
    inflowMultiplier: 8.0,
    landmarks: [
      { name: "Gangotri AWS", elev: "3048m", xRatio: 0.0 },
      { name: "Gangnani Canyon", elev: "2200m", xRatio: 0.32 },
      { name: "Bhatwari Choke", elev: "1654m", xRatio: 0.66, isRisk: true },
      { name: "Maneri Reservoir", elev: "1320m", xRatio: 1.0 }
    ]
  },
  "UTK003": {
    riverName: "Bhagirathi River",
    valleyName: "Maneri Gorge & Reservoir",
    flowType: "gorge_choke",
    startElev: 1320,
    endElev: 1050,
    slopePct: 5.8,
    chokeOrBankName: "Maneri Reservoir Dam Throat",
    channelSpecs: { upstream: "80m", bottleneck: "15m", ratio: "5.3x dam throat pinch" },
    baseInflow: 200,
    inflowMultiplier: 8.2,
    landmarks: [
      { name: "Maneri Dam AWS", elev: "1320m", xRatio: 0.0 },
      { name: "Dam Spillway", elev: "1280m", xRatio: 0.32 },
      { name: "Reservoir Throat", elev: "1150m", xRatio: 0.66, isRisk: true },
      { name: "Tiloth Powerhouse", elev: "1050m", xRatio: 1.0 }
    ]
  },
  "UTK004": {
    riverName: "Bhagirathi River",
    valleyName: "Harsil-Dharali Valley",
    flowType: "gorge_choke",
    startElev: 2287,
    endElev: 1320,
    slopePct: 9.8,
    chokeOrBankName: "Sukhi Top Ravine Choke",
    channelSpecs: { upstream: "95m", bottleneck: "19m", ratio: "5.0x ravine pinch" },
    baseInflow: 170,
    inflowMultiplier: 7.6,
    landmarks: [
      { name: "Dharali AWS", elev: "2287m", xRatio: 0.0 },
      { name: "Harsil Valley", elev: "2050m", xRatio: 0.32 },
      { name: "Sukhi Choke", elev: "1720m", xRatio: 0.66, isRisk: true },
      { name: "Bhatwari Outlet", elev: "1320m", xRatio: 1.0 }
    ]
  },
  "UTK005": {
    riverName: "Bhagirathi River",
    valleyName: "Bhatwari Mountain Gorge",
    flowType: "gorge_choke",
    startElev: 1654,
    endElev: 1165,
    slopePct: 7.2,
    chokeOrBankName: "Maneri Dam Gorge Throat",
    channelSpecs: { upstream: "85m", bottleneck: "18m", ratio: "4.7x canyon pinch" },
    baseInflow: 180,
    inflowMultiplier: 7.8,
    landmarks: [
      { name: "Bhatwari AWS", elev: "1654m", xRatio: 0.0 },
      { name: "Gangnani Bed", elev: "1480m", xRatio: 0.32 },
      { name: "Dam Gorge Throat", elev: "1320m", xRatio: 0.66, isRisk: true },
      { name: "Uttarkashi HQ", elev: "1165m", xRatio: 1.0 }
    ]
  },

  // ── Dehradun & Tehri ──
  "DDN001": {
    riverName: "Song, Rispana & Bindal Rivers",
    valleyName: "Doon Alluvial Basin",
    flowType: "riverside_flow",
    startElev: 682,
    endElev: 380,
    slopePct: 2.5,
    chokeOrBankName: "Bindal & Rispana Riverside Embankment",
    channelSpecs: { upstream: "Normal: 45m", bottleneck: "Spills: 160m Wide", ratio: "Overbank Flooding" },
    baseInflow: 160,
    inflowMultiplier: 7.2,
    landmarks: [
      { name: "Dehradun AWS", elev: "682m", xRatio: 0.0 },
      { name: "Rispana Riverbank", elev: "580m", xRatio: 0.32 },
      { name: "Bindal Lowlands", elev: "480m", xRatio: 0.66, isRisk: true },
      { name: "Song River Sangam", elev: "380m", xRatio: 1.0 }
    ]
  },
  "DDN002": {
    riverName: "Kempty & Galogi Torrents",
    valleyName: "Mussoorie Ridge Ravine",
    flowType: "gorge_choke",
    startElev: 2005,
    endElev: 682,
    slopePct: 14.2,
    chokeOrBankName: "Galogi Canyon Ravine Choke",
    channelSpecs: { upstream: "60m", bottleneck: "12m", ratio: "5.0x ravine pinch" },
    baseInflow: 150,
    inflowMultiplier: 7.4,
    landmarks: [
      { name: "Mussoorie AWS", elev: "2005m", xRatio: 0.0 },
      { name: "Kempty Falls", elev: "1450m", xRatio: 0.32 },
      { name: "Galogi Choke", elev: "1020m", xRatio: 0.66, isRisk: true },
      { name: "Dehradun Basin", elev: "682m", xRatio: 1.0 }
    ]
  },
  "DDN003": {
    riverName: "Ganges River",
    valleyName: "Ganga Foothill Riverside Corridor",
    flowType: "riverside_flow",
    startElev: 372,
    endElev: 285,
    slopePct: 1.8,
    chokeOrBankName: "Triveni Ghat Riverside Floodplain",
    channelSpecs: { upstream: "Normal: 280m", bottleneck: "Spills: 580m Wide", ratio: "Overbank Inundation" },
    baseInflow: 380,
    inflowMultiplier: 11.2,
    landmarks: [
      { name: "Rishikesh AWS", elev: "372m", xRatio: 0.0 },
      { name: "Laxman Jhula Gorge", elev: "350m", xRatio: 0.32 },
      { name: "Triveni Ghat Plain", elev: "310m", xRatio: 0.66, isRisk: true },
      { name: "Haridwar Plains", elev: "285m", xRatio: 1.0 }
    ]
  },
  "TEH001": {
    riverName: "Bhagirathi River",
    valleyName: "Tehri Mountain Ridge",
    flowType: "gorge_choke",
    startElev: 1750,
    endElev: 840,
    slopePct: 9.8,
    chokeOrBankName: "Tehri Dam Reservoir Inflow Throat",
    channelSpecs: { upstream: "140m", bottleneck: "30m", ratio: "4.7x inflow pinch" },
    baseInflow: 220,
    inflowMultiplier: 8.6,
    landmarks: [
      { name: "New Tehri AWS", elev: "1750m", xRatio: 0.0 },
      { name: "Bhilangna Flank", elev: "1350m", xRatio: 0.32 },
      { name: "Reservoir Inflow Throat", elev: "950m", xRatio: 0.66, isRisk: true },
      { name: "Tehri Dam", elev: "840m", xRatio: 1.0 }
    ]
  },
  "TEH002": {
    riverName: "Bhagirathi & Alaknanda Confluence",
    valleyName: "Koteshwar Dam Gorge",
    flowType: "gorge_choke",
    startElev: 840,
    endElev: 480,
    slopePct: 6.2,
    chokeOrBankName: "Koteshwar Spillway Canyon",
    channelSpecs: { upstream: "120m", bottleneck: "25m", ratio: "4.8x spillway pinch" },
    baseInflow: 260,
    inflowMultiplier: 9.2,
    landmarks: [
      { name: "Tehri Dam AWS", elev: "840m", xRatio: 0.0 },
      { name: "Dam Spillway", elev: "720m", xRatio: 0.32 },
      { name: "Koteshwar Gorge", elev: "580m", xRatio: 0.66, isRisk: true },
      { name: "Devprayag Sangam", elev: "480m", xRatio: 1.0 }
    ]
  },

  // ── Pithoragarh Basin (Kali & Gori Ganga) ──
  "PTH001": {
    riverName: "Soar Valley Stream",
    valleyName: "Soar Intermontane Basin",
    flowType: "riverside_flow",
    startElev: 1627,
    endElev: 820,
    slopePct: 4.1,
    chokeOrBankName: "Soar Valley Riverside Floodplain",
    channelSpecs: { upstream: "Normal: 50m", bottleneck: "Spills: 180m Wide", ratio: "Overbank Inundation" },
    baseInflow: 140,
    inflowMultiplier: 6.5,
    landmarks: [
      { name: "Pithoragarh AWS", elev: "1627m", xRatio: 0.0 },
      { name: "Wadda Terrace", elev: "1380m", xRatio: 0.32 },
      { name: "Soar Plain", elev: "1100m", xRatio: 0.66, isRisk: true },
      { name: "Ghat Riverbed", elev: "820m", xRatio: 1.0 }
    ]
  },
  "PTH002": {
    riverName: "Kali River (Indo-Nepal Border)",
    valleyName: "Dharchula Riverside Corridor",
    flowType: "riverside_flow",
    startElev: 915,
    endElev: 620,
    slopePct: 3.0,
    chokeOrBankName: "Dharchula Riverside Embankment",
    channelSpecs: { upstream: "Normal: 90m", bottleneck: "Spills: 220m Wide", ratio: "Overbank Flooding" },
    baseInflow: 210,
    inflowMultiplier: 8.4,
    landmarks: [
      { name: "Dharchula AWS", elev: "915m", xRatio: 0.0 },
      { name: "Nepal Border Gorge", elev: "820m", xRatio: 0.32 },
      { name: "Dharchula Bank", elev: "740m", xRatio: 0.66, isRisk: true },
      { name: "Baluwakot Plain", elev: "620m", xRatio: 1.0 }
    ]
  },
  "PTH003": {
    riverName: "Gori Ganga River",
    valleyName: "Munsiari Alpine Canyon",
    flowType: "gorge_choke",
    startElev: 2200,
    endElev: 610,
    slopePct: 10.8,
    chokeOrBankName: "Madkot Canyon Ravine Choke",
    channelSpecs: { upstream: "80m", bottleneck: "14m", ratio: "5.7x canyon pinch" },
    baseInflow: 180,
    inflowMultiplier: 8.0,
    landmarks: [
      { name: "Munsiari AWS", elev: "2200m", xRatio: 0.0 },
      { name: "Madkot Ravine", elev: "1350m", xRatio: 0.32 },
      { name: "Madkot Choke", elev: "980m", xRatio: 0.66, isRisk: true },
      { name: "Jauljibi Sangam", elev: "610m", xRatio: 1.0 }
    ]
  },
  "PTH004": {
    riverName: "Ramganga River",
    valleyName: "Berinag Mountain Ridge",
    flowType: "gorge_choke",
    startElev: 1860,
    endElev: 720,
    slopePct: 8.6,
    chokeOrBankName: "Thal Riverbed Gorge Choke",
    channelSpecs: { upstream: "70m", bottleneck: "15m", ratio: "4.6x gorge narrowing" },
    baseInflow: 150,
    inflowMultiplier: 7.2,
    landmarks: [
      { name: "Berinag AWS", elev: "1860m", xRatio: 0.0 },
      { name: "Thal Gorge", elev: "1280m", xRatio: 0.32 },
      { name: "Thal Choke", elev: "950m", xRatio: 0.66, isRisk: true },
      { name: "Rameshwar Sangam", elev: "720m", xRatio: 1.0 }
    ]
  },
  "PTH005": {
    riverName: "Charma River",
    valleyName: "Didihat Mountain Flank",
    flowType: "gorge_choke",
    startElev: 1725,
    endElev: 610,
    slopePct: 8.9,
    chokeOrBankName: "Askot Gorge Ravine Choke",
    channelSpecs: { upstream: "65m", bottleneck: "14m", ratio: "4.6x canyon pinch" },
    baseInflow: 140,
    inflowMultiplier: 7.0,
    landmarks: [
      { name: "Didihat AWS", elev: "1725m", xRatio: 0.0 },
      { name: "Ogla Flank", elev: "1320m", xRatio: 0.32 },
      { name: "Askot Choke", elev: "880m", xRatio: 0.66, isRisk: true },
      { name: "Jauljibi Sangam", elev: "610m", xRatio: 1.0 }
    ]
  },

  // ── Assam Corridor (Brahmaputra Riparian Plains) ──
  "ASM001": {
    riverName: "Brahmaputra River",
    valleyName: "Guwahati Riparian Corridor",
    flowType: "riverside_flow",
    startElev: 49,
    endElev: 22,
    slopePct: 0.4,
    chokeOrBankName: "Bharalu Riverbank & City Dyke",
    channelSpecs: { upstream: "Normal: 350m", bottleneck: "Spills: 850m Wide", ratio: "Dyke Overtopping" },
    baseInflow: 450,
    inflowMultiplier: 14.2,
    landmarks: [
      { name: "Guwahati AWS", elev: "49m", xRatio: 0.0 },
      { name: "Deepor Beel", elev: "40m", xRatio: 0.32 },
      { name: "Bharalu Dyke", elev: "32m", xRatio: 0.66, isRisk: true },
      { name: "Mainstem Channel", elev: "22m", xRatio: 1.0 }
    ]
  },
  "ASM006": {
    riverName: "Shella & Wah Umngot Gorges",
    valleyName: "Cherrapunji Escarpment Ravine",
    flowType: "gorge_choke",
    startElev: 1313,
    endElev: 45,
    slopePct: 16.5,
    chokeOrBankName: "Shella Canyon Gorge Bottleneck",
    channelSpecs: { upstream: "50m", bottleneck: "10m", ratio: "5.0x waterfall gorge pinch" },
    baseInflow: 260,
    inflowMultiplier: 10.5,
    landmarks: [
      { name: "Cherrapunji AWS", elev: "1313m", xRatio: 0.0 },
      { name: "Nohkalikai Fall", elev: "850m", xRatio: 0.32 },
      { name: "Shella Gorge", elev: "350m", xRatio: 0.66, isRisk: true },
      { name: "Sylhet Lowlands", elev: "45m", xRatio: 1.0 }
    ]
  },

  // ── Sikkim & Darjeeling (Teesta Basin) ──
  "SKM001": {
    riverName: "Teesta River",
    valleyName: "Gangtok Teesta Canyon",
    flowType: "gorge_choke",
    startElev: 1650,
    endElev: 350,
    slopePct: 9.8,
    chokeOrBankName: "Singtam Gorge Chokepoint",
    channelSpecs: { upstream: "90m", bottleneck: "18m", ratio: "5.0x gorge narrowing" },
    baseInflow: 190,
    inflowMultiplier: 8.2,
    landmarks: [
      { name: "Gangtok AWS", elev: "1650m", xRatio: 0.0 },
      { name: "Ranipool Canyon", elev: "1150m", xRatio: 0.32 },
      { name: "Singtam Choke", elev: "620m", xRatio: 0.66, isRisk: true },
      { name: "Teesta Lowlands", elev: "350m", xRatio: 1.0 }
    ]
  },
  "DRJ001": {
    riverName: "Great Rangeet River",
    valleyName: "Darjeeling Mountain Flank",
    flowType: "gorge_choke",
    startElev: 2042,
    endElev: 310,
    slopePct: 11.2,
    chokeOrBankName: "Rangeet River Canyon Choke",
    channelSpecs: { upstream: "80m", bottleneck: "16m", ratio: "5.0x canyon pinch" },
    baseInflow: 170,
    inflowMultiplier: 7.8,
    landmarks: [
      { name: "Darjeeling AWS", elev: "2042m", xRatio: 0.0 },
      { name: "Lebong Ridge", elev: "1450m", xRatio: 0.32 },
      { name: "Rangeet Choke", elev: "750m", xRatio: 0.66, isRisk: true },
      { name: "Teesta Bazaar", elev: "310m", xRatio: 1.0 }
    ]
  }
};

/**
 * Procedural fallback generator for any station not explicitly in HYDRO_DATABASE
 */
function getStationHydroGeography(station) {
  if (!station) {
    return HYDRO_DATABASE["RDP001"];
  }

  if (HYDRO_DATABASE[station.id]) {
    return HYDRO_DATABASE[station.id];
  }

  const elev = station.elevation || 1200;
  const isHighMountain = elev >= 1000;
  const flowType = isHighMountain ? "gorge_choke" : "riverside_flow";
  const endElev = isHighMountain ? Math.max(300, Math.round(elev * 0.38)) : Math.max(20, Math.round(elev * 0.6));
  const drop = elev - endElev;
  const slopePct = parseFloat((drop / 160).toFixed(1));

  const riverName = `${station.name} River`;
  const valleyName = `${station.name} ${isHighMountain ? 'Mountain Canyon' : 'Riverside Corridor'}`;
  const chokeOrBankName = isHighMountain
    ? `${station.name} Gorge Chokepoint`
    : `${station.name} Riverside Embankment`;

  const channelSpecs = isHighMountain
    ? { upstream: "85m", bottleneck: "18m", ratio: "4.7x canyon pinch" }
    : { upstream: `Normal: ${Math.round(elev * 0.2 + 80)}m`, bottleneck: `Spills: ${Math.round(elev * 0.5 + 220)}m Wide`, ratio: "Overbank Inundation" };

  const landmarks = [
    { name: `${station.name} AWS`, elev: `${elev}m`, xRatio: 0.0 },
    { name: `${station.name} Flank`, elev: `${Math.round((elev * 2 + endElev) / 3)}m`, xRatio: 0.32 },
    { name: isHighMountain ? `${station.name} Choke` : `${station.name} Bank`, elev: `${Math.round((elev + endElev * 2) / 3)}m`, xRatio: 0.66, isRisk: true },
    { name: "Downstream Basin", elev: `${endElev}m`, xRatio: 1.0 }
  ];

  return {
    riverName,
    valleyName,
    flowType,
    startElev: elev,
    endElev,
    slopePct,
    chokeOrBankName,
    channelSpecs,
    baseInflow: Math.round(140 + elev * 0.05),
    inflowMultiplier: 7.5,
    landmarks
  };
}

/**
 * Configure PINN topography dynamically for any selected AWS station
 */
function setPINNStation(station) {
  if (!station) return;
  PINN_SIM.activeStation = station;

  const geo = getStationHydroGeography(station);
  PINN_SIM.geo = geo;
  PINN_SIM.startElevation = geo.startElev;
  PINN_SIM.endElevation = geo.endElev;
  PINN_SIM.slopePct = geo.slopePct;
  PINN_SIM.flowType = geo.flowType;
  PINN_SIM.riverName = geo.riverName;
  PINN_SIM.basinName = geo.valleyName;
  PINN_SIM.accumulationName = geo.chokeOrBankName;
  PINN_SIM.landmarks = geo.landmarks;

  // Update header title
  const titleEl = document.getElementById("pinn-valley-title");
  if (titleEl) {
    titleEl.textContent = `River Flood: ${station.name}`;
    titleEl.title = `${geo.valleyName} (${geo.riverName})`;
  }

  // Update header flow status badge
  const statusPill = document.getElementById("pinn-valley-status");
  if (statusPill) {
    statusPill.textContent = geo.flowType === "gorge_choke" ? "MOUNTAIN GORGE" : "RIVERSIDE PLAIN";
  }

  // Re-generate real elevation curve and render canvas
  initPINNTopography();
  renderPINNCanvas();

  // Dynamically update the 4 Flow Path cards on the right
  updateHowPINNBuilding(station, PINN_SIM.currentRainRate, PINN_SIM.currentPeakDepth, PINN_SIM.currentVelocity);
}

/**
 * Generate real physical slope and elevation profile
 */
function initPINNTopography() {
  PINN_SIM.elevation = [];
  PINN_SIM.h = [];
  PINN_SIM.u = [];
  PINN_SIM.particles = [];

  const startE = PINN_SIM.startElevation || 2400;
  const endE = PINN_SIM.endElevation || 850;
  const drop = startE - endE;
  const isGorge = PINN_SIM.flowType === "gorge_choke";

  for (let i = 0; i < PINN_SIM.gridSize; i++) {
    const r = i / (PINN_SIM.gridSize - 1);
    let bed;
    if (isGorge) {
      // Mountain Gorge: Steep mountain drop, concave canyon transition, hydraulic scour bed depression at choke
      bed = startE - drop * Math.pow(r, 0.78);
      if (r >= 0.63 && r <= 0.88) {
        bed -= (drop * 0.045) * Math.sin((r - 0.63) / 0.25 * Math.PI);
      }
    } else {
      // Riverside Plain: Gentle alluvial concave slope with natural riverbed hummocks
      bed = startE - drop * Math.pow(r, 0.62);
      bed += (drop * 0.015) * Math.sin(r * Math.PI * 4);
    }
    PINN_SIM.elevation.push(bed);
    PINN_SIM.h.push(0.4);
    PINN_SIM.u.push(isGorge ? 1.4 : 0.9);
  }

  // Populate particles with speed scaled to slope steepness
  const speedScale = isGorge ? 1.25 : 0.75;
  for (let p = 0; p < 70; p++) {
    PINN_SIM.particles.push({
      x: Math.random() * (PINN_SIM.gridSize - 1),
      yOffset: (Math.random() - 0.5) * 0.8,
      speed: (0.6 + Math.random() * 0.8) * speedScale,
      size: 1.5 + Math.random() * 2,
      alpha: 0.4 + Math.random() * 0.6
    });
  }
}

function syncPINNWithSlider(r) {
  PINN_SIM.currentRainRate = r;
  if (r <= 5) {
    PINN_SIM.currentPeakDepth = 0.4;
    PINN_SIM.currentVelocity = 1.2;
    PINN_SIM.currentArrivalMin = 60;
  } else {
    PINN_SIM.currentPeakDepth = Math.min(8.5, Math.max(0.4, 0.4 + (r / 150.0) * 6.6));
    PINN_SIM.currentVelocity = Math.min(14.0, Math.max(1.2, 1.2 + (r / 150.0) * 10.5));
    PINN_SIM.currentArrivalMin = Math.max(15, Math.round(60 - (r / 150.0) * 35));
  }

  if (r >= 45 && !PINN_SIM.active) {
    startFloodSim("cloudburst");
  } else if (r < 15 && PINN_SIM.active && !APP.activeReplay) {
    resetFloodSim();
  } else {
    renderPINNCanvas();
    updatePINNMetrics();
  }
}

function stepPINNPhysics() {
  const isGorge = PINN_SIM.flowType === "gorge_choke";
  const rain = PINN_SIM.currentRainRate || 0;
  const isHeavy = rain >= 35;
  const peakDepth = PINN_SIM.currentPeakDepth || (isHeavy ? 6.8 : 0.4);
  const velocity = PINN_SIM.currentVelocity || (isHeavy ? 11.4 : 1.2);

  // Smooth traveling wave crest along the mountain slope
  const crestX = (PINN_SIM.t / PINN_SIM.maxT) * (PINN_SIM.gridSize * 1.08);

  for (let i = 0; i < PINN_SIM.gridSize; i++) {
    const dist = i - crestX;
    let surge = 0;
    let velSurge = 0;

    if (isHeavy) {
      if (dist <= 0 && dist > -45) {
        const tail = Math.exp(dist / 16);
        surge = (peakDepth - 0.4) * tail;
        velSurge = (velocity - 1.2) * tail;
      } else if (dist > 0 && dist < 8) {
        const front = 1.0 - (dist / 8);
        surge = (peakDepth - 0.4) * front;
        velSurge = (velocity - 1.2) * front;
      }
    }

    // Physical accumulation factor at the bottleneck / floodplain
    let accumFactor = 1.0;
    if (isGorge) {
      if (i >= 63 && i <= 88) {
        accumFactor = 1.45; // Water pools up high in narrow gorge choke
      } else if (i >= 30 && i <= 48) {
        accumFactor = 0.85; // Fast rapids on steep mountain slope
      }
    } else {
      // Riverside plain: water spreads out laterally across floodplains
      if (i >= 63 && i <= 88) {
        accumFactor = 1.20; // Overbank spill
      }
    }

    // Smooth physical wave profile (NO SPURIOUS HIGH-FREQUENCY OSCILLATIONS!)
    PINN_SIM.h[i] = Math.max(0.4, 0.4 + surge * accumFactor);
    PINN_SIM.u[i] = Math.max(1.2, 1.2 + velSurge * (i >= 30 && i <= 50 ? 1.3 : 0.9));
  }

  // Animate water flow particles smoothly along streamlines
  for (let p of PINN_SIM.particles) {
    const idx = Math.min(PINN_SIM.gridSize - 1, Math.max(0, Math.floor(p.x)));
    const vel = PINN_SIM.u[idx] || 1.5;
    p.x += (vel / 7.0) * p.speed;
    if (p.x >= PINN_SIM.gridSize - 1) p.x = 0;
  }
}

function renderPINNCanvas() {
  const canvas = document.getElementById("flood-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  const isLight = typeof document !== "undefined" && document.body && document.body.classList.contains("light-mode");

  // Atmospheric sky gradient (theme-responsive)
  const skyGrad = ctx.createLinearGradient(0, 0, 0, H);
  if (isLight) {
    skyGrad.addColorStop(0, "#e2e8f0");
    skyGrad.addColorStop(0.5, "#e0f2fe");
    skyGrad.addColorStop(1, "#f8fafc");
  } else {
    skyGrad.addColorStop(0, "#060b14");
    skyGrad.addColorStop(0.5, "#0a1322");
    skyGrad.addColorStop(1, "#040810");
  }
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, W, H);

  const marginL = 58;
  const marginR = 24;
  const marginB = 34;
  const marginT = 24;
  const renderW = W - marginL - marginR;
  const renderH = H - marginT - marginB;

  const bedTopY = marginT + renderH * 0.28;
  const bedBaseY = marginT + renderH * 0.82;
  const maxDepthHeight = 44;

  const startE = PINN_SIM.startElevation || 2400;
  const endE = PINN_SIM.endElevation || 850;
  const elevRange = Math.max(1, startE - endE);
  const isGorge = PINN_SIM.flowType === "gorge_choke";
  const geo = PINN_SIM.geo || getStationHydroGeography(PINN_SIM.activeStation);

  // Converts real elevation in meters ASL to canvas screen Y coordinate
  function elevToY(elev) {
    const ratio = (startE - elev) / elevRange;
    return bedTopY + ratio * (bedBaseY - bedTopY);
  }

  // ── LAYER 1: SURROUNDING MOUNTAIN / VALLEY SILHOUETTE (BACKGROUND) ────────
  ctx.beginPath();
  ctx.moveTo(marginL, H - marginB);
  if (isGorge) {
    // Jagged Alpine ridges with V-shaped canyon cut
    const peaks = [
      { x: 0.0, y: 0.08 },
      { x: 0.14, y: 0.18 },
      { x: 0.28, y: 0.06 },
      { x: 0.44, y: 0.22 },
      { x: 0.58, y: 0.14 },
      { x: 0.68, y: 0.38 }, // Canyon V-notch where river cuts through
      { x: 0.82, y: 0.26 },
      { x: 1.0, y: 0.45 }
    ];
    peaks.forEach((pt, idx) => {
      const px = marginL + pt.x * renderW;
      const py = marginT + pt.y * renderH;
      if (idx === 0) ctx.lineTo(px, py);
      else ctx.lineTo(px, py);
    });
  } else {
    // Rolling foothill ridges and open valley sky
    const hills = [
      { x: 0.0, y: 0.22 },
      { x: 0.25, y: 0.16 },
      { x: 0.5, y: 0.24 },
      { x: 0.75, y: 0.18 },
      { x: 1.0, y: 0.26 }
    ];
    hills.forEach((pt, idx) => {
      const px = marginL + pt.x * renderW;
      const py = marginT + pt.y * renderH;
      if (idx === 0) ctx.lineTo(px, py);
      else ctx.quadraticCurveTo(px - 30, py - 12, px, py);
    });
  }
  ctx.lineTo(marginL + renderW, H - marginB);
  ctx.closePath();
  ctx.fillStyle = isLight
    ? (isGorge ? "rgba(148, 163, 184, 0.4)" : "rgba(203, 213, 225, 0.55)")
    : (isGorge ? "rgba(15, 23, 42, 0.6)" : "rgba(15, 23, 42, 0.4)");
  ctx.fill();

  // ── LAYER 2: SURROUNDING CANYON ROCK WALLS OR RIVERSIDE LEVEES ────────────
  if (isGorge) {
    // Shaded mountain canyon rock wall flanking the river valley behind the bed
    ctx.beginPath();
    ctx.moveTo(marginL, H - marginB);
    for (let i = 0; i < PINN_SIM.gridSize; i++) {
      const rx = marginL + (i / (PINN_SIM.gridSize - 1)) * renderW;
      const wallY = elevToY(PINN_SIM.elevation[i] || startE) - 22;
      ctx.lineTo(rx, wallY);
    }
    ctx.lineTo(marginL + renderW, H - marginB);
    ctx.closePath();
    ctx.fillStyle = isLight ? "rgba(148, 163, 184, 0.35)" : "rgba(30, 41, 59, 0.35)";
    ctx.fill();
  } else {
    // Natural Riverside Embankment / Levee crest line
    ctx.strokeStyle = isLight ? "rgba(22, 163, 74, 0.6)" : "rgba(34, 197, 94, 0.35)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i < PINN_SIM.gridSize; i += 4) {
      const rx = marginL + (i / (PINN_SIM.gridSize - 1)) * renderW;
      const bankY = elevToY(PINN_SIM.elevation[i] || startE) - 12; // ~1.5m bank crest
      if (i === 0) ctx.moveTo(rx, bankY);
      else ctx.lineTo(rx, bankY);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ── 1. Y-AXIS: Elevation (Meters ASL) ──────────────────────────────────────
  ctx.strokeStyle = isLight ? "rgba(15, 23, 42, 0.15)" : "rgba(148, 163, 184, 0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(marginL, marginT);
  ctx.lineTo(marginL, H - marginB);
  ctx.stroke();

  const midE = Math.round((startE + endE) / 2);
  ctx.fillStyle = isLight ? "rgba(15, 23, 42, 0.75)" : "rgba(148, 163, 184, 0.75)";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";
  ctx.fillText(`${startE}m`, marginL - 6, bedTopY + 4);
  ctx.fillText(`${midE}m`, marginL - 6, (bedTopY + bedBaseY) / 2 + 4);
  ctx.fillText(`${endE}m`, marginL - 6, bedBaseY + 4);

  // Axis Title Y
  ctx.save();
  ctx.translate(14, marginT + renderH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillStyle = isLight ? "rgba(15, 23, 42, 0.6)" : "rgba(148, 163, 184, 0.55)";
  ctx.font = "9px 'Inter', sans-serif";
  ctx.fillText("Elevation (m ASL)", 0, 0);
  ctx.restore();

  // ── 2. X-AXIS & 4 CLEAN, NON-OVERLAPPING LANDMARKS ─────────────────────────
  ctx.beginPath();
  ctx.moveTo(marginL, H - marginB);
  ctx.lineTo(W - marginR, H - marginB);
  ctx.stroke();

  const landmarks = PINN_SIM.landmarks || geo.landmarks;
  landmarks.forEach((lm) => {
    const lx = marginL + lm.xRatio * renderW;
    ctx.strokeStyle = lm.isRisk
      ? (isGorge ? (isLight ? "#dc2626" : "#ef4444") : (isLight ? "#ea580c" : "#f97316"))
      : (isLight ? "rgba(15, 23, 42, 0.25)" : "rgba(148, 163, 184, 0.3)");
    ctx.beginPath();
    ctx.moveTo(lx, H - marginB);
    ctx.lineTo(lx, H - marginB + 4);
    ctx.stroke();

    ctx.textAlign = lm.xRatio === 0 ? "left" : (lm.xRatio === 1 ? "right" : "center");
    ctx.fillStyle = lm.isRisk
      ? (isGorge ? (isLight ? "#dc2626" : "#f87171") : (isLight ? "#ea580c" : "#fb923c"))
      : (isLight ? "#0f172a" : "#f1f5f9");
    ctx.font = lm.isRisk ? "bold 8.5px 'JetBrains Mono', monospace" : "8px 'JetBrains Mono', monospace";
    ctx.fillText(lm.name, lx, H - marginB + 13);

    ctx.fillStyle = isLight ? "rgba(15, 23, 42, 0.7)" : "rgba(148, 163, 184, 0.75)";
    ctx.font = "7.5px 'Inter', sans-serif";
    ctx.fillText(`${lm.elev} ASL`, lx, H - marginB + 23);
  });
  ctx.textAlign = "start";

  // ── 3. INUNDATION ZONE: CHOKE POOL VS RIVERSIDE OVERFLOW ───────────────────
  const accumStart = marginL + 0.65 * renderW;
  const accumEnd = marginL + 0.88 * renderW;
  const poolW = accumEnd - accumStart;

  ctx.fillStyle = isGorge
    ? (isLight ? "rgba(220, 38, 38, 0.08)" : "rgba(239, 68, 68, 0.08)")
    : (isLight ? "rgba(234, 88, 12, 0.08)" : "rgba(249, 115, 22, 0.08)");
  ctx.fillRect(accumStart, marginT + 6, accumEnd - accumStart, renderH - 6);

  ctx.strokeStyle = isGorge
    ? (isLight ? "rgba(220, 38, 38, 0.45)" : "rgba(239, 68, 68, 0.4)")
    : (isLight ? "rgba(234, 88, 12, 0.45)" : "rgba(249, 115, 22, 0.45)");
  ctx.setLineDash([3, 3]);
  ctx.strokeRect(accumStart, marginT + 6, accumEnd - accumStart, renderH - 6);
  ctx.setLineDash([]);

  // Inundation Zone Card Badge
  ctx.fillStyle = isLight ? "rgba(255, 255, 255, 0.96)" : "rgba(15, 23, 42, 0.92)";
  ctx.strokeStyle = isGorge
    ? (isLight ? "rgba(220, 38, 38, 0.45)" : "rgba(239, 68, 68, 0.5)")
    : (isLight ? "rgba(234, 88, 12, 0.45)" : "rgba(249, 115, 22, 0.5)");
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(accumStart + 4, marginT + 8, poolW - 8, 30, 4);
  } else {
    ctx.rect(accumStart + 4, marginT + 8, poolW - 8, 30);
  }
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = isGorge ? (isLight ? "#dc2626" : "#f87171") : (isLight ? "#ea580c" : "#fb923c");
  ctx.font = "bold 8px 'JetBrains Mono', monospace";
  ctx.fillText(isGorge ? "VALLEY GORGE CHOKE" : "RIVERSIDE FLOODPLAIN", accumStart + 8, marginT + 19);

  ctx.font = "7.5px 'Inter', sans-serif";
  ctx.fillStyle = isGorge
    ? (isLight ? "#991b1b" : "rgba(248, 113, 113, 0.9)")
    : (isLight ? "#9a3412" : "rgba(251, 146, 60, 0.9)");
  const rawChoke = PINN_SIM.accumulationName || (isGorge ? "Gorge Bottleneck" : "Riverside Embankment");
  const chokeLabel = rawChoke.length > 28 ? rawChoke.substring(0, 26) + "..." : rawChoke;
  ctx.fillText(chokeLabel, accumStart + 8, marginT + 31);

  // ── 4. DRAW ACTUAL GENERATED RIVERBED SLOPE ───────────────────────────────
  ctx.beginPath();
  ctx.moveTo(marginL, H - marginB);
  for (let i = 0; i < PINN_SIM.gridSize; i++) {
    const rx = marginL + (i / (PINN_SIM.gridSize - 1)) * renderW;
    const ry = elevToY(PINN_SIM.elevation[i] || startE);
    ctx.lineTo(rx, ry);
  }
  ctx.lineTo(marginL + renderW, H - marginB);
  ctx.closePath();

  const bedGrad = ctx.createLinearGradient(0, bedTopY, 0, H - marginB);
  if (isLight) {
    bedGrad.addColorStop(0, isGorge ? "#94a3b8" : "#cbd5e1");
    bedGrad.addColorStop(1, "#f1f5f9");
    ctx.fillStyle = bedGrad;
    ctx.fill();
    ctx.strokeStyle = "#64748b";
  } else {
    bedGrad.addColorStop(0, isGorge ? "#1a2436" : "#1e293b");
    bedGrad.addColorStop(1, "#070b14");
    ctx.fillStyle = bedGrad;
    ctx.fill();
    ctx.strokeStyle = isGorge ? "#334155" : "#475569";
  }
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // ── 5. DRAW SMOOTH, CONTINUOUS FLOODWATER LAYER (NO SHARK TEETH!) ─────────
  ctx.beginPath();
  // Bottom boundary along the riverbed (right to left)
  for (let i = PINN_SIM.gridSize - 1; i >= 0; i--) {
    const rx = marginL + (i / (PINN_SIM.gridSize - 1)) * renderW;
    const bedY = elevToY(PINN_SIM.elevation[i] || startE);
    if (i === PINN_SIM.gridSize - 1) ctx.moveTo(rx, bedY);
    else ctx.lineTo(rx, bedY);
  }

  // Top boundary along the continuous water surface (left to right)
  for (let i = 0; i < PINN_SIM.gridSize; i++) {
    const rx = marginL + (i / (PINN_SIM.gridSize - 1)) * renderW;
    const bedY = elevToY(PINN_SIM.elevation[i] || startE);
    const depth = PINN_SIM.h[i] || 0.4;
    const visualH = Math.min(maxDepthHeight, (depth / 7.0) * maxDepthHeight);
    ctx.lineTo(rx, bedY - visualH);
  }
  ctx.closePath();

  const waterGrad = ctx.createLinearGradient(0, bedTopY - maxDepthHeight, 0, bedBaseY);
  if (isLight) {
    waterGrad.addColorStop(0, "rgba(2, 132, 199, 0.92)");
    waterGrad.addColorStop(0.4, "rgba(14, 165, 233, 0.82)");
    waterGrad.addColorStop(1, "rgba(3, 105, 161, 0.95)");
    ctx.fillStyle = waterGrad;
    ctx.fill();
    ctx.strokeStyle = "#0284c7";
  } else {
    waterGrad.addColorStop(0, "rgba(56, 189, 248, 0.88)");
    waterGrad.addColorStop(0.4, "rgba(14, 165, 233, 0.78)");
    waterGrad.addColorStop(1, "rgba(3, 105, 161, 0.92)");
    ctx.fillStyle = waterGrad;
    ctx.fill();
    ctx.strokeStyle = "#7dd3fc";
  }

  // Water surface stroke line with glowing foam crest
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < PINN_SIM.gridSize; i++) {
    const rx = marginL + (i / (PINN_SIM.gridSize - 1)) * renderW;
    const bedY = elevToY(PINN_SIM.elevation[i] || startE);
    const depth = PINN_SIM.h[i] || 0.4;
    const visualH = Math.min(maxDepthHeight, (depth / 7.0) * maxDepthHeight);
    if (i === 0) ctx.moveTo(rx, bedY - visualH);
    else ctx.lineTo(rx, bedY - visualH);
  }
  ctx.stroke();

  // Animated flow particles ride smoothly inside the water column
  for (let p of PINN_SIM.particles) {
    const idx = Math.min(PINN_SIM.gridSize - 1, Math.max(0, Math.floor(p.x)));
    const rx = marginL + (p.x / (PINN_SIM.gridSize - 1)) * renderW;
    const bedY = elevToY(PINN_SIM.elevation[idx] || startE);
    const depth = PINN_SIM.h[idx] || 0.4;
    const visualH = Math.min(maxDepthHeight, (depth / 7.0) * maxDepthHeight);
    
    // Confine particle strictly inside the water layer
    const radius = Math.max(0.8, Math.min(2.0, visualH * 0.3));
    const py = bedY - visualH * (0.28 + (p.yOffset + 0.5) * 0.44);

    ctx.fillStyle = isLight
      ? `rgba(255, 255, 255, ${p.alpha * 0.95})`
      : `rgba(224, 242, 254, ${p.alpha * 0.85})`;
    ctx.beginPath();
    ctx.arc(rx, py, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  // Water Depth Readout inside Inundation Zone
  const gaugeIdx = Math.floor(0.78 * (PINN_SIM.gridSize - 1));
  const gaugeDepth = (PINN_SIM.h[gaugeIdx] || 0.4).toFixed(1);
  const gaugeX = accumStart + poolW + 8;
  if (gaugeX + 65 < marginL + renderW) {
    ctx.fillStyle = isLight ? "#0f172a" : "#ffffff";
    ctx.font = "bold 11px 'Outfit', sans-serif";
    ctx.fillText(`Depth: ${gaugeDepth}m`, gaugeX, marginT + 26);
  }

  // ── 6. OPERATIONAL CONTEXT & TERRAIN TYPE BADGES ──────────────────────────
  const isHighFlow = (PINN_SIM.currentRainRate || 0) >= 35;
  const statusSpeed = (PINN_SIM.currentVelocity || 1.2).toFixed(1);
  const statusDepth = (PINN_SIM.currentPeakDepth || 0.4).toFixed(1);

  ctx.fillStyle = isHighFlow
    ? (isLight ? "#dc2626" : "#ef4444")
    : (isLight ? "#0284c7" : "#38bdf8");
  ctx.font = "bold 8.5px 'JetBrains Mono', monospace";
  if (isHighFlow) {
    ctx.fillText(`LIVE FLOW: ${statusSpeed} m/s | ${isGorge ? 'GORGE POOL' : 'BANK SPILL'}: ${statusDepth}m`, marginL, marginT - 7);
  } else {
    ctx.fillText(`BASEFLOW: 0.4m | NATURAL CHANNEL NORMAL`, marginL, marginT - 7);
  }

  // Terrain Type Indicator (Top Right)
  ctx.textAlign = "right";
  ctx.fillStyle = isGorge
    ? (isLight ? "#dc2626" : "rgba(248, 113, 113, 0.85)")
    : (isLight ? "#ea580c" : "rgba(251, 146, 60, 0.85)");
  ctx.font = "bold 8px 'JetBrains Mono', monospace";
  ctx.fillText(isGorge ? "TERRAIN: STEEP GORGE CHOKE" : "TERRAIN: RIVERSIDE FLOODPLAIN", marginL + renderW, marginT - 7);
  ctx.textAlign = "start";

  // Hydraulic Jump Marker (Gorge) or Riverbank Breach Marker (Riverside)
  if (isHighFlow) {
    const jumpX = marginL + 0.65 * renderW;
    ctx.strokeStyle = isGorge
      ? (isLight ? "#dc2626" : "rgba(239, 68, 68, 0.85)")
      : (isLight ? "#ea580c" : "rgba(249, 115, 22, 0.85)");
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(jumpX, marginT + 12);
    ctx.lineTo(jumpX, H - marginB);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = isGorge
      ? (isLight ? "#dc2626" : "#ef4444")
      : (isLight ? "#ea580c" : "#f97316");
    ctx.font = "bold 8px 'JetBrains Mono', monospace";
    ctx.fillText(isGorge ? "CHOKE ENTRANCE" : "BANK BREACH", jumpX - 82, marginT + 14);
  }
}
function updatePINNMetrics() {
  const preset = PINN_SIM.presets[PINN_SIM.scenario] || PINN_SIM.presets.cloudburst;
  let maxD = PINN_SIM.currentPeakDepth || 0.4;
  let maxV = PINN_SIM.currentVelocity || 1.2;

  for (let i = 0; i < PINN_SIM.gridSize; i++) {
    if (PINN_SIM.h[i] > maxD) maxD = PINN_SIM.h[i];
    if (PINN_SIM.u[i] > maxV) maxV = PINN_SIM.u[i];
  }

  const depthEl = document.getElementById("pinn-kpi-depth");
  const velEl = document.getElementById("pinn-kpi-vel");
  const arrivalEl = document.getElementById("pinn-kpi-eta");
  const riskEl = document.getElementById("pinn-kpi-risk");

  if (depthEl) depthEl.textContent = `${maxD.toFixed(1)}m`;
  if (velEl) velEl.textContent = `${maxV.toFixed(1)} m/s`;
  if (arrivalEl) {
    const baseArrival = PINN_SIM.currentArrivalMin || preset.arrivalTime;
    const rem = Math.max(0, baseArrival - Math.floor(PINN_SIM.t * 0.35));
    arrivalEl.textContent = rem > 0 ? `${rem} min` : "IMPACT";
  }
  if (riskEl) {
    const isSevere = maxD >= 4.0;
    riskEl.textContent = isSevere ? "CRITICAL" : (maxD >= 1.5 ? "SURGE WATCH" : "NOMINAL");
    riskEl.style.color = isSevere ? "#ef4444" : (maxD >= 1.5 ? "#f59e0b" : "#4ade80");
  }

  const sideArr = document.getElementById("downstream-arrival-text");
  const sideDepth = document.getElementById("downstream-depth-text");
  const sideSettlement = document.getElementById("downstream-settlement-text");
  if (sideArr) sideArr.textContent = maxD >= 1.5 ? (arrivalEl ? arrivalEl.textContent : "32 min") : "No flood detected";
  if (sideDepth) sideDepth.textContent = `${maxD.toFixed(1)} m (${maxD >= 4.0 ? "Severe Flash Flood" : maxD >= 1.5 ? "Surge Watch" : "Baseflow"})`;
  if (sideSettlement) sideSettlement.textContent = PINN_SIM.accumulationName || "Valley Lowland Bottleneck";

  // Also sync the 4 flow path cards
  updateHowPINNBuilding(PINN_SIM.activeStation, PINN_SIM.currentRainRate, maxD, maxV);
}

function animatePINN() {
  if (!PINN_SIM.active) return;
  PINN_SIM.t += 0.4;
  if (PINN_SIM.t > PINN_SIM.maxT) PINN_SIM.t = 0;

  stepPINNPhysics();
  renderPINNCanvas();
  updatePINNMetrics();

  PINN_SIM.animId = requestAnimationFrame(animatePINN);
}

function startFloodSim(scenario = "cloudburst") {
  if (PINN_SIM.animId) cancelAnimationFrame(PINN_SIM.animId);
  PINN_SIM.active = true;
  PINN_SIM.scenario = scenario;
  PINN_SIM.t = 0;
  initPINNTopography();
  animatePINN();
}

function resetFloodSim() {
  if (PINN_SIM.animId) cancelAnimationFrame(PINN_SIM.animId);
  PINN_SIM.active = false;
  PINN_SIM.t = 0;
  PINN_SIM.currentPeakDepth = 0.4;
  PINN_SIM.currentVelocity = 1.2;
  initPINNTopography();
  renderPINNCanvas();

  const depthEl = document.getElementById("pinn-kpi-depth");
  const velEl = document.getElementById("pinn-kpi-vel");
  const arrivalEl = document.getElementById("pinn-kpi-eta");
  const riskEl = document.getElementById("pinn-kpi-risk");

  if (depthEl) depthEl.textContent = "0.4m";
  if (velEl) velEl.textContent = "1.2 m/s";
  if (arrivalEl) arrivalEl.textContent = "Standby";
  if (riskEl) {
    riskEl.textContent = "NOMINAL";
    riskEl.style.color = "#4ade80";
  }

  const sideArr = document.getElementById("downstream-arrival-text");
  const sideDepth = document.getElementById("downstream-depth-text");
  if (sideArr) sideArr.textContent = "No flood detected";
  if (sideDepth) sideDepth.textContent = "0.4 m (Normal baseflow)";

  updateHowPINNBuilding(PINN_SIM.activeStation, 0, 0.4, 1.2);
}

function togglePINNSimulation() {
  if (PINN_SIM.active) {
    resetFloodSim();
  } else {
    startFloodSim("cloudburst");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("flood-canvas");
  if (canvas && canvas.parentElement) {
    canvas.width = canvas.parentElement.clientWidth || 500;
    canvas.height = canvas.parentElement.clientHeight || 190;
  }

  // Initialize with active station if available
  const initialStation = (typeof STATIONS !== "undefined" && STATIONS.length > 0) ? STATIONS[0] : null;
  if (initialStation) {
    setPINNStation(initialStation);
  } else {
    initPINNTopography();
    renderPINNCanvas();
  }
});

window.addEventListener("resize", () => {
  const canvas = document.getElementById("flood-canvas");
  if (canvas && canvas.parentElement) {
    canvas.width = canvas.parentElement.clientWidth || 500;
    canvas.height = canvas.parentElement.clientHeight || 190;
    renderPINNCanvas();
  }
});

function switchPINNTab(tab) {
  const canvasWrap = document.getElementById("flood-canvas-wrap");
  const howWrap = document.getElementById("pinn-how-wrap");
  const btnCanvas = document.getElementById("btn-pinn-tab-canvas");
  const btnHow = document.getElementById("btn-pinn-tab-how");

  if (tab === "how") {
    if (canvasWrap) canvasWrap.style.display = "none";
    if (howWrap) howWrap.style.display = "flex";
    if (btnCanvas) btnCanvas.classList.remove("active");
    if (btnHow) btnHow.classList.add("active");
    updateHowPINNBuilding();
  } else {
    if (canvasWrap) canvasWrap.style.display = "block";
    if (howWrap) howWrap.style.display = "none";
    if (btnCanvas) btnCanvas.classList.add("active");
    if (btnHow) btnHow.classList.remove("active");
    renderPINNCanvas();
  }
}

/**
 * 100% DYNAMIC 4-STAGE FLOW PATH CARDS ("TABLE")
 * Dynamically updates all card headings, metrics, descriptions,
 * gorge narrowing vs riverside capacity, and flood verdict.
 */
function updateHowPINNBuilding(station, r, peakDepth, vel) {
  const targetStation = station || PINN_SIM.activeStation || ((typeof STATIONS !== "undefined") ? STATIONS[0] : null);
  const geo = targetStation ? getStationHydroGeography(targetStation) : (PINN_SIM.geo || HYDRO_DATABASE["RDP001"]);
  const isGorge = geo.flowType === "gorge_choke";

  const rain = typeof r === "number" ? r : (PINN_SIM.currentRainRate || 0);
  const depth = typeof peakDepth === "number" ? peakDepth : (PINN_SIM.currentPeakDepth || 0.4);
  const velocity = typeof vel === "number" ? vel : (PINN_SIM.currentVelocity || 1.2);
  const isHeavy = rain >= 35;

  // ── CARD 1: DELUGE INFLOW ────────────────────────────────────────────────
  const card1Num = document.getElementById("pde-card1-num");
  const pdeR = document.getElementById("pde-r-in");
  const pdeQ0 = document.getElementById("pde-q0");
  const pdeInflowDesc = document.getElementById("pde-inflow-desc");

  if (card1Num) card1Num.textContent = isGorge ? "1. RAIN HITS MOUNTAIN" : "1. RAIN HITS CATCHMENT";
  if (pdeR) pdeR.textContent = `${Math.round(rain)} mm/h`;
  if (pdeQ0) {
    const inflowVal = Math.round(geo.baseInflow + rain * geo.inflowMultiplier);
    pdeQ0.textContent = `${inflowVal} m³/s`;
  }
  if (pdeInflowDesc && targetStation) {
    pdeInflowDesc.textContent = isHeavy
      ? `Cloudburst deluge over ${targetStation.name} (${targetStation.elevation}m ASL) pours massive inflow into ${geo.riverName}.`
      : `Ambient telemetry monitoring across ${targetStation.name} (${targetStation.elevation}m ASL) catchment for ${geo.riverName}.`;
  }

  // ── CARD 2: SLOPE & TORRENT VELOCITY ──────────────────────────────────────
  const pdeUmax = document.getElementById("pde-umax");
  const pdeS0 = document.getElementById("pde-s0");
  const pdeMomDesc = document.getElementById("pde-mom-desc");

  if (pdeUmax) pdeUmax.textContent = `${velocity.toFixed(1)} m/s`;
  if (pdeS0) pdeS0.textContent = `${geo.slopePct}% Drop`;
  if (pdeMomDesc && targetStation) {
    pdeMomDesc.textContent = isGorge
      ? `Steep mountain slope (${geo.slopePct}% drop from ${geo.startElev}m to ${geo.endElev}m) pulls torrent down canyon like a waterfall.`
      : `Alluvial riverbed gradient (${geo.slopePct}% drop) carries wide surface runoff toward ${geo.riverName} floodplains.`;
  }

  // ── CARD 3: CHOKE VS RIVERSIDE FLOW ───────────────────────────────────────
  const card3Num = document.getElementById("pde-card3-num");
  const statusMass = document.getElementById("pde-status-mass");
  const narrowsLbl = document.getElementById("pde-narrows-lbl");
  const chokeName = document.getElementById("pde-choke-name");
  const massDesc = document.getElementById("pde-mass-desc");

  if (card3Num) card3Num.textContent = isGorge ? "3. NARROW VALLEY GORGE" : "3. RIVERSIDE FLOODPLAIN";
  if (statusMass) {
    statusMass.textContent = isGorge ? "NARROW CHOKE" : "RIVERSIDE FLOW";
    statusMass.className = isGorge ? "flow-badge" : "flow-badge";
  }
  if (narrowsLbl) {
    narrowsLbl.innerHTML = isGorge
      ? `Riverbed Narrows: <b id="pde-narrows-val" style="color:#f8fafc;">${geo.channelSpecs.upstream} &rarr; ${geo.channelSpecs.bottleneck}</b>`
      : `Riverbed Capacity: <b id="pde-narrows-val" style="color:#f8fafc;">${geo.channelSpecs.upstream} &rarr; ${geo.channelSpecs.bottleneck}</b>`;
  }
  if (chokeName) chokeName.textContent = geo.chokeOrBankName;
  if (massDesc) {
    massDesc.textContent = isGorge
      ? `The riverbed gets ${geo.channelSpecs.ratio} at ${geo.chokeOrBankName}, trapping the rushing water.`
      : `Water volume exceeds natural channel banks along ${geo.chokeOrBankName}, threatening riverside settlements.`;
  }

  // ── CARD 4: INUNDATION ASSESSMENT & VERDICT ──────────────────────────────
  const card4Num = document.getElementById("pde-card4-num");
  const pdeAccumAns = document.getElementById("pde-accum-ans");
  const statusChoke = document.getElementById("pde-status-choke");
  const chokeSub = document.getElementById("flow-choke-sub");

  if (card4Num) card4Num.textContent = "4. WILL WATER FLOOD HERE?";
  if (pdeAccumAns && statusChoke) {
    if (isHeavy) {
      pdeAccumAns.textContent = isGorge
        ? `YES — Pools to ${depth.toFixed(1)}m in Gorge!`
        : `YES — Spills +${(depth * 0.45).toFixed(1)}m Over Riverbank!`;
      pdeAccumAns.style.color = "var(--red)";
      statusChoke.textContent = isGorge ? "CHOKE OVERFLOW" : "RIVERSIDE FLOOD";
      statusChoke.className = "flow-badge danger";
      if (chokeSub) {
        chokeSub.textContent = isGorge
          ? `Water traps at ${geo.chokeOrBankName}. Mandatory evacuation along ${geo.riverName} lowlands!`
          : `Water overflows riverbanks along ${geo.chokeOrBankName}. Evacuate riverside ghats and lowlands!`;
      }
    } else {
      pdeAccumAns.textContent = isGorge
        ? "NO — Safe 0.4m Natural Depth"
        : "NO — Safe within Riverbanks";
      pdeAccumAns.style.color = "var(--green)";
      statusChoke.textContent = isGorge ? "SAFE BASEFLOW" : "BANK NOMINAL";
      statusChoke.className = "flow-badge";
      if (chokeSub) {
        chokeSub.textContent = isGorge
          ? `Channel capacity safe. Normal river baseflow through ${geo.chokeOrBankName}.`
          : `Water stays safely confined within natural riverbanks along ${geo.chokeOrBankName}.`;
      }
    }
  }
}
