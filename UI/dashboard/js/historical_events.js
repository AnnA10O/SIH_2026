/**
 * historical_events.js
 * Pre-computed disaster scenarios for historical replay and flash flood accumulation simulation.
 * Kedarnath 2013, Chamoli 2021, Assam Monsoon 2014
 */

const HISTORICAL_EVENTS = {

  kedarnath: {
    name: "Kedarnath Cloudburst & Flash Flood — June 16, 2013",
    epicenter: { lat: 30.734, lon: 79.066, stationId: "RDP001" },
    hazard: "cloudburst",
    tier: "severe",
    // Downstream water accumulation corridor along Mandakini River valley
    accumulation_zone: {
      name: "Mandakini River Valley Inundation Corridor",
      basin: "Mandakini / Rudraprayag",
      peak_accumulation_depth: 6.8, // meters
      estimated_inundation_area_km2: 18.2,
      critical_settlements: ["Kedarnath Shrine Base", "Rambara Gorge", "Gaurikund", "Sonprayag Confluence"],
      // Polygon coords tracing downstream valley water accumulation
      polygon: [
        [30.740, 79.060],
        [30.734, 79.070],
        [30.680, 79.055],
        [30.640, 79.030],
        [30.624, 79.003], // Sonprayag
        [30.615, 78.990],
        [30.625, 78.980],
        [30.650, 79.010],
        [30.700, 79.040],
        [30.745, 79.050]
      ]
    },
    stations_override: {
      "RDP001": { R: 148.5, R30: 231.2, R60: 298.6, RI: 42.3, L_score: 0.83, P_cb: 0.94, tier: "severe", snn_state: "ACTIVE" },
      "RDP002": { R: 91.2,  R30: 148.0, R60: 192.4, RI: 28.6, L_score: 0.71, P_cb: 0.87, tier: "warning", snn_state: "ACTIVE" },
      "RDP003": { R: 38.4,  R30: 64.5,  R60: 82.3,  RI: 11.2, L_score: 0.42, P_cb: 0.63, tier: "watch",   snn_state: "DORMANT" },
      "RDP004": { R: 62.8,  R30: 98.5,  R60: 127.6, RI: 18.4, L_score: 0.61, P_cb: 0.79, tier: "warning", snn_state: "ACTIVE" },
      "CHM001": { R: 22.4,  R30: 36.8,  R60: 47.5,  RI: 6.1,  L_score: 0.28, P_cb: 0.38, tier: "normal",  snn_state: "DORMANT" },
      "UTK001": { R: 14.2,  R30: 22.8,  R60: 29.4,  RI: 3.8,  L_score: 0.18, P_cb: 0.22, tier: "normal",  snn_state: "DORMANT" },
    },
    xai_log: [
      { title: "[SNN Gate] SNN Cloudburst Gate — FIRED (RDP001 Kedarnath)", body: "Rain surge ΔR=42.3 mm/h | V_mem=1.48 > θ=1.00\nTotal spikes: 4/5 | Fast LIF decay β=0.50 (15 min window)" },
      { title: "[1D-CNN+BiLSTM] Inference — SEVERE CLOUDBURST", body: "P(Cloudburst) = 94.2%\nThreshold exceeded (τ=0.15)\nNeural Saliency: R=0.31 | RI=0.29 | R60=0.24 | R30=0.16" },
      { title: "[Spatial Gate] Tier-2 Spatial Gate — CONFIRMED", body: "L-Score = 0.83 (threshold: 0.40)\nR_core = 148.5 mm/h vs R_bg = 25.2 mm/h\nRegime: CONFIRMED_CLOUDBURST (Hyper-localized mountain cell)" },
      { title: "[PINN SWE] Hydro Handoff — Water Accumulation Triggered", body: "Peak accumulation depth: 6.8 m at Sonprayag valley bottleneck\nFlow velocity: 11.4 m/s | Flash flood arrival: 32 min\nInundation corridor: Mandakini riverbed (18.2 km²)" },
      { title: "[Emergency Alert] TIER-4 RED ALERT", body: "Immediate riverbank evacuation ordered.\nAction window: 2 to 4 hours lead time." }
    ],
    alert_message: "[CRITICAL] SEVERE CLOUDBURST & FLASH FLOOD\nMandakini Basin · Kedarnath-Sonprayag Corridor",
    pinn_trigger: { rain_rate: 148.5, hazard: "cloudburst", peak_depth: 6.8, velocity: 11.4 }
  },

  chamoli: {
    name: "Chamoli Flash Flood — February 7, 2021",
    epicenter: { lat: 30.559, lon: 79.564, stationId: "CHM001" },
    hazard: "thunderstorm",
    tier: "severe",
    accumulation_zone: {
      name: "Rishi Ganga & Alaknanda Gorge Accumulation Basin",
      basin: "Alaknanda / Chamoli",
      peak_accumulation_depth: 5.4,
      estimated_inundation_area_km2: 14.1,
      critical_settlements: ["Raini Village Confluence", "Tapovan Barrage Site", "Joshimath Lowland"],
      polygon: [
        [30.580, 79.580],
        [30.560, 79.570],
        [30.510, 79.540],
        [30.480, 79.480],
        [30.403, 79.329], // Chamoli Town
        [30.410, 79.310],
        [30.500, 79.460],
        [30.550, 79.530],
        [30.590, 79.560]
      ]
    },
    stations_override: {
      "CHM001": { R: 112.4, R30: 178.6, R60: 231.8, RI: 35.2, L_score: 0.78, P_cb: 0.91, tier: "severe",  snn_state: "ACTIVE" },
      "CHM002": { R: 84.6,  R30: 134.5, R60: 174.2, RI: 26.3, L_score: 0.69, P_cb: 0.85, tier: "warning", snn_state: "ACTIVE" },
      "CHM003": { R: 48.2,  R30: 76.4,  R60: 98.8,  RI: 14.2, L_score: 0.51, P_cb: 0.72, tier: "warning", snn_state: "ACTIVE" },
      "CHM004": { R: 28.4,  R30: 45.8,  R60: 59.2,  RI: 8.6,  L_score: 0.36, P_cb: 0.54, tier: "watch",   snn_state: "DORMANT" },
      "RDP001": { R: 31.2,  R30: 50.4,  R60: 65.2,  RI: 9.4,  L_score: 0.33, P_cb: 0.48, tier: "watch",   snn_state: "DORMANT" },
    },
    xai_log: [
      { title: "[SNN Gate] SNN Slow Gate — FIRED (CHM001 Joshimath)", body: "Pressure drop = 2.8 hPa/h | IWV trend = 4.2 mm/h\nV_mem=1.22 > θ=1.00 | Pre-convective buildup detected" },
      { title: "[1D-CNN+BiLSTM] TIER ESCALATION", body: "P(Nowcast) = 91.2% | Level: SEVERE FLASH FLOOD" },
      { title: "[PINN SWE] Alaknanda Gorge Water Accumulation", body: "Peak water accumulation: 5.4 m at Tapovan barrage confluence\nFlow velocity: 9.2 m/s | Extent: ~14.1 km²" }
    ],
    alert_message: "[CRITICAL] FLASH FLOOD ALERT\nRishi Ganga & Alaknanda Gorges · Tapovan-Joshimath",
    pinn_trigger: { rain_rate: 112.4, hazard: "thunderstorm", peak_depth: 5.4, velocity: 9.2 }
  },

  assam: {
    name: "Assam Synoptic Monsoon — August 3, 2014",
    epicenter: { lat: 26.637, lon: 92.794, stationId: "ASM002" },
    hazard: "thunderstorm",
    tier: "watch",
    accumulation_zone: {
      name: "Brahmaputra Floodplain Uniform Sheet Flow",
      basin: "Brahmaputra Valley",
      peak_accumulation_depth: 2.1,
      estimated_inundation_area_km2: 45.0,
      critical_settlements: ["Tezpur Riparian Lowlands", "Jorhat Floodplain"],
      polygon: [
        [26.70, 92.65],
        [26.75, 93.10],
        [26.60, 93.40],
        [26.45, 92.90],
        [26.50, 92.60]
      ]
    },
    stations_override: {
      "ASM001": { R: 42.8, R30: 65.2, R60: 88.4,  RI: 4.2,  L_score: 0.11, P_cb: 0.28, tier: "normal", snn_state: "DORMANT" },
      "ASM002": { R: 48.4, R30: 72.8, R60: 95.5,  RI: 5.6,  L_score: 0.12, P_cb: 0.32, tier: "watch",  snn_state: "DORMANT" },
      "ASM003": { R: 44.4, R30: 68.8, R60: 91.0,  RI: 4.1,  L_score: 0.08, P_cb: 0.26, tier: "normal", snn_state: "DORMANT" },
      "ASM006": { R: 52.4, R30: 78.2, R60: 104.6, RI: 6.8,  L_score: 0.14, P_cb: 0.36, tier: "watch",  snn_state: "DORMANT" },
      "ASM007": { R: 46.2, R30: 70.8, R60: 93.8,  RI: 4.8,  L_score: 0.09, P_cb: 0.29, tier: "normal", snn_state: "DORMANT" },
    },
    xai_log: [
      { title: "[Synoptic] Monsoon Evaluation", body: "Widespread moderate rainfall detected across 10 stations (mean=46 mm/h).\nSpatial gradient is negligible (R_core=48.4 vs R_bg=44.1)." },
      { title: "[Spatial Gate] Tier-2 Confirmation — REJECTED", body: "L-Score = 0.12 < Threshold 0.40\nRegime: WIDESPREAD_MONSOON\nResult: Cloudburst false alarm successfully suppressed! No emergency panic issued." }
    ],
    alert_message: "[WATCH] ROUTINE MONSOON WATCH\nUniform Synoptic Rain · No Cloudburst Anomaly",
    pinn_trigger: { rain_rate: 48.4, hazard: "thunderstorm", peak_depth: 2.1, velocity: 4.2 }
  }
};
