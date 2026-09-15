/**
 * spatial_gate.js
 * Tier-2 Multi-Station Spatial Confirmation Gate
 * Computes localized precipitation gradient between core candidate station
 * and neighboring AWS stations in concentric spatial annuli (0-15km vs 15-50km).
 * 
 * L-Score Formulation:
 *   L = (R_core - R_bg) / max(R_core, 1)
 */

function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

function computeSpatialGate(coreStation, allStations) {
  if (!coreStation) {
    return { lScore: 0, rCore: 0, rBg: 0, regime: "STANDBY", neighbors: [], color: "#94a3b8" };
  }

  const rCore = coreStation.R || 0;
  const neighbors = [];

  for (const s of allStations) {
    if (s.id === coreStation.id) continue;
    const dist = haversineDistance(coreStation.lat, coreStation.lon, s.lat, s.lon);
    if (dist <= 65) {
      neighbors.push({
        id: s.id,
        name: s.name,
        dist: dist,
        R: s.R || 0,
        lat: s.lat,
        lon: s.lon
      });
    }
  }

  if (neighbors.length === 0) {
    const fallback = allStations
      .filter(s => s.id !== coreStation.id)
      .map(s => ({
        id: s.id,
        name: s.name,
        dist: haversineDistance(coreStation.lat, coreStation.lon, s.lat, s.lon),
        R: s.R || 0,
        lat: s.lat,
        lon: s.lon
      }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 4);
    neighbors.push(...fallback);
  }

  let sumWeight = 0, weightedR = 0;
  for (const n of neighbors) {
    const weight = 1 / Math.max(n.dist, 2);
    weightedR += n.R * weight;
    sumWeight += weight;
  }

  const rBg = sumWeight > 0 ? (weightedR / sumWeight) : 0;
  const lScore = Math.max(0, Math.min(1.0, (rCore - rBg) / Math.max(rCore, 1.0)));

  let regime = "WIDESPREAD";
  let color = "#38bdf8";

  if (lScore >= 0.40 && rCore >= 30) {
    regime = "CONFIRMED";
    color = "#ef4444";
  } else if (lScore >= 0.20 || (rCore >= 20 && lScore >= 0.15)) {
    regime = "ELEVATED";
    color = "#f59e0b";
  }

  return {
    lScore: parseFloat(lScore.toFixed(3)),
    rCore: parseFloat(rCore.toFixed(1)),
    rBg: parseFloat(rBg.toFixed(1)),
    regime,
    color,
    neighbors: neighbors.slice(0, 5)
  };
}

function drawAnnulusCanvas(canvasId, coreStation, spatialResult) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;

  ctx.clearRect(0, 0, w, h);

  const maxR = Math.min(cx, cy) - 18;
  const rCore = maxR * 0.35;
  const rOuter = maxR * 0.88;

  // Concentric rings
  ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);

  ctx.beginPath();
  ctx.arc(cx, cy, rOuter, 0, Math.PI * 2);
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
  ctx.beginPath();
  ctx.arc(cx, cy, rCore, 0, Math.PI * 2);
  ctx.stroke();

  // Core fill
  ctx.fillStyle = spatialResult.color ? `${spatialResult.color}22` : "rgba(6, 182, 212, 0.1)";
  ctx.beginPath();
  ctx.arc(cx, cy, rCore, 0, Math.PI * 2);
  ctx.fill();

  // Draw neighbors
  if (spatialResult.neighbors) {
    spatialResult.neighbors.forEach((n, idx) => {
      let angle = (idx / spatialResult.neighbors.length) * Math.PI * 2;
      const distRatio = Math.min(1.0, Math.max(0.2, n.dist / 60));
      const nr = rCore + distRatio * (rOuter - rCore);
      const nx = cx + Math.cos(angle) * nr;
      const ny = cy + Math.sin(angle) * nr;

      ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nx, ny);
      ctx.stroke();

      ctx.fillStyle = n.R > 10 ? "#fbbf24" : "#64748b";
      ctx.beginPath();
      ctx.arc(nx, ny, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "rgba(226, 232, 240, 0.6)";
      ctx.font = "8px 'JetBrains Mono', monospace";
      ctx.fillText(`${n.id.slice(0, 6)}: ${n.R}`, nx + 5, ny + 3);
    });
  }

  // Core center point
  ctx.fillStyle = spatialResult.color || "#ef4444";
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 9px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  const coreName = coreStation ? coreStation.name.split(" ")[0] : "CORE";
  ctx.fillText(coreName, cx, cy + 15);
  ctx.textAlign = "start";
}

function updateSpatialGateUI(stationOrId) {
  let station = stationOrId;
  if (typeof stationOrId === "string" && typeof STATIONS !== "undefined") {
    station = STATIONS.find(s => s.id === stationOrId);
  }
  if (!station && typeof STATIONS !== "undefined" && STATIONS.length > 0) {
    station = STATIONS[0];
  }

  const allStations = typeof STATIONS !== "undefined" ? STATIONS : [];
  const result = computeSpatialGate(station, allStations);

  const lScoreEl = document.getElementById("sp-l-score");
  const regimeEl = document.getElementById("sp-regime");

  if (lScoreEl) {
    lScoreEl.textContent = result.lScore.toFixed(2);
    lScoreEl.style.color = result.color;
  }
  if (regimeEl) {
    regimeEl.textContent = result.regime;
    regimeEl.style.color = result.color;
  }

  drawAnnulusCanvas("annulus-canvas", station, result);
}
