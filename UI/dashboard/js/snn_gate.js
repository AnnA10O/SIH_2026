/**
 * snn_gate.js
 * Dual-Timescale Neuromorphic Leaky Integrate-and-Fire (LIF) Edge Gate
 * Cloudburst Fast Gate (tau=15m, beta=0.50) & Thunderstorm Slow Gate (tau=60m, beta=0.88)
 */

const SNN_CONFIGS = {
  cloudburst: {
    beta: 0.50,
    v_thresh: 1.00,
    num_steps: 5,
    in_features: 4,
    hidden: 12,
    scales: [100.0, 60.0, 30.0, 40.0],
    weight_seeds: [0.8, 0.9, 1.4, 1.6],
    fc1_bias: -0.4,
    fc2_weight: 0.6,
    fc2_bias: -0.2,
  },
  thunderstorm: {
    beta: 0.88,
    v_thresh: 1.00,
    num_steps: 5,
    in_features: 5,
    hidden: 12,
    scales: [5.0, 3.0, 10.0, 4.0, 500.0],
    weight_seeds: [1.3, 1.4, 0.8, 0.9, 1.2],
    fc1_bias: -0.5,
    fc2_weight: 0.55,
    fc2_bias: -0.25,
  }
};

function buildSNNWeights(cfg) {
  const w1 = [];
  for (let i = 0; i < cfg.hidden; i++) {
    const row = [];
    for (let j = 0; j < cfg.in_features; j++) {
      row.push(cfg.weight_seeds[j] + 0.2 * Math.sin(i * (j + 1) * 0.5));
    }
    w1.push(row);
  }
  return w1;
}

function lifStep(v_mem, input_current, beta, v_thresh) {
  const v_new = beta * v_mem + input_current;
  const fired = v_new >= v_thresh;
  const v_reset = fired ? v_new - v_thresh : v_new;
  return [fired, Math.max(0, Math.min(v_reset, 3.0))];
}

function runSNN(type, raw_vals) {
  const cfg = SNN_CONFIGS[type] || SNN_CONFIGS.cloudburst;
  const w1 = buildSNNWeights(cfg);

  // Normalize inputs
  const norm_vals = raw_vals.map((v, i) => Math.min(3.0, Math.max(0.0, (v || 0) / cfg.scales[i])));

  // Compute input current
  let total_current = 0;
  for (let i = 0; i < cfg.hidden; i++) {
    let sum = cfg.fc1_bias;
    for (let j = 0; j < cfg.in_features; j++) {
      sum += (w1[i][j] || 0.8) * (norm_vals[j] || 0);
    }
    total_current += Math.max(0, sum) * (cfg.fc2_weight / cfg.hidden);
  }
  total_current += cfg.fc2_bias;

  // Simulate membrane trajectory over timesteps
  let v_mem = 0;
  let total_spikes = 0;
  const mem_history = [0];

  for (let t = 0; t < cfg.num_steps; t++) {
    const [fired, new_v] = lifStep(v_mem, total_current, cfg.beta, cfg.v_thresh);
    if (fired) total_spikes++;
    v_mem = new_v;
    mem_history.push(v_mem);
  }

  const peak_mem = Math.max(...mem_history);
  const fired = total_spikes > 0;

  return { fired, total_spikes, mem_history, peak_mem };
}

function drawMembrane(canvasId, mem_history, beta, fired, v_thresh = 1.0, strokeColor = "#c084fc") {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let y = 0.25; y <= 1.0; y += 0.25) {
    const py = H - y * (H - 20) - 10;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(W, py);
    ctx.stroke();
  }

  // Threshold Line θ = 1.0
  const threshY = H - (v_thresh / 1.5) * (H - 20) - 10;
  ctx.strokeStyle = "rgba(239, 68, 68, 0.45)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, threshY);
  ctx.lineTo(W, threshY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Membrane trace line
  const pts = mem_history || [0];
  const xStep = W / (pts.length - 1 || 1);

  ctx.strokeStyle = fired ? strokeColor : "rgba(148, 163, 184, 0.5)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  pts.forEach((v, i) => {
    const x = i * xStep;
    const y = H - (v / 1.5) * (H - 20) - 10;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Spikes indicators
  if (fired) {
    ctx.fillStyle = strokeColor;
    pts.forEach((v, i) => {
      if (v >= v_thresh * 0.9) {
        const x = i * xStep;
        ctx.beginPath();
        ctx.arc(x, 12, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }
}

let prevR = 0, prevRI = 0;
function updateSNNLive() {
  const rEl = document.getElementById("snn-R") || document.getElementById("feat-R");
  const riEl = document.getElementById("snn-RI") || document.getElementById("feat-RI");
  const drEl = document.getElementById("snn-dR");

  const R = rEl ? parseFloat(rEl.value || 0) : 0;
  const RI = riEl ? parseFloat(riEl.value || 0) : 0;
  const dR = drEl ? parseFloat(drEl.value || 0) : Math.max(0, R * 0.5);

  const delta_RI = Math.abs(RI - prevRI);
  const result = runSNN("cloudburst", [R, RI, dR, delta_RI]);
  prevR = R; prevRI = RI;

  drawMembrane("snn-membrane-canvas", result.mem_history, 0.50, result.fired);

  const vmemVal = document.getElementById("snn-vmem-val");
  if (vmemVal) vmemVal.textContent = result.peak_mem.toFixed(2);
}

function updateThunderstormLive() {
  const iwvEl = document.getElementById("ts-iwv");
  const presEl = document.getElementById("ts-pres");
  const windEl = document.getElementById("ts-wind");
  const tempEl = document.getElementById("ts-temp");

  const iwv = iwvEl ? parseFloat(iwvEl.value || 0) : 1.2;
  const pres = presEl ? parseFloat(presEl.value || 0) : 0.8;
  const wind = windEl ? parseFloat(windEl.value || 0) : 2.0;
  const temp = tempEl ? parseFloat(tempEl.value || 0) : 1.0;

  const result = runSNN("thunderstorm", [iwv, pres, wind, temp, iwv * 100]);
  drawMembrane("ts-membrane-canvas", result.mem_history, 0.88, result.fired, 1.0, "#22d3ee");

  const vmemVal = document.getElementById("ts-vmem-val");
  if (vmemVal) vmemVal.textContent = result.peak_mem.toFixed(2);
}
