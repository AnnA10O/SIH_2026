/**
 * pinn_auto_trigger.js
 * ---------------------------------------------------------
 * Watches the live risk score (#risk-score-val) already
 * computed on the AI Nowcast tab, and automatically switches
 * the PINN Flood Hydrodynamics tab between NORMAL / WATCH /
 * ALERT states — no manual button click required.
 *
 * Works with the EXISTING global functions already exported
 * by pinn_canvas.js: simulateFloodCanvas() and resetFloodCanvas().
 * Does not touch or depend on any specific version of that file.
 *
 * HOW TO USE:
 * Add this line in tabbed_dashboard.html, right after the
 * <script src="js/pinn_canvas.js"></script> tag (or wherever
 * pinn_canvas.js is included), and before </body>:
 *
 *   <script src="js/pinn_auto_trigger.js"></script>
 *
 * That's it — no other changes needed.
 * ---------------------------------------------------------
 */

(function () {
  // ---- Thresholds (tweak these two numbers if needed) ----
  const WATCH_THRESHOLD = 10;   // risk % above this -> WATCH state
  const ALERT_THRESHOLD = 50;   // risk % above this -> ALERT state (auto flood)

  // ---- Internal state, so we don't re-trigger every poll ----
  let currentState = "NORMAL";
  const POLL_INTERVAL_MS = 2000;

  function getCurrentRiskPercent() {
    const el = document.getElementById("risk-score-val");
    if (!el) return 0;
    // innerText looks like "96.4%" -> parse the number out of it
    const match = el.innerText.match(/[\d.]+/);
    return match ? parseFloat(match[0]) : 0;
  }

  function ensureAutoBadge() {
    // Create a small always-visible auto-status badge inside the PINN tab header,
    // right next to the "8 BASINS" badge, if it doesn't already exist.
    let badge = document.getElementById("pinn-auto-status-badge");
    if (badge) return badge;

    const header = document.querySelector("#tab-pinn .pinn-header-bar > div");
    if (!header) return null;

    badge = document.createElement("span");
    badge.className = "badge";
    badge.id = "pinn-auto-status-badge";
    badge.style.transition = "all 0.3s ease";
    header.appendChild(badge);
    return badge;
  }

  function setBadgeState(state, riskPct) {
    const badge = ensureAutoBadge();
    if (!badge) return;

    if (state === "ALERT") {
      badge.innerText = `AUTO-DETECTED: FLOOD ALERT (${riskPct.toFixed(1)}%)`;
      badge.style.background = "rgba(239, 68, 68, 0.18)";
      badge.style.color = "#ef4444";
      badge.style.border = "1px solid rgba(239, 68, 68, 0.5)";
      badge.style.animation = "pulse-alert 1.2s infinite";
    } else if (state === "WATCH") {
      badge.innerText = `AUTO-MONITOR: ELEVATED RISK (${riskPct.toFixed(1)}%)`;
      badge.style.background = "rgba(234, 179, 8, 0.15)";
      badge.style.color = "#eab308";
      badge.style.border = "1px solid rgba(234, 179, 8, 0.4)";
      badge.style.animation = "none";
    } else {
      badge.innerText = `AUTO-MONITOR: NORMAL (${riskPct.toFixed(1)}%)`;
      badge.style.background = "rgba(34, 197, 94, 0.15)";
      badge.style.color = "#22c55e";
      badge.style.border = "1px solid rgba(34, 197, 94, 0.3)";
      badge.style.animation = "none";
    }
  }

  function applyState(newState, riskPct) {
    if (newState === currentState) {
      // Still update badge text (risk % may have changed within same state)
      setBadgeState(newState, riskPct);
      return;
    }

    console.log(
      `[PINN Auto-Trigger] State change: ${currentState} -> ${newState} (risk score = ${riskPct.toFixed(1)}%)`
    );

    if (newState === "ALERT") {
      if (typeof window.simulateFloodCanvas === "function") {
        window.simulateFloodCanvas();
        console.log(
          `[PINN Auto-Trigger] Flood overlay AUTO-TRIGGERED for basin '${(window.PINN_SIM && window.PINN_SIM.activeRegion) || "current"}' — risk score ${riskPct.toFixed(1)}% crossed ${ALERT_THRESHOLD}% threshold.`
        );
      }
    } else if (currentState === "ALERT" && newState !== "ALERT") {
      // Dropping out of ALERT back to WATCH/NORMAL -> clear the flood overlay
      if (typeof window.resetFloodCanvas === "function") {
        window.resetFloodCanvas();
        console.log(`[PINN Auto-Trigger] Flood overlay auto-cleared — risk dropped to ${riskPct.toFixed(1)}%.`);
      }
    }

    currentState = newState;
    setBadgeState(newState, riskPct);
  }

  function pollRiskScore() {
    const riskPct = getCurrentRiskPercent();
    let newState = "NORMAL";
    if (riskPct >= ALERT_THRESHOLD) newState = "ALERT";
    else if (riskPct >= WATCH_THRESHOLD) newState = "WATCH";
    applyState(newState, riskPct);
  }

  // Inject a small CSS pulse animation for the ALERT badge
  const style = document.createElement("style");
  style.textContent = `
    @keyframes pulse-alert {
      0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
      70%  { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
  `;
  document.head.appendChild(style);

  // Start polling once the page has loaded
  document.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => {
      pollRiskScore(); // run once immediately
      setInterval(pollRiskScore, POLL_INTERVAL_MS);
      console.log(`[PINN Auto-Trigger] Started polling risk score every ${POLL_INTERVAL_MS}ms.`);
    }, 500);
  });
})();
