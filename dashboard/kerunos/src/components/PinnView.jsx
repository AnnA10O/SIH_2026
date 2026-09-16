import React, { useEffect, useRef } from "react";

export default function PinnView() {
  const containerRef = useRef(null);

  useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = `
      :root {
        --bg-dark: #060b14;
        --bg-panel: rgba(10, 18, 32, 0.95);
        --bg-card: rgba(15, 27, 48, 0.7);
        --bg-card-hover: rgba(22, 38, 66, 0.85);
        --border: rgba(99, 179, 237, 0.14);
        --border-bright: rgba(99, 179, 237, 0.35);
        
        --text-main: #f1f5f9;
        --text-sub: #94a3b8;
        --text-muted: #52677e;
        --cyan: #06b6d4;
        --blue: #38bdf8;
        --purple: #a855f7;
        --green: #22c55e;
        --yellow: #eab308;
        --orange: #f97316;
        --red: #ef4444;
        
        --font-sans: 'Inter', -apple-system, sans-serif;
        --font-heading: 'Outfit', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      }
      .pinn-header-bar {
        padding: 8px 14px; background: rgba(6, 12, 24, 0.95); border-bottom: 1px solid var(--border);
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; flex-shrink: 0;
      }
      #pinn-3d-container {
        flex: 1; width: 100%; height: 100%; min-height: 250px; position: relative; background: #040810; overflow: hidden;
      }
      .filter-chip {
        padding: 4px 10px; border-radius: 14px; font-size: 11px; font-weight: 600;
        background: transparent; border: 1px solid transparent; color: var(--text-sub); cursor: pointer; transition: all 0.25s;
      }
      .filter-chip:hover { color: var(--text-main); }
      .filter-chip.active { background: rgba(6, 182, 212, 0.2); border-color: var(--cyan); color: #67e8f9; }
      
      .btn {
        background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border);
        color: var(--text-main); font-size: 11px; font-weight: 600; padding: 5px 12px;
        border-radius: 6px; cursor: pointer; transition: all 0.25s;
        display: inline-flex; align-items: center; gap: 6px;
      }
      .btn:hover { background: rgba(255, 255, 255, 0.1); border-color: var(--border-bright); }
      .btn-primary { background: linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(56, 189, 248, 0.25)); border-color: var(--cyan); color: #67e8f9; }
      .btn-primary:hover { background: linear-gradient(135deg, rgba(6, 182, 212, 0.4), rgba(56, 189, 248, 0.4)); }

      .badge {
        font-size: 9.5px; font-weight: 700; padding: 2px 6px; border-radius: 10px;
        background: rgba(255, 255, 255, 0.08); color: var(--text-sub);
      }
    `;
    document.head.appendChild(style);

    const loadScript = (src) => {
      return new Promise((resolve) => {
        if (document.querySelector(`script[src="${src}"]`)) {
          resolve();
          return;
        }
        const script = document.createElement("script");
        script.src = src;
        script.onload = resolve;
        document.head.appendChild(script);
      });
    };

    const loadScripts = async () => {
      await loadScript("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js");
      await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js");
      await loadScript("./js/pinn_canvas.js?v=20260915_v4");
      
      if (window.initFloodCanvas) {
        window.initFloodCanvas();
      }
    };

    loadScripts();

    return () => {
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
    };
  }, []);

  return (
    <div
      id="tab-pinn"
      className="tab-pane active"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        width: "100%",
        background: "#040810",
        overflow: "hidden",
        position: "relative",
        color: "white",
        fontFamily: "var(--font-sans)"
      }}
      ref={containerRef}
    >
      <div className="pinn-header-bar" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px", padding: "8px 14px", background: "rgba(6, 12, 24, 0.95)", borderBottom: "1px solid var(--border)", flexShrink: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "13px", fontWeight: 800, color: "var(--cyan)", fontFamily: "var(--font-heading)" }}>
            UTTARAKHAND 2D SATELLITE FLOOD MAP
          </span>
          <span className="badge" style={{ background: "rgba(6, 182, 212, 0.15)", color: "var(--cyan)", border: "1px solid rgba(6, 182, 212, 0.3)" }}>
            REAL SATELLITE + DEM OVERLAY
          </span>
          <span className="badge" style={{ background: "rgba(34, 197, 94, 0.15)", color: "var(--green)", border: "1px solid rgba(34, 197, 94, 0.3)" }}>
            8 BASINS
          </span>
        </div>

        <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
          <button className="filter-chip active" id="chip-region-rudraprayag" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("rudraprayag", e.currentTarget)}>Mandakini (Rudraprayag)</button>
          <button className="filter-chip" id="chip-region-chamoli" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("chamoli", e.currentTarget)}>Alaknanda (Chamoli)</button>
          <button className="filter-chip" id="chip-region-uttarkashi" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("uttarkashi", e.currentTarget)}>Bhagirathi (Uttarkashi)</button>
          <button className="filter-chip" id="chip-region-pithoragarh" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("pithoragarh", e.currentTarget)}>Gori Ganga (Pithoragarh)</button>
          <button className="filter-chip" id="chip-region-tehri" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("tehri", e.currentTarget)}>Bhilangna (Tehri)</button>
          <button className="filter-chip" id="chip-region-pauri" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("pauri", e.currentTarget)}>Nayar (Pauri)</button>
          <button className="filter-chip" id="chip-region-nainital" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("nainital", e.currentTarget)}>Gaula (Nainital)</button>
          <button className="filter-chip" id="chip-region-almora" onClick={(e) => window.switchPINNRegion && window.switchPINNRegion("almora", e.currentTarget)}>Kosi (Almora)</button>
        </div>

        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
          <button className="btn btn-primary" style={{ padding: "5px 14px", fontSize: "11.5px", fontWeight: 700 }} onClick={() => window.simulateFloodCanvas && window.simulateFloodCanvas()}>
            ⚡ Run Simulation
          </button>
          <button className="btn" style={{ padding: "5px 14px", fontSize: "11.5px" }} onClick={() => window.resetFloodCanvas && window.resetFloodCanvas()}>
            ↺ Reset
          </button>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0, width: "100%" }}>
        <div id="pinn-leaflet-map" style={{ flex: 1, minWidth: 0, height: "100%", position: "relative", background: "#040810", zIndex: 1 }}></div>

        <div id="pinn-right-panel" style={{ width: "380px", minWidth: "380px", flexShrink: 0, background: "rgba(6, 12, 24, 0.98)", borderLeft: "1px solid var(--border)", padding: "18px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "16px", zIndex: 10 }}>
          <div style={{ borderBottom: "1px solid var(--border-bright)", paddingBottom: "8px" }}>
            <h3 style={{ fontSize: "14px", color: "var(--cyan)", margin: "0 0 4px 0", fontFamily: "var(--font-heading)" }}>EVENT CONCLUSION</h3>
            <p style={{ fontSize: "11px", color: "var(--text-muted)", margin: 0 }}>Automated Damage & Evacuation Assessment</p>
          </div>

          <div id="pinn-conclusion-content" style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "12.5px", color: "var(--text-main)" }}>
            <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>Awaiting simulation trigger...</div>
          </div>
        </div>
      </div>

      <div id="pinn-bottom-panel" style={{ background: "rgba(6, 12, 24, 0.95)", backdropFilter: "blur(12px)", borderTop: "1px solid var(--border-bright)", padding: "10px 16px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "14px", flexShrink: 0, zIndex: 100, boxShadow: "0 -4px 20px rgba(0,0,0,0.6)" }}>
        <div style={{ display: "flex", gap: "20px", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: "9px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>PEAK SURGE DEPTH</div>
            <div style={{ fontSize: "15px", fontWeight: 800, color: "var(--blue)" }} id="pinn-depth">-- m</div>
          </div>
          <div style={{ width: "1px", height: "24px", background: "var(--border)" }}></div>
          <div>
            <div style={{ fontSize: "9px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>TORRENT FLOW SPEED</div>
            <div style={{ fontSize: "15px", fontWeight: 800, color: "var(--cyan)" }} id="pinn-speed">-- m/s</div>
          </div>
          <div style={{ width: "1px", height: "24px", background: "var(--border)" }}></div>
          <div>
            <div style={{ fontSize: "9px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>FLOODED AREA</div>
            <div style={{ fontSize: "15px", fontWeight: 800, color: "var(--green)" }} id="pinn-area">-- km²</div>
          </div>
          <div style={{ width: "1px", height: "24px", background: "var(--border)" }}></div>
          <div>
            <div style={{ fontSize: "9px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>CHOKE LOCATION</div>
            <div style={{ fontSize: "13px", fontWeight: 800, color: "var(--orange)" }} id="pinn-gorge-name">--</div>
          </div>
        </div>

        <div id="pinn-towns-list" style={{ fontSize: "11px", color: "var(--text-sub)", flex: 1, minWidth: "250px", textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>AFFECTED TOWNS:</span> Loading live data...
        </div>
      </div>
    </div>
  );
}
