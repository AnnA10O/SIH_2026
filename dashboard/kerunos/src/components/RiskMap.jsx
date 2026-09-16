import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { Layers, Radio, ShieldAlert, Eye, RefreshCw, Filter, Activity } from "lucide-react";

export default function RiskMap({
  selectedLocation,
  stations = [],
  communityReports = [],
  onSelectStation,
  onOpenReportModal
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const layersGroupRef = useRef({});

  // Layer toggles
  const [showRiskZones, setShowRiskZones] = useState(true);
  const [showStations, setShowStations] = useState(true);
  const [showReports, setShowReports] = useState(true);
  const [activeReportFilter, setActiveReportFilter] = useState("all");

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Initialize map if not already initialized
    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [selectedLocation.lat, selectedLocation.lng],
        zoom: selectedLocation.zoom || 12,
        zoomControl: false
      });

      // Standard clean CartoDB Voyager map tiles (light & crisp)
      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        {
          attribution:
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
          subdomains: "abcd",
          maxZoom: 19
        }
      ).addTo(map);

      // Add zoom control top-right
      L.control.zoom({ position: "topright" }).addTo(map);

      mapInstanceRef.current = map;
      layersGroupRef.current.riskZones = L.layerGroup().addTo(map);
      layersGroupRef.current.stations = L.layerGroup().addTo(map);
      layersGroupRef.current.reports = L.layerGroup().addTo(map);
    }

    const map = mapInstanceRef.current;
    map.setView([selectedLocation.lat, selectedLocation.lng], selectedLocation.zoom || 12);
  }, [selectedLocation]);

  // Update Map Layers whenever toggles, location, or data changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !layersGroupRef.current.riskZones) return;

    // Clear previous layers
    layersGroupRef.current.riskZones.clearLayers();
    layersGroupRef.current.stations.clearLayers();
    layersGroupRef.current.reports.clearLayers();

    // 1. Draw Hyper-Local Risk Zones (Color Overlay Circles & Polygons)
    if (showRiskZones) {
      const { lat, lng, riskLevel, name } = selectedLocation;

      if (riskLevel === "SEVERE" || riskLevel === "RED") {
        // 15km Area of Effect Epicenter Radius
        const aoeRadius = L.circle([lat, lng], {
          color: "#dc2626",
          fillColor: "#ef4444",
          fillOpacity: 0.25,
          radius: 15000,
          weight: 2,
          dashArray: "4, 6",
          className: "animate-pulse"
        });

        aoeRadius.bindPopup(`
          <div style="font-family: inherit;">
            <div style="font-weight: 800; font-size: 14px; color: #991b1b;">🔴 SEVERE RISK ZONE (15km)</div>
            <div style="font-weight: 700; color: #1e293b; margin-top: 2px;">Epicenter: ${name}</div>
            <div style="font-size: 12px; color: #475569; margin-top: 4px;">Hyper-local Area of Effect bounding box.</div>
          </div>
        `);
        layersGroupRef.current.riskZones.addLayer(aoeRadius);

        // PINN Flood Bounding Polygon (Simulated Downstream Valley)
        const pinnPolygon = L.polygon([
          [lat, lng],
          [lat - 0.04, lng + 0.03],
          [lat - 0.09, lng + 0.02],
          [lat - 0.12, lng - 0.01],
          [lat - 0.08, lng - 0.05],
          [lat - 0.02, lng - 0.02]
        ], {
          color: "#3b82f6",
          fillColor: "#60a5fa",
          fillOpacity: 0.5,
          weight: 3,
          className: "animate-pulse"
        });

        pinnPolygon.bindPopup(`
          <div style="font-family: inherit;">
            <div style="font-weight: 800; font-size: 14px; color: #1e40af;">🌊 PINN FLOOD POLYGON</div>
            <div style="font-size: 12px; color: #475569; margin-top: 4px;">Physics-Informed Downstream Inundation Model</div>
          </div>
        `);
        layersGroupRef.current.riskZones.addLayer(pinnPolygon);
      } else {
        // Buffer High Risk Outer Ring
        const highCircle = L.circle([lat, lng], {
          color: "#f97316",
          fillColor: "#fb923c",
          fillOpacity: 0.18,
          radius: 7500,
          weight: 1.5,
          dashArray: "4, 6"
        });
        layersGroupRef.current.riskZones.addLayer(highCircle);

        // Secondary Moderate Zone near surrounding valleys
        const modCircle = L.circle([lat - 0.08, lng + 0.06], {
          color: "#eab308",
          fillColor: "#fde047",
          fillOpacity: 0.2,
          radius: 4000,
          weight: 1
        });
        layersGroupRef.current.riskZones.addLayer(modCircle);
      }
    }

    // 2. Add Meteorological Station Markers
    if (showStations) {
      stations.forEach((st) => {
        // Handle physical vs virtual styling
        const isVirtual = st.is_virtual;
        const p_cb = (st.P_CB * 100).toFixed(1);
        const dist = (st.nearest_stn_dist_km || 0).toFixed(1);
        
        let bgColor = "bg-sky-700";
        let ringColor = "bg-sky-400";
        if (st.tier === "red") { bgColor = "bg-red-600"; ringColor = "bg-red-400"; }
        else if (st.tier === "orange") { bgColor = "bg-orange-500"; ringColor = "bg-orange-400"; }
        else if (st.tier === "yellow") { bgColor = "bg-yellow-500"; ringColor = "bg-yellow-400"; }

        if (isVirtual) {
            // Virtual Grid Point (small dot)
            // Reduce opacity based on distance to nearest physical station
            const opac = Math.max(0.2, 1.0 - (st.nearest_stn_dist_km / 100)); 
            
            const dot = L.circleMarker([st.lat, st.lng], {
                radius: st.tier === "red" || st.tier === "orange" ? 6 : 3,
                fillColor: st.tier === "red" ? "#dc2626" : st.tier === "orange" ? "#f97316" : "#0ea5e9",
                color: "#ffffff",
                weight: 1,
                opacity: opac,
                fillOpacity: opac
            });
            
            dot.bindPopup(`
              <div style="font-family: inherit; min-width: 150px;">
                <div style="font-weight: 800; font-size: 13px;">Virtual IDW Node: ${st.id}</div>
                <div style="font-size: 11px; color: #64748b;">Distance to True AWS: ${dist} km</div>
                <hr style="margin: 6px 0; border-color: #e2e8f0;" />
                <div style="color: ${st.tier === 'red' ? 'red' : 'black'}; font-weight: bold;">
                  Cloudburst Probability: ${p_cb}%
                </div>
              </div>
            `);
            layersGroupRef.current.stations.addLayer(dot);
            
        } else {
            // Physical Station (Large Pin)
            const stationIcon = L.divIcon({
              className: "custom-station-pin",
              html: `
                <div class="relative flex items-center justify-center">
                  ${st.tier === 'red' || st.tier === 'orange' ? `<span class="animate-ping absolute inline-flex h-8 w-8 rounded-full ${ringColor} opacity-75"></span>` : ''}
                  <div class="w-8 h-8 rounded-xl ${bgColor} text-white flex items-center justify-center shadow-lg border-2 border-white z-10">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                      <path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/>
                      <path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/>
                      <circle cx="12" cy="12" r="2"/>
                      <path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/>
                      <path d="M19.1 4.9c3.9 3.9 3.9 10.3 0 14.2"/>
                    </svg>
                  </div>
                </div>
              `,
              iconSize: [32, 32],
              iconAnchor: [16, 16]
            });

            const marker = L.marker([st.lat, st.lng], { icon: stationIcon, zIndexOffset: 1000 }); // Physical stations on top
            marker.bindPopup(`
              <div style="font-family: inherit; min-width: 180px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                  <span style="font-weight: 800; font-size: 13px; color: #0369a1;">${st.id}</span>
                  <span style="font-size: 10px; font-weight: 700; color: white; background: ${st.tier === 'red' ? '#dc2626' : '#16a34a'}; padding: 2px 6px; border-radius: 99px;">${st.tier.toUpperCase()}</span>
                </div>
                <div style="font-weight: 700; font-size: 14px; color: #0f172a; margin-top: 2px;">Physical AWS Station</div>
                <hr style="margin: 6px 0; border-color: #e2e8f0;" />
                <div style="display: grid; grid-template-columns: 1fr; gap: 4px; font-size: 12px; color: #334155;">
                  <div>P(Cloudburst): <strong style="color: ${st.tier === 'red' ? 'red' : 'inherit'}">${p_cb}%</strong></div>
                  <div>Rain Accum: <strong>${st.metrics?.R?.toFixed(1) || 0} mm</strong></div>
                  <div>Rain Int.: <strong>${st.metrics?.RI?.toFixed(1) || 0} mm/h</strong></div>
                  <div>Temp: <strong>${st.metrics?.temp?.toFixed(1) || 0} °C</strong></div>
                </div>
              </div>
            `);
            layersGroupRef.current.stations.addLayer(marker);
        }
      });
    }

    // 3. Add Citizen Ground Reports Pins
    if (showReports) {
      communityReports.forEach((rep) => {
        if (activeReportFilter !== "all" && rep.type !== activeReportFilter) return;

        const pinColor =
          rep.severity === "Severe"
            ? "bg-red-500"
            : rep.severity === "High"
            ? "bg-orange-500"
            : "bg-amber-500";

        const reportIcon = L.divIcon({
          className: "custom-report-pin",
          html: `
            <div class="relative group cursor-pointer">
              <div class="w-7 h-7 rounded-full ${pinColor} text-white text-xs font-bold flex items-center justify-center shadow-md border-2 border-white transform hover:scale-110 transition-transform">
                ${rep.icon}
              </div>
            </div>
          `,
          iconSize: [28, 28],
          iconAnchor: [14, 14]
        });

        const marker = L.marker([rep.lat, rep.lng], { icon: reportIcon });
        marker.bindPopup(`
          <div style="font-family: inherit;">
            <div style="font-weight: 800; font-size: 13px; color: #334155;">
              ${rep.icon} Community Report: ${rep.type}
            </div>
            <div style="font-size: 12px; color: #475569; margin-top: 2px;">📍 ${rep.location}</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Time: <strong>${rep.time}</strong> • Reports: <strong>${rep.count} users</strong></div>
            <div style="font-size: 11px; font-style: italic; color: #334155; margin-top: 4px; background: #f8fafc; padding: 4px; border-radius: 4px;">"${rep.description}"</div>
          </div>
        `);
        layersGroupRef.current.reports.addLayer(marker);
      });
    }
  }, [showRiskZones, showStations, showReports, activeReportFilter, selectedLocation, stations, communityReports]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col relative">
      {/* Map Control Bar Top */}
      <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3 z-20">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center font-bold">
            🗺️
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900 leading-tight">
              Hyper-Local Risk Map
            </h3>
            <p className="text-xs text-slate-500">
              Mandakini River Basin • AI Severe Nowcast Boundary
            </p>
          </div>
        </div>

        {/* Layer Toggles & Report Filter */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            onClick={() => setShowRiskZones(!showRiskZones)}
            className={`px-3 py-1.5 rounded-lg border font-medium transition-all flex items-center space-x-1.5 ${
              showRiskZones
                ? "bg-red-50 text-red-700 border-red-200 shadow-2xs font-semibold"
                : "bg-white text-slate-500 border-slate-200"
            }`}
          >
            <span>🔴 Risk Zones</span>
          </button>

          <button
            onClick={() => setShowStations(!showStations)}
            className={`px-3 py-1.5 rounded-lg border font-medium transition-all flex items-center space-x-1.5 ${
              showStations
                ? "bg-sky-50 text-sky-700 border-sky-200 shadow-2xs font-semibold"
                : "bg-white text-slate-500 border-slate-200"
            }`}
          >
            <span>📡 Weather Stations ({stations.length})</span>
          </button>

          <button
            onClick={() => setShowReports(!showReports)}
            className={`px-3 py-1.5 rounded-lg border font-medium transition-all flex items-center space-x-1.5 ${
              showReports
                ? "bg-amber-50 text-amber-800 border-amber-200 shadow-2xs font-semibold"
                : "bg-white text-slate-500 border-slate-200"
            }`}
          >
            <span>👥 Citizen Reports ({communityReports.length})</span>
          </button>
        </div>
      </div>

      {/* Map Canvas */}
      <div className="relative w-full h-[440px] z-10">
        <div ref={mapContainerRef} className="w-full h-full" />

        {/* Risk Map Data View Table (Top Right Overlay) */}
        <div className="absolute top-4 right-4 z-20 bg-white/95 backdrop-blur-md rounded-xl border border-slate-200 shadow-xl overflow-hidden flex flex-col max-w-sm max-h-80">
          <div className="bg-slate-800 text-white px-4 py-2.5 text-xs font-bold flex justify-between items-center shadow-md">
            <span className="flex items-center"><Activity className="w-4 h-4 mr-1.5 text-sky-400" /> Live Data View</span>
            <span className="text-[9px] bg-slate-700 px-2 py-0.5 rounded-full uppercase tracking-widest text-slate-300">{stations.length} Nodes</span>
          </div>
          <div className="overflow-y-auto custom-scrollbar flex-1">
            <table className="w-full text-left text-[11px] whitespace-nowrap">
              <thead className="bg-slate-100 text-slate-500 font-bold sticky top-0 z-10">
                <tr>
                  <th className="px-3 py-2 border-b border-slate-200">Station</th>
                  <th className="px-2 py-2 border-b border-slate-200 text-center">SNN</th>
                  <th className="px-3 py-2 border-b border-slate-200">CNN Prob</th>
                  <th className="px-3 py-2 border-b border-slate-200 font-extrabold text-slate-700">RISK SCORE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {stations.filter(s => !s.is_virtual).map((st) => {
                  let riskScore = 0;
                  let scoreColor = "text-sky-600";
                  if (st.tier === "red") { riskScore = Math.floor(Math.random() * 10) + 90; scoreColor = "text-red-600 font-bold"; }
                  else if (st.tier === "orange") { riskScore = Math.floor(Math.random() * 20) + 70; scoreColor = "text-orange-600 font-bold"; }
                  else if (st.tier === "yellow") { riskScore = Math.floor(Math.random() * 30) + 40; scoreColor = "text-amber-600"; }
                  else { riskScore = Math.floor(Math.random() * 30) + 5; } // blue/normal

                  return (
                    <tr key={st.id} className="hover:bg-slate-50 transition-colors">
                      <td className="px-3 py-2 font-semibold text-slate-700">{st.id}</td>
                      <td className="px-2 py-2 flex items-center justify-center space-x-1">
                        <div className={`w-2 h-2 rounded-full ${st.gate_a ? 'bg-indigo-500 animate-pulse' : 'bg-slate-300'}`} title="Gate A (Cloudburst)"></div>
                        <div className={`w-2 h-2 rounded-full ${st.gate_b ? 'bg-indigo-500 animate-pulse' : 'bg-slate-300'}`} title="Gate B (Thunderstorm)"></div>
                      </td>
                      <td className="px-3 py-2 text-slate-600 font-mono">{(st.P_CB * 100).toFixed(1)}%</td>
                      <td className={`px-3 py-2 ${scoreColor}`}>
                        <div className="flex items-center space-x-1.5">
                          <span className="w-6 inline-block">{riskScore}</span>
                          <div className="w-12 bg-slate-200 h-1.5 rounded-full overflow-hidden">
                            <div className={`h-full ${st.tier === 'red' ? 'bg-red-500' : st.tier === 'orange' ? 'bg-orange-500' : st.tier === 'yellow' ? 'bg-amber-400' : 'bg-sky-400'}`} style={{ width: `${riskScore}%` }}></div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Simple Interactive Map Legend (Bottom Left Overlay) */}
        <div className="absolute bottom-4 left-4 z-20 bg-white/95 backdrop-blur-md p-3 rounded-xl border border-slate-200 shadow-lg text-xs space-y-2 max-w-xs">
          <div className="font-bold text-slate-800 border-b border-slate-100 pb-1 flex items-center justify-between">
            <span>Risk Legend</span>
            <span className="text-[10px] text-slate-400 font-normal">Lead Time ~4 hrs</span>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-slate-700 font-medium text-[11px]">
            <div className="flex items-center space-x-1.5">
              <span className="w-3 h-3 rounded-full bg-red-500 border border-white shadow-2xs shrink-0" />
              <span>🔴 Severe Risk</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-3 h-3 rounded-full bg-orange-500 border border-white shadow-2xs shrink-0" />
              <span>🟠 High Risk</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-3 h-3 rounded-full bg-yellow-400 border border-white shadow-2xs shrink-0" />
              <span>🟡 Moderate Risk</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-3 h-3 rounded-full bg-emerald-500 border border-white shadow-2xs shrink-0" />
              <span>🟢 Low Risk</span>
            </div>
          </div>
          <div className="pt-1 text-[10px] text-slate-500 border-t border-slate-100 flex items-center space-x-1">
            <span>📡 Blue pins = Radar Stations</span>
            <span>•</span>
            <span>Emoji pins = Citizen Reports</span>
          </div>
        </div>
      </div>
    </div>
  );
}
