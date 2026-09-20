import React, { useState, useEffect, useRef } from "react";
import { Send, CheckCircle2, MessageSquare, Smartphone, Zap, AlertTriangle, ShieldAlert } from "lucide-react";

const getEventType = (riskLevel) => {
  if (riskLevel === "SEVERE") return "Cloudburst";
  if (riskLevel === "HIGH") return "Extremely Heavy Rainfall";
  return "Heavy Rainfall";
};

const getTemplateForRisk = (location) => {
  if (!location) return "";
  const name = location.name || "Unknown Region";
  const window = location.timeWindow || "Next 2-6 Hours";
  const event = getEventType(location.riskLevel);
  
  if (location.riskLevel === "SEVERE") {
    return `🚨 IMMEDIATE WEATHER WARNING – ${name.toUpperCase()}

Your area is at very high risk of an imminent ${event} in the coming ${window}.

Move away from low-lying and flood-prone areas immediately. Do NOT cross flooded roads, streams or underpasses. Stay in a safe elevated location and follow instructions from local authorities.

Risk Level: 🔴 VERY HIGH
Expected: ${window}`;
  } else if (location.riskLevel === "HIGH") {
    return `🚨 Proposed Citizen Alert — ${event} Risk

⚠️ WEATHER ALERT – HIGH ${event.toUpperCase()} RISK
A potential ${event.toLowerCase()} has been detected for ${name} during ${window}.

Please:
• Avoid low-lying areas, river/stream crossings and underpasses.
• Stay indoors and away from windows if severe weather develops.
• Do not attempt to cross flooded roads or flowing water.
• Follow instructions from local authorities.

Risk Level: 🔴 HIGH
Location: ${name}
Expected Window: ${window}

This is an early warning based on weather observations and predictive analysis. Stay alert for official updates.`;
  } else {
    return `⚠️ WEATHER ADVISORY – ELEVATED RISK
A potential for ${event.toLowerCase()} has been detected for ${name} during ${window}.

Please stay alert, monitor local weather updates, and be prepared to move to higher ground if conditions worsen.

Risk Level: 🟠 MODERATE
Location: ${name}
Expected Window: ${window}`;
  }
};

const getSmsTemplate = (location) => {
  if (!location) return "";
  const name = location.name || "Unknown Region";
  const window = location.timeWindow || "next few hours";
  const event = getEventType(location.riskLevel);
  return `⚠️ WEATHER ALERT: Potential ${event.toLowerCase()} risk in ${name} during ${window}. Avoid low-lying areas, flooded roads, underpasses & water crossings. Stay indoors if severe weather develops. Follow local authority instructions. Risk: ${location.riskLevel || 'HIGH'}.`;
};

export default function AlertManager({ locations = [], defaultLocation }) {
  const [activeFormat, setActiveFormat] = useState("long"); // 'long' | 'sms'
  const [selectedLocations, setSelectedLocations] = useState(defaultLocation ? [defaultLocation] : []);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [messageContents, setMessageContents] = useState({});
  const [isSent, setIsSent] = useState(false);
  const [dispatchStatus, setDispatchStatus] = useState("");
    // Sort locations: SEVERE/HIGH at the top
  const sortedLocations = [...locations].sort((a, b) => {
    const isAffected = (loc) => loc.riskLevel === "SEVERE" || loc.riskLevel === "HIGH";
    if (isAffected(a) && !isAffected(b)) return -1;
    if (!isAffected(a) && isAffected(b)) return 1;
    return 0;
  });

  const toggleLocation = (loc) => {
    setSelectedLocations(prev => {
      const exists = prev.find(l => l.id === loc.id);
      if (exists) {
         return prev.filter(l => l.id !== loc.id);
      } else {
         return [...prev, loc];
      }
    });
  };


  // Update text areas when location or format changes
  useEffect(() => {
    setMessageContents(prev => {
      const newContents = { ...prev };
      selectedLocations.forEach(loc => {
        // Only generate default template if it hasn't been generated yet for this location/format,
        // or if we just want to reset (we won't overwrite existing edits here, but format switch will).
        // Actually, if format changes, we should regenerate all.
        newContents[loc.id] = activeFormat === "long" ? getTemplateForRisk(loc) : getSmsTemplate(loc);
      });
      return newContents;
    });
  }, [selectedLocations, activeFormat]);

  const handleMessageChange = (id, value) => {
    setMessageContents(prev => ({ ...prev, [id]: value }));
  };

  if (locations.length === 0) return null;

  const handleSimulate = async () => {
    if (selectedLocations.length === 0) return;
    setIsSent(true);
    setDispatchStatus("Dispatching...");
    
    const getTier = (rLevel) => {
      if (rLevel === "SEVERE") return "Red";
      if (rLevel === "HIGH") return "Orange";
      if (rLevel === "MODERATE") return "Yellow";
      return "Green";
    };
    
    try {
      const promises = selectedLocations.map(loc => {
        const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        return fetch(`${baseUrl}/api/alerts/send`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            region: loc.name || loc.region || "Unknown Region",
            tier: getTier(loc.riskLevel),
            risk_score: loc.riskPercent || loc.riskScore || 0,
            message: messageContents[loc.id] || ""
          })
        }).then(r => r.json());
      });

      await Promise.all(promises);
      console.log("Network push dispatched for multiple locations");
      setDispatchStatus("Success");
    } catch (err) {
      console.error("Dispatch failed", err);
      setDispatchStatus("Network Error");
    }

    setTimeout(() => {
      setIsSent(false);
      setDispatchStatus("");
    }, 5000);
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col h-full overflow-hidden">
      
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-red-100 text-red-600 rounded-lg">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">Broadcast Composer</h2>
            <p className="text-xs text-slate-500">Draft, edit, and dispatch emergency alerts</p>
          </div>
        </div>
        
        {/* Format Switcher */}
        <div className="flex bg-slate-200/60 p-1 rounded-lg">
          <button
            onClick={() => setActiveFormat("long")}
            className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center space-x-2 transition-all ${activeFormat === "long" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Full Template</span>
          </button>
          <button
            onClick={() => setActiveFormat("sms")}
            className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center space-x-2 transition-all ${activeFormat === "sms" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
          >
            <Smartphone className="w-3.5 h-3.5" />
            <span>SMS / WhatsApp</span>
          </button>
        </div>
      </div>


      <div className="flex-1 flex flex-col min-h-0 overflow-y-auto">
        {/* Multi-Location Dropdown UI */}
        <div className="px-6 py-3 border-b border-slate-100 bg-white relative">
          <label className="block text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">Affected Regions Selection</label>
          <div className="relative">
            <button 
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="w-full text-left px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm font-medium text-slate-700 flex justify-between items-center hover:bg-slate-100 transition-colors"
            >
              <span className="truncate">
                {selectedLocations.length === 0 
                  ? "Select Locations..." 
                  : selectedLocations.map(l => l.name).join(", ")}
              </span>
              <span className="ml-2 bg-sky-100 text-sky-700 py-0.5 px-2 rounded-full text-[10px] font-bold whitespace-nowrap">
                {selectedLocations.length} Selected
              </span>
            </button>
            
            {isDropdownOpen && (
              <div className="absolute z-10 top-full left-0 right-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-xl max-h-64 overflow-y-auto">
                {sortedLocations.map(loc => {
                  const isSelected = selectedLocations.some(l => l.id === loc.id);
                  const isAffected = loc.riskLevel === "SEVERE" || loc.riskLevel === "HIGH";
                  return (
                    <div 
                      key={loc.id}
                      onClick={() => toggleLocation(loc)}
                      className="px-4 py-3 border-b border-slate-100 last:border-0 hover:bg-slate-50 flex items-center space-x-3 cursor-pointer"
                    >
                      <input 
                        type="checkbox" 
                        checked={isSelected} 
                        readOnly 
                        className="w-4 h-4 text-sky-600 rounded border-slate-300 focus:ring-sky-500 cursor-pointer"
                      />
                      <div className="flex-1">
                        <div className="text-sm font-semibold text-slate-800 flex items-center space-x-2">
                          <span>{loc.name}</span>
                          {isAffected && (
                            <div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse" title="Highly Affected Area"></div>
                          )}
                        </div>
                        <div className="text-xs text-slate-500">{loc.region}</div>
                      </div>
                      <div className="text-[10px] uppercase font-bold tracking-wider">
                        <span className={loc.riskLevel === 'SEVERE' ? 'text-red-600 bg-red-50 px-1.5 py-0.5 rounded' : loc.riskLevel === 'HIGH' ? 'text-orange-600 bg-orange-50 px-1.5 py-0.5 rounded' : 'text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded'}>
                          {loc.riskLevel}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Editor Area */}
        <div className="p-6 flex flex-col space-y-4 border-b border-slate-100">
          <div className="flex flex-col space-y-4">
            {selectedLocations.length === 0 && (
              <div className="p-8 text-center text-slate-400 bg-slate-50 rounded-xl border border-dashed border-slate-200">
                Select one or more regions above to generate alert messages.
              </div>
            )}
            {selectedLocations.map(loc => (
              <div key={loc.id} className="relative flex flex-col bg-slate-50 border border-slate-200 rounded-xl overflow-hidden focus-within:ring-2 focus-within:ring-sky-500/40 focus-within:bg-white transition-all shadow-sm">
                <div className="bg-slate-100/80 px-4 py-2 border-b border-slate-200 flex justify-between items-center">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">{loc.name}</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${loc.riskLevel === 'SEVERE' ? 'bg-red-100 text-red-600' : loc.riskLevel === 'HIGH' ? 'bg-orange-100 text-orange-600' : 'bg-amber-100 text-amber-600'}`}>
                    {loc.riskLevel} RISK
                  </span>
                </div>
                <textarea
                  value={messageContents[loc.id] || ""}
                  onChange={(e) => handleMessageChange(loc.id, e.target.value)}
                  className="w-full min-h-[160px] p-4 bg-transparent text-[13px] font-medium text-slate-800 focus:outline-none resize-y leading-relaxed"
                  placeholder="Alert message content..."
                />
              </div>
            ))}
          </div>

          {/* Action Bar */}
          <div className="flex items-center justify-end pt-2">
            <button
              onClick={handleSimulate}
              disabled={isSent}
              className={`px-6 py-3 rounded-xl font-bold text-sm text-white shadow-md transition-all flex items-center justify-center space-x-2 w-full sm:w-auto ${
                isSent
                  ? "bg-emerald-600 scale-[0.98]"
                  : "bg-red-600 hover:bg-red-700 shadow-red-500/20 active:scale-95"
              }`}
            >
              {isSent ? (
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Dispatched</span>
                  {dispatchStatus && (
                    <span className="ml-2 pl-2 border-l border-white/30 text-xs opacity-90">{dispatchStatus}</span>
                  )}
                </div>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Simulate Live Public Broadcast</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* AI Justification Panel */}
        <div className="p-6 bg-slate-900 text-white">
          <div className="flex flex-col space-y-6">
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sky-400 mb-1">
                <Zap className="w-4 h-4" />
                <h3 className="text-[11px] font-bold uppercase tracking-wider">AI Threat Analysis</h3>
              </div>
              
              <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">
                <span className="font-semibold text-white">Trigger Justification:</span>{" "}
                {selectedLocations.length > 0 ? (
                  <>
                    The system has detected actionable conditions across <span className="text-amber-400 font-medium">{selectedLocations.length} selected locations</span>.
                    {selectedLocations.some(l => l.riskLevel === "SEVERE" || l.riskLevel === "HIGH") && (
                      <span className="ml-1">Overlapping anomalies indicate high probability of severe events like <span className="text-red-400 font-bold">Cloudbursts</span>.</span>
                    )}
                  </>
                ) : (
                  <>Select regions above to analyze threats.</>
                )}
              </p>

              {selectedLocations.some(l => l.riskLevel === "SEVERE" || l.riskLevel === "HIGH") && (
                <div className="flex flex-wrap gap-2 text-xs text-slate-300 mt-3">
                  <div className="flex items-center space-x-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/50">
                    <div className="w-1.5 h-1.5 bg-rose-500 rounded-full animate-pulse"></div>
                    <span>Rapid cloud cooling detected</span>
                  </div>
                  <div className="flex items-center space-x-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/50">
                    <div className="w-1.5 h-1.5 bg-rose-500 rounded-full animate-pulse"></div>
                    <span>High vertical wind shear</span>
                  </div>
                </div>
              )}
            </div>
            
            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/50 flex flex-col justify-center">
              <div className="text-[10px] text-slate-400 uppercase font-bold mb-0.5">Recommended Action</div>
              {selectedLocations.some(l => l.riskLevel === "SEVERE") ? (
                <div className="text-base font-black text-rose-500 tracking-wide">EVACUATE LOW ZONES IN AFFECTED REGIONS</div>
              ) : selectedLocations.some(l => l.riskLevel === "HIGH") ? (
                <div className="text-base font-black text-amber-500 tracking-wide">PREPARE FOR FLOODING</div>
              ) : (
                <div className="text-base font-black text-sky-400 tracking-wide">MONITOR CONDITIONS</div>
              )}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
