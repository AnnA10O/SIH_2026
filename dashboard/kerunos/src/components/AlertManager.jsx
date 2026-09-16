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

export default function AlertManager({ location }) {
  const [activeFormat, setActiveFormat] = useState("long"); // 'long' | 'sms'
  const [messageContent, setMessageContent] = useState("");
  const [isSent, setIsSent] = useState(false);
  const [dispatchStatus, setDispatchStatus] = useState("");
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [messageContent]);

  // Update text area when location or format changes
  useEffect(() => {
    if (activeFormat === "long") {
      setMessageContent(getTemplateForRisk(location));
    } else {
      setMessageContent(getSmsTemplate(location));
    }
  }, [location, activeFormat]);

  if (!location) return null;

  const handleSimulate = () => {
    setIsSent(true);
    setDispatchStatus("Dispatching...");
    
    // Convert UI risk level to standard tier
    let tier = "Green";
    if (location.riskLevel === "SEVERE") tier = "Red";
    else if (location.riskLevel === "HIGH") tier = "Orange";
    else if (location.riskLevel === "MODERATE") tier = "Yellow";
    
    // Call backend API to dispatch alert
    fetch('/api/alerts/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        region: location.name || location.region || "Unknown Region",
        tier: tier,
        risk_score: location.riskScore || 0,
        message: messageContent // pass the customized message
      })
    })
    .then(r => r.json())
    .then(data => {
      console.log("Network push dispatched", data);
      setDispatchStatus(data.network_status || "Success");
    })
    .catch(err => {
      console.error("Dispatch failed", err);
      setDispatchStatus("Network Error");
    });

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

      <div className="flex-1 flex flex-col min-h-0">
        {/* Editor Area */}
        <div className="p-6 flex-1 flex flex-col space-y-4 overflow-y-auto border-b border-slate-100">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={messageContent}
              onChange={(e) => setMessageContent(e.target.value)}
              className="w-full min-h-[120px] p-5 bg-slate-50 border border-slate-200 rounded-xl text-[13px] font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:bg-white transition-all resize-none leading-relaxed overflow-hidden"
              placeholder="Alert message content..."
            />
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
        <div className="p-6 flex-1 bg-slate-900 text-white overflow-y-auto">
          <div className="flex flex-col space-y-6">
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sky-400 mb-1">
                <Zap className="w-4 h-4" />
                <h3 className="text-[11px] font-bold uppercase tracking-wider">AI Threat Analysis</h3>
              </div>
              
              <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">
                <span className="font-semibold text-white">Trigger Justification:</span>{" "}
                {location.riskLevel === "SEVERE" || location.riskLevel === "HIGH" ? (
                  <>
                    The system has detected multiple overlapping anomalies for <span className="text-amber-400 font-medium">{location.name}</span> indicating an imminent {getEventType(location.riskLevel).toLowerCase()}. 
                    The PINN simulation predicts a <span className="text-red-400 font-bold">{location.riskPercent || 85}%</span> probability of a severe event within {location.timeWindow || "2-6 hours"}.
                  </>
                ) : (
                  <>
                    Conditions for <span className="text-amber-400 font-medium">{location.name}</span> are currently elevated. The PINN simulation predicts a {location.riskPercent || 35}% probability of moderate rainfall.
                  </>
                )}
              </p>

              {(location.riskLevel === "SEVERE" || location.riskLevel === "HIGH") && (
                <div className="flex flex-wrap gap-2 text-xs text-slate-300 mt-3">
                  <div className="flex items-center space-x-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/50">
                    <div className="w-1.5 h-1.5 bg-rose-500 rounded-full animate-pulse"></div>
                    <span>Rapid cloud cooling</span>
                  </div>
                  <div className="flex items-center space-x-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/50">
                    <div className="w-1.5 h-1.5 bg-rose-500 rounded-full animate-pulse"></div>
                    <span>High vertical wind shear</span>
                  </div>
                  <div className="flex items-center space-x-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/50">
                    <div className="w-1.5 h-1.5 bg-amber-500 rounded-full"></div>
                    <span>Ground sat: {location.riskScore || 70}%</span>
                  </div>
                </div>
              )}
            </div>
            
            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/50 flex flex-col justify-center">
              <div className="text-[10px] text-slate-400 uppercase font-bold mb-0.5">Recommended Action</div>
              {location.riskLevel === "SEVERE" ? (
                <div className="text-base font-black text-rose-500 tracking-wide">EVACUATE LOW ZONES</div>
              ) : location.riskLevel === "HIGH" ? (
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
