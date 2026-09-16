import React, { useState } from "react";
import { Smartphone, MessageSquare, Send, CheckCircle2, ShieldAlert, Sparkles, BellRing } from "lucide-react";

export default function PublicAlertPreview({ location, onTriggerBroadcast }) {
  const [activeFormat, setActiveFormat] = useState("push"); // 'push' | 'sms' | 'whatsapp'
  const [isSent, setIsSent] = useState(false);

  if (!location) return null;

  const handleSimulate = () => {
    setIsSent(true);
    
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
        risk_score: location.riskScore || 0
      })
    })
    .then(r => r.json())
    .then(data => console.log("Network push dispatched", data))
    .catch(err => console.error("Dispatch failed", err));

    if (onTriggerBroadcast) onTriggerBroadcast();
    setTimeout(() => setIsSent(false), 4000);
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded-lg bg-sky-100 text-sky-700 font-bold">
              <Smartphone className="w-4 h-4" />
            </span>
            <h3 className="text-base font-bold text-slate-900">
              Public Emergency Alert Preview
            </h3>
          </div>
          <p className="text-xs text-slate-500">
            Preview live automated alert conversion sent to citizens & pilgrims
          </p>
        </div>

        {/* Format Selector Tabs */}
        <div className="flex items-center bg-slate-100 p-1 rounded-xl text-xs font-semibold">
          <button
            onClick={() => setActiveFormat("push")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center space-x-1.5 ${
              activeFormat === "push"
                ? "bg-white text-slate-900 shadow-2xs font-bold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <span>📱 Mobile Push</span>
          </button>

          <button
            onClick={() => setActiveFormat("sms")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center space-x-1.5 ${
              activeFormat === "sms"
                ? "bg-white text-slate-900 shadow-2xs font-bold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <span>💬 SMS Alert</span>
          </button>

          <button
            onClick={() => setActiveFormat("whatsapp")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center space-x-1.5 ${
              activeFormat === "whatsapp"
                ? "bg-white text-slate-900 shadow-2xs font-bold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <span>🟢 WhatsApp</span>
          </button>
        </div>
      </div>

      {/* Main Display: Realistic Mobile Mockup */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-6 py-2">
        {/* Mobile Phone Mockup */}
        <div className="w-full max-w-sm bg-slate-900 p-3 rounded-[36px] shadow-2xl border-4 border-slate-800 relative">
          {/* Phone Top Notch */}
          <div className="w-32 h-4 bg-slate-800 rounded-b-xl mx-auto mb-3 flex items-center justify-center">
            <div className="w-3 h-3 rounded-full bg-slate-900 mr-2" />
            <div className="w-10 h-1 bg-slate-700 rounded-full" />
          </div>

          {/* Screen Container */}
          <div className="bg-slate-950 rounded-[28px] p-4 text-slate-100 min-h-[380px] flex flex-col justify-between relative overflow-hidden">
            {/* Lock Screen Time Bar */}
            <div className="text-center pt-2">
              <div className="text-3xl font-light text-slate-200 tracking-tight">15:15</div>
              <div className="text-[10px] text-slate-400 font-medium">Tuesday, September 15</div>
            </div>

            {/* Notification Card rendered depending on active format */}
            <div className="my-auto space-y-3">
              {activeFormat === "push" && (
                <div className="bg-slate-900/90 backdrop-blur-md border border-red-500/40 rounded-2xl p-4 shadow-xl space-y-2 text-left animate-in fade-in duration-300">
                  <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800 pb-2">
                    <span className="flex items-center space-x-1.5 font-bold text-sky-400">
                      ⚡ KERAUNOS EMERGENCY
                    </span>
                    <span>now</span>
                  </div>

                  <div className="space-y-1.5">
                    <div className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-red-600/90 text-white text-[10px] font-black tracking-wider uppercase">
                      🔴 SEVERE WEATHER ALERT
                    </div>
                    <div className="text-sm font-bold text-white leading-tight">
                      Heavy Rainfall and Cloudburst Expected
                    </div>
                  </div>

                  <div className="text-xs text-slate-300 space-y-1 pt-1 font-medium">
                    <div>📍 <strong>{location.name}</strong></div>
                    <div>⏰ Expected: <strong>{location.timeWindow}</strong></div>
                    <div>🌧️ <strong>Heavy rainfall expected ({location.rainfallRate})</strong></div>
                    <div>⚠️ <strong>Flash flood risk along river banks</strong></div>
                  </div>

                  <div className="text-[11px] text-amber-300 bg-amber-950/60 border border-amber-800/60 p-2 rounded-lg font-semibold mt-2">
                    Please stay alert and move to higher ground immediately.
                  </div>
                </div>
              )}

              {activeFormat === "sms" && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl text-left space-y-2 font-mono text-xs">
                  <div className="text-[10px] text-slate-400 flex items-center justify-between border-b border-slate-800 pb-1.5">
                    <span>Sender: <strong>KERAUNOS-ALERT</strong></span>
                    <span>14:15 IST</span>
                  </div>
                  <div className="text-amber-400 font-bold">
                    [URGENT GOVT ALERT] 🔴 SEVERE WEATHER WARNING:
                  </div>
                  <p className="text-slate-200 leading-relaxed text-[11px]">
                    Heavy Rainfall & Cloudburst forecasted at {location.name} between {location.timeWindow}.
                  </p>
                  <p className="text-slate-300 text-[11px]">
                    Rainfall: {location.rainfallRate}. Flash flood danger high. Evacuate low riverbed areas.
                  </p>
                </div>
              )}

              {activeFormat === "whatsapp" && (
                <div className="bg-emerald-950/90 border border-emerald-700/50 rounded-2xl p-4 shadow-xl text-left space-y-2">
                  <div className="flex items-center space-x-2 text-emerald-400 text-xs font-bold border-b border-emerald-800/80 pb-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span>Keraunos Emergency Alert System Verified</span>
                  </div>
                  <div className="text-sm font-bold text-white">
                    🚨 SEVERE WEATHER NOWCAST ALERT
                  </div>
                  <div className="text-xs text-slate-200 space-y-1">
                    <div>📍 <strong>Location:</strong> {location.name}</div>
                    <div>⏰ <strong>Time:</strong> {location.timeWindow}</div>
                    <div>🌧️ <strong>Rainfall:</strong> {location.rainfallRate}</div>
                    <div>⚠️ <strong>Hazard:</strong> Cloudburst & Flash Flood</div>
                  </div>
                  <div className="pt-2 flex gap-2">
                    <button className="w-full py-1.5 bg-emerald-600 text-white rounded-lg text-[10px] font-bold">
                      Acknowledge Alert
                    </button>
                    <button className="w-full py-1.5 bg-slate-800 text-slate-200 rounded-lg text-[10px] font-bold">
                      View Safe Map
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Bar */}
            <div className="w-24 h-1 bg-slate-600 rounded-full mx-auto mb-1" />
          </div>
        </div>

        {/* Right Info & Broadcast Trigger Panel */}
        <div className="flex-1 space-y-4 text-left">
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 space-y-3">
            <h4 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Broadcast Channel Readiness</span>
            </h4>

            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between bg-white p-2.5 rounded-xl border border-slate-200">
                <span className="font-semibold text-slate-700">📱 Mobile Push Network (Cell Broadcast)</span>
                <span className="text-emerald-700 font-bold bg-emerald-50 px-2 py-0.5 rounded-md">READY</span>
              </div>
              <div className="flex items-center justify-between bg-white p-2.5 rounded-xl border border-slate-200">
                <span className="font-semibold text-slate-700">💬 National SMS Gateway (BSNL/Airtel/Jio)</span>
                <span className="text-emerald-700 font-bold bg-emerald-50 px-2 py-0.5 rounded-md">READY</span>
              </div>
              <div className="flex items-center justify-between bg-white p-2.5 rounded-xl border border-slate-200">
                <span className="font-semibold text-slate-700">🟢 WhatsApp Public Warning API</span>
                <span className="text-emerald-700 font-bold bg-emerald-50 px-2 py-0.5 rounded-md">READY</span>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={handleSimulate}
              className={`flex-1 py-3 px-4 rounded-xl font-bold text-sm text-white shadow-md transition-all flex items-center justify-center space-x-2 cursor-pointer active:scale-95 ${
                isSent
                  ? "bg-emerald-600"
                  : "bg-sky-600 hover:bg-sky-700 shadow-sky-500/20"
              }`}
            >
              {isSent ? (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Broadcast Dispatched to 14,250 Users!</span>
                </>
              ) : (
                <>
                  <BellRing className="w-4 h-4" />
                  <span>Simulate Live Public Broadcast</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
