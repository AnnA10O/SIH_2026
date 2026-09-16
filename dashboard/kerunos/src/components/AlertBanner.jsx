import React from "react";
import { AlertTriangle, MapPin, Clock, Send, ShieldAlert, CheckCircle2 } from "lucide-react";

export default function AlertBanner({ location, onSimulateBroadcast }) {
  if (!location) return null;

  const isSevere = location.riskLevel === "SEVERE";
  const isHigh = location.riskLevel === "HIGH";
  const isModerate = location.riskLevel === "MODERATE";

  // Color mappings for Risk Badges
  const badgeConfig = {
    SEVERE: {
      bg: "bg-red-50",
      border: "border-red-200",
      accentBg: "bg-red-600",
      text: "text-red-700",
      badgeText: "🔴 SEVERE RISK",
      pulseColor: "bg-red-500",
      shadow: "shadow-red-500/10"
    },
    HIGH: {
      bg: "bg-orange-50",
      border: "border-orange-200",
      accentBg: "bg-orange-500",
      text: "text-orange-700",
      badgeText: "🟠 HIGH RISK",
      pulseColor: "bg-orange-500",
      shadow: "shadow-orange-500/10"
    },
    MODERATE: {
      bg: "bg-amber-50",
      border: "border-amber-200",
      accentBg: "bg-amber-500",
      text: "text-amber-800",
      badgeText: "🟡 MODERATE RISK",
      pulseColor: "bg-amber-500",
      shadow: "shadow-amber-500/10"
    },
    LOW: {
      bg: "bg-emerald-50",
      border: "border-emerald-200",
      accentBg: "bg-emerald-600",
      text: "text-emerald-800",
      badgeText: "🟢 LOW RISK",
      pulseColor: "bg-emerald-500",
      shadow: "shadow-emerald-500/10"
    }
  };

  const config = badgeConfig[location.riskLevel] || badgeConfig.SEVERE;

  return (
    <div
      className={`rounded-2xl border ${config.border} ${config.bg} p-5 shadow-lg ${config.shadow} transition-all relative overflow-hidden`}
    >
      {/* Decorative subtle stripe indicator */}
      <div className={`absolute top-0 left-0 bottom-0 w-2 ${config.accentBg}`} />

      <div className="pl-2 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        {/* Left Side: Hazard & Location Details */}
        <div className="space-y-2 max-w-2xl">
          <div className="flex flex-wrap items-center gap-2.5">
            <span
              className={`inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider text-white ${config.accentBg}`}
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>{config.badgeText}</span>
            </span>

            <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-white border border-slate-200 text-slate-700 shadow-2xs">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
              <span>Public Alert Broadcasted</span>
            </span>
          </div>

          <h2 className="text-xl sm:text-2xl font-black text-slate-900 tracking-tight leading-tight">
            {location.hazard}
          </h2>

          <div className="flex flex-wrap items-center gap-y-1 gap-x-4 text-xs sm:text-sm text-slate-700 font-medium">
            <div className="flex items-center space-x-1.5 text-slate-800">
              <MapPin className="w-4 h-4 text-sky-600 shrink-0" />
              <span className="font-bold">{location.name}</span>
            </div>

            <div className="flex items-center space-x-1.5 text-slate-800">
              <Clock className="w-4 h-4 text-amber-600 shrink-0" />
              <span>Expected: <strong>{location.timeWindow}</strong> (in ~{location.leadTime})</span>
            </div>
          </div>
        </div>

        {/* Right Side: Quick Action & Broadcast Info */}
        <div className="flex flex-col sm:flex-row md:flex-col items-stretch sm:items-center md:items-end gap-2 w-full md:w-auto shrink-0 pt-2 md:pt-0 border-t md:border-t-0 border-slate-200/60">
          <div className="text-right hidden md:block">
            <div className="text-xs text-slate-500">Predicted Risk Intensity</div>
            <div className={`text-2xl font-black ${config.text}`}>
              {location.riskPercent}% <span className="text-xs font-normal text-slate-500">Peak Probability</span>
            </div>
          </div>

          <button
            onClick={onSimulateBroadcast}
            className="inline-flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-md transition-all cursor-pointer active:scale-95"
          >
            <Send className="w-3.5 h-3.5 text-sky-400" />
            <span>View Broadcast Preview</span>
          </button>
        </div>
      </div>
    </div>
  );
}
