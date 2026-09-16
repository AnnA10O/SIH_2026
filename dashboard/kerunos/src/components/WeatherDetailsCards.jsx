import React from "react";
import { MapPin, Clock, Hourglass, CloudRain, AlertTriangle, ShieldAlert } from "lucide-react";

export default function WeatherDetailsCards({ location }) {
  if (!location) return null;

  const details = [
    {
      id: "location",
      label: "Location",
      value: location.name,
      subtext: location.region,
      icon: MapPin,
      iconBg: "bg-sky-100 text-sky-700",
      borderColor: "border-sky-200"
    },
    {
      id: "time",
      label: "Expected Time",
      value: location.timeWindow,
      subtext: `Peak forecasted at ${location.expectedPeak}`,
      icon: Clock,
      iconBg: "bg-amber-100 text-amber-700",
      borderColor: "border-amber-200"
    },
    {
      id: "leadTime",
      label: "Lead Time",
      value: location.leadTime,
      subtext: "Advance early warning window",
      icon: Hourglass,
      iconBg: "bg-blue-100 text-blue-700",
      borderColor: "border-blue-200"
    },
    {
      id: "rainfall",
      label: "Rainfall Intensity",
      value: location.rainfallRate,
      subtext: "Extremely heavy microburst precipitation",
      icon: CloudRain,
      iconBg: "bg-indigo-100 text-indigo-700",
      borderColor: "border-indigo-200"
    },
    {
      id: "risk",
      label: "Severe Hazard",
      value: location.hazard,
      subtext: `Risk Level: ${location.riskLevel} (${location.riskPercent}%)`,
      icon: AlertTriangle,
      iconBg: "bg-red-100 text-red-700",
      borderColor: "border-red-200"
    }
  ];

  const SnnGate = ({ gateId, active, label, iconSvg }) => (
    <div className={`flex flex-col items-center p-3 rounded-xl border ${active ? 'bg-indigo-50 border-indigo-400 shadow-md' : 'bg-slate-50 border-slate-200'}`}>
      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">{label}</span>
      <div className={`p-2 rounded-lg ${active ? 'bg-indigo-500 text-white shadow-[0_0_15px_rgba(99,102,241,0.6)] animate-bounce' : 'bg-slate-200 text-slate-400'}`}>
        {iconSvg}
      </div>
      <span className={`text-xs font-black mt-2 ${active ? 'text-indigo-700' : 'text-slate-400'}`}>{active ? 'DETECTED' : 'STANDBY'}</span>
    </div>
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center space-x-2">
          <span>Simple Weather Details</span>
          <span className="text-[10px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-semibold">
            Non-Technical View
          </span>
        </h3>
        <span className="text-xs text-slate-500">Live Telemetry Synced</span>
      </div>

      {/* Primary Telemetry Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
        {details.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.id}
              className={`bg-white rounded-2xl border ${item.borderColor} p-4 shadow-2xs hover:shadow-md transition-shadow flex flex-col justify-between space-y-2`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                  {item.label}
                </span>
                <div className={`p-2 rounded-xl ${item.iconBg}`}>
                  <Icon className="w-4 h-4" />
                </div>
              </div>

              <div>
                <div className="text-base font-black text-slate-900 leading-tight">
                  {item.value}
                </div>
                <div className="text-[11px] text-slate-500 mt-1 line-clamp-1 font-medium">
                  {item.subtext}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* SNN Gate AI Detection */}
      <div className="mt-4 pt-3 border-t border-slate-200">
        <h4 className="text-xs font-bold text-slate-700 uppercase tracking-widest mb-3 flex items-center"><ShieldAlert className="w-4 h-4 mr-1.5 text-sky-600"/> SNN Pattern Recognition</h4>
        <div className="grid grid-cols-2 gap-4">
          <SnnGate 
            gateId="a" 
            label="Gate A (Cloudburst)" 
            active={location.gate_a} 
            iconSvg={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-6 h-6"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>}
          />
          <SnnGate 
            gateId="b" 
            label="Gate B (Thunderstorm)" 
            active={location.gate_b} 
            iconSvg={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-6 h-6"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>}
          />
        </div>
      </div>
    </div>
  );
}
