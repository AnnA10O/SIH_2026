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
    </div>
  );
}
