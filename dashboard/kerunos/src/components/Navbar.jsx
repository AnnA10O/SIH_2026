import React, { useState, useEffect } from "react";
import { CloudLightning, MapPin, Radio, Activity, Menu, X, ShieldAlert } from "lucide-react";

export default function Navbar({
  locations,
  selectedLocation,
  onSelectLocation,
  mobileMenuOpen,
  setMobileMenuOpen,
  simulateDisaster,
  setSimulateDisaster
}) {
  const [timeString, setTimeString] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const options = {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZoneName: "short"
      };
      setTimeString(now.toLocaleTimeString("en-US", options));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left: Brand Logo & Title */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 focus:outline-hidden"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
            <div className="flex items-center space-x-2.5 cursor-pointer">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-blue-700 flex items-center justify-center text-white shadow-md shadow-sky-500/20">
                <CloudLightning className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-extrabold text-xl tracking-tight text-slate-900">
                    KERAUNOS
                  </span>
                  <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-semibold tracking-wider text-sky-700 bg-sky-50 border border-sky-200 rounded-full">
                    AI NOWCAST
                  </span>
                </div>
                <p className="text-xs text-slate-500 hidden md:block">
                  Severe Weather Early Warning System
                </p>
              </div>
            </div>
          </div>

          {/* Center: Active Location Dropdown */}
          <div className="flex items-center space-x-2">
            <div className="relative flex items-center bg-slate-100 hover:bg-slate-200/80 transition-colors border border-slate-200 rounded-xl px-3 py-1.5 text-sm font-medium text-slate-800 cursor-pointer shadow-xs">
              <MapPin className="w-4 h-4 text-sky-600 mr-2 shrink-0" />
              <select
                value={selectedLocation.id}
                onChange={(e) => {
                  const loc = locations.find((l) => l.id === e.target.value);
                  if (loc) onSelectLocation(loc);
                }}
                className="bg-transparent text-slate-800 text-sm font-semibold focus:outline-hidden cursor-pointer pr-4"
              >
                {locations.map((loc) => (
                  <option key={loc.id} value={loc.id}>
                    {loc.name} ({loc.riskLevel} Risk)
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Right: Live Clock & System Status */}
          <div className="flex items-center space-x-4">
            <div className="hidden lg:flex flex-col items-end text-xs text-slate-500 font-mono">
              <span className="text-slate-700 font-semibold">{timeString || "15:02:44 IST"}</span>
              <span className="text-[11px] text-slate-400">Target Lead Time: 2–6 Hours</span>
            </div>
            
            <button
              onClick={() => setSimulateDisaster(!simulateDisaster)}
              className={`flex items-center space-x-1.5 px-3 py-1.5 text-xs font-bold rounded-full shadow-xs transition-all ${
                simulateDisaster 
                  ? 'bg-red-600 text-white animate-pulse shadow-[0_0_12px_rgba(220,38,38,0.7)] border border-red-500' 
                  : 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
              }`}
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>SIMULATE DISASTER</span>
            </button>

            <div className="flex items-center space-x-2 px-3 py-1.5 bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-semibold rounded-full shadow-xs">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span>Online</span>
              <span className="hidden sm:inline text-[11px] opacity-80 border-l border-emerald-200 pl-2 ml-1">
                Radar Synced
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
