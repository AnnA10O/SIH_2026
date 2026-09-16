import React, { useState, useEffect, useRef } from "react";
import {
  Home,
  Map,
  Brain,
  Bell,
  MessageSquare,
  Star,
  BarChart3,
  ShieldCheck,
  Radio,
  CloudLightning,
  MapPin,
  Menu,
  X,
  Terminal,
  ShieldAlert
} from "lucide-react";

export default function Sidebar({
  activeTab,
  setActiveTab,
  activeAlertCount = 1,
  unreadMessageCount = 2,
  mobileMenuOpen: propMobileMenuOpen,
  setMobileMenuOpen: propSetMobileMenuOpen,
  locations = [],
  selectedLocation,
  onSelectLocation,
  simulateDisaster,
  setSimulateDisaster
}) {
  const [internalMobileMenuOpen, setInternalMobileMenuOpen] = useState(false);
  const mobileMenuOpen = propMobileMenuOpen !== undefined ? propMobileMenuOpen : internalMobileMenuOpen;
  const setMobileMenuOpen = propSetMobileMenuOpen || setInternalMobileMenuOpen;

  const [telemetry, setTelemetry] = useState({ imd: 'online', mosdac: 'online', logs: [] });
  const terminalRef = useRef(null);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/telemetry");
        if (res.ok) {
          setTelemetry(await res.json());
        }
      } catch (err) {
        console.error("Telemetry error", err);
      }
    };
    fetchTelemetry();
    const int = setInterval(fetchTelemetry, 3000);
    return () => clearInterval(int);
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [telemetry.logs]);

  const navItems = [
    { id: "map", label: "Risk Map", icon: Map },
    { id: "pinn", label: "PINN", icon: Brain, badge: "AI Physics", badgeColor: "bg-purple-600" },
    { id: "alerts", label: "Alerts", icon: Bell, badge: activeAlertCount, badgeColor: "bg-red-500" },
    { id: "chat", label: "Station Chat", icon: MessageSquare, badge: unreadMessageCount, badgeColor: "bg-sky-500" },
    { id: "feedback", label: "Feedback", icon: Star },
    { id: "reports", label: "Reports", icon: BarChart3 }
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileMenuOpen && (
        <div
          onClick={() => setMobileMenuOpen(false)}
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-20 lg:hidden"
        />
      )}

      {/* Mobile Menu Toggle (Visible only on small screens) */}
      <button
        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        className="lg:hidden fixed top-4 left-4 z-30 p-2 rounded-lg bg-white shadow-md text-slate-500 hover:text-slate-700"
        aria-label="Toggle menu"
      >
        {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-20 w-64 bg-white border-r border-slate-200 transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex flex-col h-full p-4 overflow-y-auto">
          {/* Brand Logo & Title */}
          <div className="flex items-center space-x-2.5 mb-6">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-blue-700 flex items-center justify-center text-white shadow-md shadow-sky-500/20 shrink-0">
              <CloudLightning className="w-6 h-6" />
            </div>
            <div>
              <div className="flex flex-col">
                <span className="font-extrabold text-lg tracking-tight text-slate-900 leading-tight">
                  KERAUNOS
                </span>
                <span className="inline-block px-1.5 py-0.5 text-[9px] font-semibold tracking-wider text-sky-700 bg-sky-50 border border-sky-200 rounded-full w-max">
                  AI NOWCAST
                </span>
              </div>
            </div>
          </div>

          {/* Location Selector */}
          {locations && locations.length > 0 && selectedLocation && (
            <div className="mb-6 relative flex items-center bg-slate-100 hover:bg-slate-200/80 transition-colors border border-slate-200 rounded-xl px-3 py-2 text-sm font-medium text-slate-800 cursor-pointer shadow-xs">
              <MapPin className="w-4 h-4 text-sky-600 mr-2 shrink-0" />
              <select
                value={selectedLocation.id}
                onChange={(e) => {
                  const loc = locations.find((l) => l.id === e.target.value);
                  if (loc) onSelectLocation(loc);
                }}
                className="bg-transparent text-slate-800 text-xs font-semibold focus:outline-hidden cursor-pointer w-full"
              >
                {locations.map((loc) => (
                  <option key={loc.id} value={loc.id}>
                    {loc.name} ({loc.riskLevel})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Simulate Disaster Button */}
          {setSimulateDisaster && (
            <button
              onClick={() => setSimulateDisaster(!simulateDisaster)}
              className={`w-full mb-6 flex items-center justify-center space-x-2 px-4 py-2.5 text-xs font-bold rounded-xl shadow-xs transition-all ${
                simulateDisaster 
                  ? 'bg-red-600 text-white animate-pulse shadow-[0_0_15px_rgba(220,38,38,0.7)] border border-red-500' 
                  : 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
              }`}
            >
              <ShieldAlert className="w-4 h-4" />
              <span>SIMULATE DISASTER</span>
            </button>
          )}

          <nav className="space-y-1.5 mb-auto">
            <div className="px-3 py-2 text-xs font-bold text-slate-400 uppercase tracking-wider">
              Navigation
            </div>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveTab(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? "bg-sky-50 text-sky-700 shadow-xs border border-sky-100 font-semibold"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/80"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <Icon
                      className={`w-5 h-5 ${
                        isActive ? "text-sky-600" : "text-slate-400"
                      }`}
                    />
                    <span>{item.label}</span>
                  </div>
                  {item.badge ? (
                    <span
                      className={`px-2 py-0.5 text-xs font-bold text-white rounded-full ${item.badgeColor}`}
                    >
                      {item.badge}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </nav>

          {/* System Telemetry & Quick Info Card */}
          <div className="mt-4 pt-4 border-t border-slate-100">
            <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3 text-xs text-slate-600 space-y-3 shadow-inner">
              <div className="flex items-center justify-between font-semibold text-slate-700">
                <span className="flex items-center space-x-1.5">
                  <Radio className="w-4 h-4 text-sky-600 animate-pulse" />
                  <span>API Telemetry</span>
                </span>
                <span className="text-[10px] text-emerald-600 bg-emerald-100/80 px-1.5 py-0.5 rounded-md font-bold border border-emerald-200">
                  LIVE
                </span>
              </div>
              
              <div className="flex justify-between items-center text-[10.5px]">
                <div className="flex items-center space-x-1">
                  <div className={`w-2 h-2 rounded-full ${telemetry.imd === 'online' ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]' : 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)] animate-pulse'}`} />
                  <span className="font-medium">IMD / O-Meteo</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className={`w-2 h-2 rounded-full ${telemetry.mosdac === 'online' ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]' : 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.6)]'}`} />
                  <span className="font-medium">MOSDAC</span>
                </div>
              </div>

              {/* Mini Terminal Viewer */}
              <div className="bg-slate-900 border border-slate-700 rounded-lg p-2 overflow-hidden shadow-md">
                <div className="flex items-center space-x-1 mb-1 opacity-70">
                  <Terminal className="w-3 h-3 text-sky-400" />
                  <span className="text-[9px] font-mono text-slate-400 uppercase tracking-widest">inference.log</span>
                </div>
                <div 
                  ref={terminalRef}
                  className="font-mono text-[9px] text-emerald-400 h-24 overflow-y-auto leading-tight space-y-1 custom-scrollbar"
                >
                  {telemetry.logs.length > 0 ? telemetry.logs.map((log, idx) => (
                    <div key={idx} className="break-all whitespace-pre-wrap opacity-90 hover:opacity-100 transition-opacity">
                      {log.replace(/^\[.*?\]\s*/, '')}
                    </div>
                  )) : (
                    <div className="text-slate-500 italic">Waiting for AI telemetry...</div>
                  )}
                </div>
              </div>

              <div className="pt-1 flex items-center justify-between text-[10px] text-slate-400 font-medium">
                <span>Model: Keraunos v3.4</span>
                <span className="text-sky-600 hover:underline cursor-pointer">Re-sync</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
