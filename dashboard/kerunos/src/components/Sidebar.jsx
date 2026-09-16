import React from "react";
import {
  Home,
  Map,
  Brain,
  Bell,
  MessageSquare,
  Star,
  BarChart3,
  Activity,
  ShieldCheck,
  Radio,
  CloudLightning,
  MapPin,
  Menu,
  X
} from "lucide-react";

export default function Sidebar({
  activeTab,
  setActiveTab,
  activeAlertCount = 1,
  unreadMessageCount = 2,
  mobileMenuOpen,
  setMobileMenuOpen,
  locations = [],
  selectedLocation,
  onSelectLocation
}) {
  const navItems = [
    { id: "map", label: "Risk Map", icon: Map },
    { id: "analytics", label: "Analytics", icon: Activity },
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
            <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3.5 text-xs text-slate-600 space-y-2">
              <div className="flex items-center justify-between font-semibold text-slate-700">
                <span className="flex items-center space-x-1.5">
                  <Radio className="w-4 h-4 text-sky-600" />
                  <span>AI Telemetry</span>
                </span>
                <span className="text-[10px] text-emerald-600 bg-emerald-100/80 px-1.5 py-0.5 rounded-md font-bold">
                  94.2% ACC
                </span>
              </div>
              <p className="text-slate-500 leading-relaxed text-[11px]">
                Processing Doppler Doppler & INSAT-3DR satellite vectors continuously.
              </p>
              <div className="pt-1 flex items-center justify-between text-[11px] text-slate-400">
                <span>Model: Keraunos v3.4</span>
                <span className="text-sky-600 hover:underline cursor-pointer">Specs</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
