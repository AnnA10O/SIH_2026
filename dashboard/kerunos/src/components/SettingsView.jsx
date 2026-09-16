import React, { useState } from "react";
import { Settings, Bell, Shield, Radio, Check, Save } from "lucide-react";

export default function SettingsView() {
  const [autoBroadcastThreshold, setAutoBroadcastThreshold] = useState("75");
  const [smsGatewayActive, setSmsGatewayActive] = useState(true);
  const [whatsappActive, setWhatsappActive] = useState(true);
  const [cellBroadcastActive, setCellBroadcastActive] = useState(true);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSave = () => {
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 3000);
  };

  return (
    <div className="max-w-4xl space-y-6">
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-100 pb-4">
          <div className="w-10 h-10 rounded-xl bg-slate-900 text-white flex items-center justify-center font-bold">
            <Settings className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Keraunos Early Warning System Settings
            </h2>
            <p className="text-xs text-slate-500">
              Configure alert trigger thresholds, broadcast channels, and station telemetry parameters
            </p>
          </div>
        </div>

        {/* Form Controls */}
        <div className="space-y-6 pt-2">
          {/* Automated Broadcast Threshold */}
          <div className="space-y-2">
            <label className="block text-sm font-bold text-slate-900">
              Automated Emergency Broadcast Risk Threshold (%)
            </label>
            <p className="text-xs text-slate-500">
              When AI NOWCAST risk probability exceeds this percentage, public alerts are automatically queued.
            </p>
            <div className="flex items-center space-x-4">
              <input
                type="range"
                min="50"
                max="95"
                step="5"
                value={autoBroadcastThreshold}
                onChange={(e) => setAutoBroadcastThreshold(e.target.value)}
                className="w-64 accent-sky-600 cursor-pointer"
              />
              <span className="px-3 py-1 bg-sky-100 text-sky-800 font-extrabold text-sm rounded-lg">
                {autoBroadcastThreshold}% (Severe / High)
              </span>
            </div>
          </div>

          <hr className="border-slate-100" />

          {/* Broadcast Channels Toggle */}
          <div className="space-y-3">
            <label className="block text-sm font-bold text-slate-900">
              Active Public Alert Broadcast Channels
            </label>
            
            <div className="space-y-2">
              <label className="flex items-center justify-between p-3.5 rounded-xl border border-slate-200 bg-slate-50/50 cursor-pointer hover:bg-slate-50">
                <div>
                  <div className="text-xs font-bold text-slate-800">📱 Mobile Push & Cell Broadcast</div>
                  <div className="text-[11px] text-slate-500">Direct location-based mobile push notification to all phones in valley</div>
                </div>
                <input
                  type="checkbox"
                  checked={cellBroadcastActive}
                  onChange={(e) => setCellBroadcastActive(e.target.checked)}
                  className="w-4 h-4 text-sky-600 rounded-sm cursor-pointer"
                />
              </label>

              <label className="flex items-center justify-between p-3.5 rounded-xl border border-slate-200 bg-slate-50/50 cursor-pointer hover:bg-slate-50">
                <div>
                  <div className="text-xs font-bold text-slate-800">💬 Emergency SMS Gateway</div>
                  <div className="text-[11px] text-slate-500">Bulk SMS broadcast to registered local citizens & pilgrim SIMs</div>
                </div>
                <input
                  type="checkbox"
                  checked={smsGatewayActive}
                  onChange={(e) => setSmsGatewayActive(e.target.checked)}
                  className="w-4 h-4 text-sky-600 rounded-sm cursor-pointer"
                />
              </label>

              <label className="flex items-center justify-between p-3.5 rounded-xl border border-slate-200 bg-slate-50/50 cursor-pointer hover:bg-slate-50">
                <div>
                  <div className="text-xs font-bold text-slate-800">🟢 WhatsApp Disaster Warning Template</div>
                  <div className="text-[11px] text-slate-500">Automated rich WhatsApp warning messages with interactive action links</div>
                </div>
                <input
                  type="checkbox"
                  checked={whatsappActive}
                  onChange={(e) => setWhatsappActive(e.target.checked)}
                  className="w-4 h-4 text-sky-600 rounded-sm cursor-pointer"
                />
              </label>
            </div>
          </div>

          <hr className="border-slate-100" />

          {/* Telemetry Frequencies */}
          <div className="space-y-2">
            <label className="block text-sm font-bold text-slate-900">
              Doppler Weather Radar Telemetry Sync Frequency
            </label>
            <select className="px-3.5 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-800 bg-white focus:outline-hidden">
              <option>Every 2 Minutes (High Frequency - Active Severe Mode)</option>
              <option>Every 5 Minutes (Standard Routine Mode)</option>
              <option>Every 15 Minutes (Low Power Mode)</option>
            </select>
          </div>

          {/* Save Button */}
          <div className="pt-2 flex items-center space-x-3">
            <button
              onClick={handleSave}
              className="px-6 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold shadow-md transition-all flex items-center space-x-2 cursor-pointer"
            >
              <Save className="w-4 h-4" />
              <span>Save System Settings</span>
            </button>
            {savedSuccess && (
              <span className="text-xs font-bold text-emerald-600 flex items-center space-x-1">
                <Check className="w-4 h-4" />
                <span>Settings saved successfully!</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
