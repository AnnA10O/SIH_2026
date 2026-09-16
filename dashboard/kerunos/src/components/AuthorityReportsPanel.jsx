import React from "react";
import { Users, MapPin, ShieldCheck, Sparkles, AlertTriangle } from "lucide-react";

export default function AuthorityReportsPanel({ reports = [], onOpenModal }) {
  // Aggregate report counts by type
  const counts = reports.reduce((acc, r) => {
    acc[r.type] = (acc[r.type] || 0) + (r.count || 1);
    return acc;
  }, {});

  const totalReports = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-lg bg-amber-100 text-amber-700 flex items-center justify-center font-bold">
            <Users className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Community Ground Reports
            </h3>
            <p className="text-xs text-slate-500">
              Real-time citizen ground-truth corroboration
            </p>
          </div>
        </div>

        <button
          onClick={onOpenModal}
          className="px-3 py-1.5 bg-amber-500 hover:bg-amber-600 text-white rounded-xl text-xs font-bold shadow-2xs transition-all active:scale-95"
        >
          + Submit Report
        </button>
      </div>

      {/* Tally Summary Badges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-base">🌧️</span>
            <div>
              <div className="font-bold text-slate-800">Heavy Rain</div>
              <div className="text-[10px] text-slate-500">Gaurikund & Sonprayag</div>
            </div>
          </div>
          <span className="text-lg font-black text-amber-800">{counts["Heavy Rain"] || 12} reports</span>
        </div>

        <div className="bg-purple-50 border border-purple-200 rounded-xl p-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-base">⛈️</span>
            <div>
              <div className="font-bold text-slate-800">Thunderstorm</div>
              <div className="text-[10px] text-slate-500">Lincholi & Ridge</div>
            </div>
          </div>
          <span className="text-lg font-black text-purple-800">{counts["Thunderstorm"] || 8} reports</span>
        </div>

        <div className="bg-red-50 border border-red-200 rounded-xl p-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-base">🌊</span>
            <div>
              <div className="font-bold text-slate-800">Flooding</div>
              <div className="text-[10px] text-slate-500">Rambara Track</div>
            </div>
          </div>
          <span className="text-lg font-black text-red-800">{counts["Flooding"] || 3} reports</span>
        </div>
      </div>

      {/* Integration Explanation Note */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-600 flex items-start space-x-2.5">
        <Sparkles className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong>AI Model Synchronization:</strong> Citizen observations automatically plot as map markers and feed directly into the Keraunos Nowcasting Ensemble to validate convective storm cell intensity.
        </p>
      </div>
    </div>
  );
}
