import React from "react";
import { BarChart3, ShieldCheck, CheckCircle2, FileText, Download, Award, Zap } from "lucide-react";

export default function ReportsView({ metrics }) {
  const recentAlertHistory = [
    {
      id: "K-9402",
      date: "Today, 14:15 IST",
      location: "Mandakini Valley, Kedarnath",
      hazard: "Cloudburst & Heavy Rainfall",
      leadTime: "2.5 Hours",
      status: "ACTIVE BROADCAST",
      accuracy: "96.4%"
    },
    {
      id: "K-9398",
      date: "Yesterday, 16:30 IST",
      location: "Alaknanda Basin, Joshimath",
      hazard: "Flash Flood Warning",
      leadTime: "3.1 Hours",
      status: "RESOLVED",
      accuracy: "94.8%"
    },
    {
      id: "K-9380",
      date: "12 Sep 2026, 11:00 IST",
      location: "Bhagirathi Valley, Uttarkashi",
      hazard: "Severe Microburst",
      leadTime: "4.0 Hours",
      status: "RESOLVED",
      accuracy: "93.1%"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 to-sky-950 text-white rounded-2xl p-6 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded-lg bg-sky-500/20 text-sky-400">
              <BarChart3 className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-bold tracking-tight">
              AI Nowcasting Analytics & Performance Report
            </h2>
          </div>
          <p className="text-xs text-slate-300">
            Real-time validation log of cloudburst early warnings and satellite radar accuracy
          </p>
        </div>

        <button className="px-4 py-2.5 bg-sky-600 hover:bg-sky-500 text-white rounded-xl text-xs font-bold shadow-md transition-all flex items-center space-x-2 shrink-0 cursor-pointer">
          <Download className="w-4 h-4" />
          <span>Export Monthly PDF Report</span>
        </button>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-2xs space-y-1">
          <div className="text-xs font-bold text-slate-400 uppercase">Overall Precision</div>
          <div className="text-2xl font-black text-emerald-600">94.2%</div>
          <div className="text-[11px] text-slate-500">Validated against ground AWS rain gauges</div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-2xs space-y-1">
          <div className="text-xs font-bold text-slate-400 uppercase">Average Lead Time</div>
          <div className="text-2xl font-black text-sky-600">3.4 Hours</div>
          <div className="text-[11px] text-slate-500">Target window: 2 to 6 hours</div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-2xs space-y-1">
          <div className="text-xs font-bold text-slate-400 uppercase">Incidents Early-Warned</div>
          <div className="text-2xl font-black text-indigo-600">148 Events</div>
          <div className="text-[11px] text-slate-500">Zero false-negative cloudburst alerts</div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-2xs space-y-1">
          <div className="text-xs font-bold text-slate-400 uppercase">Sensors Integrated</div>
          <div className="text-2xl font-black text-amber-600">184 Feeds</div>
          <div className="text-[11px] text-slate-500">Doppler Radar + Satellite + AWS</div>
        </div>
      </div>

      {/* Historical Alert Log Table */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
            <FileText className="w-4 h-4 text-sky-600" />
            <span>Recent Alert Audit Log</span>
          </h3>
          <span className="text-xs text-slate-500">Showing last 3 active nowcasts</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-500 uppercase font-bold text-[10px]">
              <tr>
                <th className="px-4 py-3">Alert ID</th>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Location</th>
                <th className="px-4 py-3">Hazard Type</th>
                <th className="px-4 py-3">Lead Time</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3 text-right">AI Accuracy Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
              {recentAlertHistory.map((row) => (
                <tr key={row.id} className="hover:bg-slate-50/80 transition-colors">
                  <td className="px-4 py-3 font-mono font-bold text-slate-900">{row.id}</td>
                  <td className="px-4 py-3 text-slate-500">{row.date}</td>
                  <td className="px-4 py-3 font-bold text-slate-800">{row.location}</td>
                  <td className="px-4 py-3">{row.hazard}</td>
                  <td className="px-4 py-3 text-sky-700 font-bold">{row.leadTime}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        row.status === "ACTIVE BROADCAST"
                          ? "bg-red-100 text-red-700 border border-red-200"
                          : "bg-emerald-100 text-emerald-700"
                      }`}
                    >
                      {row.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono font-bold text-emerald-600">
                    {row.accuracy}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
