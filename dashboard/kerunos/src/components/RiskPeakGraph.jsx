import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot
} from "recharts";
import { TrendingUp, Clock, AlertTriangle, Info } from "lucide-react";

export default function RiskPeakGraph({ data, peakTime = "16:00", peakPercent = 87 }) {
  // Custom Tooltip for non-technical users
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      const isPeak = dataPoint.time === peakTime;

      return (
        <div className="bg-slate-900 text-white p-3 rounded-xl shadow-xl text-xs space-y-1 border border-slate-800">
          <div className="flex items-center justify-between space-x-3 font-semibold text-slate-300">
            <span>Time: {dataPoint.time} IST</span>
            {isPeak && (
              <span className="px-1.5 py-0.5 bg-red-600 text-white font-extrabold rounded-md text-[10px]">
                PEAK
              </span>
            )}
          </div>
          <div className="text-lg font-black text-sky-400">
            {dataPoint.risk}% Risk Probability
          </div>
          <div className="text-slate-300 font-medium">
            Status: <span className="text-amber-300 font-bold">{dataPoint.status}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-4">
      {/* Header & Main Callout Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded-lg bg-red-100 text-red-600 font-bold">
              <TrendingUp className="w-4 h-4" />
            </span>
            <h3 className="text-base font-bold text-slate-900">
              Risk Peak Timeline Graph
            </h3>
          </div>
          <p className="text-xs text-slate-500">
            Predicted risk intensity curve over the next 6 hours
          </p>
        </div>

        {/* Big Prominent Callout Card */}
        <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-2.5 flex items-center space-x-3 shadow-2xs">
          <div className="w-3 h-3 rounded-full bg-red-600 animate-ping shrink-0" />
          <div>
            <div className="text-xs font-bold text-red-900 uppercase tracking-wide">
              🔴 Expected Peak
            </div>
            <div className="text-sm font-extrabold text-red-700">
              Risk peaks at <span className="underline decoration-red-400">{peakTime} IST</span> ({peakPercent}% Peak Risk)
            </div>
          </div>
        </div>
      </div>

      {/* Timeline Sequence Header */}
      <div className="bg-slate-50 rounded-xl p-3 border border-slate-200/80 flex items-center justify-between text-xs text-slate-600 font-medium overflow-x-auto">
        <span className="text-emerald-700 font-bold">Low Risk (13:00)</span>
        <span className="text-slate-400">→</span>
        <span className="text-amber-700 font-bold">Increasing (15:00)</span>
        <span className="text-slate-400">→</span>
        <span className="px-2.5 py-1 bg-red-600 text-white font-black rounded-lg shadow-xs">
          🔴 16:00 PEAK ({peakPercent}%)
        </span>
        <span className="text-slate-400">→</span>
        <span className="text-slate-700 font-bold">Decreasing (17:00)</span>
        <span className="text-slate-400">→</span>
        <span className="text-slate-500 font-bold">Subsiding (19:00)</span>
      </div>

      {/* Recharts Area Curve */}
      <div className="h-56 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.7} />
                <stop offset="50%" stopColor="#f97316" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 12, fill: "#64748b", fontWeight: 600 }}
            />
            <YAxis
              domain={[0, 100]}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 11, fill: "#94a3b8" }}
              unit="%"
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="risk"
              stroke="#dc2626"
              strokeWidth={3.5}
              fillOpacity={1}
              fill="url(#riskGradient)"
            />
            <ReferenceDot
              x={peakTime}
              y={peakPercent}
              r={7}
              fill="#dc2626"
              stroke="#ffffff"
              strokeWidth={3}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between text-xs text-slate-500 pt-1 border-t border-slate-100">
        <span className="flex items-center space-x-1">
          <Info className="w-3.5 h-3.5 text-sky-600" />
          <span>Graph updates automatically every 5 minutes from AI NOWCAST engine.</span>
        </span>
        <span className="font-semibold text-slate-700">Lead Time Window: ~2 Hours</span>
      </div>
    </div>
  );
}
