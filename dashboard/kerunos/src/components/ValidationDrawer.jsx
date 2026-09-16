import React, { useState } from "react";
import { ChevronUp, ChevronDown, Target, ShieldCheck, AlertCircle, History } from "lucide-react";

export default function ValidationDrawer({ metrics = {} }) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Safely format metrics into percentages with 1 decimal place to prevent rounding 99.7% to 100%
  const formatPct = (val) => val != null ? `${(val * 100).toFixed(1)}%` : "--%";
  const accPct = formatPct(metrics.accuracy);
  const catchPct = formatPct(metrics.catchRate);
  const farPct = formatPct(metrics.falseAlarmRate);
  // We don't have a direct historical replay metric from the live stream, so we use F1 score as a proxy for historical robustness
  const f1Pct = formatPct(metrics.f1Score);

  const cards = [
    {
      id: "accuracy",
      icon: Target,
      iconColor: "text-sky-600 bg-sky-50 border-sky-200",
      label: "Accuracy",
      value: accPct,
      note: "Live Inference Pipeline"
    },
    {
      id: "catchRate",
      icon: ShieldCheck,
      iconColor: "text-emerald-600 bg-emerald-50 border-emerald-200",
      label: "Catch Rate (POD)",
      value: catchPct,
      note: "Live Inference Pipeline"
    },
    {
      id: "falseAlarm",
      icon: AlertCircle,
      iconColor: "text-amber-600 bg-amber-50 border-amber-200",
      label: "False Alarm Rate",
      value: farPct,
      note: "Live Inference Pipeline"
    },
    {
      id: "historicalReplay",
      icon: History,
      iconColor: "text-indigo-600 bg-indigo-50 border-indigo-200",
      label: "Model F1 Score",
      value: f1Pct,
      note: "Live Inference Pipeline"
    }
  ];

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 bg-white rounded-t-2xl border-t border-slate-200 shadow-[0_-8px_30px_rgba(0,0,0,0.12)] transition-all duration-300 ease-in-out font-sans">
      {/* Clickable Header Bar */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="px-4 py-3 sm:px-6 flex flex-wrap items-center justify-between gap-3 cursor-pointer select-none bg-slate-50/80 hover:bg-slate-100/90 rounded-t-2xl transition-colors"
      >
        {/* Title & Chevron */}
        <div className="flex items-center space-x-2.5">
          <div className="p-1 rounded-md bg-slate-200 text-slate-700">
            {isExpanded ? (
              <ChevronDown className="w-5 h-5 text-slate-800" />
            ) : (
              <ChevronUp className="w-5 h-5 text-slate-800" />
            )}
          </div>
          <h3 className="text-sm font-extrabold text-slate-900 tracking-tight">
            Model performance & validation
          </h3>
        </div>

        {/* Compact Summary Statistics Inline (Visible when collapsed & expanded) */}
        <div className="flex items-center space-x-3 sm:space-x-5 text-xs font-semibold text-slate-700">
          <div className="flex items-center space-x-1">
            <span className="text-slate-400">Catch Rate:</span>
            <span className="text-emerald-700 font-extrabold">{catchPct}</span>
          </div>
          <span className="text-slate-300">•</span>
          <div className="flex items-center space-x-1">
            <span className="text-slate-400">False Alarm:</span>
            <span className="text-amber-700 font-extrabold">{farPct}</span>
          </div>
          <span className="text-slate-300 hidden sm:inline">•</span>
          <div className="flex items-center space-x-1 hidden sm:flex">
            <span className="text-slate-400">Lead Time:</span>
            <span className="text-sky-700 font-extrabold">~4.0 hrs</span>
          </div>
        </div>
      </div>

      {/* Expanded Content Body */}
      {isExpanded && (
        <div className="p-4 sm:p-6 bg-white border-t border-slate-100 animate-in fade-in duration-200">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 max-w-7xl mx-auto">
            {cards.map((card) => {
              const Icon = card.icon;
              return (
                <div
                  key={card.id}
                  className="bg-slate-50/70 rounded-xl border border-slate-200/80 p-4 flex flex-col justify-between space-y-2 shadow-2xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-600">
                      {card.label}
                    </span>
                    <div className={`p-1.5 rounded-lg border ${card.iconColor}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                  </div>

                  <div>
                    <div className="text-2xl font-black text-slate-900 leading-none tracking-tight">
                      {card.value}
                    </div>
                    <div className="text-[11px] text-slate-500 font-medium mt-1.5">
                      {card.note}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
