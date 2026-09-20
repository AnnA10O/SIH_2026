import React from "react";
import { Download, FileText, Database, Activity, Map, BarChart3, Clock, AlertTriangle } from "lucide-react";

export default function AnalyticsView() {
  const datasets = [
    {
      id: "aws-stations",
      name: "All India AWS Stations",
      filename: "all_india_aws_stations.csv",
      description: "Comprehensive metadata for all IMD Automatic Weather Stations across India, including coordinates, elevation, and commission dates.",
      size: "176 KB",
      rows: "3,820",
      icon: Map,
      color: "text-emerald-500",
      bg: "bg-emerald-500/10"
    },
    {
      id: "cloudburst-events",
      name: "Historical Cloudburst Events",
      filename: "cloudburst_events.csv",
      description: "Curated dataset of verified cloudburst events and flash floods used for training the AI model, with precipitation intensity data.",
      size: "488 KB",
      rows: "12,450",
      icon: AlertTriangle,
      color: "text-rose-500",
      bg: "bg-rose-500/10"
    },
    {
      id: "training-logs",
      name: "AI Model Training Logs",
      filename: "training_log.csv",
      description: "Epoch-by-epoch training metrics for the SNN and BiLSTM networks, including loss, CSI, and ROC-AUC convergence rates.",
      size: "3 KB",
      rows: "100",
      icon: Activity,
      color: "text-indigo-500",
      bg: "bg-indigo-500/10"
    },
    {
      id: "station-quality",
      name: "Sensor Quality Audit",
      filename: "station_quality_report.csv",
      description: "Data quality scores and staleness metrics for all integrated AWS and MOSDAC satellite feeds.",
      size: "5 KB",
      rows: "394",
      icon: ShieldCheck = Database, // Fallback to Database
      color: "text-amber-500",
      bg: "bg-amber-500/10"
    },
    {
      id: "region-scores",
      name: "Regional Vulnerability Scores",
      filename: "region_scores.csv",
      description: "Static hydrological vulnerability and slope stability scores for all monitored river basins.",
      size: "1 KB",
      rows: "14",
      icon: BarChart3,
      color: "text-sky-500",
      bg: "bg-sky-500/10"
    }
  ];

  const handleDownload = (filename) => {
    // Construct path to the public outputs directory
    const url = `/outputs/${filename}`;
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-black text-slate-900 tracking-tight flex items-center gap-2">
            <Database className="w-6 h-6 text-sky-600" />
            Statistical Analytics & Datasets
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            Download raw CSV exports of training data, historical events, and real-time sensor metrics used by the Keraunos engine.
          </p>
        </div>
      </div>

      {/* Dataset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {datasets.map((dataset) => {
          const IconComponent = dataset.icon;
          return (
            <div key={dataset.id} className="bg-white rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden flex flex-col h-full">
              <div className="p-5 flex-1 space-y-4">
                <div className="flex items-start justify-between">
                  <div className={`p-3 rounded-xl ${dataset.bg}`}>
                    <IconComponent className={`w-6 h-6 ${dataset.color}`} />
                  </div>
                  <div className="flex items-center gap-2 text-xs font-medium text-slate-400 bg-slate-50 px-2.5 py-1 rounded-full border border-slate-100">
                    <FileText className="w-3.5 h-3.5" />
                    CSV
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-bold text-slate-900 mb-1">{dataset.name}</h3>
                  <p className="text-sm text-slate-500 line-clamp-2">{dataset.description}</p>
                </div>

                <div className="flex items-center gap-4 text-xs font-semibold text-slate-600 pt-2">
                  <div className="flex items-center gap-1.5">
                    <Database className="w-4 h-4 text-slate-400" />
                    {dataset.rows} rows
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-4 h-4 text-slate-400" />
                    {dataset.size}
                  </div>
                </div>
              </div>
              
              <div className="p-4 bg-slate-50 border-t border-slate-100">
                <button
                  onClick={() => handleDownload(dataset.filename)}
                  className="w-full py-2.5 bg-white border border-slate-300 hover:border-sky-500 hover:text-sky-600 text-slate-700 rounded-lg text-sm font-bold shadow-sm transition-all flex items-center justify-center gap-2 group"
                >
                  <Download className="w-4 h-4 text-slate-400 group-hover:text-sky-500 transition-colors" />
                  Download {dataset.filename}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
