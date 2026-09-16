import React, { useMemo, useState } from "react";
import { MapPin } from "lucide-react";

export default function AuthorityReportsPanel({ reports = [], onOpenModal }) {
  const [activeTab, setActiveTab] = useState("recent"); // "recent" or "previous"
  
  // Classification Logic
  const cutoffDate = new Date("2026-09-12");
  cutoffDate.setHours(0, 0, 0, 0);

  const { recentReports, previousReports } = useMemo(() => {
    const recent = [];
    const previous = [];

    reports.forEach(report => {
      if (!report.eventDate) return;
      const eventD = new Date(report.eventDate);
      eventD.setHours(0, 0, 0, 0);
      if (eventD >= cutoffDate) {
        recent.push(report);
      } else {
        previous.push(report);
      }
    });

    // Sort descending by eventDate
    recent.sort((a, b) => new Date(b.eventDate) - new Date(a.eventDate));
    previous.sort((a, b) => new Date(b.eventDate) - new Date(a.eventDate));

    return { recentReports: recent, previousReports: previous };
  }, [reports]);

  const displayedReports = activeTab === "recent" ? recentReports : previousReports;

  return (
    <div className="space-y-6">
      
      {/* Top Actions & Tabs Area */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
        
        {/* Tabs */}
        <div className="flex bg-slate-50 p-1 rounded border border-slate-200 shrink-0">
          <button
            onClick={() => setActiveTab("recent")}
            className={`px-4 py-2 text-xs font-bold uppercase tracking-wider rounded transition-all ${
              activeTab === "recent"
                ? "bg-orange-500/10 text-orange-500 border border-orange-500/30 shadow-sm"
                : "text-slate-500 hover:text-slate-900 hover:bg-white border border-transparent"
            }`}
          >
            Recently Affected
          </button>
          <button
            onClick={() => setActiveTab("previous")}
            className={`px-4 py-2 text-xs font-bold uppercase tracking-wider rounded transition-all ${
              activeTab === "previous"
                ? "bg-white border border-slate-300 text-slate-900 shadow-sm"
                : "text-slate-500 hover:text-slate-900 hover:bg-white border border-transparent"
            }`}
          >
            Previously Affected
          </button>
        </div>

        {/* Submit Feedback Button */}
        <button
          onClick={onOpenModal}
          className="px-4 py-2 bg-slate-50 border border-slate-300 hover:brightness-95 text-slate-900 rounded shadow-sm text-xs font-bold transition-all uppercase tracking-wider"
        >
          + Submit Feedback
        </button>
      </div>

      {/* Feedback List Container */}
      <div className="bg-white rounded border border-slate-200 shadow-sm overflow-hidden">
        
        {displayedReports.length > 0 ? (
          <div className="divide-y divide-border-subtle">
            {displayedReports.map(report => (
              <div key={report.id} className="p-4 hover:bg-slate-50 transition-colors">
                
                {/* Compact Header: Date & Location */}
                <div className="flex items-center space-x-3 mb-2">
                  <div className="font-mono text-xs font-bold text-slate-900 bg-slate-50 border border-slate-200 px-1.5 py-0.5 rounded shrink-0">
                    {report.eventDate}
                  </div>
                  <div className="text-xs font-bold text-slate-900 flex items-center">
                    <MapPin className="w-3.5 h-3.5 mr-1 text-slate-500 shrink-0" />
                    {report.location}
                  </div>
                </div>

                {/* Event Type */}
                <div className="text-sm font-black text-slate-900 mb-3">
                  {report.eventType}
                </div>

                {/* Details */}
                <div className="space-y-3 text-xs text-slate-700">
                  
                  {/* Missing */}
                  {report.missingElements && report.missingElements.length > 0 && (
                    <div>
                      <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">Missing:</span>
                      <span className="font-medium text-slate-900">{report.missingElements.join(", ")}</span>
                    </div>
                  )}

                  {/* Problems Faced */}
                  {report.problemsFaced && (
                    <div>
                      <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">Problem:</span>
                      <span className="font-medium text-slate-900 italic">"{report.problemsFaced}"</span>
                    </div>
                  )}

                  {/* Information Needed */}
                  {report.missingInfo && (
                    <div>
                      <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">Information Needed:</span>
                      <span className="font-medium text-slate-900">{report.missingInfo}</span>
                    </div>
                  )}

                  {/* Warning Received */}
                  {report.warningTime && (
                    <div>
                      <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">Warning Received:</span>
                      <span className="font-medium text-slate-900">{report.warningTime}</span>
                    </div>
                  )}

                  {/* What would have helped */}
                  {report.helpfulAdditions && (
                    <div>
                      <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">What would have helped:</span>
                      <span className="font-medium text-slate-900">{report.helpfulAdditions}</span>
                    </div>
                  )}
                  
                </div>

                {/* Submission Date Note */}
                {report.submissionDate && (
                  <div className="mt-4 text-[10px] text-slate-500 border-t border-slate-200 pt-2 inline-block">
                    Submitted: {report.submissionDate}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="p-12 text-center text-slate-500 text-sm flex flex-col items-center justify-center space-y-2">
            <span className="text-2xl">📋</span>
            <span>
              {activeTab === "recent" 
                ? "No recently affected feedback yet." 
                : "No previously affected feedback yet."}
            </span>
          </div>
        )}
      </div>

    </div>
  );
}
