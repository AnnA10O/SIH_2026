import React, { useState } from "react";
import { CheckCircle2, X, Send, Calendar, MapPin, AlertTriangle } from "lucide-react";

export default function FeedbackModal({ isOpen, onClose, onSubmitReport, currentLocationName }) {
  const [eventDate, setEventDate] = useState("");
  const [location, setLocation] = useState(currentLocationName || "");
  const [eventType, setEventType] = useState("Flash Flood");
  const [missingElements, setMissingElements] = useState([]);
  const [problemsFaced, setProblemsFaced] = useState("");
  const [missingInfo, setMissingInfo] = useState("");
  const [warningTime, setWarningTime] = useState("");
  const [helpfulAdditions, setHelpfulAdditions] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [dateError, setDateError] = useState("");

  if (!isOpen) return null;

  const eventTypeOptions = [
    "Flash Flood",
    "Cloudburst",
    "Heavy Rainfall",
    "Landslide",
    "Water Level Rise",
    "Road Blockage",
    "Bridge Damage",
    "Other"
  ];

  const missingOptions = [
    "Early Warning",
    "Warning came too late",
    "Clear evacuation instructions",
    "Safe evacuation route information",
    "Communication / Network",
    "Emergency Services",
    "Transportation",
    "Shelter",
    "Food / Water",
    "Medical Assistance",
    "Information about nearby danger",
    "Other"
  ];

  const warningTimeOptions = [
    "No warning",
    "Less than 15 minutes",
    "15–30 minutes",
    "30–60 minutes",
    "1–2 hours",
    "More than 2 hours",
    "Don't know"
  ];

  const handleCheckboxToggle = (option) => {
    setMissingElements(prev =>
      prev.includes(option)
        ? prev.filter(item => item !== option)
        : [...prev, option]
    );
  };

  const getCategory = (dateString) => {
    if (!dateString) return null;
    const cutoffDate = new Date("2026-09-12");
    const selectedDate = new Date(dateString);
    return selectedDate >= cutoffDate ? "recent" : "previous";
  };

  const handleDateChange = (e) => {
    const val = e.target.value;
    const selectedDate = new Date(val);
    const today = new Date();
    
    // Reset time for fair comparison
    today.setHours(0, 0, 0, 0);
    selectedDate.setHours(0, 0, 0, 0);

    if (selectedDate > today) {
      setDateError("Please enter a valid event date. (Cannot be in the future)");
      setEventDate("");
    } else {
      setDateError("");
      setEventDate(val);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!eventDate) {
      setDateError("Please enter the date when the event occurred.");
      return;
    }

    const todayStr = new Date().toISOString().split("T")[0];

    onSubmitReport({
      eventDate,
      submissionDate: todayStr,
      location: location || "Unknown Location",
      eventType,
      missingElements,
      problemsFaced,
      missingInfo,
      warningTime: warningTime || "Don't know",
      helpfulAdditions
    });

    setIsSubmitted(true);
  };

  const handleReset = () => {
    setIsSubmitted(false);
    setEventDate("");
    setLocation(currentLocationName || "");
    setEventType("Flash Flood");
    setMissingElements([]);
    setProblemsFaced("");
    setMissingInfo("");
    setWarningTime("");
    setHelpfulAdditions("");
    setDateError("");
    onClose();
  };

  const category = getCategory(eventDate);

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded border border-slate-200 shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="bg-slate-50 border-b border-slate-200 p-4 flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded bg-sky-600/10 text-sky-600">
              <AlertTriangle className="w-5 h-5" />
            </span>
            <div>
              <h3 className="font-bold text-sm tracking-tight text-slate-900">
                Post-Event Feedback & Ground Report
              </h3>
              <p className="text-[10px] text-slate-500">Help us improve the disaster response system</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded text-slate-500 hover:text-slate-900 hover:bg-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="overflow-y-auto p-5 no-scrollbar">
          {isSubmitted ? (
            <div className="py-12 text-center space-y-4">
              <div className="w-16 h-16 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mx-auto shadow-inner">
                <CheckCircle2 className="w-10 h-10" />
              </div>
              <div className="space-y-1">
                <h4 className="text-lg font-bold text-slate-900">
                  Feedback Successfully Submitted
                </h4>
                <p className="text-xs text-slate-700 leading-relaxed max-w-sm mx-auto">
                  Your ground-level insights have been recorded. This data directly helps improve disaster readiness and early warning models for future events.
                </p>
              </div>
              <div className="pt-4">
                <button
                  onClick={handleReset}
                  className="px-8 py-2.5 bg-slate-50 border border-slate-300 hover:brightness-95 text-slate-900 font-bold text-xs rounded shadow-sm transition-all cursor-pointer uppercase tracking-wider"
                >
                  Close
                </button>
              </div>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6 text-left pb-4">
              
              {/* Event Date */}
              <div className="bg-slate-50 p-4 rounded border border-slate-200">
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  EVENT DATE
                </label>
                <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
                  <div className="relative w-full sm:w-auto">
                    <input
                      type="date"
                      value={eventDate}
                      onChange={handleDateChange}
                      className="w-full sm:w-48 px-3.5 py-2 pl-9 rounded border border-slate-300 bg-white text-slate-900 text-xs font-medium focus:ring-1 focus:ring-sky-600 focus:outline-hidden"
                    />
                    <Calendar className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                  </div>
                  {dateError && (
                    <span className="text-xs text-red-600 font-bold bg-red-50 px-2 py-1 rounded">
                      {dateError}
                    </span>
                  )}
                </div>

                {/* Auto-Classification Banner */}
                {category === "recent" && (
                  <div className="mt-4 p-3 border border-orange-500/30 bg-orange-500/10 rounded flex items-center justify-between animate-in fade-in zoom-in-95">
                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-wider text-orange-500 mb-0.5">Recently Affected</div>
                      <div className="text-xs text-slate-900 font-semibold">Current / Recent Event</div>
                    </div>
                    <div className="text-xs font-mono font-bold text-orange-500 bg-orange-500/20 px-2 py-1 rounded">
                      {eventDate}
                    </div>
                  </div>
                )}
                
                {category === "previous" && (
                  <div className="mt-4 p-3 border border-blue-500/30 bg-blue-500/10 rounded flex items-center justify-between animate-in fade-in zoom-in-95">
                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-wider text-blue-500 mb-0.5">Previously Affected</div>
                      <div className="text-xs text-slate-900 font-semibold">Historical Event</div>
                    </div>
                    <div className="text-xs font-mono font-bold text-blue-500 bg-blue-500/20 px-2 py-1 rounded">
                      {eventDate}
                    </div>
                  </div>
                )}
              </div>

              {/* Location & Type */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                    EVENT LOCATION
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      placeholder="e.g. Kedarnath Valley"
                      className="w-full px-3.5 py-2 pl-9 rounded border border-slate-300 bg-white text-slate-900 text-xs font-medium focus:ring-1 focus:ring-sky-600 focus:outline-hidden"
                      required
                    />
                    <MapPin className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                  </div>
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                    EVENT TYPE
                  </label>
                  <select
                    value={eventType}
                    onChange={(e) => setEventType(e.target.value)}
                    className="w-full px-3.5 py-2 rounded border border-slate-300 bg-white text-slate-900 text-xs font-medium focus:ring-1 focus:ring-sky-600 focus:outline-hidden"
                  >
                    {eventTypeOptions.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Warning Time */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  HOW MUCH WARNING DID YOU RECEIVE?
                </label>
                <select
                  value={warningTime}
                  onChange={(e) => setWarningTime(e.target.value)}
                  className="w-full px-3.5 py-2 rounded border border-slate-300 bg-white text-slate-900 text-xs font-medium focus:ring-1 focus:ring-sky-600 focus:outline-hidden"
                  required
                >
                  <option value="" disabled>Select warning time...</option>
                  {warningTimeOptions.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>

              {/* What was missing */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  WHAT WAS MISSING? (Select all that apply)
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {missingOptions.map(opt => (
                    <label key={opt} className="flex items-center space-x-2 cursor-pointer p-2 rounded border border-slate-200 hover:bg-slate-50 transition-colors">
                      <input
                        type="checkbox"
                        checked={missingElements.includes(opt)}
                        onChange={() => handleCheckboxToggle(opt)}
                        className="w-3.5 h-3.5 rounded text-sky-600 bg-white border-slate-300 focus:ring-sky-600 focus:ring-offset-white"
                      />
                      <span className="text-xs text-slate-900 font-medium">{opt}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Problems Faced */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  WHAT PROBLEMS DID YOU FACE?
                </label>
                <textarea
                  value={problemsFaced}
                  onChange={(e) => setProblemsFaced(e.target.value)}
                  rows={3}
                  placeholder="Describe the main difficulties you experienced during the event..."
                  className="w-full px-3.5 py-2 rounded border border-slate-300 bg-white text-slate-900 text-xs font-medium focus:ring-1 focus:ring-sky-600 focus:outline-hidden resize-none"
                  required
                />
              </div>

              {/* Missing Info */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  WHAT INFORMATION DID YOU NEED BUT COULD NOT GET?
                </label>
                <textarea
                  value={missingInfo}
                  onChange={(e) => setMissingInfo(e.target.value)}
                  rows={2}
                  placeholder="For example: water level, safe route, evacuation timing, nearby danger..."
                  className="w-full px-3.5 py-2 rounded border border-slate-300 bg-white text-slate-900 text-xs font-medium focus:ring-1 focus:ring-sky-600 focus:outline-hidden resize-none"
                />
              </div>

              {/* What would have helped */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  WHAT WOULD HAVE HELPED YOU DURING THE EVENT?
                </label>
                <textarea
                  value={helpfulAdditions}
                  onChange={(e) => setHelpfulAdditions(e.target.value)}
                  rows={2}
                  placeholder="Suggest improvements or tools that would have assisted you..."
                  className="w-full px-3.5 py-2 rounded border border-slate-300 bg-white text-slate-900 text-xs font-medium focus:ring-1 focus:ring-sky-600 focus:outline-hidden resize-none"
                />
              </div>

              <div className="pt-2 border-t border-slate-200 mt-6 pt-4">
                <button
                  type="submit"
                  className="w-full py-3 bg-sky-600 text-white font-bold text-xs uppercase tracking-wider rounded shadow-sm hover:bg-sky-700 transition-all flex items-center justify-center space-x-2 cursor-pointer"
                >
                  <Send className="w-4 h-4" />
                  <span>Submit Feedback</span>
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
