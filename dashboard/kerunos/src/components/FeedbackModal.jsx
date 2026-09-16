import React, { useState } from "react";
import { CheckCircle2, X, MapPin, Send, MessageSquare, Star, ShieldCheck } from "lucide-react";

export default function FeedbackModal({ isOpen, onClose, onSubmitReport, currentLocationName }) {
  const [selectedWeather, setSelectedWeather] = useState("Heavy Rain");
  const [description, setDescription] = useState("");
  const [userLocation, setUserLocation] = useState(currentLocationName || "Mandakini Valley");
  const [isSubmitted, setIsSubmitted] = useState(false);

  if (!isOpen) return null;

  const weatherOptions = [
    { id: "Normal", label: "Normal", icon: "🌤️", severity: "Low" },
    { id: "Heavy Rain", label: "Heavy Rain", icon: "🌧️", severity: "High" },
    { id: "Thunderstorm", label: "Severe Rain / Thunderstorm", icon: "⛈️", severity: "High" },
    { id: "Flooding", label: "Flooding", icon: "🌊", severity: "Severe" },
    { id: "Other Emergency", label: "Other Emergency", icon: "⚠️", severity: "Severe" }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    const selectedOption = weatherOptions.find((w) => w.id === selectedWeather);

    onSubmitReport({
      type: selectedWeather,
      icon: selectedOption?.icon || "🌧️",
      severity: selectedOption?.severity || "High",
      location: userLocation,
      description: description || "Direct user weather observation submitted.",
      time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false }) + " IST",
      count: 1
    });

    setIsSubmitted(true);
  };

  const handleReset = () => {
    setIsSubmitted(false);
    setDescription("");
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl border border-slate-200 shadow-2xl max-w-md w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="bg-slate-900 text-white p-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded-lg bg-amber-500/20 text-amber-400">
              <Star className="w-4 h-4" />
            </span>
            <h3 className="font-bold text-sm tracking-tight">
              Citizen Ground Weather Report
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        {isSubmitted ? (
          <div className="p-6 text-center space-y-4">
            <div className="w-16 h-16 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mx-auto shadow-inner">
              <CheckCircle2 className="w-10 h-10" />
            </div>
            <div className="space-y-1">
              <h4 className="text-lg font-bold text-slate-900">
                Thank you for your report!
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed max-w-xs mx-auto">
                Your feedback helps improve real-time weather situation awareness and validates the AI prediction model.
              </p>
            </div>
            <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-xs text-slate-700 flex items-center justify-between">
              <span>Map Marker Placed:</span>
              <span className="font-bold text-sky-700">📍 {userLocation}</span>
            </div>
            <button
              onClick={handleReset}
              className="w-full py-2.5 bg-slate-900 hover:bg-slate-800 text-white font-bold text-xs rounded-xl shadow-md transition-all cursor-pointer"
            >
              Done & Close
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="p-5 space-y-4 text-left">
            <div>
              <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                How is the weather in your area?
              </label>
              <div className="space-y-2">
                {weatherOptions.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setSelectedWeather(opt.id)}
                    className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl border text-xs font-semibold transition-all ${
                      selectedWeather === opt.id
                        ? "bg-sky-50 border-sky-400 text-sky-900 shadow-2xs font-bold"
                        : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <span className="flex items-center space-x-2.5">
                      <span className="text-base">{opt.icon}</span>
                      <span>{opt.label}</span>
                    </span>
                    {selectedWeather === opt.id && (
                      <span className="w-2 h-2 rounded-full bg-sky-600" />
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
                📍 Current Location
              </label>
              <input
                type="text"
                value={userLocation}
                onChange={(e) => setUserLocation(e.target.value)}
                placeholder="Enter current valley or landmark..."
                className="w-full px-3.5 py-2 rounded-xl border border-slate-200 text-xs font-medium focus:ring-2 focus:ring-sky-500 focus:outline-hidden"
              />
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
                Optional Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                placeholder="Describe what you are observing (e.g. river water level, wind gust, visibility)..."
                className="w-full px-3.5 py-2 rounded-xl border border-slate-200 text-xs font-medium focus:ring-2 focus:ring-sky-500 focus:outline-hidden resize-none"
              />
            </div>

            <button
              type="submit"
              className="w-full py-3 bg-sky-600 hover:bg-sky-700 text-white font-bold text-xs rounded-xl shadow-md shadow-sky-500/20 transition-all flex items-center justify-center space-x-2 cursor-pointer active:scale-95"
            >
              <Send className="w-4 h-4" />
              <span>Submit Ground Report</span>
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
