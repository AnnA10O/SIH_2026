import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import RiskMap from "./components/RiskMap";
import RiskPeakGraph from "./components/RiskPeakGraph";
import WeatherDetailsCards from "./components/WeatherDetailsCards";
import AlertManager from "./components/AlertManager";
import StationChat from "./components/StationChat";
import AuthorityReportsPanel from "./components/AuthorityReportsPanel";
import FeedbackModal from "./components/FeedbackModal";
import ReportsView from "./components/ReportsView";
import ValidationDrawer from "./components/ValidationDrawer";
import PinnView from "./components/PinnView";

import {
  monitoringLocations,
  riskTimelineData,
  weatherStations,
  initialCommunityReports,
  initialStationMessages,
  aiModelMetrics
} from "./data/mockData";

export default function App() {
  const [activeTab, setActiveTab] = useState("map");
  const [selectedLocation, setSelectedLocation] = useState(monitoringLocations[0]);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showLanding, setShowLanding] = useState(true);

  useEffect(() => {
    // When switching tabs, fire a resize event so that canvases (like PINN 3D) 
    // recalculate their width properly instead of overflowing because they were 
    // initialized while hidden (display: none).
    const timer = setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 50);
    return () => clearTimeout(timer);
  }, [activeTab]);

  // Dynamic state for Station Chat
  const [messages, setMessages] = useState(initialStationMessages);

  // Dynamic state for Citizen Ground Reports
  const [reports, setReports] = useState(initialCommunityReports);

  // Modal control
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);

  // Handler for sending new message in Station Chat
  const handleSendMessage = ({ sender, senderCode, text, isPriority }) => {
    const newMsg = {
      id: `msg-${Date.now()}`,
      sender,
      senderCode,
      time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false }) + " IST",
      text,
      isPriority: !!isPriority,
      status: "delivered"
    };
    setMessages((prev) => [...prev, newMsg]);
  };

  // Handler for submitting new Citizen Ground Report
  const handleSubmitReport = (newReportData) => {
    const newReport = {
      id: `rep-${Date.now()}`,
      ...newReportData,
      lat: selectedLocation.lat + (Math.random() * 0.04 - 0.02),
      lng: selectedLocation.lng + (Math.random() * 0.04 - 0.02)
    };
    setReports((prev) => [newReport, ...prev]);
  };

  // Scroll to section or change tab when clicking alert broadcast action
  const handleTriggerBroadcastPreview = () => {
    setActiveTab("alerts");
    const el = document.getElementById("public-alert-section");
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <>
      {/* LANDING PAGE OVERLAY */}
      {showLanding && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black">
          <iframe 
            src="./landing.html" 
            className="w-full h-full border-none absolute inset-0" 
            title="Landing Page"
          />
          <button 
            onClick={() => setShowLanding(false)}
            className="absolute bottom-10 left-1/2 -translate-x-1/2 px-8 py-4 bg-sky-500 hover:bg-sky-400 text-white font-bold rounded-full shadow-[0_0_20px_rgba(14,165,233,0.5)] transition-all hover:scale-105 active:scale-95 flex items-center space-x-2 z-10 font-sans tracking-wide uppercase"
          >
            <span>Enter System</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
          </button>
        </div>
      )}

      <div className="h-screen w-screen bg-slate-50 text-slate-800 flex overflow-hidden antialiased">
        {/* Left Navigation Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        activeAlertCount={selectedLocation.riskLevel === "SEVERE" ? 1 : 0}
        unreadMessageCount={messages.length}
        mobileMenuOpen={mobileMenuOpen}
        setMobileMenuOpen={setMobileMenuOpen}
        locations={monitoringLocations}
        selectedLocation={selectedLocation}
        onSelectLocation={setSelectedLocation}
      />

      {/* Main Workspace Area */}
      <main className="flex-1 lg:pl-64 min-w-0 h-full overflow-y-auto relative bg-slate-50">

        {/* TAB 2: FULL-SCREEN RISK MAP VIEW */}
        {activeTab === "map" && (
          <div className="h-full w-full animate-in fade-in duration-300">
            <iframe 
              src="/UI/realtime/index.html" 
              className="w-full h-full border-none" 
              title="Realtime Simulation UI (Map)"
            />
          </div>
        )}

        {/* TAB: PINN PHYSICS SIMULATION VIEW (Always mounted, hidden when inactive) */}
        <div 
          className="h-full w-full animate-in fade-in duration-300 overflow-hidden" 
          style={{ display: activeTab === "pinn" ? "block" : "none" }}
        >
          <PinnView />
        </div>

        {/* TAB 3: ALERTS VIEW */}
        {activeTab === "alerts" && (
            <div className="p-4 sm:p-6 lg:p-8 h-full animate-in fade-in duration-300 flex flex-col">
              <AlertManager locations={monitoringLocations} defaultLocation={selectedLocation} />
            </div>
        )}

        {/* TAB 4: STATION CHAT VIEW */}
        {activeTab === "chat" && (
          <div className="h-full flex flex-col p-4 sm:p-6 lg:p-8 animate-in fade-in duration-300">
            <div className="shrink-0 mb-4">
              <h2 className="text-xl font-bold text-slate-900">Station Communication Mesh</h2>
              <p className="text-xs text-slate-500">Direct encrypted channel between monitoring stations</p>
            </div>
            <div className="flex-1 min-h-0">
              <StationChat
                messages={messages}
                onSendMessage={handleSendMessage}
              />
            </div>
          </div>
        )}

        {/* TAB 5: FEEDBACK VIEW */}
        {activeTab === "feedback" && (
          <div className="p-4 sm:p-6 lg:p-8 space-y-6 animate-in fade-in duration-300 h-full flex flex-col">
            <div className="shrink-0">
              <h2 className="text-xl font-bold text-slate-900 font-sans">Citizen Ground Truth Feedback System</h2>
              <p className="text-xs text-slate-500">Submit or review crowd-sourced ground observations</p>
            </div>
            <div className="flex-1 min-h-0">
              <AuthorityReportsPanel
                reports={reports}
                onOpenModal={() => setIsReportModalOpen(true)}
              />
            </div>
          </div>
        )}

        {/* TAB 6: REPORTS VIEW */}
        {activeTab === "reports" && (
          <div className="p-4 sm:p-6 lg:p-8 h-full animate-in fade-in duration-300">
            <ReportsView metrics={aiModelMetrics} />
          </div>
        )}

        {/* TAB 7: STATISTICAL ANALYTICS VIEW */}
        {activeTab === "analytics" && (
          <div className="h-full w-full animate-in fade-in duration-300 relative">
             <iframe 
               src="./weather_statistical_analytics.html" 
               className="w-full h-full border-none absolute inset-0 bg-slate-50" 
               title="Statistical Analytics"
             />
          </div>
        )}

        {/* Fixed Collapsible Bottom Drawer: Model performance & validation */}
        <ValidationDrawer />
      </main>

      {/* Ground Feedback Modal */}
      <FeedbackModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        onSubmitReport={handleSubmitReport}
        currentLocationName={selectedLocation.name}
      />
    </div>
    </>
  );
}
