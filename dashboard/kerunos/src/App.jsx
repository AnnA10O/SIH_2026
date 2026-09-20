import React, { useState, useEffect, useRef } from "react";
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
import { initialCommunityReports, monitoringLocations as defaultLocations } from "./data/mockData";

// Removed all static imports from mockData.js

export default function App() {
  const [activeTab, setActiveTab] = useState("map");
  const [monitoringLocations, setMonitoringLocations] = useState(defaultLocations);
  const [selectedLocation, setSelectedLocation] = useState(defaultLocations[0]);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showLanding, setShowLanding] = useState(true);
  const [liveStations, setLiveStations] = useState([]);
  const [aiModelMetrics, setAiModelMetrics] = useState({ f1Score: 0, accuracy: 0, falseAlarmRate: 0 });
  const [backendOnline, setBackendOnline] = useState(false);
  const retryDelayRef = useRef(5000); // start at 5s, back off to 30s
  const retryTimerRef = useRef(null);

  // Backend polling is defined below, after simulateDisaster/simulateTargetRef are declared;

  useEffect(() => {
    const handleMessage = (event) => {
      if (event.data && event.data.type === 'SWITCH_TAB') {
        setActiveTab(event.data.tab);
        if (event.data.region) {
          // Delay to ensure the PINN 3D engine is mounted and visible before triggering region switch
          setTimeout(() => {
            if (window.switchPINNRegion) {
              window.switchPINNRegion(event.data.region).then(() => {
                  if (window.simulateFloodCanvas) window.simulateFloodCanvas();
              });
            }
          }, 300);
        }
      } else if (event.data && event.data.type === 'SWITCH_TAB_AND_PREFILL') {
        setActiveTab(event.data.tab);
        if (event.data.stationData) {
            try {
                const sData = JSON.parse(decodeURIComponent(event.data.stationData));
                const mappedLoc = {
                   id: sData.id || "Unknown",
                   name: `Uttarakhand Stn ${sData.id.split('-')[1] || ''}`,
                   riskLevel: sData.tier === 'red' ? 'SEVERE' : (sData.tier === 'orange' ? 'HIGH' : 'MODERATE'),
                   timeWindow: "Next 1-2 Hours"
                };
                setSelectedLocation(mappedLoc);
            } catch(e) {}
        }
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

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
  const [pinnTime, setPinnTime] = useState(0);
  const [simulateDisaster, setSimulateDisaster] = useState(false);
  const simulateTargetRef = useRef(null);
  
  useEffect(() => {
    if (simulateDisaster) {
       simulateTargetRef.current = "UK-" + (Math.floor(Math.random() * 8) + 1);
    }
    const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    fetch(`${baseUrl}/api/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active: simulateDisaster, target_station: simulateTargetRef.current || "UK-6" })
    }).catch(console.error);
  }, [simulateDisaster]);

  // Backend polling — placed here so simulateDisaster & simulateTargetRef are in scope
  useEffect(() => {
    const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    let intervalId = null;

    const fetchNowcast = async () => {
      try {
        const res = await fetch(`${baseUrl}/api/nowcast`);
        if (res.ok) {
          let data = await res.json();
          setBackendOnline(true);
          retryDelayRef.current = 5000;

          if (simulateDisaster && data.length > 0) {
            let targetIdx = data.findIndex(d => !d.is_virtual);
            if (targetIdx !== -1) simulateTargetRef.current = data[targetIdx].id;
          } else {
            simulateTargetRef.current = null;
          }

          setLiveStations(data);

          if (data && data.length > 0) {
            const dynamicLocations = data
              .filter(d => !d.is_virtual)
              .map(d => ({
                id: d.id,
                name: expandStationName(d.id),
                lat: d.lat,
                lng: d.lng,
                zoom: 11,
                riskLevel: d.tier.toUpperCase(),
                gate_a: d.gate_a,
                gate_b: d.gate_b,
                P_CB: d.P_CB,
                metrics: d.metrics
              }));

            if (dynamicLocations.length > 0) setMonitoringLocations(dynamicLocations);

            const maxProb = Math.max(...data.map(d => d.P_CB || 0));
            let level = "LOW";
            if (maxProb >= 0.85) level = "SEVERE";
            else if (maxProb >= 0.65) level = "HIGH";
            else if (maxProb >= 0.35) level = "MODERATE";

            setSelectedLocation(prev => {
              const base = prev || dynamicLocations[0];
              return { ...base, riskLevel: level, riskPercent: Math.round(maxProb * 100), riskScore: maxProb };
            });
          }

          if (intervalId) clearInterval(intervalId);
          intervalId = setInterval(fetchNowcast, 15000);
        } else {
          throw new Error(`HTTP ${res.status}`);
        }
      } catch (err) {
        console.warn(`Backend offline, retrying in ${retryDelayRef.current / 1000}s...`, err.message);
        setBackendOnline(false);
        if (intervalId) clearInterval(intervalId);
        retryTimerRef.current = setTimeout(() => {
          retryDelayRef.current = Math.min(retryDelayRef.current * 2, 30000);
          fetchNowcast();
        }, retryDelayRef.current);
      }
    };

    fetchNowcast();

    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${baseUrl}/api/metrics`);
        if (res.ok) setAiModelMetrics(await res.json());
      } catch (err) { console.error("Failed to fetch metrics:", err); }
    };
    fetchMetrics();

    return () => {
      if (intervalId) clearInterval(intervalId);
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Helper for expanding names
  const expandStationName = (id) => {
    if (id.startsWith("UK-")) return "Uttarakhand Stn " + id.split("-")[1];
    if (id.startsWith("AS-")) return "Assam Stn " + id.split("-")[1];
    if (id.startsWith("SK-")) return "Sikkim Stn " + id.split("-")[1];
    if (id.startsWith("V-")) return "Virtual Node " + id.split("-")[1];
    return id;
  };

  const [messages, setMessages] = useState([]);

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

  if (!selectedLocation) {
    return <div className="h-screen w-screen flex items-center justify-center bg-slate-900 text-white font-mono animate-pulse">Establishing secure handshake with inference engine...</div>;
  }

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
        simulateDisaster={simulateDisaster}
        setSimulateDisaster={setSimulateDisaster}
      />

      {/* Main Workspace Area */}
      <main className="flex-1 lg:pl-64 min-w-0 h-full overflow-y-auto relative bg-slate-50">

        {/* TAB 2: FULL-SCREEN RISK MAP VIEW */}
        {activeTab === "map" && (
          <div className="h-full w-full animate-in fade-in duration-300">
            <RiskMap
              selectedLocation={selectedLocation}
              stations={liveStations}
              communityReports={communityReports}
              onSelectStation={(stn) => setSelectedLocation(stn)}
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
          <div className="h-full w-full animate-in fade-in duration-300 relative p-6">
            <h2 className="text-xl font-bold text-slate-900 mb-2">Statistical Analytics</h2>
            <p className="text-sm text-slate-500">Analytics dashboard — coming soon in a future update.</p>
          </div>
        )}

        {/* Fixed Collapsible Bottom Drawer: Model performance & validation */}
        <ValidationDrawer metrics={aiModelMetrics} />
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
