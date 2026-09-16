import React, { useState, useEffect, useRef } from "react";
import Sidebar from "./components/Sidebar";
import AlertBanner from "./components/AlertBanner";
import RiskMap from "./components/RiskMap";
import RiskPeakGraph from "./components/RiskPeakGraph";
import WeatherDetailsCards from "./components/WeatherDetailsCards";
import PublicAlertPreview from "./components/PublicAlertPreview";
import StationChat from "./components/StationChat";
import AuthorityReportsPanel from "./components/AuthorityReportsPanel";
import FeedbackModal from "./components/FeedbackModal";
import ReportsView from "./components/ReportsView";
import ValidationDrawer from "./components/ValidationDrawer";
import PinnView from "./components/PinnView";

// Removed all static imports from mockData.js

export default function App() {
  const [activeTab, setActiveTab] = useState("map");
  const [monitoringLocations, setMonitoringLocations] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [liveStations, setLiveStations] = useState([]);
  const [aiModelMetrics, setAiModelMetrics] = useState({ f1Score: 0, accuracy: 0, falseAlarmRate: 0 });

  // Fetch live predictions from the inference server
  useEffect(() => {
    const fetchNowcast = async () => {
      try {
        // Our backend runs on localhost:8000 when launched via run_dashboard.py
        const res = await fetch("http://localhost:8000/api/nowcast");
        if (res.ok) {
          let data = await res.json();
          
          // --- Mock Disaster Injection ---
          if (simulateDisaster && data.length > 0) {
            let targetIdx = data.findIndex(d => !d.is_virtual);
            if (targetIdx !== -1) {
                simulateTargetRef.current = data[targetIdx].id;
                data[targetIdx] = {
                    ...data[targetIdx],
                    gate_a: true,
                    gate_b: true,
                    P_CB: 0.98,
                    tier: 'red',
                    metrics: {
                        ...data[targetIdx].metrics,
                        cape: 4200,
                        rain: 95.5,
                        wind: 75
                    }
                };
            }
          } else {
              simulateTargetRef.current = null;
          }

          setLiveStations(data);
          
          // Compute the region's overall risk score and populate dynamic locations
          if (data && data.length > 0) {
            const dynamicLocations = data
              .filter(d => !d.is_virtual) // Only feature trusted physical towers in the sidebar locations list
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

            if (dynamicLocations.length > 0) {
              setMonitoringLocations(dynamicLocations);
            }

            const maxProb = Math.max(...data.map(d => d.P_CB || 0));
            let level = "LOW";
            if (maxProb >= 0.85) level = "SEVERE";
            else if (maxProb >= 0.65) level = "HIGH";
            else if (maxProb >= 0.35) level = "MODERATE";
            
            setSelectedLocation(prev => {
              // If it's the first load, grab the most dangerous physical station
              const base = prev || dynamicLocations.sort((a, b) => b.riskScore - a.riskScore)[0] || dynamicLocations[0];
              return {
                ...base,
                riskLevel: level,
                riskPercent: Math.round(maxProb * 100),
                riskScore: maxProb
              };
            });
          }
        }
      } catch (err) {
        console.error("Failed to fetch live nowcast data:", err);
      }
    };
    fetchNowcast();
    const interval = setInterval(fetchNowcast, 15000); // 15 sec polling

    // Fetch AI metrics once on mount
    const fetchMetrics = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/metrics");
        if (res.ok) setAiModelMetrics(await res.json());
      } catch (err) {
        console.error("Failed to fetch metrics:", err);
      }
    };
    fetchMetrics();

    return () => clearInterval(interval);
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
  const [reports, setReports] = useState([]);

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
          <div className="p-4 sm:p-6 lg:p-8 space-y-6 animate-in fade-in duration-300">
            <h2 className="text-xl font-bold text-slate-900">Active Emergency Weather Alerts</h2>
            <AlertBanner
              location={selectedLocation}
              onSimulateBroadcast={handleTriggerBroadcastPreview}
            />
            <PublicAlertPreview
              location={selectedLocation}
              onTriggerBroadcast={handleTriggerBroadcastPreview}
            />
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
  );
}
