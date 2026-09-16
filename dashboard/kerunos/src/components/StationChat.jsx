import React, { useState } from "react";
import { Send, AlertTriangle, Radio, CheckCheck } from "lucide-react";

export default function StationChat({ messages = [], onSendMessage }) {
  const [inputText, setInputText] = useState("");
  const [isPriorityMode, setIsPriorityMode] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    onSendMessage({
      sender: "Kedarnath Weather Station",
      senderCode: "KED-01",
      text: inputText,
      isPriority: isPriorityMode
    });

    setInputText("");
    setIsPriorityMode(false);
  };

  const handleSendPriorityQuick = () => {
    onSendMessage({
      sender: "Kedarnath Weather Station",
      senderCode: "KED-01",
      text: "⚠️ PRIORITY: Rapid rainfall increase detected. Please verify local conditions & initiate public alert protocol.",
      isPriority: true
    });
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs flex flex-col h-[520px] overflow-hidden">
      {/* Header Bar */}
      <div className="p-4 bg-slate-900 text-white flex items-center justify-between z-10 shadow-xs">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-sky-600 text-white flex items-center justify-center font-bold">
            <Radio className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-bold tracking-tight">
                Station Communication Console
              </h3>
              <span className="px-2 py-0.5 text-[10px] bg-slate-800 text-sky-400 font-mono rounded-md border border-slate-700">
                DISPATCH MESH
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Kedarnath Station (KED-01) ↔ Joshimath Station (JSH-04) ↔ State Disaster HQ
            </p>
          </div>
        </div>

        {/* Station Online Status Indicator */}
        <div className="flex items-center space-x-2 bg-slate-800/90 border border-slate-700 px-3 py-1.5 rounded-full text-xs">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          <span className="text-emerald-400 font-semibold">🟢 Station Channel Online</span>
        </div>
      </div>

      {/* Message Feed Stream */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4 bg-slate-50/60">
        {messages.map((msg) => {
          const isKedarnath = msg.sender.includes("Kedarnath");

          return (
            <div
              key={msg.id}
              className={`flex flex-col ${
                isKedarnath ? "items-end" : "items-start"
              }`}
            >
              {/* Message Bubble Container */}
              <div className="max-w-xl space-y-1">
                {/* Station Sender & Timestamp Info */}
                <div className={`flex items-center space-x-2 text-[11px] text-slate-500 ${isKedarnath ? "justify-end" : "justify-start"}`}>
                  <span className="font-bold text-slate-700">{msg.sender}</span>
                  <span className="text-slate-400">({msg.senderCode})</span>
                  <span>•</span>
                  <span>{msg.time}</span>
                </div>

                {/* Message Body */}
                <div
                  className={`p-3.5 rounded-2xl text-xs sm:text-sm shadow-2xs leading-relaxed ${
                    msg.isPriority
                      ? "bg-red-50 border-2 border-red-500 text-red-950 font-medium rounded-tr-2xs shadow-red-500/10"
                      : isKedarnath
                      ? "bg-sky-600 text-white font-normal rounded-tr-2xs"
                      : "bg-white border border-slate-200 text-slate-800 rounded-tl-2xs"
                  }`}
                >
                  {msg.isPriority && (
                    <div className="flex items-center space-x-1.5 text-red-700 font-extrabold text-xs uppercase tracking-wider mb-1.5 pb-1 border-b border-red-200">
                      <AlertTriangle className="w-4 h-4 shrink-0 text-red-600" />
                      <span>HIGH PRIORITY ALERT DISPATCH</span>
                    </div>
                  )}
                  <p>{msg.text}</p>
                </div>

                {/* Delivery Status indicator */}
                <div className={`flex items-center space-x-1 text-[10px] text-slate-400 ${isKedarnath ? "justify-end" : "justify-start"}`}>
                  <CheckCheck className="w-3 h-3 text-sky-500" />
                  <span>Synced to Station Logs</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Control & Input Footer */}
      <div className="p-3 bg-white border-t border-slate-200 space-y-2">
        {/* Priority Toggle Bar & Quick Priority Button */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => setIsPriorityMode(!isPriorityMode)}
            className={`px-3 py-1 rounded-lg text-xs font-bold transition-all flex items-center space-x-1.5 ${
              isPriorityMode
                ? "bg-red-600 text-white shadow-xs"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{isPriorityMode ? "Priority Mode ACTIVE" : "Toggle Priority Mode"}</span>
          </button>

          <button
            type="button"
            onClick={handleSendPriorityQuick}
            className="px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 border border-red-300 rounded-lg text-xs font-bold transition-all flex items-center space-x-1"
          >
            <span>🔴 Send Priority Alert</span>
          </button>
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex items-center space-x-2">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={
              isPriorityMode
                ? "Type HIGH PRIORITY EMERGENCY message..."
                : "Type station dispatch message..."
            }
            className={`flex-1 px-4 py-2.5 rounded-xl border text-xs sm:text-sm focus:outline-hidden transition-all ${
              isPriorityMode
                ? "border-red-400 bg-red-50/50 focus:ring-2 focus:ring-red-500"
                : "border-slate-200 bg-slate-50 focus:bg-white focus:ring-2 focus:ring-sky-500"
            }`}
          />
          <button
            type="submit"
            className="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold shadow-md transition-all flex items-center space-x-1.5 cursor-pointer"
          >
            <span>Send</span>
            <Send className="w-3.5 h-3.5" />
          </button>
        </form>
      </div>
    </div>
  );
}
