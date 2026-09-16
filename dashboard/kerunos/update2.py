import sys

with open('src/components/AlertManager.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace messageContent state
content = content.replace(
    'const [messageContent, setMessageContent] = useState("");',
    'const [messageContents, setMessageContents] = useState({});'
)

# Update useEffect
useEffect_old = '''  // Update text area when location or format changes
  useEffect(() => {
    if (selectedLocations.length === 0) {
      setMessageContent("");
      return;
    }
    const combinedMsg = selectedLocations.map(loc => {
      if (activeFormat === "long") {
        return getTemplateForRisk(loc);
      } else {
        return getSmsTemplate(loc);
      }
    }).join("\\n\\n----------------------------------------\\n\\n");
    setMessageContent(combinedMsg);
  }, [selectedLocations, activeFormat]);'''

useEffect_new = '''  // Update text areas when location or format changes
  useEffect(() => {
    setMessageContents(prev => {
      const newContents = { ...prev };
      selectedLocations.forEach(loc => {
        // Only generate default template if it hasn't been generated yet for this location/format,
        // or if we just want to reset (we won't overwrite existing edits here, but format switch will).
        // Actually, if format changes, we should regenerate all.
        newContents[loc.id] = activeFormat === "long" ? getTemplateForRisk(loc) : getSmsTemplate(loc);
      });
      return newContents;
    });
  }, [selectedLocations, activeFormat]);

  const handleMessageChange = (id, value) => {
    setMessageContents(prev => ({ ...prev, [id]: value }));
  };'''

content = content.replace(useEffect_old, useEffect_new)

# Update Auto-resize textarea - actually we can remove the auto-resize useEffect since there are multiple
auto_resize_old = '''  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = ${textareaRef.current.scrollHeight}px;
    }
  }, [messageContent]);'''

content = content.replace(auto_resize_old, '')
content = content.replace('const textareaRef = useRef(null);\n\n', '')

# Update handleSimulate
simulate_old = '''    try {
      const promises = selectedLocations.map(loc => {
        return fetch('/api/alerts/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            region: loc.name || loc.region || "Unknown Region",
            tier: getTier(loc.riskLevel),
            risk_score: loc.riskScore || 0,
            message: messageContent
          })
        }).then(r => r.json());
      });'''

simulate_new = '''    try {
      const promises = selectedLocations.map(loc => {
        return fetch('/api/alerts/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            region: loc.name || loc.region || "Unknown Region",
            tier: getTier(loc.riskLevel),
            risk_score: loc.riskScore || 0,
            message: messageContents[loc.id] || ""
          })
        }).then(r => r.json());
      });'''

content = content.replace(simulate_old, simulate_new)

# Update textarea render block
textarea_old = '''          <div className="relative">
            <textarea
              ref={textareaRef}
              value={messageContent}
              onChange={(e) => setMessageContent(e.target.value)}
              className="w-full min-h-[120px] p-5 bg-slate-50 border border-slate-200 rounded-xl text-[13px] font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:bg-white transition-all resize-none leading-relaxed overflow-hidden"
              placeholder="Alert message content..."
            />
          </div>'''

textarea_new = '''          <div className="flex flex-col space-y-4">
            {selectedLocations.length === 0 && (
              <div className="p-8 text-center text-slate-400 bg-slate-50 rounded-xl border border-dashed border-slate-200">
                Select one or more regions above to generate alert messages.
              </div>
            )}
            {selectedLocations.map(loc => (
              <div key={loc.id} className="relative flex flex-col bg-slate-50 border border-slate-200 rounded-xl overflow-hidden focus-within:ring-2 focus-within:ring-sky-500/40 focus-within:bg-white transition-all shadow-sm">
                <div className="bg-slate-100/80 px-4 py-2 border-b border-slate-200 flex justify-between items-center">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">{loc.name}</span>
                  <span className={	ext-[10px] font-bold px-2 py-0.5 rounded }>
                    {loc.riskLevel} RISK
                  </span>
                </div>
                <textarea
                  value={messageContents[loc.id] || ""}
                  onChange={(e) => handleMessageChange(loc.id, e.target.value)}
                  className="w-full min-h-[160px] p-4 bg-transparent text-[13px] font-medium text-slate-800 focus:outline-none resize-y leading-relaxed"
                  placeholder="Alert message content..."
                />
              </div>
            ))}
          </div>'''

content = content.replace(textarea_old, textarea_new)


with open('src/components/AlertManager.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print('Update successful')
