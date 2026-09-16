import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# In-memory storage for the latest alert
latest_alert = None

# This is the full-screen Web UI that will be displayed
HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>KERAUNOS Receiver Station</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: white; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; text-align: center; transition: background-color 0.5s;}
        #alert-container { display: none; padding: 50px; border-radius: 20px; max-width: 900px; width: 90%; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7); }
        
        .red-alert { background-color: #ef4444; animation: pulseRed 1.5s infinite; }
        .orange-alert { background-color: #f97316; animation: pulseOrange 2s infinite; }
        .yellow-alert { background-color: #eab308; color: #1e293b; }
        .green-alert { background-color: #22c55e; }
        
        @keyframes pulseRed {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); transform: scale(1); }
            50% { transform: scale(1.02); }
            70% { box-shadow: 0 0 0 40px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); transform: scale(1); }
        }
        @keyframes pulseOrange {
            0% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.7); }
            70% { box-shadow: 0 0 0 40px rgba(249, 115, 22, 0); }
            100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0); }
        }

        h1 { font-size: 3.5rem; margin-top: 0; text-transform: uppercase; letter-spacing: 3px; border-bottom: 2px solid rgba(255,255,255,0.2); padding-bottom: 20px;}
        .yellow-alert h1 { border-bottom: 2px solid rgba(0,0,0,0.2); }
        .region { font-size: 3rem; font-weight: bold; margin: 30px 0; }
        .details { font-size: 1.8rem; background: rgba(0,0,0,0.15); padding: 30px; border-radius: 15px; text-align: left; line-height: 1.6;}
        
        #idle { display: block; }
        #idle h2 { font-size: 3rem; color: #94a3b8; letter-spacing: 4px;}
        .loader { border: 6px solid #1e293b; border-top: 6px solid #3b82f6; border-radius: 50%; width: 50px; height: 50px; animation: spin 2s linear infinite; margin: 40px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="idle">
        <h2>KERAUNOS REMOTE TERMINAL</h2>
        <p style="font-size: 1.5rem; color: #64748b;">Waiting for incoming emergency broadcasts...</p>
        <div class="loader"></div>
    </div>
    
    <div id="alert-container">
        <h1 id="alert-title">EMERGENCY ALERT</h1>
        <div class="region" id="alert-region">UNKNOWN REGION</div>
        <div class="details">
            <p><strong>Threat Level:</strong> <span id="alert-tier">N/A</span></p>
            <p><strong>Risk Score:</strong> <span id="alert-score">N/A</span>/100</p>
        </div>
    </div>

    <!-- Audio element for siren -->
    <audio id="siren" loop>
      <!-- We use a data URI for a short repeating alarm beep so no external file is needed! -->
      <source src="data:audio/mp3;base64,//NExAAAAANIAAAAAExBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq//NExAAAAANIAAAAAExBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq" type="audio/mpeg">
    </audio>

    <script>
        let currentAlertId = null;
        let isFlashing = false;
        let siren = document.getElementById('siren');
        
        // This simulates a loud repeating beep using the web audio API (very reliable)
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        function playSiren() {
            if(audioCtx.state === 'suspended') { audioCtx.resume(); }
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'square';
            osc.frequency.setValueAtTime(800, audioCtx.currentTime); // High pitch
            osc.frequency.setValueAtTime(600, audioCtx.currentTime + 0.3); // High pitch
            gain.gain.setValueAtTime(0, audioCtx.currentTime);
            gain.gain.linearRampToValueAtTime(1, audioCtx.currentTime + 0.1);
            gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.5);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.5);
        }

        let beepInterval;

        function updateUI(alert) {
            const container = document.getElementById('alert-container');
            const idle = document.getElementById('idle');
            
            idle.style.display = 'none';
            container.style.display = 'block';
            
            document.getElementById('alert-region').innerText = alert.region;
            document.getElementById('alert-tier').innerText = alert.tier.toUpperCase();
            document.getElementById('alert-score').innerText = alert.risk_score;
            
            // Apply styling
            container.className = '';
            document.body.style.backgroundColor = '#0f172a';
            
            clearInterval(beepInterval);

            if(alert.tier === 'Red') {
                container.classList.add('red-alert');
                document.getElementById('alert-title').innerText = 'CRITICAL EMERGENCY ALERT';
                document.body.style.backgroundColor = '#7f1d1d';
                beepInterval = setInterval(playSiren, 600);
            } else if(alert.tier === 'Orange') {
                container.classList.add('orange-alert');
                document.getElementById('alert-title').innerText = 'SEVERE WEATHER WARNING';
                beepInterval = setInterval(playSiren, 1000);
            } else if(alert.tier === 'Yellow') {
                container.classList.add('yellow-alert');
                document.getElementById('alert-title').innerText = 'WEATHER ADVISORY';
            } else {
                container.classList.add('green-alert');
                document.getElementById('alert-title').innerText = 'ALL CLEAR';
            }
        }

        // Poll the server every 2 seconds for new alerts
        setInterval(() => {
            fetch('/api/latest_alert')
                .then(r => r.json())
                .then(data => {
                    if(data && data.region && data.timestamp !== currentAlertId) {
                        currentAlertId = data.timestamp;
                        updateUI(data);
                    }
                })
                .catch(e => console.log('Waiting for alerts...'));
        }, 2000);
    </script>
</body>
</html>
"""

class ReceiverUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Keep console clean

    def do_GET(self):
        # Serve the UI
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_UI.encode('utf-8'))
        
        # API for the UI to poll the latest alert
        elif self.path == '/api/latest_alert':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response_data = latest_alert if latest_alert else {}
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global latest_alert
        if self.path == '/receive_alert':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                alert_data = json.loads(post_data.decode('utf-8'))
                # Add timestamp to act as a unique ID for the frontend
                alert_data['timestamp'] = time.time()
                latest_alert = alert_data
                
                print(f"[OK] Alert received for {alert_data.get('region')} ({alert_data.get('tier')})")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "success"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, ReceiverUIHandler)
    print(f"==================================================")
    print(f" KERAUNOS RECEIVER STATION STARTED ")
    print(f"==================================================")
    print(f" 1. Open a browser on this PC.")
    print(f" 2. Go to: http://localhost:{port}/")
    print(f" 3. Wait for alerts from the main dashboard.")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()

if __name__ == '__main__':
    run_server(8080)
