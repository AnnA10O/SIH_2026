"""KERAUNOS Dashboard Runner — Dual UI Launcher

Allows launching either the New Dashboard (One-Glance Console) or the Previous Dashboard (Mission Control).
"""

import http.server
import os
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PORT = 8000

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format, *args):
        # Keep console output clean
        pass

    def do_POST(self):
        if self.path == "/api/alerts/send":
            import json
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                alert_data = json.loads(post_data.decode('utf-8'))
                
                from src.alert_sender import send_alert_to_receiver
                receiver_ip = "10.210.48.78"
                receiver_port = 8080
                
                # send_alert_to_receiver also saves it to history
                result = send_alert_to_receiver(receiver_ip, receiver_port, alert_data)
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return

        if hasattr(super(), 'do_POST'):
            return super().do_POST()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/api/alerts/current":
            import json
            try:
                alerts_file = BASE_DIR / "outputs" / "alert_history.json"
                if not alerts_file.exists():
                    data = None
                else:
                    with open(alerts_file, "r") as f:
                        history = json.load(f)
                        data = history[-1] if history else None
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return

        if self.path == "/api/alerts/history":
            import json
            try:
                alerts_file = BASE_DIR / "outputs" / "alert_history.json"
                if not alerts_file.exists():
                    data = []
                else:
                    with open(alerts_file, "r") as f:
                        history = json.load(f)
                        data = list(reversed(history[-5:]))
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return

        if self.path == "/api/nowcast":
            import json
            from src.inference_server import handle_api_request
            try:
                data = handle_api_request()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return

        if self.path.startswith("/api/pinn"):
            import json
            try:
                pinn_file = BASE_DIR / "outputs" / "pinn_3d_multi_region_FINAL.json"
                if not pinn_file.exists():
                    pinn_file = BASE_DIR / "outputs" / "pinn_3d_simulation.json"
                
                with open(pinn_file, "r") as f:
                    pinn_data = json.load(f)

                # Check if specific region requested (e.g. /api/pinn/rudraprayag)
                parts = [p for p in self.path.split("/") if p]
                if len(parts) >= 3:
                    region = parts[2].lower()
                    if region in pinn_data:
                        pinn_data = pinn_data[region]

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(pinn_data).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return

        return super().do_GET()


def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        httpd.serve_forever()


def main():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    print("=" * 66)
    print("       KERAUNOS CLOUDBURST EARLY WARNING DASHBOARD LAUNCHER        ")
    print("=" * 66)
    print(f" Local Web Server active at: http://localhost:{PORT}/")
    print("-" * 66)
    print(" Select which UI you would like to run:")
    print("  [1] Older UI (Mission Control & PINN Handoff)  -> dashboard/older_ui.html")
    print("  [2] Newer UI (Kerunos AI NOWCAST Dashboard)    -> dashboard/newer_ui.html")
    print("  [3] Realtime Simulation UI                     -> UI/realtime/index.html")
    print("  [4] Unified Tabbed Dashboard UI                -> dashboard/tabbed_dashboard.html")
    print("  [5] Launch All UIs (in separate browser tabs)")
    print("  [6] Exit")
    print("-" * 66)

    try:
        choice = input(" Enter choice [1-6] (default: 1): ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "6"

    url_older = f"http://localhost:{PORT}/dashboard/older_ui.html"
    url_newer = f"http://localhost:{PORT}/dashboard/newer_ui.html"
    url_realtime = f"http://localhost:{PORT}/UI/realtime/index.html"
    url_tabbed = f"http://localhost:{PORT}/dashboard/tabbed_dashboard.html"

    if choice in ("", "1"):
        print(f"\n>> Opening Option 1 [Older UI]: {url_older}")
        webbrowser.open(url_older)
    elif choice == "2":
        print(f"\n>> Opening Option 2 [Newer UI (Kerunos)]: {url_newer}")
        webbrowser.open(url_newer)
    elif choice == "3":
        print(f"\n>> Opening Option 3 [Realtime Simulation UI]: {url_realtime}")
        webbrowser.open(url_realtime)
    elif choice == "4":
        print(f"\n>> Opening Option 4 [Unified Tabbed Dashboard UI]: {url_tabbed}")
        webbrowser.open(url_tabbed)
    elif choice == "5":
        print(f"\n>> Opening Older UI: {url_older}")
        webbrowser.open(url_older)
        time.sleep(0.4)
        print(f">> Opening Newer UI (Kerunos): {url_newer}")
        webbrowser.open(url_newer)
        time.sleep(0.4)
        print(f">> Opening Realtime Simulation UI: {url_realtime}")
        webbrowser.open(url_realtime)
        time.sleep(0.4)
        print(f">> Opening Unified Tabbed Dashboard UI: {url_tabbed}")
        webbrowser.open(url_tabbed)
    elif choice == "6":
        print("\nExiting launcher.")
        sys.exit(0)
    else:
        print(f"\n>> Unknown option '{choice}', opening Older UI by default.")
        webbrowser.open(url_older)

    print("\n[INFO] Server is running in the background. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server. Goodbye!")


if __name__ == "__main__":
    main()
