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

    def do_GET(self):
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
    print("  [1] New Dashboard (One-Glance Console)     -> UI/dashboard/new_ui.html")
    print("  [2] Previous Dashboard (Mission Control)   -> UI/dashboard/previous_ui.html")
    print("  [3] Realtime Simulation UI                 -> UI/realtime/index.html")
    print("  [4] Launch All (in browser tabs)")
    print("  [5] Exit")
    print("-" * 66)

    try:
        choice = input(" Enter choice [1-5] (default: 1): ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "5"

    url_new = f"http://localhost:{PORT}/UI/dashboard/new_ui.html"
    url_prev = f"http://localhost:{PORT}/UI/dashboard/previous_ui.html"
    url_realtime = f"http://localhost:{PORT}/UI/realtime/index.html"

    if choice in ("", "1"):
        print(f"\n>> Opening New Dashboard: {url_new}")
        webbrowser.open(url_new)
    elif choice == "2":
        print(f"\n>> Opening Previous Dashboard: {url_prev}")
        webbrowser.open(url_prev)
    elif choice == "3":
        print(f"\n>> Opening Realtime Simulation UI: {url_realtime}")
        webbrowser.open(url_realtime)
    elif choice == "4":
        print(f"\n>> Opening New Dashboard: {url_new}")
        webbrowser.open(url_new)
        time.sleep(0.5)
        print(f">> Opening Previous Dashboard: {url_prev}")
        webbrowser.open(url_prev)
        time.sleep(0.5)
        print(f">> Opening Realtime Simulation UI: {url_realtime}")
        webbrowser.open(url_realtime)
    elif choice == "5":
        print("\nExiting.")
        sys.exit(0)
    else:
        print(f"\n>> Unknown option '{choice}', opening New Dashboard by default.")
        webbrowser.open(url_new)

    print("\n[INFO] Server is running in the background. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server. Goodbye!")


if __name__ == "__main__":
    main()
