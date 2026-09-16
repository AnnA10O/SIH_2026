import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Paths
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
HISTORY_FILE = OUTPUT_DIR / "alert_history.json"

# Make sure outputs dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Templates based on tiers
TEMPLATES = {
    "Red": "Immediate Evacuation Required. Flash flood/cloudburst imminent in this region. Move to higher ground immediately.",
    "Orange": "Be Prepared. High probability of severe weather or flash flooding. Keep emergency kits ready and avoid low-lying areas.",
    "Yellow": "Be Aware. Moderate risk of heavy rainfall. Stay tuned to local advisories and monitor river levels.",
    "Green": "All Clear. No immediate threats detected. Normal conditions."
}

def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def send_alert_to_receiver(receiver_ip: str, receiver_port: int, alert_data: dict):
    """
    Sends the alert via HTTP POST to the receiver PC and saves the alert history
    to outputs/alert_history.json with network success/failure status.
    """
    url = f"http://{receiver_ip}:{receiver_port}/receive_alert"
    
    # Send HTTP request
    network_status = "Failed"
    try:
        response = requests.post(url, json=alert_data, timeout=5.0)
        if response.status_code == 200:
            network_status = "Success"
        else:
            network_status = f"Failed (HTTP {response.status_code})"
    except requests.exceptions.RequestException as e:
        network_status = f"Failed ({type(e).__name__})"

    # Format the alert with the network status and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_record = {
        "id": f"alert-{int(time.time())}",
        "timestamp": timestamp,
        "region": alert_data.get("region", "Unknown"),
        "tier": alert_data.get("tier", "Green"),
        "risk_score": alert_data.get("risk_score", 0.0),
        "precaution": TEMPLATES.get(alert_data.get("tier", "Green"), TEMPLATES["Green"]),
        "network_target": f"{receiver_ip}:{receiver_port}",
        "network_status": network_status,
        "raw_data": alert_data
    }

    # Load, append, and save
    history = load_history()
    history.append(alert_record)
    
    # Keep only the last 50 alerts to prevent file from growing indefinitely
    if len(history) > 50:
        history = history[-50:]
        
    save_history(history)
    print(f"[{timestamp}] Alert ({alert_record['tier']}) saved. Network Push: {network_status}")
    return alert_record

if __name__ == "__main__":
    # Test firing an alert
    test_alert = {
        "region": "Uttarkashi Basin",
        "tier": "Orange",
        "risk_score": 82.5
    }
    
    # Receiver IP provided by user
    receiver_ip = "10.210.48.78"
    receiver_port = 8080 # Default port, can be changed if needed
    
    send_alert_to_receiver(receiver_ip, receiver_port, test_alert)
