"""
FastAPI Local Backend Server for PINN 3D Hydrodynamic Simulation Data
PS 26077 -- AI Hyper-Local Early Warning System (MoES / NCMRWF)
"""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Keraunos PINN 3D Hydrodynamic API", version="1.0.0")

# Enable CORS so dashboard HTML (file:// or localhost) can fetch smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JSON_PATH = Path(__file__).resolve().parent.parent / "outputs" / "pinn_3d_multi_region_FINAL.json"
ALERTS_JSON_PATH = Path(__file__).resolve().parent.parent / "outputs" / "alert_history.json"


def load_pinn_data():
    if not JSON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PINN simulation dataset file missing at: {JSON_PATH}"
        )
    try:
        with open(JSON_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PINN simulation dataset JSON: {e}"
        )


@app.get("/api/pinn/regions")
def get_regions():
    """Returns list of all available Uttarakhand region keys and names."""
    data = load_pinn_data()
    return {
        "count": len(data),
        "regions": [
            {
                "key": key,
                "name": val.get("region_name", key),
                "choke": val.get("choke_location", "N/A"),
                "peak_depth_m": val.get("peak_water_depth_m", 0.0),
                "peak_speed_m_s": val.get("peak_velocity_m_s", 0.0)
            }
            for key, val in data.items()
        ]
    }


@app.get("/api/pinn/all")
def get_all_regions():
    """Returns full multi-region simulation dataset."""
    return load_pinn_data()


@app.get("/api/pinn/{region_name}")
def get_region_data(region_name: str):
    """Returns 51x51 grid tensors and hydrodynamic metrics for a specific basin."""
    data = load_pinn_data()
    key = region_name.lower().strip()
    if key not in data:
        valid_keys = list(data.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Region '{region_name}' not found. Available regions: {valid_keys}"
        )
    return data[key]


@app.get("/api/alerts/history")
def get_alert_history():
    """Returns the last 5 alerts from history."""
    if not ALERTS_JSON_PATH.exists():
        return []
    try:
        with open(ALERTS_JSON_PATH, "r") as f:
            history = json.load(f)
            # Return last 5 in reverse chronological order
            return list(reversed(history[-5:]))
    except Exception as e:
        print(f"Error reading alert history: {e}")
        return []

@app.get("/api/alerts/current")
def get_current_alert():
    """Returns the most recent active alert."""
    if not ALERTS_JSON_PATH.exists():
        return None
    try:
        with open(ALERTS_JSON_PATH, "r") as f:
            history = json.load(f)
            if not history:
                return None
            return history[-1]
    except Exception as e:
        print(f"Error reading current alert: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    print(f"[PINN API Server] Starting server on http://localhost:8000...")
    print(f"[PINN API Server] Serving dataset from: {JSON_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
