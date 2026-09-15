import re
import sys
import pypdf
import pandas as pd
import numpy as np
from pathlib import Path

# Load GAGAN stations
gagan_stations = [
    {"id": 1,  "name": "Madurai",     "lat": 9.837,  "lon": 78.091, "state": "Tamil Nadu"},
    {"id": 3,  "name": "Bangalore",   "lat": 13.033, "lon": 77.567, "state": "Karnataka"},
    {"id": 4,  "name": "Hyderabad",   "lat": 17.450, "lon": 78.470, "state": "Telangana"},
    {"id": 5,  "name": "Bhopal",      "lat": 23.284, "lon": 77.404, "state": "Madhya Pradesh"},
    {"id": 6,  "name": "Lucknow",     "lat": 26.761, "lon": 80.883, "state": "Uttar Pradesh"},
    {"id": 8,  "name": "Agatti",      "lat": 10.824, "lon": 72.176, "state": "Lakshadweep"},
    {"id": 10, "name": "Mumbai",      "lat": 19.089, "lon": 72.845, "state": "Maharashtra"},
    {"id": 11, "name": "Bagdogra",    "lat": 26.685, "lon": 88.326, "state": "West Bengal"},
    {"id": 12, "name": "Lengpui",     "lat": 23.840, "lon": 92.624, "state": "Mizoram"},
    {"id": 13, "name": "Guwahati",    "lat": 26.120, "lon": 91.590, "state": "Assam"},
    {"id": 14, "name": "Kolkata",     "lat": 22.645, "lon": 88.448, "state": "West Bengal"},
    {"id": 15, "name": "Raipur",      "lat": 21.242, "lon": 81.633, "state": "Chhattisgarh"},
    {"id": 16, "name": "Vizag",       "lat": 17.728, "lon": 83.226, "state": "Andhra Pradesh"},
    {"id": 17, "name": "Gaya",        "lat": 24.747, "lon": 84.945, "state": "Bihar"},
    {"id": 18, "name": "Trivandrum",  "lat": 8.536,  "lon": 76.905, "state": "Kerala"},
    {"id": 21, "name": "Ahmedabad2",  "lat": 23.064, "lon": 72.620, "state": "Gujarat"},
    {"id": 26, "name": "Bhubaneswar", "lat": 20.254, "lon": 85.809, "state": "Odisha"},
    {"id": 28, "name": "Jodhpur",     "lat": 26.264, "lon": 73.051, "state": "Rajasthan"},
]

print("Reading AWS_LIST_IN_SITU.pdf...")
reader = pypdf.PdfReader("AWS_LIST_IN_SITU.pdf")
raw_stations = []

for page_idx, page in enumerate(reader.pages):
    text = page.extract_text()
    if not text:
        continue
    tokens = [t.strip() for t in text.splitlines() if t.strip()]
    for i, tok in enumerate(tokens):
        m = re.match(r'^ISRO-?\s*(\d{4})$', tok)
        if m:
            isro_id = f"ISRO{m.group(1)}"
        elif tok == "ISRO" and i + 1 < len(tokens) and re.match(r'^\d{4}$', tokens[i+1]):
            isro_id = f"ISRO{tokens[i+1]}"
        else:
            isro_id = None

        if isro_id:
            window = tokens[max(0, i-15):min(len(tokens), i+25)]
            win_str = " ".join(window)
            coords = re.findall(r'(\d{1,2}\.\d{2,4})', win_str)
            lat, lon = None, None
            if len(coords) >= 2:
                for c_idx in range(len(coords)-1):
                    val1 = float(coords[c_idx])
                    val2 = float(coords[c_idx+1])
                    if (6.0 <= val1 <= 38.0) and (68.0 <= val2 <= 98.0):
                        lat, lon = val1, val2
                        break
            if lat is not None and lon is not None:
                # Try to extract station name or location clue
                raw_stations.append({
                    "station_id": isro_id,
                    "lat": lat,
                    "lon": lon,
                    "page": page_idx + 1,
                    "details": win_str[:120]
                })

df_aws = pd.DataFrame(raw_stations).drop_duplicates(subset=["station_id"])
print(f"Extracted {len(df_aws)} AWS stations.")

# Also load the specific quality-assessed Assam stations from station_quality_report.csv
quality_csv = Path("outputs/station_quality_report.csv")
if quality_csv.exists():
    q_df = pd.read_csv(quality_csv)
    print(f"Loaded {len(q_df)} quality-assessed regional stations.")
else:
    q_df = pd.DataFrame()

# Save complete AWS stations CSV
out_csv = Path("outputs/all_india_aws_stations.csv")
df_aws.to_csv(out_csv, index=False)
print(f"Saved to {out_csv}")
