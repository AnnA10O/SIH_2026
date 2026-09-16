import requests
import json
import os

def main():
    # Public domain geojson of Indian states
    url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        os.makedirs("data", exist_ok=True)
        with open("data/indian_states.geojson", "w") as f:
            json.dump(data, f)
        print("GeoJSON saved to data/indian_states.geojson")
    else:
        print(f"Failed to download. HTTP {resp.status_code}")

if __name__ == "__main__":
    main()
