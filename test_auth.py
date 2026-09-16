import requests
import json
import sys

API_BASE = "https://mosdac.gov.in/catalog-app"
with open("d:\\SIH\\mosdac_api\\config.json", "r") as f:
    config = json.load(f)
creds = config.get("user_credentials", {})

print("Sleeping 35 seconds to clear rolling rate limit...")
import time
time.sleep(35)
print("Testing MOSDAC Auth...")
try:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.post("https://mosdac.gov.in/download_api/gettoken", json=creds, headers=headers, timeout=15)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
sys.stdout.flush()
