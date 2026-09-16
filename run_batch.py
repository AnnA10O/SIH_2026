import os
import time
import subprocess

print("Waiting 65 seconds for MOSDAC rolling rate limits to clear...")
time.sleep(65)
print("Resuming downloads with rate limit protection active...")

commands = [
    "python d:\\SIH\\mosdac_downloader.py --mode events --dataset AUTO_HEM --limit 1",
    "python d:\\SIH\\mosdac_downloader.py --mode events --dataset AUTO_OLR --limit 1",
    "python d:\\SIH\\mosdac_downloader.py --mode events --dataset AUTO_UTH --limit 1",
    "python d:\\SIH\\mosdac_downloader.py --mode events --dataset AUTO_CTP --limit 1"
]

for cmd in commands:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
