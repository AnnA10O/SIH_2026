import sys, os, time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

LOG_FILE = Path(r"C:\Users\LOQ\.gemini\antigravity-ide\brain\a049bc4c-0913-4cd0-abe4-60d08579cd2f\.system_generated\tasks\task-877.log")
SAT_DIR  = Path(r"d:\SIH\data\raw\satellite")

print("=" * 60)
print("       MOSDAC SATELLITE DOWNLOAD STATUS MONITOR")
print("=" * 60)

# Check log file modification time
if LOG_FILE.exists():
    mtime = LOG_FILE.stat().st_mtime
    age_sec = int(time.time() - mtime)
    print(f"Log file: {LOG_FILE}")
    print(f"Last log update: {age_sec}s ago")
    print("-" * 60)
    print("Latest 12 log lines:")
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
            for l in lines[-12:]:
                print(f"  {l}")
    except Exception as e:
        print(f"  Error reading log: {e}")
else:
    print(f"Log file not found at {LOG_FILE}")

print("-" * 60)
print("Current Data on Disk (d:\\SIH\\data\\raw\\satellite):")
total_files = 0
total_bytes = 0

if SAT_DIR.exists():
    for sub in sorted(SAT_DIR.iterdir()):
        if sub.is_dir() and sub.name != "logs":
            files = list(sub.rglob("*.h5"))
            size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
            total_files += len(files)
            total_bytes += sum(f.stat().st_size for f in files)
            if len(files) > 0:
                print(f"  {sub.name:<22} : {len(files):>4} files | {size_mb:>8.1f} MB")

    total_gb = total_bytes / (1024 * 1024 * 1024)
    print("-" * 60)
    print(f"TOTAL: {total_files} files | {total_gb:.2f} GB")
else:
    print("  Satellite directory does not exist yet.")

print("=" * 60)
