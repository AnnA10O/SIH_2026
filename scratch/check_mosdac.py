import json
from mosdac_downloader import search_date_files, get_era_dataset_id, load_credentials

creds = load_credentials()
date_strs = ["2026-09-17", "2026-09-16", "2026-09-15", "2026-08-01", "2024-07-31", "2024-08-01"]
for date_str in date_strs:
    era_dataset = get_era_dataset_id("AUTO_UTH", date_str)
    entries = search_date_files(era_dataset, date_str, creds, count=5)
    print(f"Date: {date_str}, Entries: {len(entries)}")
