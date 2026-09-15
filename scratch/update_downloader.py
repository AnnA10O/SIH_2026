import re

with open('d:/SIH/mosdac_downloader.py', 'r') as f:
    code = f.read()

new_imports = """import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np"""

code = code.replace("import pandas as pd\nfrom pathlib import Path\nfrom datetime import datetime", new_imports)

era_logic = """
def get_era_dataset_id(dataset_id: str, date_str: str) -> str:
    \"\"\"Dynamically select the product ID based on operational eras if a generic prefix is used.\"\"\"
    if not dataset_id.startswith("AUTO_"):
        return dataset_id
        
    product = dataset_id.replace("AUTO_", "")
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year = d.year
    month = d.month
    
    if year < 2014:
        # Kalpana-1 era
        if product == "HEM": return "K1VHR_L2B_HEM"
        if product == "QPE": return "K1VHR_L2B_QPE"
        return f"K1VHR_L2B_{product}"
    elif year < 2016 or (year == 2016 and month <= 9):
        # INSAT-3D era
        return f"3DIMG_L2B_{product}"
    else:
        # INSAT-3DR era
        return f"3RIMG_L2B_{product}"

def run_downloader"""

code = code.replace("def run_downloader", era_logic)

events_logic_old = """    if mode == "events":
        if not EVENTS_CSV.exists():
            print(f"ERROR: {EVENTS_CSV} does not exist. Run pipeline.py first.")
            return
        df = pd.read_csv(EVENTS_CSV)
        non_normal = df[df.final_label != "NORMAL"]
        event_dates = sorted(pd.to_datetime(non_normal["timestamp"]).dt.strftime("%Y-%m-%d").unique())
        target_dates = event_dates
        print(f"Mode: TARGETED EVENTS ({len(target_dates)} dates identified from ground truth)")
        for d in target_dates:
            print(f"  - Event Date: {d}")"""

events_logic_new = """    if mode == "events":
        if not EVENTS_CSV.exists():
            print(f"ERROR: {EVENTS_CSV} does not exist. Run pipeline.py first.")
            return
        df = pd.read_csv(EVENTS_CSV)
        
        pos_mask = df.final_label != "NORMAL"
        pos_dates = pd.to_datetime(df[pos_mask]["timestamp"]).dt.strftime("%Y-%m-%d").unique()
        neg_dates = pd.to_datetime(df[~pos_mask]["timestamp"]).dt.strftime("%Y-%m-%d").unique()
        
        # Sample negative dates to maintain roughly 104:1 imbalance relative to positive events
        # Since downloading 104x dates is too expensive (thousands of files), we sample a comparable large volume 
        # of negative dates (e.g., all available negative dates in the CSV, or max 1000) to ensure feature availability
        # doesn't perfectly correlate with positive events.
        np.random.seed(42)
        n_neg = min(len(neg_dates), len(pos_dates) * 10) # 10x dates gives plenty of negative rows (multiple basins per date)
        sampled_neg = np.random.choice(neg_dates, size=n_neg, replace=False)
        
        target_dates = sorted(list(set(pos_dates) | set(sampled_neg)))
        print(f"Mode: TARGETED EVENTS ({len(pos_dates)} pos dates, {len(sampled_neg)} neg dates sampled for unbiased availability)")"""

code = code.replace(events_logic_old, events_logic_new)

process_loop_old = """        entries = search_date_files(dataset_id, d, token)
        print(f"  Files available in MOSDAC archive: {len(entries)}")

        if not entries:
            print(f"  No scenes found for {dataset_id} on {d}")
            continue"""

process_loop_new = """        era_dataset = get_era_dataset_id(dataset_id, d)
        entries = search_date_files(era_dataset, d, token)
        print(f"  Files available in MOSDAC archive for {era_dataset}: {len(entries)}")

        if not entries:
            print(f"  No scenes found for {era_dataset} on {d}")
            continue"""

code = code.replace(process_loop_old, process_loop_new)

file_dest_old = """            filename = f"{dataset_id}_{time_tag}_{rec_id}.h5"
            dest = DOWNLOAD_BASE / dataset_id / d / filename"""

file_dest_new = """            filename = f"{era_dataset}_{time_tag}_{rec_id}.h5"
            dest = DOWNLOAD_BASE / era_dataset / d / filename"""

code = code.replace(file_dest_old, file_dest_new)

with open('d:/SIH/mosdac_downloader.py', 'w') as f:
    f.write(code)
print("Updated mosdac_downloader.py successfully!")
