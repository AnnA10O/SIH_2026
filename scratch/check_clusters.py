import pandas as pd
import sys
sys.path.insert(0, 'd:/SIH')
from src.phase_d_training import load_imd_parquets, build_feature_matrix

events = load_imd_parquets()
df = events.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.sort_values(['region_name', 'timestamp'])

# Only cluster the positive events to find the discrete storms
pos_df = df[df['final_label'].isin({'CONFIRMED_CLOUDBURST', 'CANDIDATE_CLOUDBURST'})].dropna(subset=['timestamp']).copy()

# Group by region, and within each region, find gaps > 7 days
clusters = []
cluster_id = 0
for region, group in pos_df.groupby('region_name'):
    group = group.sort_values('timestamp')
    gap = group['timestamp'].diff() > pd.Timedelta('7 days')
    # cumsum creates a unique ID for each contiguous block of events (separated by >3 days)
    block_id = gap.cumsum()
    
    for b_id, block in group.groupby(block_id):
        duration = (block['timestamp'].max() - block['timestamp'].min()).days
        clusters.append({
            'region': region,
            'cluster_id': cluster_id,
            'n_events': len(block),
            'duration_days': duration,
            'start': block['timestamp'].min(),
            'end': block['timestamp'].max()
        })
        cluster_id += 1

cluster_df = pd.DataFrame(clusters)
print(f"Total independent positive storm clusters (>3 day gap): {len(cluster_df)}")
print("\nCluster duration distribution (days):")
print(cluster_df['duration_days'].value_counts().sort_index())

print("\nCluster size distribution (number of positive records):")
print(cluster_df['n_events'].value_counts().sort_index().head(10))

# Print the top 5 largest clusters by duration to check for mega-clusters
print("\nTop 5 largest clusters (by duration):")
print(cluster_df.sort_values('duration_days', ascending=False).head(5).to_string())
