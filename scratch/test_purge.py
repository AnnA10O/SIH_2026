import numpy as np
import pandas as pd

def greedy_split(cluster_sizes, train_frac=0.7, val_frac=0.15):
    # cluster_sizes: list of (cluster_id, n_events)
    # returns: train_ids, val_ids, test_ids
    
    total_events = sum(s[1] for s in cluster_sizes)
    target_train = int(total_events * train_frac)
    target_val = int(total_events * val_frac)
    target_test = total_events - target_train - target_val
    
    train_ids, val_ids, test_ids = [], [], []
    curr_train, curr_val, curr_test = 0, 0, 0
    
    # Sort clusters descending by size
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)
    
    for cid, size in sorted_clusters:
        # Compute deficit (target - current)
        def_train = target_train - curr_train
        def_val = target_val - curr_val
        def_test = target_test - curr_test
        
        # We assign to the bucket with the largest relative deficit, or just absolute deficit
        # Using absolute deficit is fine since we sorted by size.
        max_def = max(def_train, def_val, def_test)
        if max_def == def_train:
            train_ids.append(cid)
            curr_train += size
        elif max_def == def_val:
            val_ids.append(cid)
            curr_val += size
        else:
            test_ids.append(cid)
            curr_test += size
            
    print(f"Target: Train={target_train}, Val={target_val}, Test={target_test}")
    print(f"Actual: Train={curr_train}, Val={curr_val}, Test={curr_test}")
    return train_ids, val_ids, test_ids

cluster_sizes = [
    (1, 274), (2, 19), (3, 16), (4, 6), (5, 2), (6, 1)
] + [(i, 1) for i in range(7, 22)]

greedy_split(cluster_sizes)
