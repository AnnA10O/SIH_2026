import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("d:/SIH")
sys.path.insert(0, str(ROOT))

from src.dataset_builder import load_imd_parquets, build_feature_matrix
from src.train_neural_nowcaster_v2 import CloudburstCNNBiLSTM
from src.config import NEURAL_MODEL_PT

def plot_cm(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("Loading data...")
    splits = np.load(ROOT / "outputs" / "split_assignments.npz")
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    df = load_imd_parquets()
    X, y, _, _, _ = build_feature_matrix(df)
    
    # Check features matching the model
    checkpoint = torch.load(NEURAL_MODEL_PT, map_location='cpu')
    model_config = checkpoint.get('model_config', {'in_features': 15})
    model = CloudburstCNNBiLSTM(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        logits = model(X_t).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()
    
    threshold = 0.36 # Using same threshold or compute optimal
    preds = (probs > threshold).astype(int)
    
    cm_train = confusion_matrix(y[train_idx], preds[train_idx])
    cm_val = confusion_matrix(y[val_idx], preds[val_idx])
    cm_test = confusion_matrix(y[test_idx], preds[test_idx])
    
    print("Train Confusion Matrix:")
    print(cm_train)
    plot_cm(cm_train, 'Train Confusion Matrix', ROOT / 'outputs' / 'cm_train.png')
    
    print("Val Confusion Matrix:")
    print(cm_val)
    plot_cm(cm_val, 'Validation Confusion Matrix', ROOT / 'outputs' / 'cm_val.png')
    
    print("Test Confusion Matrix:")
    print(cm_test)
    plot_cm(cm_test, 'Test Confusion Matrix', ROOT / 'outputs' / 'cm_test.png')

if __name__ == "__main__":
    main()
