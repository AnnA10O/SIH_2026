import re
import matplotlib.pyplot as plt
import numpy as np

log_file = r'C:\Users\LOQ\.gemini\antigravity-ide\brain\d5cf6d17-e659-4f18-b865-8da2f6d14451\.system_generated\tasks\task-5390.log'

epochs = []
train_losses = []
val_losses = []
val_csis = []

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Example line: "    1 | 2.00e-03 |     0.0019 |     0.0011 |    0.443 |    0.383 |    0.347 |     0.6182"
        match = re.match(r'\s*(\d+)\s*\|\s*[\d.e-]+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|.*?\|\s*([\d.]+)\s*\|', line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))
            val_csis.append(float(match.group(4)))

if epochs:
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/training_loss_curve.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_csis, label='Val CSI', marker='^', color='green')
    plt.title('Validation CSI Curve')
    plt.xlabel('Epoch')
    plt.ylabel('CSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/training_csi_curve.png')
    print("Curves saved to outputs/training_loss_curve.png and outputs/training_csi_curve.png")
else:
    print("No epochs found in log yet.")
