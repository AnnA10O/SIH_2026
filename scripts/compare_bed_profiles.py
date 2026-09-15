import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pinn_swe import SharedSWEPINN

def main():
    os.makedirs("outputs", exist_ok=True)
    
    y_vals = np.linspace(0.0, 1.0, 100)
    x_val = 0.5
    
    x_t = torch.tensor([[x_val]] * 100, dtype=torch.float32)
    y_t = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)
    
    # 1. Old Analytic Formula
    # z_thalweg = 1.0 - 0.75 * y_t + 0.15 * (y_t**2)
    z_old_nondim = (1.0 - 0.75 * y_vals + 0.15 * (y_vals**2))
    
    # 2. New Real DEM Profile via SharedSWEPINN
    z_new_nondim, dz_dx, dz_dy = SharedSWEPINN.compute_bed_profile(x_t, y_t)
    z_new_nondim = z_new_nondim.detach().cpu().numpy().flatten()
    
    # Convert non-dimensional z_tilde back to physical meters (610m to 3583m)
    z_min, z_max = 610.0, 3583.0
    z_old_meters = z_min + z_old_nondim * (z_max - z_min)
    z_new_meters = z_min + z_new_nondim * (z_max - z_min)
    
    # Known Anchor Points
    anchors = [
        ("Kedarnath Shrine", 0.05, 3583.0),
        ("Rambara Gorge", 0.32, 2700.0),
        ("Sonprayag Choke", 0.66, 1829.0),
        ("Rudraprayag", 0.95, 610.0)
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Mandakini River Valley Bed Elevation: Synthetic vs Real SRTM 30m DEM", fontsize=14, fontweight='bold')
    
    # Subplot 1: Real Elevation in Meters
    ax1.plot(y_vals, z_old_meters, 'r--', label='Synthetic Analytic Formula (Old)', linewidth=2.0)
    ax1.plot(y_vals, z_new_meters, 'b-', label='Real SRTM 30m DEM Profile (New)', linewidth=2.5)
    
    for name, y_a, z_a in anchors:
        ax1.scatter([y_a], [z_a], color='gold', edgecolor='black', s=80, zorder=5)
        ax1.annotate(f"{name}\n({z_a:.0f}m)", (y_a, z_a), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))
        
    ax1.set_ylabel("Elevation (meters MSL)", fontsize=11, fontweight='bold')
    ax1.set_title("Physical Elevation along Thalweg (Kedarnath -> Sonprayag -> Rudraprayag)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Subplot 2: Non-Dimensional z_tilde
    ax2.plot(y_vals, z_old_nondim, 'r--', label='Synthetic z_tilde', linewidth=2.0)
    ax2.plot(y_vals, z_new_nondim, 'b-', label='Real DEM z_tilde', linewidth=2.5)
    ax2.set_xlabel("Normalized Distance along River Corridor y_t (0 = Kedarnath, 1 = Rudraprayag)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Normalized Bed Elevation z_tilde", fontsize=11, fontweight='bold')
    ax2.set_title("Non-Dimensional Inputs fed to 2D SWE-PINN Solver", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = "outputs/dem_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[compare_bed_profiles] Saved comparison plot to: {output_path}")

if __name__ == "__main__":
    main()
