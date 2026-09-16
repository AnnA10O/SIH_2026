import os
import sys
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pinn_swe import SharedSWEPINN

def main():
    print("=== SMOKE TEST: 2D SWE-PINN with Real SRTM 30m DEM (Rudraprayag) ===")
    
    # 1. Test Bed Profile Gradients directly
    x_t = torch.rand(10, 1)
    y_t = torch.rand(10, 1)
    z_t, dz_dx, dz_dy = SharedSWEPINN.compute_bed_profile(x_t, y_t)
    
    print(f"[Bed Profile Test] z_t min={z_t.min().item():.4f}, max={z_t.max().item():.4f}")
    print(f"[Bed Profile Test] dz_dx min={dz_dx.min().item():.4f}, max={dz_dx.max().item():.4f}")
    print(f"[Bed Profile Test] dz_dy min={dz_dy.min().item():.4f}, max={dz_dy.max().item():.4f}")
    
    assert not torch.isnan(z_t).any(), "NaN found in z_t"
    assert not torch.isnan(dz_dx).any(), "NaN found in dz_dx"
    assert not torch.isnan(dz_dy).any(), "NaN found in dz_dy"
    assert not (dz_dy == 0).all(), "dz_dy gradients are all zero!"
    print("-> Bed profile autograd check PASSED!\n")
    
    # 2. Run train_pinn smoke test
    pinn = SharedSWEPINN()
    train_metrics = pinn.train_pinn(epochs_adam=50, epochs_lbfgs=5, hazard_type="cloudburst")
    print(f"[Train Metrics] Result dict: {train_metrics}")
    
    assert not math.isnan(train_metrics["final_pde_loss"]), "Final PDE loss is NaN!"
    
    # 3. Run simulate_inundation smoke test
    sim_result = pinn.simulate_inundation(hazard_type="cloudburst", region="rudraprayag")
    
    print("\n[Inundation Sim Result Summary]")
    print(f"Region: {sim_result.get('region_name', 'rudraprayag')}")
    print(f"Peak Water Depth: {sim_result.get('peak_water_depth_m', sim_result.get('max_depth_m', 'N/A'))} m")
    print(f"Max Flow Velocity: {sim_result.get('max_velocity_ms', 'N/A')} m/s")
    
    print("\n-> Smoke test COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    import math
    main()
