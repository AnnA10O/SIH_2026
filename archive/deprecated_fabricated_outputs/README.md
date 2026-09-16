# Deprecated Fabricated PINN Outputs

These JSON files (`pinn_3d_multi_region.json`, `pinn_3d_multi_region_FINAL.json`, `pinn_3d_simulation.json`) were originally stored in `outputs/` and served by `scripts/serve_pinn.py`. 

They have been relocated to this archive because they contain fabricated, hardcoded data (e.g., peak depths of ~3.9 meters) that artificially bypass the actual `src/pinn_swe.py` physics solver. 

They are preserved here solely as evidence for **Doubt 9** of the codebase audit, documenting that the dashboard UI was intentionally disconnected from the under-trained physics model in order to display exaggerated, "realistic-looking" flood depths.
