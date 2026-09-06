from src.phase_a_region_discovery import run_region_discovery

scores_df, best = run_region_discovery()
print("\n=== ALL REGIONS WITH PASSING AWS STATIONS ===")
for _, r in scores_df.iterrows():
    if r.n_aws_pass > 0:
        print(f"{r.region_name} (GAGAN #{r.gagan_station_id}): score={r.composite_score:.3f}, n_aws={r.n_aws_pass}, cov={r.gagan_coverage_pct:.1f}%")
        print(f"   AWS stations ({r.n_aws_pass}): {r.aws_station_ids}\n")
