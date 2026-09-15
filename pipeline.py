"""Master pipeline runner — executes Phase A → B → C → D end-to-end or individually."""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_all():
    from src.phase_a_region_discovery import run_region_discovery
    from src.phase_c_labeling import run_event_labeling
    from src.phase_d_training import run_training

    print("\n" + "=" * 60)
    print("  Cloudburst Nowcasting -- Full Pipeline Run")
    print("  PS 26077 | MoES / NCMRWF | MOSDAC Data Foundation")
    print("=" * 60)

    # Phase A + B (Phase B is embedded inside Phase A)
    scores, best_region = run_region_discovery()
    if not best_region:
        print("\nPipeline halted: no valid region found in Phase A.")
        sys.exit(1)

    # Phase C
    events = run_event_labeling(best_region)
    if events.empty:
        print("\nPipeline halted: no events labeled in Phase C.")
        sys.exit(1)

    # Phase D
    run_training()

    print("\n" + "=" * 60)
    print("  Pipeline complete. Outputs:")
    from src.config import (REGION_SCORES_CSV, STATION_QUALITY_CSV,
                             CLOUDBURST_EVENTS_CSV, TRAINING_REPORT_MD)
    for f in [REGION_SCORES_CSV, STATION_QUALITY_CSV,
              CLOUDBURST_EVENTS_CSV, TRAINING_REPORT_MD]:
        exists = "OK" if Path(f).exists() else "MISSING"
        print(f"  [{exists}] {f}")
    print("=" * 60)


def run_phase_a():
    from src.phase_a_region_discovery import run_region_discovery
    run_region_discovery()


def run_phase_b():
    from src.phase_b_quality_gate import run_quality_gate
    from src.config import AWS_CSV
    run_quality_gate(str(AWS_CSV))


def run_phase_c():
    from src.phase_a_region_discovery import run_region_discovery
    from src.phase_c_labeling import run_event_labeling
    _, best = run_region_discovery()
    run_event_labeling(best)


def run_phase_d():
    from src.phase_d_training import run_training
    run_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cloudburst Nowcasting Pipeline Runner"
    )
    parser.add_argument(
        "--phase", choices=["a", "b", "c", "d", "all"],
        default="all",
        help="Which phase to run (default: all)"
    )
    args = parser.parse_args()

    dispatch = {"a": run_phase_a, "b": run_phase_b,
                "c": run_phase_c, "d": run_phase_d, "all": run_all}
    dispatch[args.phase]()
