"""KERAUNOS Dashboard Launcher — Subfolder Forwarder
Executes the main root run_dashboard.py regardless of working directory.
"""
import sys
from pathlib import Path

# Locate root directory and run_dashboard.py
ROOT_DIR = Path(__file__).resolve().parent.parent
ROOT_LAUNCHER = ROOT_DIR / "run_dashboard.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(ROOT_LAUNCHER), run_name="__main__")
