"""Set up the local Python environment before running master.py.

Run this once from the repo root:

    python run_first.py

This creates .venv if it does not already exist and installs requirements.txt.
After that, run the pipeline with master.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


# Use the Python executable inside .venv. Windows puts it in Scripts; Mac and Linux put it in bin.
def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


# Run a setup command from the repo root and stop immediately if it fails.
def run_command(command: list[str], label: str) -> None:
    print(f"\n==> {label}", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


# Make sure the Python version can run the package code.
def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Please use Python 3.10 or newer.")


# Create .venv only when it is missing.
def create_environment() -> Path:
    python_path = venv_python()
    if python_path.exists():
        print(f"Using existing environment: {VENV}", flush=True)
        return python_path

    run_command([sys.executable, "-m", "venv", str(VENV)], "Creating .venv")
    if not python_path.exists():
        raise FileNotFoundError(f"Could not find the virtual-environment Python at {python_path}")
    return python_path


# Install the packages needed by the build and analysis scripts.
def install_requirements(python_path: Path) -> None:
    if not REQUIREMENTS.exists():
        raise FileNotFoundError(f"Missing {REQUIREMENTS}")
    run_command([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS)], "Installing requirements")


def main() -> None:
    check_python_version()
    python_path = create_environment()
    install_requirements(python_path)

    print("\nSetup complete.", flush=True)
    if os.name == "nt":
        print("Activate the environment with:", flush=True)
        print(r"  .venv\Scripts\activate", flush=True)
    else:
        print("Activate the environment with:", flush=True)
        print("  source .venv/bin/activate", flush=True)
    print("Then run the full pipeline with:", flush=True)
    print("  python master.py", flush=True)


if __name__ == "__main__":
    main()
