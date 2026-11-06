#!/usr/bin/env python3
"""Create the project virtual environment and install its dependencies.

Works on Windows, macOS, and Linux. Run with:
    python setup_venv.py
or, on macOS/Linux:
    python3 setup_venv.py
"""

from pathlib import Path
import subprocess
import sys
import venv


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / "venv"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"


def run(command: list[str]) -> None:
    """Run a command and stop with a helpful error if it fails."""
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> int:
    if not REQUIREMENTS_FILE.is_file():
        print(f"Error: requirements.txt was not found at {REQUIREMENTS_FILE}")
        return 1

    if not VENV_DIR.exists():
        print(f"Creating virtual environment: {VENV_DIR}")
        try:
            venv.EnvBuilder(with_pip=True).create(VENV_DIR)
        except Exception as exc:
            print(f"Could not create the virtual environment: {exc}")
            return 1
    else:
        print(f"Using existing virtual environment: {VENV_DIR}")

    # The interpreter location differs between Windows and POSIX systems.
    if sys.platform == "win32":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"

    if not venv_python.is_file():
        print(f"Error: virtual-environment Python was not found at {venv_python}")
        return 1

    try:
        run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    except subprocess.CalledProcessError as exc:
        print(f"\nInstallation failed (exit code {exc.returncode}).")
        print("Check your internet connection and Python/package compatibility, then run this script again.")
        return exc.returncode or 1

    print("\nDependencies installed successfully in venv.")
    print("Activate the environment with:")
    if sys.platform == "win32":
        print(r"  venv\Scripts\activate")
    else:
        print("  source venv/bin/activate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
