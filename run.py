"""Start-up file for Visual Studio (F5): launches the MoshFlow GUI.

Works with any Python 3.7 or newer, including the one Visual Studio installs with
its Python workload (3.7 in VS 2019, 3.9 in VS 2022). If that Python lacks the
packages, it uses a ready local `.venv`, or offers to create one and install
requirements.txt into it (no administrator rights needed).
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

MIN_PYTHON = (3, 7)
REQUIRED = (
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PySide6", "PySide6"),
)
REEXEC_GUARD = "MOSHFLOW_REEXEC"
ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _missing_packages():
    return [package for module, package in REQUIRED if importlib.util.find_spec(module) is None]


def _venv_python(folder):
    for rel in (Path("Scripts") / "python.exe", Path("bin") / "python"):
        if (folder / rel).is_file():
            return folder / rel
    return None


def _is_current(executable):
    try:
        return Path(executable).resolve() == Path(sys.executable).resolve()
    except OSError:
        return False


def _python_is_ready(executable):
    check = (
        "import importlib.util, sys; "
        "ok = sys.version_info >= {} and all("
        "importlib.util.find_spec(m) is not None for m in ('numpy', 'cv2', 'PySide6')); "
        "sys.exit(0 if ok else 1)"
    ).format(MIN_PYTHON)
    try:
        return subprocess.call([str(executable), "-c", check],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def _find_ready_venv():
    for folder in (VENV, ROOT / "venv", ROOT / "env"):
        python = _venv_python(folder)
        if python is not None and not _is_current(python) and _python_is_ready(python):
            return python
    return None


def _reexec(executable, args):
    env = os.environ.copy()
    env[REEXEC_GUARD] = "1"
    return subprocess.call([str(executable), str(Path(__file__).resolve())] + list(args), env=env)


def _set_up_venv():
    """Asks, then creates .venv with this Python and installs requirements.txt into it.
    Returns the new environment's python, or None."""
    if not sys.stdin or not sys.stdin.isatty():
        return None
    try:
        answer = input("\nCreate .venv here and install requirements.txt into it now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if answer not in ("", "y", "yes"):
        return None
    if _venv_python(VENV) is None and subprocess.call([sys.executable, "-m", "venv", str(VENV)]) != 0:
        return None
    python = _venv_python(VENV)
    if python is None:
        return None
    subprocess.call([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    if subprocess.call([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]) != 0:
        return None
    return python


def _print_setup_guide(version_too_old, missing):
    say = lambda text="": print(text, file=sys.stderr)   # noqa: E731
    say("MoshFlow could not start with the active Python environment.")
    say("  Active Python: {} ({})".format(sys.version.split()[0], sys.executable))
    if version_too_old:
        say("  Problem: MoshFlow needs Python {}.{} or newer.".format(*MIN_PYTHON))
    if missing:
        say("  Missing packages: {}".format(", ".join(missing)))
    say()
    say("Easiest on Windows: double-click run_windows.bat in the project folder.")
    say()
    say("Or in Visual Studio:")
    say("  1. Solution Explorer -> Python Environments -> right-click the active environment")
    say("     -> Install from requirements.txt (or Add Environment -> Virtual environment).")
    say("  2. Press F5 again.")


def launch(args=None):
    args = sys.argv[1:] if args is None else list(args)
    version_too_old = sys.version_info < MIN_PYTHON
    missing = _missing_packages()

    if version_too_old or missing:
        if os.environ.get(REEXEC_GUARD) != "1":
            python = _find_ready_venv()
            if python is not None:
                print("MoshFlow: using the local environment {}".format(python))
                return _reexec(python, args)
        _print_setup_guide(version_too_old, missing)
        python = None if version_too_old or os.environ.get(REEXEC_GUARD) == "1" else _set_up_venv()
        if python is None:
            if sys.stdin and sys.stdin.isatty():
                input("\nPress Enter to close.")
            return 1
        return _reexec(python, args)

    from moshflow.__main__ import main

    return main()


if __name__ == "__main__":
    sys.exit(launch())
