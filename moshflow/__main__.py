"""`python -m moshflow` starts the GUI."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from moshflow.ui.main_window import run
    except ImportError as e:
        print(f"cannot start the GUI ({e}). Install the requirements: pip install -r requirements.txt",
              file=sys.stderr)
        return 1
    return run()


if __name__ == "__main__":
    sys.exit(main())
