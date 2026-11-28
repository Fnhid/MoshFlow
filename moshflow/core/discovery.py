"""EffectDiscovery: imports every effect module so it can @register itself."""

from __future__ import annotations

import importlib
import pkgutil
import traceback
from dataclasses import dataclass


@dataclass(frozen=True)
class DiscoveryFailure:
    module: str
    error: str


def discover_effects(package: str = "moshflow.effects") -> list[DiscoveryFailure]:
    """Imports all modules of `package` except those starting with '_'.

    A broken effect module does not stop the program: its error is returned
    so the UI and CLI can show it.
    """
    pkg = importlib.import_module(package)
    failures = []
    for module in pkgutil.iter_modules(pkg.__path__):
        if module.name.startswith("_"):
            continue
        name = f"{package}.{module.name}"
        try:
            importlib.import_module(name)
        except Exception:
            failures.append(DiscoveryFailure(name, traceback.format_exc(limit=3)))
    return failures
