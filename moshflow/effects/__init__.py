"""Transition effects: one module per effect.

Every module here (except those starting with '_') is imported at start-up by
moshflow.core.discovery, and its @register-ed class becomes available in the
UI and the CLI. To write a new effect, copy _template.py.
"""

from __future__ import annotations

