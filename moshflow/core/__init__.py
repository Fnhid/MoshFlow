"""Core engine: project model, timeline resolution and rendering.

The core depends only on the interfaces in moshflow.api; it never imports a
concrete effect or OpenCV.
"""

from __future__ import annotations

