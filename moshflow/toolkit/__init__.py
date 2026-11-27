"""Shared image helpers, written by the team with plain NumPy arithmetic.

No library image-processing functions are used here. Helpers that more than
one effect needs are added to this package:
  color, resize, frames  colour, resizing and window helpers
  motion, warp           block matching and motion compensation
  codec                  hierarchical ME, P-frame modes, dead-zone residual, coordinate advection
See docs/API.md, "Toolkit".
"""
