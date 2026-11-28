"""Colour conversion and pixel-value helpers, implemented by hand."""

from __future__ import annotations

import numpy as np


def bgr_to_y(frame: np.ndarray) -> np.ndarray:
    """Luma (ITU-R BT.601) of a BGR frame as an H x W float32 array."""
    img = frame.astype(np.float32)
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]


def to_uint8(values: np.ndarray) -> np.ndarray:
    """Rounds and saturates float pixel values to the 0..255 range."""
    return (np.clip(values, 0.0, 255.0) + 0.5).astype(np.uint8)
