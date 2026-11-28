"""Motion compensation: moving a picture along a vector field, with plain NumPy arithmetic."""

from __future__ import annotations

import numpy as np


def warp(image: np.ndarray, field: np.ndarray, sampling: str = "nearest") -> np.ndarray:
    """Backward warping: out(x, y) = image(x + dx, y + dy), with (dx, dy) = field[y, x].

    image: H x W or H x W x C array (any dtype). field: H x W x 2 float32, as made
    by motion.to_field or motion.smooth_field. Sample positions outside the image
    are clamped to the nearest edge pixel.

    sampling = "nearest" rounds the position to the nearest pixel (blocky codec
    look); "bilinear" mixes the four surrounding pixels (smooth, for Liquid Melt).

    Returns a float32 array of the same shape as image. Never modifies image.
    """
    height, width = image.shape[:2]
    source = image.astype(np.float32)
    xs = np.clip(np.arange(width, dtype=np.float32)[None, :] + field[..., 0], 0, width - 1)
    ys = np.clip(np.arange(height, dtype=np.float32)[:, None] + field[..., 1], 0, height - 1)

    if sampling == "nearest":
        return source[(ys + 0.5).astype(np.intp), (xs + 0.5).astype(np.intp)]
    if sampling != "bilinear":
        raise ValueError(f"sampling must be 'nearest' or 'bilinear', got {sampling!r}")

    x0, y0 = xs.astype(np.intp), ys.astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = xs - x0, ys - y0
    if source.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    upper = source[y0, x0] * (1 - wx) + source[y0, x1] * wx
    lower = source[y1, x0] * (1 - wx) + source[y1, x1] * wx
    return (upper * (1 - wy) + lower * wy).astype(np.float32)
