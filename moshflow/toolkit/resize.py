"""Image resizing, implemented by hand."""

import numpy as np


def _source_coords(dst: int, src: int) -> np.ndarray:
    # Map destination pixel centres onto source pixel centres.
    coords = (np.arange(dst, dtype=np.float32) + 0.5) * (src / dst) - 0.5
    return np.clip(coords, 0, src - 1)


def nearest(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to size = (width, height)."""
    width, height = size
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (width, height):
        return frame
    xs = (_source_coords(width, src_w) + 0.5).astype(np.intp)
    ys = (_source_coords(height, src_h) + 0.5).astype(np.intp)
    return frame[ys[:, None], xs[None, :]]


def bilinear(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of an H x W x C uint8 image to size = (width, height)."""
    width, height = size
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (width, height):
        return frame

    xs = _source_coords(width, src_w)
    ys = _source_coords(height, src_h)
    x0 = xs.astype(np.intp)
    y0 = ys.astype(np.intp)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    wx = (xs - x0)[None, :, None]
    wy = (ys - y0)[:, None, None]

    # Pick the four neighbours first and convert only them, not the whole source
    # frame: the result is the same, and shrinking a large frame is much faster.
    rows0, rows1 = frame[y0], frame[y1]
    top = rows0[:, x0].astype(np.float32) * (1 - wx) + rows0[:, x1].astype(np.float32) * wx
    bottom = rows1[:, x0].astype(np.float32) * (1 - wx) + rows1[:, x1].astype(np.float32) * wx
    out = top * (1 - wy) + bottom * wy
    return (out + 0.5).astype(np.uint8)
