"""Converts BGR frames to QPixmap using hand-written resizing for display."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QPixmap

from moshflow.toolkit import resize


def fit_size(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Largest (width, height) inside (max_w, max_h) that preserves aspect ratio."""
    if src_w <= 0 or src_h <= 0 or max_w <= 0 or max_h <= 0:
        return 1, 1
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def frame_to_pixmap(frame: np.ndarray, max_size: tuple[int, int] | None = None) -> QPixmap:
    """Turns an (H, W, 3) uint8 BGR array into a QPixmap without Qt image processing."""
    src_h, src_w = frame.shape[:2]
    if max_size is not None:
        target = fit_size(src_w, src_h, max_size[0], max_size[1])
        if target != (src_w, src_h):
            frame = resize.bilinear(frame, target)
    height, width = frame.shape[:2]
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixmap = QPixmap()
    pixmap.loadFromData(header + rgb.tobytes(), "PPM")
    return pixmap
