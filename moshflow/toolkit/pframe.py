"""P-frame helpers: the motion and colour correction between two frames of a clip.

A P-frame is a motion field F (block matching) plus a residual
E(x) = second(x) - first(x + F(x)). Decoding it onto a picture P gives
P_new(x) = P(x + F(x)) + E(x).
"""

import numpy as np

from moshflow.api import FrameSource
from moshflow.toolkit import color, motion, warp

SEARCH = 8
CONFIDENCE = 0.7   # keep a block vector only if it cuts the block error below 70 %
_BLOCK = 8         # block size at half resolution


def p_frame(clip: FrameSource, first: int, second: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(F, E): the motion field (H, W, 2) and colour residual (H, W, 3) from `first` to `second`."""
    field = motion_field(clip, first, second, size)
    predicted = warp.warp(clip.frame(first), field, "bilinear")
    return field, clip.frame(second).astype(np.float32) - predicted


def motion_field(clip: FrameSource, first: int, second: int, size: tuple[int, int]) -> np.ndarray:
    """Smooth motion field (H, W, 2) from one frame to another, matched at half size.

    Vectors that do not explain the motion clearly better than (0, 0) are dropped,
    because a noisy vector repeated over many frames becomes a streak.
    """
    reference = color.bgr_to_y(clip.frame(first))[::2, ::2]
    current = color.bgr_to_y(clip.frame(second))[::2, ::2]
    vectors = motion.estimate(reference, current, _BLOCK, SEARCH // 2)
    vectors = _confident(vectors, reference, current)
    return motion.smooth_field(vectors * 2, size, _BLOCK * 2, 1.0)


def block_mean(values: np.ndarray, block: int) -> np.ndarray:
    """Mean of an H x W array over each block; partial blocks at the edges use what they have."""
    height, width = values.shape
    rows, cols = -(-height // block), -(-width // block)
    padded = np.zeros((rows * block, cols * block), np.float32)
    counts = np.zeros_like(padded)
    padded[:height, :width] = values
    counts[:height, :width] = 1.0
    sums = padded.reshape(rows, block, cols, block).sum(axis=(1, 3))
    return sums / counts.reshape(rows, block, cols, block).sum(axis=(1, 3))


def _confident(vectors: np.ndarray, reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Block vectors with unreliable ones set to (0, 0): error(v) must be < CONFIDENCE * error(0)."""
    height, width = current.shape
    field = motion.to_field(vectors, (width, height), _BLOCK)
    moved = _block_sum(np.abs(current - warp.warp(reference, field)))
    still = _block_sum(np.abs(current - reference))
    out = vectors.copy()
    out[~(moved < CONFIDENCE * still)] = 0
    return out


def _block_sum(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    rows, cols = -(-height // _BLOCK), -(-width // _BLOCK)
    padded = np.zeros((rows * _BLOCK, cols * _BLOCK), np.float32)
    padded[:height, :width] = values
    return padded.reshape(rows, _BLOCK, cols, _BLOCK).sum(axis=(1, 3))
