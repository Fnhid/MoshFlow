"""Block-matching motion estimation, written with plain NumPy arithmetic.

Conventions (see docs/API.md, "Motion vectors"):
- A vector v = (dx, dy) says where a pixel of the current frame came from:
  current(x, y) ~ reference(x + dx, y + dy). An object that moved 2 pixels to
  the right therefore has v = (-2, 0).
- Block vectors: int32 array of shape (rows, cols, 2) with rows = ceil(H / block),
  cols = ceil(W / block); [..., 0] is dx and [..., 1] is dy. The last row and
  column of blocks may be smaller than block x block.
- A field: float32 array of shape (H, W, 2), one vector per pixel, same order.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_MAX_WORKERS = 8
_THREAD_MIN_WORK = 2_000_000  # pixels * candidates needed to outweigh thread startup


def estimate(reference: np.ndarray, current: np.ndarray, block: int, search: int) -> np.ndarray:
    """Full-search block matching on two H x W float32 luma images.

    For every block of `current`, tries every d = (dx, dy) with |dx|, |dy| <= search
    and keeps the one with the smallest SAD(d) = sum |current(x) - reference(x + d)|
    over the block. Candidates that reach outside the image are skipped. Ties go to
    the smaller |dx| + |dy|, so flat areas get (0, 0).

    Returns block vectors (see the module docstring).
    """
    height, width = current.shape
    rows, cols = -(-height // block), -(-width // block)
    top = np.arange(rows) * block
    bottom = np.minimum(top + block, height)
    left = np.arange(cols) * block
    right = np.minimum(left + block, width)

    # One shift of the frame per candidate: a slice of the padded reference.
    ref_pad = np.ascontiguousarray(_pad_edges(reference.astype(np.float32), search))
    cur_f = np.ascontiguousarray(current, dtype=np.float32)
    candidates = _candidates(search)
    best = np.zeros((rows, cols, 2), np.int32)

    row_slices = {dy: _valid_span(top, bottom, height, dy) for dy in range(-search, search + 1)}
    col_slices = {dx: _valid_span(left, right, width, dx) for dx in range(-search, search + 1)}

    workers = min(rows, os.cpu_count() or 1, _MAX_WORKERS)
    use_threads = workers > 1 and height * width * len(candidates) >= _THREAD_MIN_WORK
    band_count = workers if use_threads else 1
    band_rows = -(-rows // band_count)
    bands = [(r0, min(r0 + band_rows, rows)) for r0 in range(0, rows, band_rows)]

    def run_band(band: tuple[int, int]) -> None:
        _estimate_band(
            ref_pad, cur_f, best, band, block, search, cols, candidates, row_slices, col_slices
        )

    if use_threads:
        with ThreadPoolExecutor(max_workers=len(bands)) as pool:
            list(pool.map(run_band, bands))
    else:
        for band in bands:
            run_band(band)
    return best


def to_field(vectors: np.ndarray, size: tuple[int, int], block: int) -> np.ndarray:
    """Expands block vectors to a per-pixel field: every pixel takes its block's vector.

    size is (width, height). Returns an H x W x 2 float32 field.
    """
    width, height = size
    rows = np.arange(height) // block
    cols = np.arange(width) // block
    return vectors[rows[:, None], cols[None, :]].astype(np.float32)


def smooth_field(vectors: np.ndarray, size: tuple[int, int], block: int, smoothness: float) -> np.ndarray:
    """A per-pixel field without block edges, for Liquid Melt.

    Interpolates the block vectors bilinearly between block centres, then mixes:
    (1 - smoothness) * to_field(...) + smoothness * interpolated. smoothness = 0
    gives exactly to_field(vectors, size, block).
    """
    width, height = size
    rows, cols = vectors.shape[:2]
    # Position of every pixel centre in block units, where block centres sit on whole numbers.
    ys = np.clip((np.arange(height) + 0.5) / block - 0.5, 0, rows - 1)
    xs = np.clip((np.arange(width) + 0.5) / block - 0.5, 0, cols - 1)
    y0, x0 = ys.astype(np.intp), xs.astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, cols - 1)
    wy = (ys - y0).astype(np.float32)[:, None, None]
    wx = (xs - x0).astype(np.float32)[None, :, None]

    v = vectors.astype(np.float32)
    upper = v[y0[:, None], x0[None, :]] * (1 - wx) + v[y0[:, None], x1[None, :]] * wx
    lower = v[y1[:, None], x0[None, :]] * (1 - wx) + v[y1[:, None], x1[None, :]] * wx
    smooth = upper * (1 - wy) + lower * wy

    blocky = to_field(vectors, size, block)
    return ((1 - smoothness) * blocky + smoothness * smooth).astype(np.float32)


def _estimate_band(
    ref_pad: np.ndarray,
    current: np.ndarray,
    best: np.ndarray,
    band: tuple[int, int],
    block: int,
    search: int,
    cols: int,
    candidates: list[tuple[int, int]],
    row_slices: dict[int, tuple[int, int]],
    col_slices: dict[int, tuple[int, int]],
) -> None:
    """Runs full-search block matching on one horizontal slice of block rows."""
    height, width = current.shape
    r0, r1 = band
    n_rows = r1 - r0
    y0 = r0 * block
    y1 = min(r1 * block, height)
    bh = y1 - y0
    cur_band = current[y0:y1]

    best_sad = np.full((n_rows, cols), np.inf, np.float32)
    best_band = best[r0:r1]
    # Reused per-band work arrays; padding beyond (bh, width) stays zero for partial blocks.
    diff_buf = np.zeros((n_rows * block, cols * block), np.float32)
    diff_valid = diff_buf[:bh, :width]
    diff_4d = diff_buf.reshape(n_rows, block, cols, block)
    row_sum = np.empty((n_rows, block, cols), np.float32)
    sad = np.empty((n_rows, cols), np.float32)

    for dx, dy in candidates:
        vr0, vr1 = row_slices[dy]
        ar0 = max(r0, vr0) - r0
        ar1 = min(r1, vr1) - r0
        if ar0 >= ar1:
            continue
        c0, c1 = col_slices[dx]
        if c0 >= c1:
            continue
        shifted = ref_pad[search + y0 + dy:search + y1 + dy, search + dx:search + dx + width]
        np.subtract(cur_band, shifted, out=diff_valid)
        np.abs(diff_valid, out=diff_valid)
        diff_4d.sum(axis=3, out=row_sum)
        row_sum.sum(axis=1, out=sad)
        sad_sub = sad[ar0:ar1, c0:c1]
        best_sub = best_sad[ar0:ar1, c0:c1]
        better = sad_sub < best_sub
        if better.any():
            best_sub[better] = sad_sub[better]
            best_band[ar0:ar1, c0:c1][better] = (dx, dy)


def _valid_span(starts: np.ndarray, ends: np.ndarray, limit: int, delta: int) -> tuple[int, int]:
    """Index range [first, stop) of blocks whose shifted span stays inside [0, limit]."""
    first = 0
    count = len(starts)
    while first < count and int(starts[first]) + delta < 0:
        first += 1
    stop = count
    while stop > first and int(ends[stop - 1]) + delta > limit:
        stop -= 1
    return first, stop


def _candidates(search: int) -> list[tuple[int, int]]:
    """Every (dx, dy) in the search range, nearest to (0, 0) first, so ties keep the smaller one."""
    span = range(-search, search + 1)
    return sorted(((dx, dy) for dy in span for dx in span), key=lambda d: abs(d[0]) + abs(d[1]))


def _pad_edges(image: np.ndarray, n: int) -> np.ndarray:
    """The image with n extra pixels on every side, copied from the nearest edge pixel."""
    height, width = image.shape
    ys = np.clip(np.arange(-n, height + n), 0, height - 1)
    xs = np.clip(np.arange(-n, width + n), 0, width - 1)
    return image[ys[:, None], xs[None, :]]
