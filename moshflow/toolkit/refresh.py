"""Block refresh: a block shows either the old picture or the new one, never a mix.

When a real decoder loses a keyframe, the new clip does not fade in. It arrives in
whole blocks: an encoder sends a fresh (intra) block where its motion prediction
fails, and the next keyframe replaces everything at once. Mixing the two pictures
with a weight (out = P + h (B - P)) looks like a crossfade instead.

Use `thresholds` once per transition and `switched` every frame:

    block = refresh.block_size(ctx.size)                     # 8 at 720p
    limits = refresh.thresholds(ctx.size, block, ctx.seed, refresh.incoming_motion(ctx, block))
    ...
    taken = refresh.switched(share, block, limits, ctx.size)  # share: 0..1 per pixel
    picture = np.where(taken, new_frame, picture)

A block switches when its average share reaches its own threshold, so blocks switch
at different moments: where the share is highest first, and with a `priority`,
where the new clip moves first. Once the share is 1
everywhere, every block has switched.
"""

import numpy as np

from moshflow.toolkit import color

LOW, HIGH = 0.15, 0.85   # thresholds lie between these
JITTER = 0.1             # share of randomness in the order when a priority is given
BLOCK_AT_720P = 8        # switching block size at 720p; scaled with the frame height


def thresholds(size: tuple[int, int], block: int, seed: int, priority: np.ndarray | None = None) -> np.ndarray:
    """One threshold per block, (rows, cols) float32, between LOW and HIGH.

    Without `priority` the order is random. With a (rows, cols) `priority` (for
    example how much the new clip moves in each block) the blocks with the highest
    priority get the lowest thresholds and switch first, as an encoder sends fresh
    blocks where things move, while still areas keep the old picture longest.
    """
    width, height = size
    rows, cols = -(-height // block), -(-width // block)
    noise = np.random.default_rng(seed).random((rows, cols))
    if priority is None:
        order = noise
    else:
        # Smoothed over the neighbouring blocks, so B spreads out from where it moves
        # in connected patches instead of single blocks scattered over still areas.
        flat = _spread(np.asarray(priority, np.float64)).reshape(-1)
        ranks = np.empty(flat.size)
        ranks[np.argsort(-flat, kind="stable")] = np.arange(flat.size)
        order = (1.0 - JITTER) * (ranks / max(flat.size - 1, 1)).reshape(rows, cols) + JITTER * noise
    return (LOW + (HIGH - LOW) * order).astype(np.float32)


def block_size(size: tuple[int, int]) -> int:
    """The switching block size for a frame size: 8 pixels at 720p, 16 at 1440p."""
    return max(4, int(round(BLOCK_AT_720P * size[1] / 720)))


def _spread(grid: np.ndarray, passes: int = 2) -> np.ndarray:
    """Averages every block with its 8 neighbours, `passes` times (edges repeated)."""
    for _ in range(passes):
        padded = np.pad(grid, 1, mode="edge")
        rows, cols = grid.shape
        grid = sum(padded[r:r + rows, c:c + cols] for r in range(3) for c in range(3)) / 9.0
    return grid


def block_mean(values: np.ndarray, block: int) -> np.ndarray:
    """Mean of an (H, W) map over every block, (rows, cols). Edge blocks may be smaller."""
    height, width = values.shape
    rows, cols = -(-height // block), -(-width // block)
    padded = np.zeros((rows * block, cols * block), np.float32)
    counts = np.zeros_like(padded)
    padded[:height, :width] = values
    counts[:height, :width] = 1.0
    sums = padded.reshape(rows, block, cols, block).sum(axis=(1, 3))
    return sums / counts.reshape(rows, block, cols, block).sum(axis=(1, 3))


def switched(share, block: int, limits: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(H, W, 1) bool: True in every block whose mean share has reached its threshold.

    `share` is a per-pixel (H, W) map, or one number for the whole picture.
    """
    if np.ndim(share) == 2:
        means = block_mean(np.asarray(share, np.float32), block)
    else:
        means = np.full(limits.shape, float(share), np.float32)
    return to_pixels(means >= limits, block, size)


def to_pixels(blocks: np.ndarray, block: int, size: tuple[int, int]) -> np.ndarray:
    """(H, W, 1) bool from a (rows, cols) bool map: every pixel takes its block's value."""
    width, height = size
    return np.repeat(np.repeat(blocks, block, axis=0), block, axis=1)[:height, :width, None]


def incoming_motion(ctx, block: int, look: int = 8) -> np.ndarray:
    """How much clip B changes in each block, (rows, cols): the mean |luma difference|
    between B's first frame in the window and `look` frames later (or the window's
    last frame, if that is sooner). Use it as the `priority` of `thresholds`."""
    b = ctx.incoming
    first = min(ctx.b_window.start, len(b) - 1)
    last = min(first + look, len(b) - 1, max(ctx.b_window.stop - 1, first))
    difference = np.abs(color.bgr_to_y(b.frame(last)) - color.bgr_to_y(b.frame(first)))
    return block_mean(difference, block)
