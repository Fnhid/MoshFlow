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

from __future__ import annotations

import numpy as np

from moshflow.toolkit import color, motion, warp

LOW, HIGH = 0.15, 0.85   # thresholds lie between these
JITTER = 0.05            # share of randomness in the order when a priority is given
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


CHANGE_FULL = 1500.0     # summed mean residual (grey levels) of a block at which B has fully taken it
NOISE = 6.0              # residual below this is compression noise, not change
STEP_CAP = 0.05          # one frame adds at most this share of CHANGE_FULL to a block
SPREAD = 0.94            # a block reaches this share of its neighbour's total each frame
COARSE = 4               # the takeover's motion search runs on every 4th pixel


class Takeover:
    """How far clip B has overwritten the old picture, block by block, as B plays.

    Every frame, each block adds up B's residual there: what B's motion does not
    explain, as a codec measures it (a camera pan is motion, not change, so it does
    not count). A block becomes B, whole, once that total reaches the block's
    threshold. Nothing is taken on a timer: a block of the old picture stays until
    B's own change wears it away, as in a real decoder. One frame can add only a
    capped amount (so a scene cut in B does not replace everything at once), and each
    block's total seeps one block outwards per frame (times SPREAD), so B fills the
    background from where it moves. Only B's next keyframe (progress 1, the
    transition's last frame) replaces whatever is left.

        takeover = refresh.Takeover(ctx, block)
        ...
        taken = takeover.step(b_before, b_now, progress)      # (H, W, 1) bool
        picture = np.where(taken, b_now, picture)
    """

    def __init__(self, ctx, block: int, change_full: float = CHANGE_FULL):
        self.block, self.size, self.full = block, ctx.size, change_full
        self.limits = thresholds(ctx.size, block, ctx.seed, incoming_motion(ctx, block))
        self.change = np.zeros(self.limits.shape, np.float32)

    def step(self, before, now, progress: float) -> np.ndarray:
        """Adds B's change from `before` to `now` (either may be None) and returns the
        blocks B has taken so far. `progress` runs from 0 to 1 over the moshed frames."""
        if before is not None and now is not None:
            # A scene cut changes everything at once; capping one frame's share keeps it
            # from taking the whole picture in one frame.
            self.change += np.minimum(block_mean(_residual(before, now), self.block), STEP_CAP * self.full)
            # B's change seeps one block outwards per frame, so B fills the background
            # from where it moves instead of the still areas waiting for the keyframe.
            self.change = np.maximum(self.change, SPREAD * _neighbour_max(self.change))
        if progress >= 1.0:                      # B's next keyframe: every block is B
            return np.ones((self.size[1], self.size[0], 1), bool)
        return to_pixels(self.change / self.full >= self.limits, self.block, self.size)


def _residual(before, now) -> np.ndarray:
    """|now - before moved by its motion| in luma, minus noise, (H, W).

    The motion is found by block matching on every COARSE-th pixel (8 x 8 blocks,
    search 4 there: 32 x 32 blocks and +-16 pixels at full size), which is enough
    to tell a camera pan or a moving object from new content, and cheap."""
    y_before = color.bgr_to_y(np.asarray(before, np.float32))
    y_now = color.bgr_to_y(np.asarray(now, np.float32))
    height, width = y_now.shape
    vectors = motion.estimate(y_before[::COARSE, ::COARSE], y_now[::COARSE, ::COARSE], 8, 4)
    field = motion.to_field(vectors * COARSE, (width, height), 8 * COARSE)
    predicted = warp.warp(y_before, field)
    return np.maximum(np.abs(y_now - predicted) - NOISE, 0.0)


def _neighbour_max(grid: np.ndarray) -> np.ndarray:
    """The largest value among each cell's 8 neighbours and itself (edges repeated)."""
    padded = np.pad(grid, 1, mode="edge")
    rows, cols = grid.shape
    out = grid.copy()
    for r in range(3):
        for c in range(3):
            out = np.maximum(out, padded[r:r + rows, c:c + cols])
    return out
