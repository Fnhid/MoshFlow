"""Vector Swap: each clip is pushed around by the other clip's motion.

Like the motion-vector transfer of tiberiuiancu/datamoshing (style_transfer.py,
built on FFglitch): one clip's motion vectors are *added* to the other clip's own
motion, block by block and frame by frame. The clip keeps its own motion and
colours, so it stays recognisable, and is pushed around by the other's motion.

Without a codec, this is done with a displacement map D (one vector per pixel),
as in ychalier/transflow: the frame shown is the clip's real frame read through
the map, out(x) = frame(x + D(x)) with bilinear sampling. Each frame the added
motion F moves the map along, D_new(x) = F(x) + D(x + F(x)), so the push builds
up over time. The picture itself is never re-warped from the previous output, so
no blur or noise accumulates, and each clip keeps playing at its real speed.

The transition:
- Before the cut, A's real frames are shown through D, and B's motion is added
  with a strength s that rises from 0 to `swap` over the first 60 % of A's part:
  A plays on but is dragged by the way B moves. B's motion here is B's first
  frames (only the vectors; nothing of B's picture is shown before the cut).
- At the cut, B's real frames are shown through the same map, so B appears
  distorted exactly the way A was and the shapes line up across the cut.
- After the cut, A's motion (A's last frames, the motion A would have continued)
  is added instead, so B is dragged the way A moved, and over B's part the map
  is let go (shown D is scaled by 1 - r, r a smoothstep from 0 to 1): pixels slide
  back to where they belong and the transition lands exactly on B's real frame.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times becomes a random streak, so vectors that do not explain the motion
clearly better than (0, 0) are dropped, and block vectors are interpolated
between block centres so no hard block edges appear.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, motion, warp

BLOCK = 16           # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7     # keep a block's vector only if it cuts the block's error to below 70 %
SWAP_RISE = 0.6      # share of A's part after which the other clip's motion is fully added
NO_B_PART = 1 / 3    # with no B window, the share of the end over which the map is let go


@register
class VectorSwap(TransitionEffect):
    info = EffectInfo("vector_swap", "Vector Swap (draft)", "Viktor Tost",
                      "Each clip is dragged by the other clip's motion; B arrives warped like A, then settles.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("swap", "Swap", 1.0, 0.0, 1.0),
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        a, b = ctx.outgoing, ctx.incoming
        width, height = ctx.size
        a_frames, b_frames = len(ctx.a_window), len(ctx.b_window)
        a_tail = list(ctx.a_window) or [len(a) - 1]    # A's motion to borrow after the cut
        first_b = min(ctx.b_window.start, len(b) - 1)
        release_from = a_frames if b_frames else round(ctx.length * (1 - NO_B_PART))

        displacement = np.zeros((height, width, 2), np.float32)
        a_field = b_field = np.zeros((height, width, 2), np.float32)
        for i in range(ctx.length):
            if progress.cancelled:
                return
            if i < a_frames:
                # Before the cut: A's real frame, pushed by B's motion.
                frame = a.frame(ctx.a_window[i])
                b_field = _next_field(b, min(first_b + i + 1, len(b) - 1), b_field, ctx.size)
                added = p["swap"] * _smoothstep((i + 1) / (SWAP_RISE * a_frames)) * b_field
            else:
                # After the cut: B's real frame, pushed by A's motion.
                k = i - a_frames
                frame = b.frame(ctx.b_window[k])
                a_field = _next_field(a, a_tail[min(k, len(a_tail) - 1)], a_field, ctx.size)
                added = p["swap"] * a_field
            added = p["gain"] * added
            displacement = added + warp.warp(displacement, added, "bilinear")

            # Let go of the map over the last part, so the end is B's real frame.
            release = 0.0
            if i >= release_from:
                release = _smoothstep((i + 1 - release_from) / (ctx.length - release_from))
            out = warp.warp(frame, (1.0 - release) * displacement, "bilinear")
            if i == ctx.length - 1 and b_frames:
                out = frame.astype(np.float32)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _smoothstep(t: float) -> float:
    """0 at t <= 0, 1 at t >= 1, with a gentle start and end: t^2 (3 - 2 t)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _next_field(clip, index: int, previous: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Only the motion from frame index - 1 to frame index (see _next_p_frame)."""
    if index <= 0 or _same(clip.frame(index - 1), clip.frame(index)):
        return previous
    return _field(clip, index - 1, index, size)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two frames are (almost) identical, i.e. a repeated frame."""
    return float(np.abs(a.astype(np.float32) - b).mean()) < 0.5


def _field(clip, first: int, second: int, size: tuple[int, int]) -> np.ndarray:
    """Motion from frame `first` to frame `second` of a clip, as a smooth field (H, W, 2).

    Block matching runs at half size (every second pixel, half the block and search)
    to keep it fast, and the vectors are doubled back to full size.
    """
    reference = color.bgr_to_y(clip.frame(first))[::2, ::2]
    current = color.bgr_to_y(clip.frame(second))[::2, ::2]
    vectors = motion.estimate(reference, current, BLOCK // 2, SEARCH // 2)
    vectors = _confident(vectors, reference, current, BLOCK // 2)
    return motion.smooth_field(vectors * 2, size, BLOCK, 1.0)


def _confident(vectors: np.ndarray, reference: np.ndarray, current: np.ndarray, block: int) -> np.ndarray:
    """Block vectors with the unreliable ones set to (0, 0).

    For each block, error(v) = sum of |current(x) - reference(x + v)| over its pixels.
    A vector is kept only if error(v) < CONFIDENCE * error(0): the match must be
    clearly better than "nothing moved", otherwise it is just noise.
    """
    height, width = current.shape
    field = motion.to_field(vectors, (width, height), block)
    moved = _block_sums(np.abs(current - warp.warp(reference, field)), block)
    still = _block_sums(np.abs(current - reference), block)
    out = vectors.copy()
    out[~(moved < CONFIDENCE * still)] = 0
    return out


def _block_sums(values: np.ndarray, block: int) -> np.ndarray:
    """Sum of an H x W array over each block; partial blocks at the edges sum what they have."""
    height, width = values.shape
    rows, cols = -(-height // block), -(-width // block)
    padded = np.zeros((rows * block, cols * block), np.float32)
    padded[:height, :width] = values
    return padded.reshape(rows, block, cols, block).sum(axis=(1, 3))
