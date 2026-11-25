"""Vector Swap: each clip is decoded with the other clip's motion vectors.

A codec stores a P-frame as a motion field F (one vector per pixel, from block
matching: F(x) says where the content at x came from) plus a residual E, the
colour the motion cannot predict. Decoding it onto a picture P gives

    P_new(x) = P(x + F(x)) + E(x)

so each pixel is copied from where the motion says and then corrected.

Vector Swap keeps each clip's residuals but trades the motion vectors:
- Before the cut, A's own P-frames are decoded, but their vectors slide from A's
  own motion to B's: F = (1 - s) F_A + s F_B, with s rising from 0 to `swap`. A's
  colour changes keep coming while its pixels move the way B moves, so A breaks
  apart along B's motion. B's motion here is B's first frames (only the vectors;
  nothing of B's picture is shown before the cut).
- After the cut, B's own P-frames are decoded on the same picture, with A's
  vectors sliding back to B's: s falls from `swap` to 0. B's colour changes paint
  B in while its pixels still move the way A moved, then B settles into its own
  motion. A's motion here is A's last frames, the motion A would have continued.
- Both clips keep their real speed: frame k of the window uses the P-frame from
  frame k - 1 to k. A repeated (duplicate) frame has no P-frame of its own, so
  the previous motion carries on and no residual is added.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times becomes a random streak, so vectors that do not explain the motion
clearly better than (0, 0) are dropped, and block vectors are interpolated
between block centres so no hard block edges appear. Pixels are read with
bilinear sampling, so fractional vectors do not leave a zig-zag.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, motion, warp

BLOCK = 16           # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7     # keep a block's vector only if it cuts the block's error to below 70 %


@register
class VectorSwap(TransitionEffect):
    info = EffectInfo("vector_swap", "Vector Swap (draft)", "Viktor Tost",
                      "Clip A moves with clip B's motion, then B arrives moving like A, and they trade back.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("swap", "Swap", 1.0, 0.0, 1.0),
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("residual", "Residual", 1.0, 0.0, 2.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        a, b = ctx.outgoing, ctx.incoming
        width, height = ctx.size
        a_frames, b_frames = len(ctx.a_window), len(ctx.b_window)
        a_tail = list(ctx.a_window) or [len(a) - 1]    # A's motion to borrow after the cut
        first_b = min(ctx.b_window.start, len(b) - 1)

        picture = _last_clean_a(ctx).astype(np.float32)
        zero_field = np.zeros((height, width, 2), np.float32)
        a_field, b_field = zero_field, zero_field
        for i in range(ctx.length):
            if progress.cancelled:
                return
            if i < a_frames:
                # Before the cut: A's P-frame, vectors sliding from A's to B's.
                a_field, error = _next_p_frame(a, ctx.a_window[i], a_field, ctx.size)
                b_field = _next_field(b, min(first_b + i + 1, len(b) - 1), b_field, ctx.size)
                s = p["swap"] * _smoothstep((i + 1) / a_frames)
                field = (1.0 - s) * a_field + s * b_field
            else:
                # After the cut: B's P-frame, vectors sliding from A's back to B's.
                k = i - a_frames
                b_field, error = _next_p_frame(b, ctx.b_window[k], b_field, ctx.size)
                a_field = _next_field(a, a_tail[min(k, len(a_tail) - 1)], a_field, ctx.size)
                s = p["swap"] * (1.0 - _smoothstep((k + 1) / b_frames))
                field = s * a_field + (1.0 - s) * b_field

            picture = warp.warp(picture, p["gain"] * field, "bilinear")
            if error is not None:
                picture = picture + p["residual"] * error
            picture = np.clip(picture, 0.0, 255.0)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _smoothstep(t: float) -> float:
    """0 at t <= 0, 1 at t >= 1, with a gentle start and end: t^2 (3 - 2 t)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _next_p_frame(clip, index: int, previous: np.ndarray, size: tuple[int, int]):
    """The P-frame that turns frame index - 1 into frame index: (F, E).

    For the first frame or a repeated (duplicate) frame there is none: the previous
    motion carries on and E is None.
    """
    if index <= 0 or _same(clip.frame(index - 1), clip.frame(index)):
        return previous, None
    return _p_frame(clip, index - 1, index, size)


def _next_field(clip, index: int, previous: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Only the motion from frame index - 1 to frame index (see _next_p_frame)."""
    if index <= 0 or _same(clip.frame(index - 1), clip.frame(index)):
        return previous
    return _field(clip, index - 1, index, size)


def _last_clean_a(ctx) -> np.ndarray:
    """A's frame just before the window (A's first frame if there is none)."""
    return ctx.outgoing.frame(max(ctx.a_window.start - 1, 0))


def _p_frame(clip, first: int, second: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The P-frame that turns frame `first` of a clip into frame `second`: (F, E).

    E(x) = second(x) - first(x + F(x)), the colour the motion alone cannot explain.
    """
    field = _field(clip, first, second, size)
    predicted = warp.warp(clip.frame(first), field, "bilinear")
    return field, clip.frame(second).astype(np.float32) - predicted


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
