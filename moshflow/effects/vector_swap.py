"""Vector Swap: each clip's blocks slide along the other clip's motion vectors.

Emulates FFglitch-style motion-vector transfer (as in tiberiuiancu/datamoshing's
style_transfer.py and Datamosher-Pro's "Motion Transfer"): a clip is decoded
from its own P-frames, but with the other clip's motion vectors.

A codec stores a P-frame per 16 x 16 macroblock: one whole-pixel motion vector v
(found by block matching: the block's content came from v pixels away) and a
residual, the colour correction for what the motion could not predict. The
decoder copies each block from where its vector points in the previous picture
and adds the residual:

    P_new(x) = P(x + v(block of x)) + E(x)

Here the blocks are copied exactly (nearest pixel, whole-pixel vectors), which
gives the crisp sliding-tile codec look. With the clip's own vectors its full
residual fits; with foreign vectors it does not, so that share of the residual
is compressed the way an encoder does: each 8 x 8 block keeps only its average
correction (the DC coefficient). A residual that no longer fits the moved
picture then becomes a soft colour patch, not black-and-white edge noise:
E = (1 - s) E_full + s E_dc, with s how far the vectors are swapped.

The transition:
- Before the cut, A plays normally at first. Then its blocks take B's vectors
  instead of their own, v = (1 - s) v_A + s v_B with s rising to `swap` over the
  first 60 % of A's part, while A's own residuals keep coming: A breaks into
  tiles that slide the way B moves. B's vectors here come from B's first frames
  (nothing of B's picture is shown yet).
- After the cut, B's I-frame is dropped, as in a datamosh transition. B's
  P-frames are decoded on the same picture with A's vectors (A's last frames),
  which trade back to B's own after 40 % of B's part.
- B emerges block by block, never blended: a block turns into B', B offset by
  how far the vectors are still swapped, B(x + s (v_A - v_B)), so B arrives
  moving the way A moved, once the total of how much B changed there, plus a
  slowly growing time share, reaches the block's own threshold: where B moves
  first, still areas last, with no fixed moment at which A has to be gone, and reaches 1 on the last frame,
  which is B's real frame.
- Both clips keep their real speed: frame k uses the P-frame from k - 1 to k. A
  repeated (duplicate) frame has no P-frame; the previous vectors carry on.

Vectors that do not explain the motion clearly better than (0, 0) are dropped,
so flat or blurry areas do not slide at random.
"""

from __future__ import annotations

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, motion, refresh, warp

BLOCK = 16           # macroblock size: one motion vector per block
SEARCH = 8           # search range of the block matching, at full size
DC_BLOCK = 8         # residual blocks that keep only their average (DC) correction
CONFIDENCE = 0.7     # keep a block's vector only if it cuts the block's error to below 70 %
SWAP_RISE = 0.6      # share of A's part after which A's blocks follow B's vectors fully
SWAP_HOLD = 0.4      # share of B's part during which B's blocks follow A's vectors
NO_B_PART = 1 / 3    # with no B window, the share of the end over which B emerges


@register
class VectorSwap(TransitionEffect):
    info = EffectInfo("vector_swap", "Vector Swap", "Viktor Tost",
                      "A's blocks slide along B's motion vectors, then B's blocks arrive sliding like A's.")

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
        rows, cols = -(-height // BLOCK), -(-width // BLOCK)
        a_frames, b_frames = len(ctx.a_window), len(ctx.b_window)
        a_tail = list(ctx.a_window) or [len(a) - 1]    # A's vectors to borrow after the cut
        first_b = min(ctx.b_window.start, len(b) - 1)
        emerge_from = a_frames if b_frames else round(ctx.length * (1 - NO_B_PART))
        fine = refresh.block_size(ctx.size)              # switching blocks: 8 x 8 at 720p
        limits = refresh.thresholds(ctx.size, fine, ctx.seed, refresh.incoming_motion(ctx, fine))
        takeover = refresh.Takeover(ctx, fine)

        picture = a.frame(max(ctx.a_window.start - 1, 0)).astype(np.float32)
        zero = np.zeros((rows, cols, 2), np.float32)
        vec_a, vec_b = zero, zero
        for i in range(ctx.length):
            if progress.cancelled:
                return
            if i < a_frames:
                # Before the cut: A's P-frame, its vectors turning into B's.
                index = ctx.a_window[i]
                real = a.frame(index).astype(np.float32)
                vec_a, error = _p_frame(a, index, vec_a, ctx.size)
                vec_b = _vectors_to(b, min(first_b + i + 1, len(b) - 1), vec_b)
                taken = _smoothstep((i + 1) / (SWAP_RISE * a_frames))
                s = p["swap"] * taken
                vectors = (1.0 - s) * vec_a + s * vec_b
            else:
                # After the cut: B's P-frame on the same picture, with A's vectors
                # trading back to B's own.
                k = i - a_frames
                real = b.frame(ctx.b_window[k]).astype(np.float32)
                vec_b, error = _p_frame(b, ctx.b_window[k], vec_b, ctx.size)
                vec_a = _vectors_to(a, a_tail[min(k, len(a_tail) - 1)], vec_a)
                s = p["swap"] * (1.0 - _smoothstep(((k + 1) / b_frames - SWAP_HOLD) / (1.0 - SWAP_HOLD)))
                vectors = s * vec_a + (1.0 - s) * vec_b

            # Decode: copy every block from where its whole-pixel vector points, then
            # add the block-averaged residual.
            vectors = np.round(p["gain"] * vectors)
            picture = warp.warp(picture, motion.to_field(vectors, ctx.size, BLOCK), "nearest")
            if error is not None:
                # With its own vectors a clip's full residual fits the moved picture;
                # with foreign vectors it no longer does, and only its block-averaged
                # (DC) version is used, so it becomes a colour patch, not edge noise.
                error = (1.0 - s) * error + s * _dc_only(error)
                picture = picture + p["residual"] * error
            picture = np.clip(picture, 0.0, 255.0)

            if i < a_frames:
                # A plays normally until the swap takes over: a block shows A's real
                # frame until the swap has reached that block.
                swapped = refresh.switched(taken, fine, limits, ctx.size)
                picture = np.where(swapped, picture, real)
            out = picture
            if i >= emerge_from:
                # B arrives block by block where it changes (see the docstring).
                target = real if i >= a_frames else b.frame(first_b).astype(np.float32)
                offset = np.round(p["gain"] * s * (vec_a - vec_b)) if i >= a_frames else 0.0 * vec_a
                arriving = warp.warp(target, motion.to_field(offset, ctx.size, BLOCK), "nearest")
                v = (i + 1 - emerge_from) / (ctx.length - emerge_from)
                before = b.frame(ctx.b_window[i - a_frames] - 1) if i >= a_frames and ctx.b_window[i - a_frames] > 0 else None
                arrived = takeover.step(before, real if i >= a_frames else None, v)
                picture = np.where(arrived, arriving, picture)
                out = target if i == ctx.length - 1 else picture

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _smoothstep(t: float) -> float:
    """0 at t <= 0, 1 at t >= 1, with a gentle start and end: t^2 (3 - 2 t)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _p_frame(clip, index: int, previous: np.ndarray, size: tuple[int, int]):
    """A clip's own P-frame from frame index - 1 to index: (block vectors, residual).

    The residual is what the clip's own vectors cannot predict. For the first frame or a repeated (duplicate) frame
    there is no P-frame: the previous vectors carry on and the residual is None.
    """
    if index <= 0 or _same(clip.frame(index - 1), clip.frame(index)):
        return previous, None
    vectors = _vectors(clip, index - 1, index)
    predicted = warp.warp(clip.frame(index - 1), motion.to_field(vectors, size, BLOCK), "nearest")
    error = clip.frame(index).astype(np.float32) - predicted
    return vectors, error


def _vectors_to(clip, index: int, previous: np.ndarray) -> np.ndarray:
    """Only the block vectors from frame index - 1 to index (see _p_frame)."""
    if index <= 0 or _same(clip.frame(index - 1), clip.frame(index)):
        return previous
    return _vectors(clip, index - 1, index)


def _dc_only(error: np.ndarray) -> np.ndarray:
    """Each DC_BLOCK x DC_BLOCK block of an H x W x 3 residual replaced by its average."""
    height, width = error.shape[:2]
    out = np.empty_like(error)
    counts = _block_sums(np.ones((height, width), np.float32), DC_BLOCK)
    ys = (np.arange(height) // DC_BLOCK)[:, None]
    xs = (np.arange(width) // DC_BLOCK)[None, :]
    for c in range(error.shape[2]):
        means = _block_sums(error[..., c], DC_BLOCK) / counts
        out[..., c] = means[ys, xs]
    return out


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two frames are (almost) identical, i.e. a repeated frame."""
    return float(np.abs(a.astype(np.float32) - b).mean()) < 0.5


def _vectors(clip, first: int, second: int) -> np.ndarray:
    """Whole-pixel motion vectors per BLOCK x BLOCK block, from frame `first` to `second`.

    Block matching runs at half size (every second pixel, half the block and search)
    to keep it fast, and the vectors are doubled back to full size.
    """
    reference = color.bgr_to_y(clip.frame(first))[::2, ::2]
    current = color.bgr_to_y(clip.frame(second))[::2, ::2]
    vectors = motion.estimate(reference, current, BLOCK // 2, SEARCH // 2)
    return (_confident(vectors, reference, current, BLOCK // 2) * 2).astype(np.float32)


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
