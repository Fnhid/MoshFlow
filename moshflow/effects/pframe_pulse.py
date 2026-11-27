"""P-frame Pulse: the cut is moshed in pulses of duplicated P-frames.

Built like the references for datamoshing without a codec:
- the NumPy/OpenCV implementation of "Datamoshing with Optical Flow" (Careaga,
  Reddy, Aksoy, SIGGRAPH Asia 2023 posters), SuperSonicHub1/datamoshing-with-
  optical-flow: each frame the previous OUTPUT is moved by the motion between the
  two latest source frames, sampled at the nearest pixel, and a 0/1 mask moved
  the same way lets the live frame show only where the warp reached outside the
  picture;
- real-time datamosh shaders (Buck, Unreal / Three.js examples): the same
  feedback loop, kept stable by a small trickle of the fresh frame;
- moshy's ppulse: every n frames the current P-frame is duplicated, so its
  motion is applied to the picture again and again (the bloom).

Per output frame (both clips keep their real speed):
- Motion F is measured between the clip's previous and current frame by block
  matching (F(x) says where the content at x came from).
- Decode: out(x) <- out(x + F(x)), mask(x) <- mask(x + F(x)) (0 where x + F(x) is
  outside the picture), then out <- mask out + (1 - mask) frame. Nothing is
  re-blurred: pixels are copied, so the damage stays sharp and builds up.
- Pulse: every `interval` frames the current F is held for `hold` frames and
  applied again on each of them (times `gain`), with nothing new coming in: a
  duplicated P-frame. The clip keeps running underneath, so after the pulse the
  picture snaps into the clip's current motion with the bloom baked in.
- Trickle: out <- out + t (frame - out), with t = `heal` at the start (A mostly
  intact) falling to 0 at the cut (pure mosh), then rising to 1 at the end, so
  B takes over through the moving, damaged picture. Pulses get no trickle, so
  none may run into the last LAND part, where the picture settles into B.
- The cut: B's I-frame is dropped; nothing is reset, and B's motion drives the
  damaged picture of A from then on. The last frame is B's real frame.

Block matching is noisy in flat or blurry areas, and a wrong vector applied many
times becomes a random streak, so vectors that do not explain the motion clearly
better than (0, 0) are dropped, and block vectors are interpolated between block
centres.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

BLOCK = 16           # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7     # keep a block's vector only if it cuts the block's error to below 70 %
LAND = 0.2           # no pulse may run into this last share of the transition


@register
class PFramePulse(TransitionEffect):
    info = EffectInfo("pframe_pulse", "P-frame Pulse", "Viktor Tost",
                      "The cut moshes in pulses of duplicated P-frames that bloom and pull B in.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("interval", "Interval (frames)", 15, 3, 60),
            IntParam("hold", "Hold (frames)", 8, 1, 59),
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("heal", "Heal", 0.3, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        width, height = ctx.size
        hold = min(p["hold"], p["interval"] - 1)
        cut = len(ctx.a_window) / ctx.length            # where the cut is, 0..1
        out = frames.last_clean_a(ctx).astype(np.float32)
        mask = np.ones((height, width), np.float32)
        field = np.zeros((height, width, 2), np.float32)
        held_left = 0
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            u = (i + 1) / ctx.length
            frame = clip.frame(index).astype(np.float32)

            if (held_left == 0 and i % p["interval"] == p["interval"] - hold
                    and u + hold / ctx.length <= 1.0 - LAND):
                held_left = hold                         # a pulse starts on this frame
            if held_left > 0:
                # Duplicated P-frame: the held motion again, nothing new comes in.
                held_left -= 1
                out, mask = _decode(out, mask, p["gain"] * field)
            else:
                # The clip's own P-frame: its motion moves the picture, and the
                # live frame shows where the motion reached outside it.
                field = _field_at(clip, index, field, ctx.size)
                out, mask = _decode(out, mask, field)
                out = mask[..., None] * out + (1.0 - mask[..., None]) * frame
                out = out + _trickle(u, cut, p["heal"]) * (frame - out)

            if i == ctx.length - 1:
                out = frame.copy()
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _decode(out: np.ndarray, mask: np.ndarray, field: np.ndarray):
    """Moves the picture and its mask by the field, copying nearest pixels.

    out(x) <- out(x + F(x)); mask(x) <- mask(x + F(x)), and 0 where x + F(x) falls
    outside the picture, so the live frame can show through there.
    """
    height, width = mask.shape
    xs = np.arange(width, dtype=np.float32)[None, :] + field[..., 0]
    ys = np.arange(height, dtype=np.float32)[:, None] + field[..., 1]
    inside = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
    out = warp.warp(out, field, "nearest")
    mask = warp.warp(mask, field, "nearest") * inside
    return out, mask


def _trickle(u: float, cut: float, heal: float) -> float:
    """How much of the live frame trickles in at progress u.

    `heal` at the start, falling to 0 at the cut, then rising to 1 at the end.
    """
    if u <= cut:
        return heal * (1.0 - _smoothstep(u / cut)) if cut > 0 else 0.0
    v = _smoothstep((u - cut) / (1.0 - cut))
    return v * v


def _smoothstep(t: float) -> float:
    """0 at t <= 0, 1 at t >= 1, with a gentle start and end: t^2 (3 - 2 t)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _field_at(clip, index: int, previous: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The motion from frame index - 1 to index. For the clip's first frame or a
    repeated (duplicate) frame there is none, and the previous motion carries on."""
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
