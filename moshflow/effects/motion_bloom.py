"""Motion Bloom: clip B's P-frames are played on clip A's last picture.

Emulates the classic datamosh transition made with real video files: clip B's
I-frame is dropped, so B's P-frames are decoded on top of A's last picture, and
one P-frame is duplicated many times (moshy's pdupe), so its motion is applied
again and again and the picture swells ("blooms"). Both clips keep their real
speed: nothing is slowed down or stretched.

A P-frame here is what a codec stores: a motion field F (one vector per pixel,
measured by block matching: F(x) says where the content at x came from) and a
residual E, the colour correction for what the motion could not predict:
E(x) = B_next(x) - B_prev(x + F(x)). Decoding it onto a picture P gives

    P_new(x) = P(x + F(x)) + E(x)

so each pixel is copied from where the motion says and then gets the same
colour difference that B had at that spot.

The transition, frame by frame:
- Before the cut, A plays clean. Nothing of B is used yet.
- Bloom (the first half after the cut): B's I-frame is gone, so the decoder
  keeps A's last picture. B's strongest opening P-frame (people's own movement in
  B's first second, without the camera's, see _opening_motions) is decoded on it
  again and again, `gain` times its motion each frame: A's picture swells and
  smears in one steady direction.
- Decode (the rest): B's own P-frames, one per frame at B's real speed, motion
  plus `residual` times the residual, so B's moving parts form inside A's picture.
  Where a P-frame fails (B moves faster than the search reaches, its residual is
  large) the whole block becomes B's real block, as an encoder sends a fresh
  (intra) block there.
- Takeover: every block adds up how much B changed there; once that total
  (times `emerge`) plus a slowly growing time share reaches the block's
  threshold, the block becomes B's real block. B wears A's picture away where it
  moves first and still areas last, with no fixed moment at which A has to be
  gone; the last frame is B.
- A block shows A's decoded picture or B's frame, never a mix of both.
- With no B window (alignment "before"), the last third of A's window is moshed
  with B's opening motion and refreshed into B's first frame.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times becomes a random streak, so vectors that do not explain the motion
clearly better than (0, 0) are dropped, and block vectors are interpolated
between block centres so no hard block edges appear. Pixels are read with
bilinear sampling, so fractional vectors do not leave a zig-zag.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, motion, refresh, warp

BLOCK = 16           # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7     # keep a block's vector only if it cuts the block's error to below 70 %
LOOK_AHEAD = 30      # B frames searched for its opening motions
LOOK_STEP = 3        # ... every third frame pair, which is enough to find the strongest
OPENING_COUNT = 1    # the bloom repeats B's single strongest opening motion
BLOOM_SHARE = 0.5    # share of the moshed frames that repeat that P-frame
INTRA_LEVEL = 40.0   # mean |residual| (grey levels) above which a block is sent fresh
NO_B_PART = 1 / 3    # with no B window, the share of A's window that is moshed


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom", "Viktor Tost",
                      "Clip B's P-frames play on clip A's last picture, one repeated, until B's blocks arrive.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("residual", "Residual", 1.0, 0.0, 2.0),
            FloatParam("emerge", "Emerge", 1.0, 0.0, 3.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        a, b = ctx.outgoing, ctx.incoming
        a_frames, b_frames = len(ctx.a_window), len(ctx.b_window)
        first_b = min(ctx.b_window.start, len(b) - 1)
        start = a_frames if b_frames else a_frames - max(1, round(a_frames * NO_B_PART))
        moshed = ctx.length - start                      # frames after the mosh starts
        bloom_end = round(BLOOM_SHARE * moshed)
        fine = refresh.block_size(ctx.size)              # B arrives in 8 x 8 blocks at 720p
        takeover = refresh.Takeover(ctx, fine, refresh.CHANGE_FULL / max(p["emerge"], 1e-3))
        taken = np.zeros((ctx.size[1], ctx.size[0], 1), bool)
        bloom = None
        picture = None

        for i in range(ctx.length):
            if progress.cancelled:
                return
            if i < start:                                 # A plays clean up to the mosh
                progress.report(i + 1, ctx.length)
                yield a.frame(ctx.a_window[i]).copy()
                continue
            if picture is None:                           # the decoder keeps A's last picture
                last_a = ctx.a_window[start - 1] if start > 0 else max(ctx.a_window.start - 1, 0)
                picture = a.frame(last_a).astype(np.float32)
                bloom = p["gain"] * _opening_motions(b, first_b, ctx.size)[0]
            m = i - start
            index = ctx.b_window[i - a_frames] if i >= a_frames else first_b
            target = b.frame(index).astype(np.float32)

            if m < bloom_end:
                # Duplicated P-frame: the same motion again, nothing new comes in.
                picture = warp.warp(picture, bloom, "bilinear")
            elif i >= a_frames and index > 0 and not _same(b.frame(index - 1), b.frame(index)):
                # B's own P-frame at B's real speed: motion, then residual.
                field, error = _p_frame(b, index - 1, index, ctx.size)
                picture = warp.warp(picture, p["gain"] * field, "bilinear") + p["residual"] * error
                fresh = refresh.block_mean(np.abs(color.bgr_to_y(error)), fine) > INTRA_LEVEL
                taken |= refresh.to_pixels(fresh, fine, ctx.size)
            picture = np.clip(picture, 0.0, 255.0)

            # Takeover: B overwrites A where it changes, still areas last.
            before = b.frame(index - 1) if i >= a_frames and index > 0 else None
            taken |= takeover.step(before, target, m / max(moshed - 1, 1))
            picture = np.where(taken, target, picture)
            out = target if i == ctx.length - 1 else picture
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _p_frame(clip, first: int, second: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The P-frame that turns frame `first` of a clip into frame `second`: (F, E).

    E(x) = second(x) - first(x + F(x)), the colour the motion alone cannot explain.
    """
    field = _field(clip, first, second, size)
    predicted = warp.warp(clip.frame(first), field, "bilinear")
    return field, clip.frame(second).astype(np.float32) - predicted


def _opening_motions(clip, start: int, size: tuple[int, int]) -> list[np.ndarray]:
    """The motions to repeat before the cut: people's own movement in B's first second.

    For each non-repeated frame pair in B's first LOOK_AHEAD frames, the field's
    average vector (the camera moving everything the same way) is subtracted, so
    what remains is how things move relative to each other. Repeating a camera pan
    would only slide A's picture off the screen; repeating local motion melts it.
    The OPENING_COUNT pairs with the most local motion are kept, in time order.
    """
    found = []
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1), LOOK_STEP):
        if _same(clip.frame(first), clip.frame(first + 1)):
            continue
        field = _field(clip, first, first + 1, size)
        local = field - field.mean(axis=(0, 1))
        amount = float(((local[..., 0] ** 2 + local[..., 1] ** 2) ** 0.5).mean())
        found.append((amount, first, local))
    if not found:
        width, height = size
        return [np.zeros((height, width, 2), np.float32)]
    strongest = sorted(found, key=lambda item: -item[0])[:OPENING_COUNT]
    return [local for _, _, local in sorted(strongest, key=lambda item: item[1])]


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
