"""Keyframe Drop: what a decoder shows when the I-frame of clip B is deleted.

In a compressed video an I-frame stores a whole picture. The P-frames after it store
motion vectors ("this block comes from over there in the last picture") and a residual
(the correction to add afterwards). Without B's I-frame the decoder still holds A's last
picture, so B's vectors and residuals are applied to A's pixels.

For one pixel (x, y) of the picture R kept between output frames, in clip B:

1. Motion:   R(x, y) = R(x + dx, y + dy), where (dx, dy) is the vector of its block.
   A block whose vector hardly beats standing still keeps (0, 0).
2. Residual: R(x, y) += B_now(x, y) - B_before(x + dx, y + dy). If B was predicted as
   100 and is really 120, the pixel gets 20 brighter, whatever A's value is.
3. Takeover: every block adds up how much B changed there; once that total (scaled by
   `heal`) plus a slowly growing time share reaches the block's threshold, the whole
   block becomes B's real block, as an encoder sends a fresh (intra) block. So B wears
   A away where it moves first, still areas keep A longest, and nothing has to vanish
   at a fixed moment. A block is A's or B's, never a mix of both.

Frames of A inside the window are intact.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, refresh, warp

SKIP_GAIN = 1.0    # a vector must lower the mean luma error of its block by this many levels


@register
class KeyframeDrop(TransitionEffect):
    info = EffectInfo("keyframe_drop", "Keyframe Drop", "Nicolai Hansen",
                      "Clip B's motion drags the last picture of clip A until B's next keyframe.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("block", "Block size", 16, 4, 64),
            IntParam("search", "Search range", 8, 1, 32),
            FloatParam("heal", "Healing", 0.005, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        block, search, heal = ctx.params["block"], ctx.params["search"], ctx.params["heal"]
        picture = frames.last_clean_a(ctx).astype(np.float32)
        previous = None    # the B frame before the current one
        fine = refresh.block_size(ctx.size)              # B arrives in 8 x 8 blocks at 720p
        takeover = refresh.Takeover(ctx, fine, refresh.CHANGE_FULL * 0.005 / max(heal, 1e-5))
        b_frames = len(ctx.b_window)

        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            current = clip.frame(index).astype(np.float32)
            if i < len(ctx.a_window):
                picture = current
            else:
                # B's frame 0 is the dropped I-frame: nothing is decoded and A's picture stays.
                if previous is not None:
                    vectors = _useful_vectors(color.bgr_to_y(previous), color.bgr_to_y(current), block, search)
                    field = motion.to_field(vectors, ctx.size, block)
                    residual = current - warp.warp(previous, field)
                    picture = np.clip(warp.warp(picture, field) + residual, 0, 255)
                k = i - len(ctx.a_window)
                taken = takeover.step(previous, current, k / max(b_frames - 1, 1))
                picture = np.where(taken, current, picture)
                previous = current

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _useful_vectors(reference: np.ndarray, current: np.ndarray, block: int, search: int) -> np.ndarray:
    """Block vectors, with (0, 0) where moving is not clearly better than standing still.

    Without this, noise in flat areas gives random vectors that stir the picture.
    """
    height, width = current.shape
    vectors = motion.estimate(reference, current, block, search)
    moved = warp.warp(reference, motion.to_field(vectors, (width, height), block))
    gain = np.abs(current - reference) - np.abs(current - moved) - SKIP_GAIN
    rows, cols = vectors.shape[:2]
    padded = np.zeros((rows * block, cols * block), np.float32)
    padded[:height, :width] = gain
    vectors[padded.reshape(rows, block, cols, block).sum(axis=(1, 3)) < 0] = 0
    return vectors
