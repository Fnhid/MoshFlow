"""Keyframe Drop: what a decoder shows when the I-frame of clip B is deleted.

In a compressed video an I-frame stores a whole picture. The P-frames after it store
motion vectors ("this block comes from over there in the last picture"), a dead-zone
quantized residual (only the colour differences too large for the quantizer to drop),
and INTRA blocks where prediction fails. Without B's I-frame the decoder still holds
A's last picture, so B's P-frames are decoded onto A's pixels.

For one pixel (x, y) of the picture R kept between output frames, in clip B:

1. Motion:   R(x, y) = R(x + dx, y + dy), where (dx, dy) comes from coarse-to-fine
   block matching with 5-neighbour median filtering. Blocks whose vector hardly beats
   standing still become SKIP blocks with (0, 0).
2. Residual: R(x, y) += Q(B_now(x, y) - B_before(x + dx, y + dy)), quantized with an
   H.263 dead zone and chroma attenuation so smooth areas keep A's colours while B's
   moving brightness contours etch in. Blocks where prediction fails switch to INTRA.
3. Healing:  R += heal * (B - R) in moving INTER blocks. Default heal = 0.0 keeps A's
   exact colours until B's next keyframe; raising heal blends moving blocks towards B.

Where B is still, A's picture stays. After the window B's own frames play again, as if
its next I-frame had arrived. Frames of A inside the window are intact.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import codec, color, frames

QUANT = 12.0
CHROMA_SCALE = 0.15
INTRA_THRESH = 24.0


@register
class KeyframeDrop(TransitionEffect):
    info = EffectInfo(
        "keyframe_drop",
        "Keyframe Drop",
        "Nicolai Hansen",
        "Clip B's motion drags the last picture of clip A until B's next keyframe.",
    )

    @classmethod
    def param_specs(cls):
        return [
            IntParam("block", "Block size", 16, 4, 64),
            IntParam("search", "Search range", 16, 1, 32),
            FloatParam("heal", "Healing", 0.0, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        if progress.cancelled:
            return
        block, search, heal = ctx.params["block"], ctx.params["search"], ctx.params["heal"]
        picture = frames.last_clean_a(ctx).astype(np.float32)
        previous = None    # the B frame before the current one

        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            current = clip.frame(index)
            if i < len(ctx.a_window):
                picture = current.astype(np.float32)
            else:
                # B's frame 0 is the dropped I-frame: nothing is decoded and A's picture stays.
                if previous is not None and not codec.frames_equal(previous, current):
                    pf = codec.analyze_pframe(
                        previous,
                        current,
                        block=block,
                        search=search,
                        quant=QUANT,
                        chroma_scale=CHROMA_SCALE,
                        intra_thresh=INTRA_THRESH,
                    )
                    picture = codec.decode_pframe(picture, pf, current, sampling="nearest")
                    if heal > 0.0:
                        moving_blocks = (pf.modes == codec.MODE_INTER) & (
                            (pf.vectors[..., 0] != 0) | (pf.vectors[..., 1] != 0)
                        )
                        if moving_blocks.any():
                            moving = codec.expand_blocks(moving_blocks, ctx.size, block)
                            cur_f = current.astype(np.float32)
                            picture[moving] += heal * (cur_f[moving] - picture[moving])
                previous = current

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)
