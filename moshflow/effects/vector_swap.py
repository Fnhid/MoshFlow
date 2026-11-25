"""Vector Swap: each clip is decoded with the other clip's motion vectors.

A codec stores a P-frame as motion vectors F plus a residual E (the colour the
motion cannot predict) and decodes it as P_new(x) = P(x + F(x)) + E(x). Vector
Swap keeps each clip's residuals but trades their motion vectors: clip A moves
the way B moves, then clip B arrives moving the way A moved, and both trade back.

SKELETON: render() is a plain cut until the steps are written.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, frames


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
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(clip.frame(index).astype(np.float32))
