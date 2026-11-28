"""Template for a new effect.

1. Copy this file to moshflow/effects/<your_effect>.py (no leading underscore;
   files starting with '_' are not loaded).
2. Rename the class, fill in `info` and `param_specs`, and write `render`.
3. Run `python -m pytest tests/test_effect_contract.py` and
   `python -m moshflow.cli try --effect <id> a.mp4 b.mp4 -o test.mp4`.

Rules (checked by the contract test):
- yield exactly ctx.length frames, each H x W x 3 uint8 (W, H = ctx.size);
- never modify input frames (they are read-only; use .astype(np.float32) for maths);
- use only ctx.seed for randomness, e.g. rng = np.random.default_rng(ctx.seed);
- check progress.cancelled in loops and return early when it is set;
- no file I/O, no UI, no OpenCV, no library image-processing functions.
"""

from __future__ import annotations

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color


@register
class MyEffect(TransitionEffect):
    info = EffectInfo("my_effect", "My Effect", "[Your name]",
                      "[One sentence: what the transition looks like]")

    @classmethod
    def param_specs(cls):
        return [FloatParam("strength", "Strength", 0.5, 0.0, 1.0)]

    def render(self, ctx, progress):
        # Frames of A to replace: ctx.a_window; frames of B to replace: ctx.b_window.
        # Any other frame can be read too, e.g. the last frame of A before the window.
        reference = ctx.outgoing.frame(max(ctx.a_window.start - 1, 0)).astype(np.float32)
        for i in range(ctx.length):
            if progress.cancelled:
                return
            out = reference  # TODO: compute output frame i
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)
