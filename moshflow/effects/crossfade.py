"""Reference effect: linear blend from clip A to clip B."""

from __future__ import annotations

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register


@register
class Crossfade(TransitionEffect):
    info = EffectInfo("crossfade", "Crossfade", "Yuhyeon Cha",
                      "Linear blend from clip A to clip B.")

    @classmethod
    def param_specs(cls):
        return [FloatParam("ease", "Ease", 1.0, 0.2, 5.0)]

    def render(self, ctx, progress):
        # While A's frames run out, B is held on its first frame, and vice versa.
        last_a, first_b = ctx.a_window.stop - 1, ctx.b_window.start
        src_a = list(ctx.a_window) + [last_a] * len(ctx.b_window)
        src_b = [first_b] * len(ctx.a_window) + list(ctx.b_window)
        for i in range(ctx.length):
            if progress.cancelled:
                return
            t = ((i + 1) / (ctx.length + 1)) ** ctx.params["ease"]
            a = ctx.outgoing.frame(src_a[i]).astype(np.float32)
            b = ctx.incoming.frame(src_b[i]).astype(np.float32)
            progress.report(i + 1, ctx.length)
            yield ((1.0 - t) * a + t * b + 0.5).astype(np.uint8)
