"""Reference effect: no transition at all."""

from __future__ import annotations

from moshflow.api import EffectInfo, TransitionEffect, register


@register
class HardCut(TransitionEffect):
    info = EffectInfo("hard_cut", "Hard Cut", "Yuhyeon Cha",
                      "No transition: clip A plays to its end, then clip B starts.")

    @classmethod
    def param_specs(cls):
        return []

    def render(self, ctx, progress):
        frames = [(ctx.outgoing, i) for i in ctx.a_window] + [(ctx.incoming, i) for i in ctx.b_window]
        for source, i in frames:
            if progress.cancelled:
                return
            yield source.frame(i)
