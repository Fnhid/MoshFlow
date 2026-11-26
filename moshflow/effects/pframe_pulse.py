"""P-frame Pulse: the transition moshes in rhythmic bursts of repeated P-frames.

Like moshy's ppulse: every few frames the current P-frame is duplicated a few
times, so the picture blooms in pulses, then snaps back into the clip's motion.

SKELETON: render() is a plain cut until the steps are written.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames


@register
class PFramePulse(TransitionEffect):
    info = EffectInfo("pframe_pulse", "P-frame Pulse (draft)", "Viktor Tost",
                      "The cut moshes in rhythmic bursts: P-frames repeat in pulses that pull B in.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("interval", "Interval (frames)", 10, 3, 60),
            IntParam("hold", "Hold (frames)", 5, 1, 30),
            FloatParam("gain", "Gain", 1.5, 0.0, 4.0),
            FloatParam("heal", "Heal", 0.25, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(clip.frame(index).astype(np.float32))
