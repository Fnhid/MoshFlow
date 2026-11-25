"""Pulse Mosh: B's motion pulses through A's last picture, then B lands block by block.

Idea taken from moshy's ppulse (duplicate a P-frame every n frames, for a rhythm)
and from how a codec recovers after a dropped I-frame (intra refresh).

Pulses. The picture starts as A's last clean frame. Every `period` frames a new
pulse starts: one motion field F from block matching between two of B's frames
(camera movement removed, so people move against each other instead of the whole
picture sliding away). During the pulse the picture is moved by F every frame,
P(x) <- P(x + gain * s * F(x)), where s falls from 1 to 0 across the period. A
pixel is smeared hard on the beat and drifts less before the next one.

Landing. There is no cross-fade. After the cut each 16 x 16 block gets a switch
time between 0 and 1. A block shows the smeared picture until its time comes, then
it shows B's real pixels at full strength, as an encoder's intra refresh would. The
times are random per block (ctx.seed) and earlier where B moves a lot, scaled by
`refresh`. On the last frame every block has switched, so the result is B itself.
Before the switch B's residual (B_now(x) - B_before(x + F(x))) is added to the
smeared picture, so B's edges and colours already show up inside the mess.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, warp
from moshflow.toolkit.pframe import block_mean, motion_field, p_frame

LOOK_AHEAD = 40      # B frames searched for pulses
NO_B_PART = 1 / 3    # with no B window, the share of the end in which B lands
CHANGE_FULL = 16.0   # residual (grey levels) at which a block lands earliest


@register
class PulseMosh(TransitionEffect):
    info = EffectInfo("pulse_mosh", "Pulse Mosh", "분배혀",
                      "B's motion pulses through A's last picture on a beat, then B lands block by block.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("period", "Beat (frames)", 6, 2, 30),
            FloatParam("gain", "Pulse strength", 1.5, 0.0, 4.0),
            FloatParam("refresh", "Landing speed", 1.0, 0.2, 2.0),
            IntParam("block", "Landing block", 16, 8, 48),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        period, gain, block = p["period"], p["gain"], p["block"]
        width, height = ctx.size
        b = ctx.incoming
        picture = frames.last_clean_a(ctx).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        pulses = _pulses(b, first_b, ctx.size)

        a_frames = len(ctx.a_window)
        land_from = a_frames if len(ctx.b_window) else round(ctx.length * (1 - NO_B_PART))
        rng = np.random.default_rng(ctx.seed)
        rows, cols = -(-height // block), -(-width // block)
        luck = rng.random((rows, cols)).astype(np.float32)
        busy = np.zeros((rows, cols), np.float32)   # most B has changed in a block so far

        pairs = frames.window_frames(ctx)
        for i, (clip, index) in enumerate(pairs):
            if progress.cancelled:
                return
            beat, phase = divmod(i, period)
            field = pulses[beat % len(pulses)]
            strength = gain * (1.0 - phase / period)
            picture = warp.warp(picture, strength * field, "nearest")

            target = b.frame(index).astype(np.float32) if clip is b else b.frame(first_b).astype(np.float32)
            if clip is b and index > 0:
                step, error = p_frame(b, index - 1, index, ctx.size)
                ramp = (i + 1 - a_frames) / max(ctx.length - a_frames, 1)
                picture = picture + ramp * error
                busy = np.maximum(busy, block_mean(np.abs(color.bgr_to_y(error)) / CHANGE_FULL, block))
            picture = np.clip(picture, 0.0, 255.0)

            out = picture
            if i >= land_from:
                v = (i + 1 - land_from) / (ctx.length - land_from)
                # earlier switch time where B moves, random elsewhere; refresh speeds both up
                when = (0.15 + 0.85 * luck * (1.0 - 0.6 * np.minimum(busy, 1.0))) / p["refresh"]
                landed = (when <= v) | (i == ctx.length - 1)
                mask = np.repeat(np.repeat(landed, block, axis=0), block, axis=1)[:height, :width]
                out = np.where(mask[..., None], target, picture)
                picture = np.where(mask[..., None], target, picture)
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _pulses(clip, start: int, size: tuple[int, int]) -> list[np.ndarray]:
    """Motion fields to pulse with: B's local motion in its first LOOK_AHEAD frames.

    The field's average vector (the camera) is subtracted. Pairs with the most local
    motion come first. A still clip gives one zero field.
    """
    found = []
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1)):
        field = motion_field(clip, first, first + 1, size)
        local = field - field.mean(axis=(0, 1))
        amount = float(((local[..., 0] ** 2 + local[..., 1] ** 2) ** 0.5).mean())
        if amount > 0.0:
            found.append((amount, local))
    if not found:
        width, height = size
        return [np.zeros((height, width, 2), np.float32)]
    found.sort(key=lambda item: -item[0])
    return [local for _, local in found[:8]]
