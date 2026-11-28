"""Pulse Mosh: B's P-frames pulse on a beat over A's colours, with no alpha cross-fade.

Emulates moshy's `ppulse` (repeating a P-frame every `period` frames for a rhythmic
datamosh) combined with real MPEG-4 / Xvid P-frame decoding from `toolkit.codec`:

1. Motion pulses: every `period` frames a new motion field F is captured from clip B
   (via hierarchical block matching up to +-24 px, with 5-neighbour median filtering
   and camera pan removed before the cut). Repeating F with nearest-neighbour warping
   drags and stretches A's crisp macroblocks without blurring or washing out A's colours.
2. Quantized residual: after the cut, B's prediction error E = B_curr - B_prev(x + F)
   is dead-zone quantized (H.263 style) with heavy chroma attenuation, so smooth areas
   keep A's exact colours while B's moving edges etch into A's palette.
3. No fade: blocks only switch to B's real pixels when B's own prediction fails
   (Xvid-style clustered INTRA blocks), or on exit:
   - "snap": keeps A's moshed colours until the transition ends (next I-frame snap).
   - "intra": B wears A away block by block where B changes, with no timer; the last
     frame is B's next keyframe.
"""

from __future__ import annotations

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import codec, color, frames, motion, refresh, warp

SEARCH = 24
LOOK_AHEAD = 36
QUANT = 14.0
CHROMA_SCALE = 0.15
INTRA_THRESH = 22.0


@register
class PulseMosh(TransitionEffect):
    info = EffectInfo(
        "pulse_mosh",
        "Pulse Mosh",
        "Yuhyeon Cha",
        "Repeats B's P-frame motion on a beat over A's colours, ending by I-frame snap or INTRA clusters.",
    )

    @classmethod
    def param_specs(cls):
        return [
            IntParam("period", "Beat (frames)", 6, 2, 30),
            FloatParam("gain", "Pulse strength", 1.4, 0.0, 4.0),
            FloatParam("residual", "Edge etch", 0.9, 0.0, 2.0),
            IntParam("block", "Block size", 16, 8, 32),
            ChoiceParam("exit", "Exit style", "intra", ("intra", "snap")),
        ]

    def render(self, ctx, progress):
        if progress.cancelled:
            return
        p = ctx.params
        period = p["period"]
        gain = p["gain"]
        residual_scale = p["residual"]
        block = p["block"]
        exit_style = p["exit"]

        width, height = ctx.size
        b = ctx.incoming
        picture = frames.last_clean_a(ctx).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        a_frames = len(ctx.a_window)
        b_len = len(ctx.b_window)
        pulses = _opening_pulses(b, first_b, ctx.size, block) if a_frames > 0 else []
        rows, cols = -(-height // block), -(-width // block)
        intra_mask = np.zeros((rows, cols), bool)
        takeover = refresh.Takeover(ctx, refresh.block_size(ctx.size))

        held_vectors = np.zeros((rows, cols, 2), np.int32)
        held_residual = np.zeros((height, width, 3), np.float32)

        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return

            beat, phase = divmod(i, period)
            decay = 1.0 - 0.5 * (phase / period)

            if i < a_frames:
                # Before the cut: pulse A's pixels along B's strongest local motions.
                field = gain * decay * pulses[beat % len(pulses)]
                picture = warp.warp(picture, field, sampling="nearest")
            else:
                b_pos = i - a_frames
                target = b.frame(index).astype(np.float32)
                if index > 0 and not codec.frames_equal(b.frame(index - 1), b.frame(index)):
                    pf = codec.analyze_pframe(
                        b.frame(index - 1),
                        b.frame(index),
                        block=block,
                        search=SEARCH,
                        quant=QUANT,
                        chroma_scale=CHROMA_SCALE,
                        intra_thresh=INTRA_THRESH,
                    )
                    if phase == 0 or not held_vectors.any():
                        held_vectors = pf.vectors
                        held_residual = pf.residual
                    # Mix rhythmic held pulse with live B motion so B's subject drives A.
                    live_field = motion.to_field(pf.vectors, ctx.size, block)
                    pulse_field = motion.to_field(held_vectors, ctx.size, block)
                    field = 0.65 * gain * decay * pulse_field + 0.55 * live_field
                    picture = warp.warp(picture, field, sampling="nearest")
                    res = 0.5 * held_residual + 0.5 * pf.residual
                    picture = np.clip(picture + residual_scale * res, 0.0, 255.0)
                else:
                    pulse_field = motion.to_field(held_vectors, ctx.size, block)
                    picture = warp.warp(picture, gain * decay * pulse_field, sampling="nearest")

                # "intra" exit: B wears A away where B changes, nothing on a timer, and
                # the last frame is B's next keyframe.
                if exit_style == "intra" and b_len > 1:
                    taken = takeover.step(b.frame(index - 1) if index > 0 else None, target, (b_pos + 1) / b_len)
                    picture = np.where(taken, target, picture)
                    if b_pos == b_len - 1:
                        intra_mask = np.ones((rows, cols), bool)

                if intra_mask.any():
                    mask_px = codec.expand_blocks(intra_mask, ctx.size, block)
                    picture = np.where(mask_px[..., None], target, picture)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _opening_pulses(
    clip, start: int, size: tuple[int, int], block: int
) -> list[np.ndarray]:
    """Local motion fields from B's opening frames, camera pan subtracted."""
    found = []
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1), 3):
        if codec.frames_equal(clip.frame(first), clip.frame(first + 1)):
            continue
        ref_y = color.bgr_to_y(clip.frame(first))
        cur_y = color.bgr_to_y(clip.frame(first + 1))
        vectors = codec.estimate_hierarchical(ref_y, cur_y, block=block, search=SEARCH)
        field = motion.to_field(vectors, size, block)
        local = field - field.mean(axis=(0, 1))
        amount = float(np.abs(local).mean())
        if amount > 0.15:
            found.append((amount, first, local))
    if not found:
        width, height = size
        return [np.zeros((height, width, 2), np.float32)]
    strongest = sorted(found, key=lambda item: -item[0])[:6]
    return [local for _, _, local in sorted(strongest, key=lambda item: item[1])]
