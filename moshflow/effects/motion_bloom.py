"""Motion Bloom: clip B's P-frames are played on clip A's last picture.

Emulates the classic datamosh transition: clip B's I-frame is dropped, so B's
P-frames are decoded on top of A's last picture, and opening P-frames are repeated
(as moshy's pdupe/ppulse do on real video files) so the motion blooms. Both clips
keep their real speed: nothing is slowed down or stretched.

Uses `toolkit.codec` for hierarchical block matching, dead-zone quantized residuals,
and clustered INTRA macroblocks instead of alpha cross-fades, so clip A's colours
stay vivid and sharp until B breaks through as real macroblock clusters:

- A's part of the window (before the cut): A's own P-frames keep A moving while B's
  strongest opening local motions (camera pan removed) repeat in 4-frame bursts,
  making A's pixels bloom without showing B yet.
- B's part of the window (after the cut): B's own P-frames decode onto the picture
  with nearest-neighbour sampling and dead-zone quantized residuals (mostly luma).
  Blocks where B's prediction fails switch to INTRA immediately, and `emerge` grows
  clustered INTRA patches organically from active regions (zero opacity fade).
- With no B window (alignment "before"), B emerges over the last third towards its
  first frame.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import codec, color, frames, motion, warp

BLOCK = 16
SEARCH = 24
LOOK_AHEAD = 30
OPENING_COUNT = 6
OPENING_HOLD = 4
QUANT = 14.0
CHROMA_SCALE = 0.15
INTRA_THRESH = 22.0
NO_B_PART = 1 / 3


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo(
        "motion_bloom",
        "Motion Bloom",
        "Viktor Tost",
        "Clip B's P-frames play on clip A's last picture, one repeated, until B emerges.",
    )

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("residual", "Residual", 1.0, 0.0, 2.0),
            FloatParam("emerge", "Emerge", 1.0, 0.0, 3.0),
        ]

    def render(self, ctx, progress):
        if progress.cancelled:
            return
        p = ctx.params
        gain = p["gain"]
        residual_scale = p["residual"]
        emerge = p["emerge"]

        a, b = ctx.outgoing, ctx.incoming
        width, height = ctx.size
        picture = frames.last_clean_a(ctx).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        a_frames = len(ctx.a_window)
        openings = _opening_motions(b, first_b, ctx.size) if a_frames > 0 else []

        emerge_from = a_frames if len(ctx.b_window) else round(ctx.length * (1 - NO_B_PART))
        rows, cols = -(-height // BLOCK), -(-width // BLOCK)
        rng = np.random.default_rng(ctx.seed)
        jitter = rng.random((rows, cols)).astype(np.float32) * 0.25

        intra_mask = np.zeros((rows, cols), bool)
        activity = np.zeros((rows, cols), np.float32)
        held_field = openings[0] if openings else np.zeros((height, width, 2), np.float32)

        for i in range(ctx.length):
            if progress.cancelled:
                return

            if i < a_frames:
                # Before the cut: repeat B's strongest opening local motions on A
                # while A's own P-frame motion keeps A alive up to the cut.
                field = openings[(i // OPENING_HOLD) % len(openings)]
                index = first_b
                picture = warp.warp(picture, gain * field, sampling="nearest")

                a_index = ctx.a_window[i]
                if a_index > 0 and not codec.frames_equal(a.frame(a_index - 1), a.frame(a_index)):
                    pf_a = codec.analyze_pframe(
                        a.frame(a_index - 1),
                        a.frame(a_index),
                        block=BLOCK,
                        search=SEARCH,
                        quant=QUANT,
                        chroma_scale=0.35,
                        intra_thresh=255.0,
                    )
                    picture = codec.decode_pframe(
                        picture, pf_a, a.frame(a_index), sampling="nearest", apply_intra=False
                    )
            else:
                index = ctx.b_window[i - a_frames]
                if index > 0 and not codec.frames_equal(b.frame(index - 1), b.frame(index)):
                    pf = codec.analyze_pframe(
                        b.frame(index - 1),
                        b.frame(index),
                        block=BLOCK,
                        search=SEARCH,
                        quant=QUANT,
                        chroma_scale=CHROMA_SCALE,
                        intra_thresh=INTRA_THRESH,
                    )
                    held_field = motion.smooth_field(pf.vectors, ctx.size, BLOCK, 0.35)
                    ramp = (i + 1 - a_frames) / max(ctx.length - a_frames, 1)
                    picture = warp.warp(picture, gain * held_field, sampling="nearest")
                    picture = np.clip(picture + residual_scale * ramp * pf.residual, 0.0, 255.0)
                    intra_mask = intra_mask | (pf.modes == codec.MODE_INTRA)
                    activity = np.maximum(activity, pf.mad / 24.0)
                else:
                    picture = warp.warp(picture, gain * held_field, sampling="nearest")

            target = b.frame(index).astype(np.float32)

            # Emerge: grow clustered INTRA macroblocks organically from active regions
            # instead of alpha-blending A and B into grey mud.
            if i >= emerge_from and emerge > 0.0:
                span = max(ctx.length - emerge_from, 1)
                v = (i + 1 - emerge_from) / span
                start = max(0.0, 0.55 - 0.25 * emerge)
                if v > start:
                    u = (v - start) / max(1.0 - start, 1e-6)
                    share = min(1.0, (u ** (1.8 / max(emerge, 0.25))))
                    scores = np.minimum(activity, 1.0) + jitter
                    intra_mask = codec.cluster_grow(intra_mask, scores, share)
                if i == ctx.length - 1:
                    intra_mask = np.ones((rows, cols), bool)

            if intra_mask.any():
                mask_px = codec.expand_blocks(intra_mask, ctx.size, BLOCK)
                picture = np.where(mask_px[..., None], target, picture)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _opening_motions(clip, start: int, size: tuple[int, int]) -> list[np.ndarray]:
    """Local motions from B's opening frames with camera pan subtracted."""
    found = []
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1), 2):
        if codec.frames_equal(clip.frame(first), clip.frame(first + 1)):
            continue
        ref_y = color.bgr_to_y(clip.frame(first))
        cur_y = color.bgr_to_y(clip.frame(first + 1))
        vectors = codec.estimate_hierarchical(ref_y, cur_y, block=BLOCK, search=SEARCH)
        field = motion.smooth_field(vectors, size, BLOCK, 0.5)
        local = field - field.mean(axis=(0, 1))
        amount = float(np.abs(local).mean())
        if amount > 0.15:
            found.append((amount, first, local))
    if not found:
        width, height = size
        return [np.zeros((height, width, 2), np.float32)]
    strongest = sorted(found, key=lambda item: -item[0])[:OPENING_COUNT]
    return [local for _, _, local in sorted(strongest, key=lambda item: item[1])]
