"""Bake Smear: A's sharp pixels are sucked into B's motion focus while B breaks through.

Combines three pixel-domain datamosh mechanisms with zero alpha cross-fading:

1. Coordinate-map advection (transflow style): instead of resampling pixels every
   frame (which blurs A into mud), we advect an (H, W, 2) coordinate map by B's
   wide-range hierarchical motion plus a radial pull towards B's motion centre c.
   Each output frame samples A's last clean frame once at the advected coordinates,
   so A's exact colours and sharpness survive even under heavy stretching.
2. Dead-zone luma etching: B's prediction residual is dead-zone quantized with almost
   no chroma (chroma_scale = 0.08) and accumulated in a separate buffer advected with
   the same flow. B's moving subject etches its brightness contours straight into A's
   colour palette.
3. Hole & INTRA emergence (Careaga/Aksoy + Xvid): wherever suction/motion uncovers
   out-of-bounds holes or B's prediction fails (INTRA macroblocks), B's real pixels
   break through as crisp patches—never a geometric wipe or opacity fade.
"""

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import codec, color, frames, motion, warp

BLOCK = 16
SEARCH = 24
FOCUS_LOOK = 20
PULL_STEP = 0.045
QUANT = 14.0
CHROMA_SCALE = 0.08
INTRA_THRESH = 20.0
SAME = 0.5


@register
class BakeSmear(TransitionEffect):
    info = EffectInfo(
        "bake_smear",
        "Bake Smear",
        "Yuhyeon Cha",
        "Sucks A's crisp pixels into B's motion focus while B's edges and INTRA blocks break through.",
    )

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Motion strength", 1.3, 0.0, 4.0),
            FloatParam("pull", "Suction", 1.0, 0.0, 3.0),
            FloatParam("residual", "Edge etch", 0.9, 0.0, 2.0),
            ChoiceParam("exit", "Exit style", "intra", ("intra", "snap")),
        ]

    def render(self, ctx, progress):
        if progress.cancelled:
            return
        p = ctx.params
        gain = p["gain"]
        pull = p["pull"]
        residual_scale = p["residual"]
        exit_style = p["exit"]

        b = ctx.incoming
        width, height = ctx.size
        source_a = frames.last_clean_a(ctx).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        a_frames = len(ctx.a_window)

        cx, cy = _focus(b, first_b, ctx.size)
        xs = np.arange(width, dtype=np.float32)[None, :]
        ys = np.arange(height, dtype=np.float32)[:, None]
        radial = np.empty((height, width, 2), np.float32)
        radial[..., 0] = xs - cx
        radial[..., 1] = ys - cy
        far = float(np.sqrt(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2)) + 1e-6

        rows, cols = -(-height // BLOCK), -(-width // BLOCK)
        bx = (np.arange(cols, dtype=np.float32) + 0.5) * BLOCK
        by = (np.arange(rows, dtype=np.float32) + 0.5) * BLOCK
        # Blocks closer to B's motion focus get higher priority when INTRA clusters grow.
        near_focus = 1.0 - np.sqrt((bx[None, :] - cx) ** 2 + (by[:, None] - cy) ** 2) / far

        rng = np.random.default_rng(ctx.seed)
        jitter = rng.random((rows, cols)).astype(np.float32) * 0.25

        coords = codec.identity_coords(ctx.size)
        valid = np.ones((height, width), bool)
        etch = np.zeros((height, width, 3), np.float32)
        intra_mask = np.zeros((rows, cols), bool)
        activity = np.zeros((rows, cols), np.float32)
        last_field = np.zeros((height, width, 2), np.float32)

        for i in range(ctx.length):
            if progress.cancelled:
                return
            t = (i + 1) / ctx.length
            index = ctx.b_window[i - a_frames] if i >= a_frames else min(first_b + i, len(b) - 1)
            target = b.frame(index).astype(np.float32)

            if index > 0 and not _same(b.frame(index - 1), b.frame(index)):
                pf = codec.analyze_pframe(
                    b.frame(index - 1),
                    b.frame(index),
                    block=BLOCK,
                    search=SEARCH,
                    quant=QUANT,
                    chroma_scale=CHROMA_SCALE,
                    intra_thresh=INTRA_THRESH,
                )
                last_field = motion.smooth_field(pf.vectors, ctx.size, BLOCK, 0.55)
                step_residual = pf.residual
                if i >= a_frames:
                    intra_mask = intra_mask | (pf.modes == codec.MODE_INTRA)
                activity = np.maximum(activity, pf.mad / 24.0)
            else:
                step_residual = None

            # Pull A's coordinates toward B's focal point while advecting by B's motion.
            # When B is in the window, start radial suction and hole-tracking at the cut
            # so B emerges smoothly from the outer border instead of popping in at once.
            has_b = a_frames < ctx.length
            in_b = i >= a_frames
            if has_b:
                if in_b:
                    b_len = ctx.length - a_frames
                    u = (i - a_frames + 1) / b_len
                    suction = PULL_STEP * pull * (4.0 * u * (1.0 - 0.65 * u)) * radial
                else:
                    suction = 0.0
            else:
                suction = PULL_STEP * pull * (4.0 * t * (1.0 - t)) * radial
            total_field = gain * last_field + suction
            if has_b and not in_b:
                coords, _ = codec.advect_coords(coords, total_field, mask=None, sampling="nearest")
            else:
                coords, valid = codec.advect_coords(coords, total_field, mask=valid, sampling="nearest")

            # Advect the luma-etched residual buffer alongside the coordinates.
            etch = warp.warp(etch, total_field, sampling="nearest") * 0.90
            if step_residual is not None:
                etch = np.clip(etch + residual_scale * step_residual, -160.0, 160.0)

            # Sample A once at the advected coordinates (stays crisp; zero repeated blur).
            moshed = np.clip(codec.sample_coords(source_a, coords, sampling="nearest") + etch, 0.0, 255.0)

            # Uncovered holes from suction/motion reveal B's real pixels directly.
            if i >= a_frames and exit_style == "intra":
                b_len = max(ctx.length - a_frames, 1)
                u = (i - a_frames + 1) / b_len
                if u > 0.35:
                    share = ((u - 0.35) / 0.65) ** 1.7
                    scores = 0.45 * near_focus + 0.55 * np.minimum(activity, 1.0) + jitter
                    intra_mask = codec.cluster_grow(intra_mask, scores, share)
                if i == ctx.length - 1:
                    intra_mask = np.ones((rows, cols), bool)

            show_b = (~valid) if i >= a_frames else np.zeros((height, width), bool)
            if intra_mask.any():
                show_b = show_b | codec.expand_blocks(intra_mask, ctx.size, BLOCK)

            out = np.where(show_b[..., None], target, moshed)
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _focus(clip, start: int, size: tuple[int, int]) -> tuple[float, float]:
    """Centre of B's strongest motion over its opening frames (image centre if still)."""
    width, height = size
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    total = sx = sy = 0.0
    for first in range(start, min(start + FOCUS_LOOK, len(clip) - 1), 2):
        if _same(clip.frame(first), clip.frame(first + 1)):
            continue
        ref_y = color.bgr_to_y(clip.frame(first))
        cur_y = color.bgr_to_y(clip.frame(first + 1))
        vectors = codec.estimate_hierarchical(ref_y, cur_y, block=BLOCK, search=SEARCH)
        field = motion.to_field(vectors, size, BLOCK)
        speed = np.abs(field[..., 0]) + np.abs(field[..., 1])
        total += float(speed.sum())
        sx += float((speed * xs).sum())
        sy += float((speed * ys).sum())
    if total < 1e-3:
        return width / 2.0, height / 2.0
    return sx / total, sy / total


def _same(first: np.ndarray, second: np.ndarray) -> bool:
    """True when two frames are effectively identical (repeated frame)."""
    return float(np.abs(first.astype(np.float32) - second).mean()) < SAME
