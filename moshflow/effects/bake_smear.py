"""Bake Smear: every output frame is the reference for the next, and B keeps feeding in.

moshy's "bake" writes a mosh out as real video, so the next step starts from the
smeared picture instead of the clean clip. Here the same happens frame by frame:
the previous output frame is decoded forward with B's motion, so B's movement
accumulates in the picture instead of being replaced each frame.

One output frame, from the previous output P:

1. Decode B's P-frame (motion F, residual E, measured between two of B's frames):
   P(x) <- P(x + gain * F(x)) + residual * E(x). During A's part A's own P-frame is
   decoded too, so A keeps moving up to the cut.
2. Feed B in, from the first frame, not only at the end. Where B moves, B's real
   pixel replaces the smeared one at once: with m = min(|F(x)| / 1.5, 1) and the
   share s = bleed * m, P <- P + s (B - P). A moving person in B therefore walks
   through A's smear from the start, and what B has not touched yet stays A's.
3. Suction. The focus c is where B moves most (the centre of |F| over B's first
   frames). The smeared picture is pulled into c: P(x) <- P(x + k (x - c)) with
   k = 0.05 * pull * 4 t (1 - t), so content shrinks towards c, fastest mid-way.
4. B emerges from c. An 8 x 8 block switches to B's real pixels when its distance
   from c, normalised to the farthest corner and mixed with a random jitter per
   block (ctx.seed), falls inside a radius R(t) that grows from 0 as t^1.5 and
   covers the whole picture on the last frame. B grows out of the point where
   the picture is being sucked in, with a ragged, block-sized edge, not a fade.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, warp
from moshflow.toolkit.pframe import motion_field, p_frame

BLOCK = 8
MOTION_FULL = 1.5    # vector length (pixels) at which B replaces the smear completely
SAME = 0.5           # mean grey-level difference below which two frames count as repeated
JITTER = 0.3         # how much of a block's switch distance is random
FOCUS_LOOK = 20      # B frames searched for the focus
PULL_STEP = 0.05     # largest per-frame shrink at pull = 1


@register
class BakeSmear(TransitionEffect):
    info = EffectInfo("bake_smear", "Bake Smear", "분배혀",
                      "Each frame is built on the last one, so B's motion smears A while B walks through it.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Smear strength", 1.2, 0.0, 4.0),
            FloatParam("bleed", "B feed-in", 0.6, 0.0, 1.0),
            FloatParam("residual", "Residual", 1.0, 0.0, 2.0),
            FloatParam("pull", "Suction", 1.0, 0.0, 3.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        a, b = ctx.outgoing, ctx.incoming
        width, height = ctx.size
        picture = ctx.outgoing.frame(max(ctx.a_window.start - 1, 0)).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        a_frames = len(ctx.a_window)
        rng = np.random.default_rng(ctx.seed)
        rows, cols = -(-height // BLOCK), -(-width // BLOCK)
        luck = rng.random((rows, cols)).astype(np.float32)
        cx, cy = _focus(b, first_b, ctx.size)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        radial = np.stack([xs - cx, ys - cy], axis=-1)
        far = float(np.sqrt(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2))
        block_x = (np.arange(cols, dtype=np.float32) + 0.5) * BLOCK
        block_y = (np.arange(rows, dtype=np.float32) + 0.5) * BLOCK
        distance = np.sqrt((block_x[None, :] - cx) ** 2 + (block_y[:, None] - cy) ** 2) / far
        reach = distance * (1.0 - JITTER) + luck * JITTER
        field = np.zeros((height, width, 2), np.float32)
        error = np.zeros((height, width, 3), np.float32)

        for i in range(ctx.length):
            if progress.cancelled:
                return
            # B's index: its real window after the cut, its opening frames before it.
            index = ctx.b_window[i - a_frames] if i >= a_frames else min(first_b + i, len(b) - 1)
            if index > 0 and not _same(b.frame(index - 1), b.frame(index)):
                field, error = p_frame(b, index - 1, index, ctx.size)
            else:
                error = np.zeros_like(error)
            target = b.frame(index).astype(np.float32)

            picture = warp.warp(picture, p["gain"] * field, "bilinear") + p["residual"] * error
            if i < a_frames:
                a_index = ctx.a_window[i]
                if a_index > 0 and not _same(a.frame(a_index - 1), a.frame(a_index)):
                    a_field, a_error = p_frame(a, a_index - 1, a_index, ctx.size)
                    picture = warp.warp(picture, a_field, "bilinear") + a_error
            picture = np.clip(picture, 0.0, 255.0)

            t = (i + 1) / ctx.length
            speed = np.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2)
            share = (p["bleed"] * np.minimum(speed / MOTION_FULL, 1.0))[..., None]
            picture = picture + share * (target - picture)
            picture = warp.warp(picture, PULL_STEP * p["pull"] * 4 * t * (1 - t) * radial, "bilinear")

            reached = reach <= (t ** 1.5) * (1.0 + JITTER)
            if i == ctx.length - 1:
                reached = np.ones_like(reached)
            mask = np.repeat(np.repeat(reached, BLOCK, axis=0), BLOCK, axis=1)[:height, :width]
            picture = np.where(mask[..., None], target, picture)
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _focus(clip, start: int, size: tuple[int, int]) -> tuple[float, float]:
    """The point where B moves most: the centre of |F| over B's first frames (image centre if B is still)."""
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    total = sx = sy = 0.0
    for first in range(start, min(start + FOCUS_LOOK, len(clip) - 1)):
        field = motion_field(clip, first, first + 1, size)
        speed = np.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2)
        total += float(speed.sum())
        sx += float((speed * xs).sum())
        sy += float((speed * ys).sum())
    if total < 1e-3:
        return width / 2, height / 2
    return sx / total, sy / total


def _same(first: np.ndarray, second: np.ndarray) -> bool:
    """True if two frames are (almost) identical, i.e. a repeated frame."""
    return float(np.abs(first.astype(np.float32) - second).mean()) < SAME
