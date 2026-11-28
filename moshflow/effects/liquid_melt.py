"""Liquid Melt: clip A's picture melts sideways as pixelated liquid and clip B pours in
behind it (Survey & Plan, 13.3: a lost I-frame with smooth motion in place of blocks).

The picture starts as the last clean frame of clip A. Every output frame:

1. Motion: block matching between two consecutive frames of the clip that is
   playing, interpolated into a smooth per-pixel flow F (no block edges).
2. Current: F is added to a velocity that keeps part of its old value (`inertia`),
   and every pixel is pushed sideways by `melt` times the speed at that spot, so
   the picture streams like a liquid and keeps flowing after the push.
3. The picture is moved by the velocity. `look` "pixel" copies the nearest whole
   pixel, so the currents leave jagged pixel streaks; "liquid" mixes neighbours.
4. Imprint: every pixel's brightness moves towards clip B's brightness, while A's
   colour stays, so B's shapes appear inside the melting picture.
5. Pixelation ("pixel" look): the frame is drawn in square cells of `pixel` size,
   each cell showing the colour at its centre. Cells grow from 1 to `pixel` while
   the picture melts and shrink back to 1 at the end, so it starts and ends sharp.
6. Pour-in: over the last `settle` share, clip B flows in from the side the
   current comes from, cell by cell, behind a ragged front that runs faster where
   the video moved most.

One pixel, direction "right": if the flow there is 3 pixels per frame and melt is 2,
its colour is copied from 6 pixels to its left, so colour streams right. With 8-pixel
cells, the pixel at (13, 21) shows the picture's colour at (12, 20), the centre of its
cell. It turns into clip B when the pour-in front passes its cell.
"""

from __future__ import annotations

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import codec, color, frames, motion, warp

CUT_LEVEL = 40.0    # mean luma difference above which two frames are a scene cut, not motion
IMPRINT_QUANT = 14.0


@register
class LiquidMelt(TransitionEffect):
    info = EffectInfo("liquid_melt", "Liquid Melt", "Casper Bach",
                      "Clip A melts sideways as pixelated liquid while clip B pours in behind it.")

    @classmethod
    def param_specs(cls):
        return [
            ChoiceParam("look", "Look", "pixel", ("pixel", "liquid")),
            ChoiceParam("direction", "Direction", "right", ("right", "left")),
            IntParam("pixel", "Pixel size", 8, 1, 32),
            FloatParam("melt", "Sideways melt", 2.0, 0.0, 6.0),
            FloatParam("strength", "Motion strength", 1.5, 0.0, 3.0),
            FloatParam("inertia", "Inertia", 0.6, 0.0, 0.95),
            FloatParam("imprint", "B shows through", 0.6, 0.0, 1.0),
            FloatParam("settle", "Pour-in", 0.35, 0.0, 1.0),
            IntParam("block", "Block size", 16, 8, 64),
            IntParam("search", "Search range", 8, 2, 32),
            FloatParam("smoothness", "Smoothness", 1.0, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        width, height = ctx.size
        sign = 1.0 if p["direction"] == "right" else -1.0
        pixel_look = p["look"] == "pixel"
        sampling = "nearest" if pixel_look else "bilinear"
        rng = np.random.default_rng(ctx.seed)
        b = ctx.incoming

        source_a = frames.last_clean_a(ctx).astype(np.float32)
        coords = codec.identity_coords(ctx.size)
        picture = source_a.copy()
        velocity = np.zeros((height, width, 2), np.float32)
        activity = np.zeros((height, width), np.float32)
        limit = 2.0 * p["block"]
        settle_frames = int(round(p["settle"] * ctx.length))
        settle_start = ctx.length - settle_frames
        first_b = b.frame(ctx.b_window.start if len(ctx.b_window) else 0)
        # Position along the current, 0 where B enters, 1 where A leaves.
        along = np.arange(width, dtype=np.float32) / max(width - 1, 1)
        if sign < 0:
            along = along[::-1].copy()
        ragged = _curve(height, max(height // 6, 2), rng)                # 0..1 per row

        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            t = (i + 1) / ctx.length
            target = (clip.frame(index) if clip is b else first_b).astype(np.float32)

            # Steps 1 to 3: the playing clip's motion drives a sideways liquid current.
            velocity *= p["inertia"]
            if index > 0:
                flow = _flow(clip.frame(index - 1), clip.frame(index), ctx.size, p)
                speed = np.abs(flow[..., 0]) + np.abs(flow[..., 1])     # |dx| + |dy|
                velocity += p["strength"] * flow
                velocity[..., 0] -= sign * p["melt"] * speed           # colour travels with sign
                velocity = np.clip(velocity, -limit, limit)
                activity += speed
            if pixel_look:
                coords, _ = codec.advect_coords(coords, velocity, sampling="nearest")
                picture = codec.sample_coords(source_a, coords, sampling="nearest")
            else:
                picture = warp.warp(picture, velocity, sampling=sampling)

            # Step 4: B's strong brightness contours etch into A's colours via dead-zone
            # quantization so smooth areas keep A's exact palette instead of turning grey.
            w = p["imprint"] * t * t * (3.0 - 2.0 * t)
            if w > 0.0:
                luma_diff = codec.quantize_residual(target - picture, quant=IMPRINT_QUANT, chroma_scale=0.0)
                out = np.clip(picture + w * luma_diff, 0.0, 255.0)
            else:
                out = picture

            # Step 6: B pours in from the side, behind a ragged front.
            if i >= settle_start:
                s = (i - settle_start + 1) / settle_frames
                busy = _row_activity(activity)                           # 0..1 per row
                when = along[None, :] * 0.75 + 0.25 * (0.6 * ragged + 0.4 * (1.0 - busy))[:, None]
                if pixel_look:
                    alpha = (when < s).astype(np.float32)
                else:
                    alpha = np.clip((s - when) / 0.03 + 0.5, 0.0, 1.0)
                out = out + alpha[..., None] * (target - out)

            # Step 5: draw in pixel cells that grow while melting and shrink at the end.
            if pixel_look and p["pixel"] > 1:
                cell = 1 + int(round((p["pixel"] - 1) * 4.0 * t * (1.0 - t)))
                if i == ctx.length - 1:
                    cell = 1
                out = _pixelate(out, cell)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _flow(previous, current, size, p) -> np.ndarray:
    """Smooth H x W x 2 flow from previous to current; zero across a scene cut."""
    reference = color.bgr_to_y(previous)
    target = color.bgr_to_y(current)
    width, height = size
    if np.abs(target - reference).sum() / target.size > CUT_LEVEL:
        return np.zeros((height, width, 2), np.float32)
    vectors = codec.estimate_hierarchical(reference, target, block=p["block"], search=p["search"])
    return motion.smooth_field(vectors, size, p["block"], p["smoothness"])


def _pixelate(image: np.ndarray, cell: int) -> np.ndarray:
    """Every pixel takes the colour at the centre of its cell x cell square."""
    if cell <= 1:
        return image
    height, width = image.shape[:2]
    ys = np.minimum((np.arange(height) // cell) * cell + cell // 2, height - 1)
    xs = np.minimum((np.arange(width) // cell) * cell + cell // 2, width - 1)
    return image[ys[:, None], xs[None, :]]


def _row_activity(activity: np.ndarray) -> np.ndarray:
    """How much each row moved during the transition, scaled to 0..1."""
    rows = activity.sum(axis=1)
    return rows / max(float(rows.max()), 1e-6)


def _curve(length: int, spacing: int, rng) -> np.ndarray:
    """Random knots in 0..1 every `spacing` samples, joined with smoothstep weights
    w * w * (3 - 2w), so neighbouring rows stay close and there are no corners."""
    knots = rng.random(length // spacing + 2).astype(np.float32)
    position = np.arange(length, dtype=np.float32) / spacing
    low = position.astype(np.intp)
    w = position - low
    w = w * w * (3.0 - 2.0 * w)
    return (1.0 - w) * knots[low] + w * knots[low + 1]
