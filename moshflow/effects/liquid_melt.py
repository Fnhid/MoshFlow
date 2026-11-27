"""Liquid Melt: clip A's picture melts sideways as pixelated liquid where the video
moves, until clip B's next keyframe arrives (Survey & Plan, 13.3: a lost I-frame
with smooth motion in place of blocks).

`timing` "at the cut" (the default): A's part of the window plays clean; at the cut
A's last frame starts to melt and B plays at its real speed, so B runs from the first
moshed frame. "stretch" melts the whole window, starting from the last clean frame
of A, and plays B slower so it lands on the right frame. With no B part ("before"
alignment) A's own last frames drive the melt.

Every melting frame:

1. Motion: block matching between two consecutive frames of the clip that is
   playing, interpolated into a smooth per-pixel flow F (no block edges). A vector
   only counts if it explains its block clearly better than standing still and the
   block changed by more than noise, so still or flat areas do not move.
2. Current: F is added to a velocity that keeps part of its old value (`inertia`),
   and every pixel is pushed sideways by `melt` times the speed at that spot, so
   the picture streams like a liquid where the video moves and keeps flowing after
   the push.
3. The picture is moved by the velocity. `look` "pixel" copies the nearest whole
   pixel, so the currents leave jagged pixel streaks; "liquid" mixes neighbours.
4. Imprint: where the video moved, a pixel's brightness moves towards clip B's
   brightness while A's colour stays, more as the transition goes on (fully once the
   video moved 8 pixels there in total, at 480p), so B's shapes appear inside the
   melting picture. Where nothing moved, A's picture is untouched.
5. Pixelation ("pixel" look): where the picture flows faster than 1 pixel per frame
   it is drawn in square cells of up to `pixel` size, each cell showing the colour
   at its centre; still parts stay sharp. Cells grow while the picture melts and
   shrink back to 1 at the end, so it ends sharp.
6. End: the melt runs to the window's last frame, which is clip B's own frame, like
   B's next keyframe arriving. Optionally (`settle` above 0), over the last `settle`
   share B flows in where the video moved most first and spreads outwards from there
   instead: the movement is let seep outwards over the block grid, the blocks are
   ranked, and the ranks are interpolated between block centres, so B's edge is
   smooth like a liquid.

Block size, search range and pixel size are given for 480p; larger frames scale
them (x2 at 1080p), so the effect looks the same at every resolution.

One pixel, direction "right": if the flow there is 3 pixels per frame and melt is 2,
its colour is copied from 6 pixels to its left, so colour streams right. With 8-pixel
cells, the pixel at (13, 21) shows the picture's colour at (12, 20), the centre of its
cell. On the window's last frame it shows clip B.
"""

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

CUT_LEVEL = 40.0    # mean luma difference above which two frames are a scene cut, not motion
USEFUL_GAIN = 2.0   # luma levels per pixel a vector must save over standing still to count
NOISE_FLOOR = 4.0   # mean change per pixel that is only sensor noise


@register
class LiquidMelt(TransitionEffect):
    info = EffectInfo("liquid_melt", "Liquid Melt", "Casper Bach",
                      "Clip A melts sideways as pixelated liquid while clip B pours in behind it.")

    @classmethod
    def param_specs(cls):
        return [
            ChoiceParam("timing", "B timing", "at the cut", ("at the cut", "stretch")),
            ChoiceParam("look", "Look", "pixel", ("pixel", "liquid")),
            ChoiceParam("direction", "Direction", "right", ("right", "left")),
            IntParam("pixel", "Pixel size", 8, 1, 32),
            FloatParam("melt", "Sideways melt", 2.0, 0.0, 6.0),
            FloatParam("strength", "Motion strength", 1.5, 0.0, 3.0),
            FloatParam("inertia", "Inertia", 0.6, 0.0, 0.95),
            FloatParam("imprint", "B shows through", 0.6, 0.0, 1.0),
            FloatParam("settle", "Pour-in", 0.0, 0.0, 1.0),
            IntParam("block", "Block size", 16, 8, 64),
            IntParam("search", "Search range", 8, 2, 32),
            FloatParam("smoothness", "Smoothness", 1.0, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        width, height = ctx.size
        # Block size, search range and pixel size are given for 480p; larger frames
        # scale them (x2 at 1080p), so the effect looks the same at every resolution.
        scale = max(1, int(round(min(width, height) / 480)))
        cell_max = p["pixel"] * scale
        sign = 1.0 if p["direction"] == "right" else -1.0
        pixel_look = p["look"] == "pixel"
        sampling = "nearest" if pixel_look else "bilinear"
        rng = np.random.default_rng(ctx.seed)
        a, b = ctx.outgoing, ctx.incoming

        plan = _plan(ctx, p["timing"])
        melt_frames = sum(1 for step in plan if step[0] == "melt")
        picture = _start_picture(ctx, p["timing"]).astype(np.float32)
        velocity = np.zeros((height, width, 2), np.float32)
        activity = np.zeros((height, width), np.float32)
        limit = 2.0 * p["block"] * scale
        settle_frames = int(round(p["settle"] * melt_frames))
        settle_start = melt_frames - settle_frames
        n = 2 * max(p["block"] // 2, 1) * scale               # block size in pixels
        jitter = rng.random((-(-height // n), -(-width // n))).astype(np.float32)
        pour_order = None                                     # when each pixel's B arrives, 0..1
        done, previous = 0, None

        for out_index, (mode, clip, index, b_index) in enumerate(plan):
            if progress.cancelled:
                return
            if mode == "clean":                               # A plays clean until the cut
                progress.report(out_index + 1, ctx.length)
                yield a.frame(index)
                continue
            i = done
            done += 1
            t = done / melt_frames
            target = b.frame(b_index).astype(np.float32)

            # Steps 1 to 3: the playing clip's motion drives a sideways liquid current.
            velocity *= p["inertia"]
            if index > 0 and (clip, index) != previous:
                flow = _flow(clip.frame(index - 1), clip.frame(index), ctx.size, p, scale)
                speed = np.abs(flow[..., 0]) + np.abs(flow[..., 1])     # |dx| + |dy|
                velocity += p["strength"] * flow
                velocity[..., 0] -= sign * p["melt"] * speed           # colour travels with sign
                velocity = np.clip(velocity, -limit, limit)
                activity += speed
            previous = (clip, index)
            picture = warp.warp(picture, velocity, sampling=sampling)

            # Step 4: B's brightness shows through A's colours where B has moved: fully
            # once B moved 8 pixels (at 480p) there in total, not at all where B is still.
            moved = _upsample(np.clip(_block_means(activity, n) / (8.0 * scale), 0.0, 1.0), ctx.size, n)
            w = p["imprint"] * t * t * (3.0 - 2.0 * t) * moved
            out = picture + (w * (color.bgr_to_y(target) - color.bgr_to_y(picture)))[..., None]

            # Step 6: B pours in where it moved most first and spreads outwards from
            # there, behind a soft liquid edge; the last pixel goes on the last frame.
            if i >= settle_start:
                s = (i - settle_start + 1) / settle_frames
                if pour_order is None:
                    pour_order = _pour_order(_block_means(activity, n), jitter, ctx.size, n)
                if pixel_look:
                    alpha = (pour_order < s).astype(np.float32)
                else:
                    alpha = np.clip((s - pour_order) / 0.08 + 0.5, 0.0, 1.0)
                out = out + alpha[..., None] * (target - out)

            # Step 5: draw in pixel cells that grow while melting and shrink at the end.
            if pixel_look and cell_max > 1:
                # Only where the picture flows (more than 1 pixel per frame, decided per
                # cell), so the parts of A that B does not move stay sharp.
                cell = 1 + int(round((cell_max - 1) * 4.0 * t * (1.0 - t)))
                if done == melt_frames:
                    cell = 1
                flowing = np.abs(velocity[..., 0]) + np.abs(velocity[..., 1])
                where = (_pixelate(flowing, cell) > 1.0).astype(np.float32)[..., None]
                out = out + where * (_pixelate(out, cell) - out)

            # The last frame is B's own frame: B's next keyframe arrives and the melt ends.
            if done == melt_frames:
                out = target

            progress.report(out_index + 1, ctx.length)
            yield color.to_uint8(out)


def _plan(ctx, timing: str) -> list[tuple]:
    """(mode, clip whose motion drives the melt, its frame, frame of B shown) for
    every output frame; mode is "clean" (A plays untouched) or "melt".

    "at the cut" with a B part: A's window plays clean and B's window melts, with B
    at its real speed. "stretch": every frame melts and B runs from its frame 0 to
    the last frame of its window, slower when the window starts in A. With no B
    part ("before" alignment) A's own last frames drive the melt towards B's first
    frame, because B starts at frame 0 after the cut."""
    a, b = ctx.outgoing, ctx.incoming
    if not len(ctx.b_window):
        return [("melt", a, k, 0) for k in ctx.a_window]
    if timing == "at the cut":
        return [("clean", a, k, None) for k in ctx.a_window] + [("melt", b, k, k) for k in ctx.b_window]
    last = ctx.b_window.stop - 1
    if ctx.length == 1:
        return [("melt", b, last, last)]
    return [("melt", b, j, j) for j in (int(round(last * k / (ctx.length - 1))) for k in range(ctx.length))]


def _start_picture(ctx, timing: str):
    """The picture that melts: A's last frame inside the window when the melt starts
    at the cut, otherwise the last clean frame of A before the window."""
    if timing == "at the cut" and len(ctx.b_window) and len(ctx.a_window):
        return ctx.outgoing.frame(ctx.a_window.stop - 1)
    return frames.last_clean_a(ctx)


def _flow(previous, current, size, p, scale: int = 1) -> np.ndarray:
    """Smooth H x W x 2 flow from previous to current; zero across a scene cut.

    Block matching runs on every (2 x scale)-th pixel of the luma, with half the
    block and search, and the vectors are scaled back up to full size. At 480p
    (scale 1) that is every second pixel; at 1080p (scale 2) every fourth, so a
    block and the search cover the same part of the picture at both sizes.
    """
    stride = 2 * scale
    before, now = color.bgr_to_y(previous), color.bgr_to_y(current)
    reference, target = before[::stride, ::stride], now[::stride, ::stride]
    width, height = size
    if np.abs(target - reference).sum() / target.size > CUT_LEVEL:
        return np.zeros((height, width, 2), np.float32)
    half_block = max(p["block"] // 2, 1)
    half_search = max(p["search"] // 2, 1)
    n = stride * half_block
    vectors = stride * motion.estimate(reference, target, half_block, half_search)
    # A vector only counts if it explains its block clearly better than standing
    # still, and the block changed by more than noise; on flat or noisy areas block
    # matching picks random vectors, which would melt still parts of the picture.
    moved_error = _block_means(np.abs(now - warp.warp(before, motion.to_field(vectors, size, n))), n)
    still_error = _block_means(np.abs(now - before), n)
    rows, cols = moved_error.shape
    useful = ((still_error - moved_error >= USEFUL_GAIN) & (still_error > NOISE_FLOOR)).astype(np.int32)
    vectors = vectors[:rows, :cols] * useful[..., None]
    return motion.smooth_field(vectors, size, n, p["smoothness"])


def _pixelate(image: np.ndarray, cell: int) -> np.ndarray:
    """Every pixel takes the colour at the centre of its cell x cell square."""
    if cell <= 1:
        return image
    height, width = image.shape[:2]
    ys = np.minimum((np.arange(height) // cell) * cell + cell // 2, height - 1)
    xs = np.minimum((np.arange(width) // cell) * cell + cell // 2, width - 1)
    return image[ys[:, None], xs[None, :]]


def _block_means(values: np.ndarray, n: int) -> np.ndarray:
    """Mean of an H x W map over every n x n block (edge pixels repeated to fill the
    last blocks). Returns a rows x cols array."""
    height, width = values.shape
    rows, cols = -(-height // n), -(-width // n)
    ys = np.minimum(np.arange(rows * n), height - 1)
    xs = np.minimum(np.arange(cols * n), width - 1)
    padded = values[ys[:, None], xs[None, :]]
    return padded.reshape(rows, n, cols, n).sum(axis=(1, 3)) / (n * n)


def _upsample(grid: np.ndarray, size: tuple[int, int], n: int) -> np.ndarray:
    """A rows x cols grid to H x W, bilinear between block centres (motion.smooth_field)."""
    knots = np.zeros(grid.shape + (2,), np.float32)
    knots[..., 0] = grid
    return motion.smooth_field(knots, size, n, 1.0)[..., 0]


def _pour_order(busy: np.ndarray, jitter: np.ndarray, size: tuple[int, int], n: int) -> np.ndarray:
    """When B arrives at every pixel in the pour-in, 0..1. The movement per block is
    let seep outwards over the grid (12 rounds of 3 x 3 averaging), so every block
    learns how close it is to where B moved; the blocks are ranked, most moved first,
    rank / number of blocks is each block's turn, and the turns are interpolated
    bilinearly between block centres, so the edge of B is smooth like a liquid."""
    near = busy.astype(np.float32)
    for _ in range(12):
        near = _blur3(near)
    near = near + 1e-4 * float(near.max() + 1.0) * jitter
    values = near.ravel().tolist()
    order = sorted(range(len(values)), key=lambda k: -values[k])
    turns = np.zeros(len(values), np.float32)
    for rank, k in enumerate(order):
        turns[k] = rank / max(len(values) - 1, 1)
    smooth = _upsample(turns.reshape(busy.shape), size, n)
    low, high = float(smooth.min()), float(smooth.max())
    return (smooth - low) / max(high - low, 1e-6) * 0.999


def _blur3(values: np.ndarray) -> np.ndarray:
    """Mean of each 3 x 3 neighbourhood of a rows x cols grid, edges repeated."""
    rows, cols = values.shape
    padded = values[np.clip(np.arange(-1, rows + 1), 0, rows - 1)][:, np.clip(np.arange(-1, cols + 1), 0, cols - 1)]
    total = np.zeros_like(values, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy:dy + rows, dx:dx + cols]
    return total / 9.0
