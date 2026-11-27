"""DCT Crush: a dropped I-frame decoded block by block in a JPEG-style codec
(Survey & Plan, 3.1 and 6.4).

Both clips are coded the way JPEG and MPEG code a picture, by hand: BGR to Y, Cb,
Cr (BT.601), then every block x block square to frequency coefficients with the
DCT, F = C . f . C^T. F[0, 0] (DC) is the block's average colour; the other
coefficients (AC) are its texture and edges.

The decoder keeps one grid of coefficient blocks, the picture it believes in. It
starts as clip A's last frame before the mosh. For every frame of clip B:

1. Motion: block matching between B's previous and current frame gives every
   block a vector and a residual (what the motion does not explain). The average
   vector is the camera's motion; a block's own movement is its vector minus it.
2. The coefficient grid is moved along B's vectors (bilinear between blocks), so
   A's blocks travel the way B moves, camera pans included.
3. Switch: every block adds up activity = residual / 32 + own movement / block
   size. When the activity around it (its 5 x 5 block neighbourhood, so regions
   switch together) reaches 1 / `sensitivity`, the block takes B's texture (AC)
   at once; at 1 / `colour` it takes B's colour (DC) too and follows B from then
   on. Blocks where B is still keep A's picture, as a decoder skips them.
4. Crush: blocks with B's texture but A's colour are rounded to the JPEG table
   times `crush`, so the damage sits where the switch happens. Colour gets
   `chroma` times coarser steps.
5. Refresh: over the last `heal` share the remaining blocks switch to B's block,
   each all at once (never half blended), like an encoder's intra refresh. The
   activity seeps outwards over the grid and the blocks are ranked by it, so B
   spreads out from where it moved at an even pace, the same number of blocks
   every frame. The last frame is exactly B.

`timing` "at the cut" lets A's part of the window play clean and starts the mosh at
B's first frame, with B at its real speed. "stretch" (and any window with no B part)
moshes the whole window and plays B slower so it lands on the right frame.

Block size and search range are given for 480p. Larger frames scale both (x2 at
1080p), so a block covers the same part of the picture at every resolution.

One pixel: the hand in B moves 6 pixels to the right of the camera's motion and
leaves a residual of 32, so its 8 x 8 block gains 6 / 8 + 1 = 1.75 activity a
frame. With sensitivity 0.5 the hand's region reaches 2 after about two frames and
switches to B's texture in one step, while the still wall next to it keeps A's
picture.
"""

import math

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, motion, warp

# Standard JPEG quantization tables (ITU-T T.81, Annex K) for brightness and colour.
LUMA_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float32)
CHROMA_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], np.float32)

CUT_LEVEL = 40.0    # mean luma difference above which two frames are a scene cut, not motion
USEFUL_GAIN = 2.0   # luma levels per pixel a vector must save over standing still to count
NOISE_FLOOR = 4.0   # mean residual per pixel that is only sensor noise


@register
class DctCrush(TransitionEffect):
    info = EffectInfo("dct_crush", "DCT Crush", "Casper Bach",
                      "Clip B's moving parts refresh clip A's JPEG blocks; still parts keep A until B's next keyframe.")

    @classmethod
    def param_specs(cls):
        return [
            ChoiceParam("timing", "B timing", "at the cut", ("at the cut", "stretch")),
            IntParam("block", "Block size", 8, 4, 32),
            FloatParam("sensitivity", "Texture refresh", 0.5, 0.0, 3.0),
            FloatParam("colour", "Colour refresh", 0.25, 0.0, 3.0),
            FloatParam("crush", "Crush", 6.0, 0.0, 30.0),
            FloatParam("chroma", "Colour crush", 2.0, 1.0, 8.0),
            FloatParam("heal", "Refresh", 0.3, 0.05, 0.6),
            IntParam("search", "Search range", 8, 2, 32),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        width, height = ctx.size
        # Block size and search are set for 480p; larger frames scale them up, so a
        # block covers the same part of the picture at every resolution.
        scale = max(1, int(round(min(width, height) / 480)))
        n = max(2 * (p["block"] // 2), 4) * scale             # block size in pixels
        stride = 2 * scale                                    # block matching on every stride-th pixel
        basis = _dct_matrix(n)
        steps = [_table(LUMA_TABLE, n), _table(CHROMA_TABLE, n) * p["chroma"], _table(CHROMA_TABLE, n) * p["chroma"]]
        is_dc = np.zeros((n, n), np.float32)
        is_dc[0, 0] = 1.0
        a, b = ctx.outgoing, ctx.incoming
        plan, mosh_frames = _plan(ctx, p["timing"])
        start = _to_ycbcr(_first_a(ctx).astype(np.float32))
        state = [_dct(plane, basis) for plane in start]       # the picture the decoder believes in
        rows, cols = state[0].shape[:2]
        busy = np.zeros((rows, cols), np.float32)             # all activity so far, per block
        b_texture = np.zeros((rows, cols), np.float32)        # 1 = block shows B's AC
        b_colour = np.zeros((rows, cols), np.float32)         # 1 = block shows B's DC too
        jitter = np.random.default_rng(ctx.seed).random((rows, cols)).astype(np.float32)
        turns = None                                          # when each block refreshes, 0..1
        done, previous_j = 0, None

        for i, (kind, j) in enumerate(plan):
            if progress.cancelled:
                return
            if kind == "a":                                   # A plays clean until the cut
                progress.report(i + 1, ctx.length)
                yield a.frame(j)
                continue
            done += 1
            t = done / mosh_frames
            frame_b = b.frame(j)
            coeff_b = [_dct(plane, basis) for plane in _to_ycbcr(frame_b.astype(np.float32))]

            if j > 0 and j != previous_j:
                found = _block_motion(b.frame(j - 1), frame_b, n, p["search"], stride)
                if found is not None:
                    vectors, residual = found
                    # Step 1: the block's own movement, without the camera's.
                    camera = vectors.reshape(-1, 2).sum(axis=0) / (rows * cols)
                    own = np.abs(vectors[..., 0] - camera[0]) + np.abs(vectors[..., 1] - camera[1])
                    busy = busy + residual / 32.0 + own / n
                    # Step 2: A's stale blocks travel with B's motion.
                    for c in range(3):
                        state[c] = _shift_blocks(state[c], vectors / n)
            previous_j = j

            # Step 3: a block switches, all at once, when the activity so far around it
            # (its 5 x 5 neighbourhood, so whole regions switch together) is enough:
            # texture at 1 / sensitivity, colour at 1 / colour (never before texture).
            around = _blur3(_blur3(busy))
            b_texture = np.maximum(b_texture, (around * p["sensitivity"] >= 1.0).astype(np.float32))
            b_colour = np.maximum(b_colour, b_texture * (around * p["colour"] >= 1.0).astype(np.float32))

            # Step 5: refresh, block by block. Over the last `heal` share every block
            # switches to B's block at once (never half blended), spreading outwards
            # from where B moved, the same number of blocks every frame.
            s = (t - (1.0 - p["heal"])) / p["heal"]
            if s >= 1.0:                                      # every block is B: B's own frame
                progress.report(i + 1, ctx.length)
                yield frame_b
                continue
            if s > 0.0:
                if turns is None:
                    turns = _refresh_turns(busy, jitter)
                turn = (turns < s).astype(np.float32)
                b_texture = np.maximum(b_texture, turn)
                b_colour = np.maximum(b_colour, turn)

            # Switched blocks follow B from now on; the others keep A's stale block.
            take = b_colour[..., None, None] * is_dc + b_texture[..., None, None] * (1.0 - is_dc)
            for c in range(3):
                state[c] = state[c] + take * (coeff_b[c] - state[c])

            # Step 4: crush the mixed blocks (B's texture in A's colour).
            ramp = _smooth(t / 0.15)
            crush = p["crush"] * ramp * (b_texture - b_colour)
            out = []
            for c in range(3):
                step = steps[c] * np.maximum(crush, 1e-3)[..., None, None]
                rounded = _round(state[c] / step) * step
                use = (crush > 0.05).astype(np.float32)[..., None, None]
                out.append(_idct(state[c] + use * (rounded - state[c]), basis, start[c].shape))

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(_to_bgr(out))


def _plan(ctx, timing: str) -> tuple[list[tuple[str, int]], int]:
    """("a", frame of A) or ("b", frame of B) for every output frame, and how many
    frames are moshed.

    "at the cut" with a B part: A's window plays clean, B's window is moshed at B's
    real speed. Otherwise every frame is moshed and B runs from frame 0 to the last
    frame of its window (slower if the window starts in A); with no B part it holds
    B's first frame, because B starts at frame 0 after the cut."""
    if timing == "at the cut" and len(ctx.b_window):
        plan = [("a", k) for k in ctx.a_window] + [("b", k) for k in ctx.b_window]
        return plan, len(ctx.b_window)
    last = ctx.b_window.stop - 1 if len(ctx.b_window) else 0
    if ctx.length == 1:
        return [("b", last)], 1
    return [("b", int(round(last * k / (ctx.length - 1)))) for k in range(ctx.length)], ctx.length


def _first_a(ctx):
    """The picture the mosh starts from: A's last frame inside the window, or the last
    clean frame before it when A has no frames in the window."""
    a = ctx.outgoing
    if len(ctx.a_window):
        return a.frame(ctx.a_window.stop - 1)
    return a.frame(max(ctx.a_window.start - 1, 0))


def _block_motion(previous, current, n: int, search: int, stride: int):
    """B's block vectors (in pixels) and each block's mean residual after motion
    compensation. Block matching runs on every stride-th pixel, with the block and
    search shrunk by stride and the search set for 480p (search / 2 at that step);
    the vectors are scaled back up. None across a scene cut."""
    before, now = color.bgr_to_y(previous), color.bgr_to_y(current)
    small_before, small_now = before[::stride, ::stride], now[::stride, ::stride]
    if np.abs(small_now - small_before).sum() / small_now.size > CUT_LEVEL:
        return None
    vectors = stride * motion.estimate(small_before, small_now, n // stride, max(search // 2, 1))
    height, width = now.shape
    predicted = warp.warp(before, motion.to_field(vectors, (width, height), n))
    moved_error = _blocks(np.abs(now - predicted), n).sum(axis=(2, 3)) / (n * n)
    still_error = _blocks(np.abs(now - before), n).sum(axis=(2, 3)) / (n * n)
    rows, cols = moved_error.shape
    vectors = vectors[:rows, :cols].astype(np.float32)
    # A vector only counts if it explains the block clearly better than standing
    # still; on flat or noisy areas block matching picks random vectors.
    useful = (still_error - moved_error >= USEFUL_GAIN).astype(np.float32)
    vectors = vectors * useful[..., None]
    error = moved_error * useful + still_error * (1.0 - useful)
    residual = np.maximum(error - NOISE_FLOOR, 0.0)
    return vectors, residual


def _shift_blocks(grid: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Backward warp of a rows x cols grid of coefficient blocks:
    out(r, c) = grid(r + dy, c + dx), with (dx, dy) = moves[r, c] in blocks,
    mixing the four nearest blocks' coefficients (bilinear)."""
    rows, cols = grid.shape[:2]
    ys = np.clip(np.arange(rows, dtype=np.float32)[:, None] + moves[..., 1], 0, rows - 1)
    xs = np.clip(np.arange(cols, dtype=np.float32)[None, :] + moves[..., 0], 0, cols - 1)
    y0, x0 = ys.astype(np.intp), xs.astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, cols - 1)
    wy, wx = (ys - y0)[..., None, None], (xs - x0)[..., None, None]
    upper = grid[y0, x0] * (1 - wx) + grid[y0, x1] * wx
    lower = grid[y1, x0] * (1 - wx) + grid[y1, x1] * wx
    return (upper * (1 - wy) + lower * wy).astype(np.float32)


def _refresh_turns(busy: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    """When every block switches in the final refresh, 0..1: the activity is let
    seep outwards over the grid (12 rounds of 3 x 3 averaging), so every block
    learns how close it is to where B moved; then the blocks are ranked, most
    active first, and turn = rank / number of blocks, so the same number of
    blocks switch every frame. A tiny random jitter breaks ties in calm areas."""
    near = busy.astype(np.float32)
    for _ in range(12):
        near = _blur3(near)
    near = near + 1e-4 * float(near.max() + 1.0) * jitter
    values = near.ravel().tolist()
    order = sorted(range(len(values)), key=lambda k: -values[k])
    turns = np.zeros(len(values), np.float32)
    for rank, k in enumerate(order):
        turns[k] = rank / len(values)
    return turns.reshape(busy.shape)


def _blur3(values: np.ndarray) -> np.ndarray:
    """Mean of each 3 x 3 neighbourhood of a rows x cols grid, edges repeated."""
    rows, cols = values.shape
    padded = values[np.clip(np.arange(-1, rows + 1), 0, rows - 1)][:, np.clip(np.arange(-1, cols + 1), 0, cols - 1)]
    total = np.zeros_like(values, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy:dy + rows, dx:dx + cols]
    return total / 9.0


def _smooth(x):
    """0 below 0, 1 above 1, and the smoothstep x * x * (3 - 2x) in between."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _dct_matrix(n: int) -> np.ndarray:
    """The n x n DCT-II matrix: C[u, x] = a(u) * cos((2x + 1) * u * pi / 2n),
    a(0) = sqrt(1/n), a(u) = sqrt(2/n). It is orthonormal, so its inverse is C^T."""
    rows = []
    for u in range(n):
        scale = math.sqrt((1.0 if u == 0 else 2.0) / n)
        rows.append([scale * math.cos((2 * x + 1) * u * math.pi / (2 * n)) for x in range(n)])
    return np.array(rows, np.float32)


def _table(table: np.ndarray, n: int) -> np.ndarray:
    """The 8 x 8 JPEG table stretched or shrunk to n x n by nearest entry."""
    idx = np.arange(n) * 8 // n
    return table[idx[:, None], idx[None, :]]


def _blocks(plane: np.ndarray, n: int) -> np.ndarray:
    """H x W plane as rows x cols x n x n blocks; the edge pixels are repeated to fill
    the last row and column of blocks."""
    height, width = plane.shape
    rows, cols = -(-height // n), -(-width // n)
    ys = np.minimum(np.arange(rows * n), height - 1)
    xs = np.minimum(np.arange(cols * n), width - 1)
    padded = plane[ys[:, None], xs[None, :]]
    return padded.reshape(rows, n, cols, n).transpose(0, 2, 1, 3)


def _times(left: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """left . block for every block: out[.., u, y] = sum_x left[u, x] * block[.., x, y],
    added up one x at a time so large frames need little memory."""
    out = np.zeros(blocks.shape[:2] + (left.shape[0], blocks.shape[3]), np.float32)
    for x in range(left.shape[1]):
        out += left[None, None, :, x, None] * blocks[:, :, None, x, :]
    return out


def _dct(plane: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """F = C . f . C^T for every block of the plane."""
    blocks = _blocks(plane, basis.shape[0])
    rows_done = _times(basis, blocks)                                 # C . f
    return _times(basis, rows_done.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)   # (C . (C . f)^T)^T


def _idct(coeff: np.ndarray, basis: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """f = C^T . F . C for every block, put back together and cut to shape."""
    inverse = basis.T.copy()
    rows_done = _times(inverse, coeff)
    blocks = _times(inverse, rows_done.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    rows, cols, n = blocks.shape[0], blocks.shape[1], blocks.shape[2]
    plane = blocks.transpose(0, 2, 1, 3).reshape(rows * n, cols * n)
    return plane[:shape[0], :shape[1]]


def _round(values: np.ndarray) -> np.ndarray:
    """Round half away from zero, by hand: sign * floor(|v| + 0.5)."""
    sign = (values >= 0).astype(np.float32) * 2.0 - 1.0
    return sign * (np.abs(values) + 0.5).astype(np.int32).astype(np.float32)


def _to_ycbcr(bgr: np.ndarray) -> list[np.ndarray]:
    """BT.601 full range: Y, Cb, Cr planes from a float BGR image."""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return [y, cb, cr]


def _to_bgr(planes: list[np.ndarray]) -> np.ndarray:
    """Inverse of _to_ycbcr, as an H x W x 3 float BGR image."""
    y, cb, cr = planes[0], planes[1] - 128.0, planes[2] - 128.0
    out = np.zeros(y.shape + (3,), np.float32)
    out[..., 0] = y + 1.772 * cb
    out[..., 1] = y - 0.344136 * cb - 0.714136 * cr
    out[..., 2] = y + 1.402 * cr
    return out
