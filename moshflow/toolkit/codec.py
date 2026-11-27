"""Codec-style P-frame and coordinate-map helpers, written with plain NumPy arithmetic.

Provides the shared building blocks for authentic datamoshing without alpha cross-fades
or repeated bilinear blur:

- Hierarchical (coarse-to-fine) block matching with 5-neighbour median refinement,
  so fast motion (up to +-24..32 px/frame) is tracked without full-search cost.
- Xvid-style macroblock classification (SKIP / INTER / INTRA) and H.263 dead-zone
  quantized residuals, so clip A's colours survive while clip B's edges and new
  content break through as clustered blocks rather than a fade.
- Coordinate-map advection (transflow style) with hole tracking (Careaga/Aksoy style),
  so the source frame is sampled once per output frame with no accumulated blur.
"""

from dataclasses import dataclass

import numpy as np

from moshflow.toolkit import color, motion, warp

MODE_SKIP = 0
MODE_INTER = 1
MODE_INTRA = 2


@dataclass(frozen=True)
class PFrame:
    """One predicted frame from `reference` to `current`.

    vectors:  (rows, cols, 2) int32 block motion vectors, zero on SKIP blocks.
    modes:    (rows, cols) int8 block types (MODE_SKIP, MODE_INTER, MODE_INTRA).
    residual: (H, W, 3) float32 dead-zone quantized colour correction for INTER blocks
              (zero on SKIP and INTRA blocks).
    mad:      (rows, cols) float32 per-block mean absolute luma prediction error.
    block:    block size in pixels.
    """

    vectors: np.ndarray
    modes: np.ndarray
    residual: np.ndarray
    mad: np.ndarray
    block: int


def downsample2(image: np.ndarray) -> np.ndarray:
    """2x2 box-average downsample of a 2D float32 luma image (pads odd edges by 1)."""
    height, width = image.shape
    pad_h = height + (height & 1)
    pad_w = width + (width & 1)
    if pad_h != height or pad_w != width:
        buf = np.empty((pad_h, pad_w), np.float32)
        buf[:height, :width] = image
        if pad_h > height:
            buf[height, :width] = image[height - 1]
        if pad_w > width:
            buf[:, width] = buf[:, width - 1]
    else:
        buf = image.astype(np.float32, copy=False)
    return 0.25 * (
        buf[0::2, 0::2] + buf[0::2, 1::2] + buf[1::2, 0::2] + buf[1::2, 1::2]
    )


def median_vectors(vectors: np.ndarray) -> np.ndarray:
    """5-neighbour (cross) median filter on (rows, cols, 2) int32 block vectors.

    Uses a pairwise min/max sorting network so no library filter is called. Removes
    isolated single-block outlier spikes while preserving sharp motion boundaries.
    """
    rows, cols = vectors.shape[:2]
    if rows < 2 or cols < 2:
        return vectors.copy()
    padded = np.empty((rows + 2, cols + 2, 2), vectors.dtype)
    padded[1:-1, 1:-1] = vectors
    padded[0, 1:-1] = vectors[0]
    padded[-1, 1:-1] = vectors[-1]
    padded[:, 0] = padded[:, 1]
    padded[:, -1] = padded[:, -2]

    a = padded[1:-1, 1:-1]
    b = padded[0:-2, 1:-1]
    c = padded[2:, 1:-1]
    d = padded[1:-1, 0:-2]
    e = padded[1:-1, 2:]

    # 5-element median via 6 compare-exchange steps (a,b), (c,d), min/max elimination.
    a, b = np.minimum(a, b), np.maximum(a, b)
    c, d = np.minimum(c, d), np.maximum(c, d)
    # Discard the smaller of the two pair minima, and the larger of the two pair maxima.
    low = np.maximum(a, c)
    high = np.minimum(b, d)
    # Median of the remaining three values (low, high, e).
    lo2 = np.minimum(low, high)
    hi2 = np.maximum(low, high)
    return np.minimum(np.maximum(lo2, e), hi2).astype(vectors.dtype)


def estimate_hierarchical(
    reference: np.ndarray,
    current: np.ndarray,
    block: int = 16,
    search: int = 24,
    smooth: bool = True,
) -> np.ndarray:
    """Coarse-to-fine block matching on two H x W float32 luma images.

    Covers +-search pixels by running full search on a downsampled pyramid level,
    then refining each block within +-2 pixels at each finer level. Falls back to
    `motion.estimate` on small frames or small search ranges.
    """
    ref = reference.astype(np.float32, copy=False)
    cur = current.astype(np.float32, copy=False)
    height, width = cur.shape
    eff_search = min(search, max(min(height, width) // 2, 1)) if search > 0 else 0
    if eff_search <= 8 or min(height, width) < block * 2 or block < 4:
        vectors = motion.estimate(ref, cur, block, eff_search)
        return median_vectors(vectors) if smooth else vectors

    # Build pyramid levels (up to 3 levels: full, 1/2, 1/4).
    refs = [ref]
    curs = [cur]
    blocks = [block]
    while len(refs) < 3 and blocks[-1] >= 4 and min(refs[-1].shape) >= blocks[-1] * 2:
        next_search = -(-eff_search // (1 << len(refs)))
        refs.append(downsample2(refs[-1]))
        curs.append(downsample2(curs[-1]))
        blocks.append(blocks[-1] // 2)
        if next_search <= 6:
            break

    top = len(refs) - 1
    scale = 1 << top
    coarse_search = max(-(-eff_search // scale), 2)
    vectors = motion.estimate(refs[top], curs[top], blocks[top], coarse_search)
    if smooth:
        vectors = median_vectors(vectors)

    for level in range(top - 1, -1, -1):
        lvl_ref, lvl_cur, lvl_block = refs[level], curs[level], blocks[level]
        lvl_h, lvl_w = lvl_cur.shape
        rows, cols = -(-lvl_h // lvl_block), -(-lvl_w // lvl_block)
        init = _upsample_vectors(vectors, rows, cols) * 2
        radius = 1 if top >= 2 else 2
        vectors = _refine_vectors(lvl_ref, lvl_cur, init, lvl_block, radius=radius, limit=eff_search)
        if smooth:
            vectors = median_vectors(vectors)
    return vectors


def _upsample_vectors(vectors: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Matches a coarse (cr, cc, 2) vector grid to finer (rows, cols, 2) shape."""
    cr, cc = vectors.shape[:2]
    ry = np.minimum(np.arange(rows), cr - 1)
    cx = np.minimum(np.arange(cols), cc - 1)
    return vectors[ry[:, None], cx[None, :]].astype(np.int32)


def _refine_vectors(
    reference: np.ndarray,
    current: np.ndarray,
    initial: np.ndarray,
    block: int,
    radius: int,
    limit: int,
) -> np.ndarray:
    """Searches (dx, dy) in initial[r, c] + [-radius, +radius]^2 for every block."""
    height, width = current.shape
    rows, cols = initial.shape[:2]
    step = 2 if block >= 8 else 1
    s_block = block // step

    cur_s = current[::step, ::step]
    sh, sw = cur_s.shape
    py = np.arange(0, height, step, dtype=np.int32)[:, None]
    px = np.arange(0, width, step, dtype=np.int32)[None, :]
    br = (py[:, 0] // block)[:, None]
    bc = (px[0, :] // block)[None, :]

    best = np.clip(initial, -limit, limit).astype(np.int32)
    best_sad = np.full((rows, cols), np.inf, np.float32)

    offsets = sorted(
        ((ox, oy) for oy in range(-radius, radius + 1) for ox in range(-radius, radius + 1)),
        key=lambda d: abs(d[0]) + abs(d[1]),
    )
    # Also test (0, 0) explicitly so stationary blocks can recover if coarse drifted.
    candidates: list[tuple[int, int] | None] = [None] + offsets

    diff_buf = np.zeros((rows * s_block, cols * s_block), np.float32)

    for cand in candidates:
        if cand is None:
            cand_vx = np.zeros((rows, cols), np.int32)
            cand_vy = np.zeros((rows, cols), np.int32)
        else:
            ox, oy = cand
            cand_vx = np.clip(initial[..., 0] + ox, -limit, limit)
            cand_vy = np.clip(initial[..., 1] + oy, -limit, limit)

        sx = px + cand_vx[br, bc]
        sy = py + cand_vy[br, bc]
        valid_px = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        sx_c = np.clip(sx, 0, width - 1)
        sy_c = np.clip(sy, 0, height - 1)

        diff = np.abs(cur_s - reference[sy_c, sx_c])
        # Penalize out-of-bounds pixels heavily so blocks stay inside the frame.
        diff = np.where(valid_px, diff, 255.0)
        diff_buf[:sh, :sw] = diff
        sad = diff_buf.reshape(rows, s_block, cols, s_block).sum(axis=(1, 3))

        better = sad < best_sad
        if better.any():
            best_sad[better] = sad[better]
            best[..., 0][better] = cand_vx[better]
            best[..., 1][better] = cand_vy[better]
    return best


def quantize_residual(
    error: np.ndarray,
    quant: float = 14.0,
    chroma_scale: float = 0.25,
) -> np.ndarray:
    """Dead-zone quantizes a BGR prediction residual (H x W x 3 float32).

    Matches MPEG-4 / H.263 inter quantization: small errors below 2.5 * quant become
    exactly zero so smooth areas keep clip A's colours untouched. Chroma differences
    are quantized more coarsely (`chroma_scale`), so B's brightness contours etch into
    A's colour palette instead of tinting A towards B.
    """
    if quant <= 0.0:
        return error.astype(np.float32, copy=True)

    err_y = 0.114 * error[..., 0] + 0.587 * error[..., 1] + 0.299 * error[..., 2]
    err_c = error - err_y[..., None]

    q_y = _deadzone(err_y, quant)
    if chroma_scale <= 0.0:
        return np.stack([q_y, q_y, q_y], axis=-1)
    q_c = _deadzone(err_c, quant / max(chroma_scale, 1e-3)) * chroma_scale
    return (q_y[..., None] + q_c).astype(np.float32)


def _deadzone(values: np.ndarray, q: float) -> np.ndarray:
    """H.263 inter dead-zone quantizer + inverse: dead zone is 2.5 * q, step is 2 * q."""
    mag = np.abs(values)
    level = np.floor(np.maximum(mag - 0.5 * q, 0.0) / (2.0 * q))
    recon = np.where(level > 0.0, level * (2.0 * q) + q, 0.0)
    return np.where(values < 0.0, -recon, recon).astype(np.float32)


def analyze_pframe(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    block: int = 16,
    search: int = 24,
    quant: float = 14.0,
    chroma_scale: float = 0.2,
    intra_thresh: float = 18.0,
    skip_thresh: float = 2.5,
    smooth_mv: bool = True,
) -> PFrame:
    """Encodes the P-frame turning `ref_bgr` into `cur_bgr` (Xvid-like gating).

    Per block:
    - Vectors that do not reduce block MAD below 85% of standing still (and by at least
      1.0 grey level) are reset to (0, 0).
    - SKIP:  mv == (0, 0) and still MAD < skip_thresh -> copies A untouched.
    - INTRA: moved MAD > intra_thresh and exceeds the block's own deviation from its
             mean (lowered by 2.5 per INTRA neighbour, so INTRA patches cluster).
    - INTER: warps by mv and adds the dead-zone quantized residual.
    """
    height, width = cur_bgr.shape[:2]
    ref_y = color.bgr_to_y(ref_bgr)
    cur_y = color.bgr_to_y(cur_bgr)

    vectors = estimate_hierarchical(ref_y, cur_y, block=block, search=search, smooth=smooth_mv)
    field_raw = motion.to_field(vectors, (width, height), block)
    pred_y = warp.warp(ref_y, field_raw, "nearest")

    mad_moved = block_mean(np.abs(cur_y - pred_y), block)
    mad_still = block_mean(np.abs(cur_y - ref_y), block)

    # Drop vectors that barely beat standing still.
    useful = (mad_moved < 0.85 * mad_still) & ((mad_still - mad_moved) >= 1.0)
    vectors = vectors.copy()
    vectors[~useful] = 0
    mad = np.where(useful, mad_moved, mad_still)

    # Block deviation from its own mean (Xvid's dev16, normalized per pixel).
    mean_y = block_mean(cur_y, block)
    mean_px = expand_blocks(mean_y, (width, height), block)
    dev = block_mean(np.abs(cur_y - mean_px), block)

    rows, cols = vectors.shape[:2]
    still = (vectors[..., 0] == 0) & (vectors[..., 1] == 0)
    modes = np.full((rows, cols), MODE_INTER, np.int8)
    modes[still & (mad < skip_thresh)] = MODE_SKIP

    # Xvid-style intra decision with neighbour clustering (left & top lower threshold).
    if intra_thresh < 255.0 and (mad > max(intra_thresh - 5.0, 4.0)).any():
        bias = max(intra_thresh * 0.25, 1.5)
        neighbor_drop = 2.5
        for r in range(rows):
            for c in range(cols):
                if modes[r, c] == MODE_SKIP:
                    continue
                n_intra = (1 if r > 0 and modes[r - 1, c] == MODE_INTRA else 0) + (
                    1 if c > 0 and modes[r, c - 1] == MODE_INTRA else 0
                )
                thresh = max(intra_thresh - neighbor_drop * n_intra, 4.0)
                if mad[r, c] > thresh and dev[r, c] < (mad[r, c] - bias + 0.5 * n_intra):
                    modes[r, c] = MODE_INTRA

    field = motion.to_field(vectors, (width, height), block)
    pred_bgr = warp.warp(ref_bgr, field, "nearest")
    raw_error = cur_bgr.astype(np.float32) - pred_bgr
    q_error = quantize_residual(raw_error, quant=quant, chroma_scale=chroma_scale)

    mode_px = expand_blocks(modes, (width, height), block)
    residual = np.where((mode_px == MODE_INTER)[..., None], q_error, 0.0).astype(np.float32)
    return PFrame(vectors=vectors, modes=modes, residual=residual, mad=mad, block=block)


def decode_pframe(
    picture: np.ndarray,
    pframe: PFrame,
    current_b: np.ndarray,
    gain: float = 1.0,
    residual_scale: float = 1.0,
    sampling: str = "nearest",
    apply_intra: bool = True,
    smooth_field: float = 0.0,
) -> np.ndarray:
    """Decodes one P-frame onto `picture` (H x W x 3 float32), returning the new buffer."""
    height, width = picture.shape[:2]
    if smooth_field > 0.0:
        field = motion.smooth_field(pframe.vectors, (width, height), pframe.block, smooth_field)
    else:
        field = motion.to_field(pframe.vectors, (width, height), pframe.block)
    warped = warp.warp(picture, gain * field, sampling=sampling)
    out = np.clip(warped + residual_scale * pframe.residual, 0.0, 255.0)
    if apply_intra and (pframe.modes == MODE_INTRA).any():
        intra_px = expand_blocks(pframe.modes == MODE_INTRA, (width, height), pframe.block)
        out = np.where(intra_px[..., None], current_b.astype(np.float32), out)
    return out.astype(np.float32)


def identity_coords(size: tuple[int, int]) -> np.ndarray:
    """Initial (H, W, 2) float32 coordinate map where coords[y, x] = (x, y)."""
    width, height = size
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    coords = np.empty((height, width, 2), np.float32)
    coords[..., 0] = xs
    coords[..., 1] = ys
    return coords


def advect_coords(
    coords: np.ndarray,
    field: np.ndarray,
    mask: np.ndarray | None = None,
    sampling: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """Advects a coordinate map backward along `field` (transflow style).

    Returns (new_coords, valid_mask) where valid_mask[y, x] is False wherever the
    lookup stepped outside the image or landed on a previous hole (Careaga/Aksoy).
    """
    height, width = coords.shape[:2]
    xs = np.arange(width, dtype=np.float32)[None, :] + field[..., 0]
    ys = np.arange(height, dtype=np.float32)[:, None] + field[..., 1]
    in_bounds = (xs >= 0.0) & (xs <= width - 1) & (ys >= 0.0) & (ys <= height - 1)

    new_coords = warp.warp(coords, field, sampling=sampling)
    if mask is None:
        return new_coords, in_bounds
    warped_mask = warp.warp(mask.astype(np.float32), field, sampling="nearest") > 0.5
    return new_coords, in_bounds & warped_mask


def sample_coords(image: np.ndarray, coords: np.ndarray, sampling: str = "nearest") -> np.ndarray:
    """Samples `image` once at the accumulated (x, y) positions in `coords`."""
    height, width = image.shape[:2]
    source = image.astype(np.float32)
    xs = np.clip(coords[..., 0], 0.0, width - 1)
    ys = np.clip(coords[..., 1], 0.0, height - 1)
    if sampling == "nearest":
        return source[(ys + 0.5).astype(np.intp), (xs + 0.5).astype(np.intp)]
    if sampling != "bilinear":
        raise ValueError(f"sampling must be 'nearest' or 'bilinear', got {sampling!r}")
    x0, y0 = xs.astype(np.intp), ys.astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = xs - x0, ys - y0
    if source.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
    bot = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def cluster_grow(mask: np.ndarray, scores: np.ndarray, target_fraction: float) -> np.ndarray:
    """Grows an existing boolean INTRA block mask organically from its borders & high-score blocks.

    Blocks adjacent to an already-INTRA block receive a priority boost, producing
    clustered codec-like rectangular patches rather than uniform random confetti.
    """
    rows, cols = mask.shape
    total = rows * cols
    want = int(round(np.clip(target_fraction, 0.0, 1.0) * total))
    if want <= 0:
        return mask.copy()
    if want >= total:
        return np.ones((rows, cols), bool)

    out = mask.copy()
    if int(out.sum()) >= want:
        return out

    # Neighbour bonus: blocks touching an active INTRA block grow first.
    adj = np.zeros((rows, cols), np.float32)
    if rows > 1:
        adj[1:, :] += out[:-1, :]
        adj[:-1, :] += out[1:, :]
    if cols > 1:
        adj[:, 1:] += out[:, :-1]
        adj[:, :-1] += out[:, 1:]

    priority = scores.astype(np.float32) + 0.45 * adj
    active = ~out
    vals = priority[active]
    need = want - int(out.sum())
    if need >= vals.size:
        return np.ones((rows, cols), bool)

    # Find threshold via binary search using only min/max/sum (no library sort).
    lo = float(vals.min())
    hi = float(vals.max())
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if int((vals >= mid).sum()) >= need:
            lo = mid
        else:
            hi = mid
    picked = active & (priority >= lo)
    out = out | picked
    return out


def expand_blocks(grid: np.ndarray, size: tuple[int, int], block: int) -> np.ndarray:
    """Expands a (rows, cols) block array to (H, W) by repeating each block."""
    width, height = size
    ry = np.arange(height) // block
    cx = np.arange(width) // block
    return grid[ry[:, None], cx[None, :]]


def block_mean(values: np.ndarray, block: int) -> np.ndarray:
    """Mean of an H x W array over each block; partial edge blocks average their own pixels."""
    height, width = values.shape
    rows, cols = -(-height // block), -(-width // block)
    padded = np.zeros((rows * block, cols * block), np.float32)
    counts = np.zeros_like(padded)
    padded[:height, :width] = values
    counts[:height, :width] = 1.0
    sums = padded.reshape(rows, block, cols, block).sum(axis=(1, 3))
    return sums / counts.reshape(rows, block, cols, block).sum(axis=(1, 3))
