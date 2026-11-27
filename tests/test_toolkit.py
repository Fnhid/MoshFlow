"""The expected behaviour of the shared toolkit helpers.

A helper that raises NotImplementedError is skipped rather than failed, so a new
helper can be added here before it is written.
"""

import numpy as np
import pytest

from helpers import ArraySource, moving_clip
from moshflow.api import TransitionContext
from moshflow.toolkit import codec, color, frames, motion, warp


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


def luma(frame):
    return color.bgr_to_y(frame)


# --- frames ------------------------------------------------------------------

def make_context(a_window, b_window, length=10):
    clip_a, clip_b = ArraySource(moving_clip(length, seed=1)), ArraySource(moving_clip(length, seed=2))
    return TransitionContext(clip_a, clip_b, a_window, b_window, params={}, seed=0)


def test_window_frames_walks_a_then_b():
    ctx = make_context(range(8, 10), range(0, 3))
    order = [(clip is ctx.outgoing, index) for clip, index in frames.window_frames(ctx)]
    assert order == [(True, 8), (True, 9), (False, 0), (False, 1), (False, 2)]


def test_last_clean_a_is_the_frame_before_the_window():
    ctx = make_context(range(8, 10), range(0, 3))
    assert frames.last_clean_a(ctx) is ctx.outgoing.frame(7)
    whole = make_context(range(0, 10), range(0, 3))
    assert frames.last_clean_a(whole) is whole.outgoing.frame(0)


# --- motion.estimate -----------------------------------------------------------

def test_estimate_shape_covers_partial_blocks():
    image = np.zeros((30, 50), np.float32)
    vectors = call(motion.estimate, image, image, 16, 4)
    assert vectors.shape == (2, 4, 2)          # ceil(30 / 16) rows, ceil(50 / 16) columns
    assert vectors.dtype == np.int32


def test_estimate_finds_a_shift_to_the_right():
    clip = moving_clip(4, size=(64, 48))       # the texture moves 2 pixels right per frame
    vectors = call(motion.estimate, luma(clip[0]), luma(clip[1]), 8, 4)
    # Each pixel came from 2 pixels to its left: v = (-2, 0). The first column of
    # blocks is left out, because the true source lies outside the image there.
    assert np.array_equal(vectors[:, 1:], np.tile(np.int32([-2, 0]), (6, 7, 1)))


def test_estimate_prefers_no_motion_on_flat_images():
    flat = np.full((32, 32), 128, np.float32)
    vectors = call(motion.estimate, flat, flat, 8, 3)
    assert not vectors.any()


def test_estimate_matches_the_slide_example():
    # The 2 x 2 example from the proposal: the block 50 60 / 70 80 moves from (1, 1) to (2, 2).
    reference = np.full((6, 6), 10, np.float32)
    current = np.full((6, 6), 10, np.float32)
    reference[1:3, 1:3] = [[50, 60], [70, 80]]
    current[2:4, 2:4] = [[50, 60], [70, 80]]
    vectors = call(motion.estimate, reference, current, 2, 1)
    assert tuple(vectors[1, 1]) == (-1, -1)


def _reference_sad_estimate(reference: np.ndarray, current: np.ndarray, block: int, search: int) -> np.ndarray:
    height, width = current.shape
    rows, cols = -(-height // block), -(-width // block)
    out = np.zeros((rows, cols, 2), np.int32)
    span = range(-search, search + 1)
    candidates = sorted(((dx, dy) for dy in span for dx in span), key=lambda d: abs(d[0]) + abs(d[1]))
    for r in range(rows):
        y0, y1 = r * block, min((r + 1) * block, height)
        for c in range(cols):
            x0, x1 = c * block, min((c + 1) * block, width)
            cur = current[y0:y1, x0:x1]
            best_sad = np.inf
            best_v = (0, 0)
            for dx, dy in candidates:
                if y0 + dy < 0 or y1 + dy > height or x0 + dx < 0 or x1 + dx > width:
                    continue
                sad = float(np.abs(cur - reference[y0 + dy:y1 + dy, x0 + dx:x1 + dx]).sum())
                if sad < best_sad:
                    best_sad = sad
                    best_v = (dx, dy)
            out[r, c] = best_v
    return out


@pytest.mark.parametrize("shape,block,search", [
    ((30, 50), 16, 6),      # partial bottom and right blocks
    ((10, 12), 16, 4),      # smaller than one block
    ((37, 41), 8, 10),      # search larger than block
    ((24, 24), 8, 0),       # zero search range
])
def test_estimate_matches_exact_full_search_on_edge_cases(shape, block, search):
    rng = np.random.default_rng(42)
    reference = rng.integers(0, 256, shape, dtype=np.uint8).astype(np.float32)
    current = np.roll(reference, (2, -3), axis=(0, 1))
    reference.flags.writeable = False
    current.flags.writeable = False
    got = call(motion.estimate, reference, current, block, search)
    expected = _reference_sad_estimate(reference, current, block, search)
    assert np.array_equal(got, expected)


def test_estimate_threaded_path_matches_single_band():
    # Large enough to exercise the threaded band path, with a non-divisible height.
    rng = np.random.default_rng(11)
    reference = rng.integers(0, 256, (180, 260), dtype=np.uint8).astype(np.float32)
    current = np.roll(reference, (-4, 5), axis=(0, 1))
    got = call(motion.estimate, reference, current, 16, 6)
    expected = _reference_sad_estimate(reference, current, 16, 6)
    assert np.array_equal(got, expected)


# --- motion.to_field / smooth_field -------------------------------------------

def test_to_field_gives_every_pixel_its_block_vector():
    vectors = np.int32([[[1, 0], [2, 0]], [[3, 0], [4, 0]]])     # 2 x 2 blocks of 2 x 2 pixels
    field = call(motion.to_field, vectors, (3, 3), 2)           # a 3 x 3 image: the last blocks are partial
    assert field.shape == (3, 3, 2) and field.dtype == np.float32
    assert field[0, 0, 0] == 1 and field[0, 2, 0] == 2 and field[2, 0, 0] == 3 and field[2, 2, 0] == 4


def test_smooth_field_with_no_smoothness_is_the_block_field():
    vectors = np.int32([[[1, 0], [5, -2]], [[0, 3], [2, 2]]])
    blocky = call(motion.to_field, vectors, (8, 8), 4)
    smooth = call(motion.smooth_field, vectors, (8, 8), 4, 0.0)
    assert np.array_equal(smooth, blocky)


def test_smooth_field_has_no_jump_at_block_edges():
    vectors = np.int32([[[0, 0], [8, 0]]])                       # one row of two blocks
    smooth = call(motion.smooth_field, vectors, (16, 8), 8, 1.0)
    steps = np.abs(np.diff(smooth[4, :, 0]))
    assert steps.max() < 8                                       # the block field jumps by 8 at x = 8


# --- warp ----------------------------------------------------------------------

def test_warp_reads_from_x_plus_v_and_clamps_at_the_edge():
    image = np.float32([[10, 20, 30, 40]])
    field = np.zeros((1, 4, 2), np.float32)
    field[..., 0] = 1
    out = call(warp.warp, image, field)
    assert out.dtype == np.float32
    assert out.tolist() == [[20, 30, 40, 40]]


def test_warp_bilinear_mixes_neighbours():
    image = np.float32([[10, 20, 30, 40]])
    field = np.zeros((1, 4, 2), np.float32)
    field[..., 0] = 0.5
    out = call(warp.warp, image, field, sampling="bilinear")
    assert np.allclose(out, [[15, 25, 35, 40]])


def test_warp_keeps_colour_channels_and_the_input():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    image.flags.writeable = False
    field = np.zeros((4, 5, 2), np.float32)
    field[..., 1] = -1
    out = call(warp.warp, image, field)
    assert out.shape == (4, 5, 3)
    assert np.array_equal(out[1:], image[:-1]) and np.array_equal(out[0], image[0])


# --- codec ---------------------------------------------------------------------

def test_hierarchical_estimate_recovers_large_shift():
    yy, xx = np.mgrid[0:128, 0:160].astype(np.float32)
    ref = (
        128.0
        + 60.0 * np.sin(xx * 0.13 + yy * 0.07)
        + 45.0 * np.cos(xx * 0.05 - yy * 0.17)
        + 20.0 * np.sin(xx * 0.31)
    ).astype(np.float32)
    cur = np.roll(ref, (10, 14), axis=(0, 1))
    vectors = codec.estimate_hierarchical(ref, cur, block=16, search=24)
    # Interior blocks (away from the wrapped border) should recover (-14, -10).
    interior = vectors[2:-1, 2:-1]
    assert np.all(interior[..., 0] == -14)
    assert np.all(interior[..., 1] == -10)


def test_median_vectors_removes_single_spike_and_keeps_step():
    grid = np.zeros((5, 6, 2), np.int32)
    grid[:, 3:, 0] = 10
    grid[2, 1, 0] = 25  # isolated outlier spike on the left half
    cleaned = codec.median_vectors(grid)
    assert cleaned[2, 1, 0] == 0
    assert np.all(cleaned[:, :3, 0] == 0)
    assert np.all(cleaned[:, 3:, 0] == 10)


def test_quantize_residual_zeroes_deadzone_and_suppresses_chroma():
    err = np.zeros((4, 4, 3), np.float32)
    err[0, 0] = [10.0, 10.0, 10.0]         # luma 10 < 2.5 * 12 = 30 -> dead zone
    err[1, 1] = [60.0, 60.0, 60.0]         # pure luma 60 -> survives
    err[2, 2] = [80.0, -20.0, 0.0]         # strong chroma, near-zero luma
    q = codec.quantize_residual(err, quant=12.0, chroma_scale=0.1)
    assert np.all(q[0, 0] == 0.0)
    assert np.all(q[1, 1] > 40.0)
    # Chroma-only residual should be strongly suppressed relative to raw error.
    assert np.abs(q[2, 2]).max() < 15.0


def test_analyze_pframe_classifies_skip_inter_and_intra():
    rng = np.random.default_rng(3)
    ref = rng.integers(30, 220, (64, 64, 3), dtype=np.uint8)
    cur = ref.copy()
    # Replace bottom-right 32x32 with a brand-new flat bright patch (not in ref -> INTRA).
    cur[32:, 32:] = np.uint8([250, 250, 250])
    pf = codec.analyze_pframe(ref, cur, block=16, search=8, quant=12.0, intra_thresh=18.0)
    assert np.all(pf.modes[:2, :2] == codec.MODE_SKIP)
    assert np.any(pf.modes[2:, 2:] == codec.MODE_INTRA)
    # SKIP and INTRA blocks must carry zero residual.
    assert not pf.residual[:32, :32].any()


def test_advect_and_sample_coords_tracks_holes_without_blur():
    img = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    coords = codec.identity_coords((10, 8))
    field = np.zeros((8, 10, 2), np.float32)
    field[..., 0] = -2.0  # read 2 pixels left -> left 2 columns step out of bounds
    coords, valid = codec.advect_coords(coords, field)
    assert not valid[:, :2].any()
    assert valid[:, 2:].all()
    sampled = codec.sample_coords(img, coords, sampling="nearest")
    assert np.array_equal(sampled[:, 2:], img[:, :-2])


def test_cluster_grow_expands_from_existing_seed():
    mask = np.zeros((4, 4), bool)
    mask[1, 1] = True
    scores = np.zeros((4, 4), np.float32)
    grown = codec.cluster_grow(mask, scores, target_fraction=0.3)
    assert grown[1, 1]
    # Newly activated blocks should touch (1, 1) since scores are uniform.
    neighbours = [(0, 1), (2, 1), (1, 0), (1, 2)]
    assert any(grown[r, c] for r, c in neighbours)
    assert not grown[3, 3]

