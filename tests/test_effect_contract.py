"""The contract every effect must follow. It runs on every effect automatically,
so a new effect is covered as soon as its file exists in moshflow/effects/."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import ArraySource, FakeProgress, moving_clip
from moshflow.api import TransitionContext, all_effects, default_params
from moshflow.core.discovery import discover_effects

failures = discover_effects()
EFFECTS = all_effects()
EFFECT_IDS = [cls.info.id for cls in EFFECTS]

# (frames of A in the window, frames of B in the window, length of each clip)
WINDOWS = {
    "center": (5, 5, 20),
    "before": (6, 0, 20),
    "after": (0, 6, 20),
    "whole_clips": (6, 6, 6),   # the transition uses both clips completely
    "one_frame": (0, 1, 20),
}


def make_context(a_len, b_len, clip_len, seed=7, params=None, cls=None):
    a = ArraySource(moving_clip(clip_len, seed=1))
    b = ArraySource(moving_clip(clip_len, seed=2))
    return TransitionContext(
        outgoing=a, incoming=b,
        a_window=range(clip_len - a_len, clip_len),
        b_window=range(0, b_len),
        params=params if params is not None else default_params(cls.param_specs()),
        seed=seed,
    )


def test_all_effect_modules_load():
    assert not failures, "\n".join(f"{f.module}\n{f.error}" for f in failures)


def test_reference_effects_exist():
    assert {"hard_cut", "crossfade"} <= set(EFFECT_IDS)


@pytest.mark.parametrize("cls", EFFECTS, ids=EFFECT_IDS)
def test_info_and_params_are_valid(cls):
    assert cls.info.id and cls.info.name and cls.info.author and cls.info.description
    ids = [spec.id for spec in cls.param_specs()]
    assert len(ids) == len(set(ids)), "parameter ids must be unique"


@pytest.mark.parametrize("window", WINDOWS.values(), ids=WINDOWS.keys())
@pytest.mark.parametrize("cls", EFFECTS, ids=EFFECT_IDS)
def test_yields_exact_frames_of_the_right_shape(cls, window):
    ctx = make_context(*window, cls=cls)
    frames = list(cls().render(ctx, FakeProgress()))
    assert len(frames) == ctx.length
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (ctx.size[1], ctx.size[0], 3)
        assert frame.dtype == np.uint8


@pytest.mark.parametrize("cls", EFFECTS, ids=EFFECT_IDS)
def test_same_seed_gives_the_same_frames(cls):
    first = list(cls().render(make_context(5, 5, 20, cls=cls), FakeProgress()))
    second = list(cls().render(make_context(5, 5, 20, cls=cls), FakeProgress()))
    for x, y in zip(first, second):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("cls", EFFECTS, ids=EFFECT_IDS)
def test_stops_when_cancelled(cls):
    ctx = make_context(5, 5, 20, cls=cls)
    assert list(cls().render(ctx, FakeProgress(cancelled=True))) == []
