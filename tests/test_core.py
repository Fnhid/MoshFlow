"""Parameters, toolkit and timeline."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from helpers import ArraySource, moving_clip
from moshflow.api import BoolParam, ChoiceParam, FloatParam, IntParam, validate_params
from moshflow.app.store import project_from_dict, project_to_dict
from moshflow.core.discovery import discover_effects
from moshflow.core.project import Clip, Project, Transition
from moshflow.core.timeline import ClipNode, TransitionNode, resolve
from moshflow.toolkit import color, resize

SPECS = [IntParam("n", "N", 4, 1, 8), FloatParam("x", "X", 0.5, 0.0, 1.0),
         ChoiceParam("c", "C", "a", ("a", "b")), BoolParam("b", "B", False)]


def test_params_fill_defaults_and_convert():
    assert validate_params(SPECS, {"n": 2.0, "x": 1}) == {"n": 2, "x": 1.0, "c": "a", "b": False}


@pytest.mark.parametrize("values", [{"n": 9}, {"n": 1.5}, {"x": "big"}, {"c": "z"}, {"b": 1}, {"nope": 1}])
def test_params_reject_bad_values(values):
    with pytest.raises(ValueError):
        validate_params(SPECS, values)


def test_bilinear_keeps_a_constant_image_and_the_same_size():
    image = np.full((10, 20, 3), 77, dtype=np.uint8)
    assert resize.bilinear(image, (20, 10)) is image
    out = resize.bilinear(image, (7, 31))
    assert out.shape == (31, 7, 3) and out.dtype == np.uint8 and (out == 77).all()


def test_bilinear_interpolates_known_values():
    image = np.array([[[0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
    out = resize.bilinear(image, (4, 1))
    assert out[0, :, 0].tolist() == [0, 25, 75, 100]


def test_nearest_picks_existing_pixels():
    image = np.array([[[10, 10, 10], [200, 200, 200]]], dtype=np.uint8)
    out = resize.nearest(image, (4, 2))
    assert out.shape == (2, 4, 3) and out[0, :, 0].tolist() == [10, 10, 200, 200]


def test_luma_weights():
    frame = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert color.bgr_to_y(frame)[0].tolist() == pytest.approx([29.07, 149.685, 76.245], abs=0.01)


def test_to_uint8_rounds_and_saturates():
    assert color.to_uint8(np.array([-5.0, 1.6, 300.0])).tolist() == [0, 2, 255]


def graph_for(duration, alignment, lengths=(20, 20)):
    sources = [ArraySource(moving_clip(n)) for n in lengths]
    project = Project()
    for i in range(len(lengths)):
        project = project.with_clip(Clip(f"clip{i}.mp4"))
    for i in range(len(project.transitions)):
        project = project.with_transition(i, Transition(duration=duration, alignment=alignment))
    return resolve(project, sources)


@pytest.mark.parametrize("alignment, a_len, b_len", [("center", 5, 5), ("before", 10, 0), ("after", 0, 10)])
def test_alignment_windows(alignment, a_len, b_len):
    graph = graph_for(1.0, alignment)         # 1 s at 10 fps = 10 frames
    first, transition, last = graph.nodes
    assert isinstance(first, ClipNode) and isinstance(transition, TransitionNode) and isinstance(last, ClipNode)
    assert first.frames == range(0, 20 - a_len)
    assert transition.a_window == range(20 - a_len, 20)
    assert transition.b_window == range(0, b_len)
    assert last.frames == range(b_len, 20)
    assert graph.total_frames == 40            # a transition replaces frames, it adds none


def test_zero_length_transition_is_skipped():
    assert [type(n) for n in graph_for(0.0, "center").nodes] == [ClipNode, ClipNode]


def test_clip_fully_used_by_its_transitions_has_no_clip_node():
    graph = graph_for(1.0, "center", lengths=(20, 10, 20))
    assert [type(n) for n in graph.nodes] == [ClipNode, TransitionNode, TransitionNode, ClipNode]


def test_transitions_longer_than_a_clip_share_it():
    graph = graph_for(1.0, "center", lengths=(20, 4, 20))     # the middle clip needs 10, has 4
    first, into, out_of, last = graph.nodes
    assert into.b_window == range(0, 2) and out_of.a_window == range(2, 4)
    assert graph.total_frames == 20 + 4 + 20 - 5 - 5 + 5 + 5   # nothing added, nothing lost
    graph = graph_for(3.0, "after", lengths=(20, 12))          # needs 30 frames of B, has 12
    assert graph.nodes[-1].b_window == range(0, 12)


def test_project_needs_one_transition_per_cut():
    project = Project(clips=[Clip("a"), Clip("b")], transitions=[])
    with pytest.raises(ValueError):
        project.validate()


def test_project_add_remove_and_move():
    project = Project()
    for name in "abc":
        project = project.with_clip(Clip(name))
    assert len(project.transitions) == 2
    moved = project.with_clip_moved(2, 0)
    assert [c.path for c in moved.clips] == ["c", "a", "b"]
    removed = moved.without_clip(1)
    assert [c.path for c in removed.clips] == ["c", "b"] and len(removed.transitions) == 1
    removed.validate()
    assert [c.path for c in project.clips] == ["a", "b", "c"]      # the original is untouched


def test_project_add_at_an_index():
    project = Project().with_clip(Clip("a")).with_clip(Clip("b")).with_clip(Clip("c"), index=0)
    assert [c.path for c in project.clips] == ["c", "a", "b"] and len(project.transitions) == 2


def test_project_data_cannot_be_changed_in_place():
    transition = Transition("crossfade", params={"ease": 2.0})
    with pytest.raises(FrozenInstanceError):
        transition.duration = 3.0
    with pytest.raises(TypeError):
        transition.params["ease"] = 1.0
    assert transition.with_param("ease", 1.0).params == {"ease": 1.0} and transition.params == {"ease": 2.0}


def test_project_dict_round_trip():
    project = (Project().with_clip(Clip("a.mp4", start=1.0)).with_clip(Clip("b.mp4", end=3.0))
               .with_transition(0, Transition("crossfade", 2.0, "after", {"ease": 2.0}, 5)))
    data = json.loads(json.dumps(project_to_dict(project)))
    assert project_from_dict(data) == project


def test_newer_project_file_is_refused():
    with pytest.raises(ValueError, match="version"):
        project_from_dict({"version": 99})


def test_broken_effect_file_is_reported_and_does_not_stop_the_others(tmp_path, monkeypatch):
    package = tmp_path / "some_effects"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "good.py").write_text("LOADED = True\n")
    (package / "broken.py").write_text("raise RuntimeError('boom')\n")
    (package / "_skipped.py").write_text("raise RuntimeError('must not be imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    failures = discover_effects("some_effects")
    assert [f.module for f in failures] == ["some_effects.broken"]
    assert "boom" in failures[0].error


def test_new_transitions_play_clip_a_to_its_end():
    # The mosh runs over the start of clip B, so clip A is never cut short.
    assert Transition().alignment == "after"


def test_clips_with_another_aspect_ratio_get_black_bars():
    from moshflow.core.conform import fit
    square = np.full((30, 30, 3), 200, np.uint8)
    out = fit(square, (60, 30))                       # 1:1 into 2:1
    assert out.shape == (30, 60, 3)
    assert out[:, :15].max() == 0 and out[:, 45:].max() == 0   # bars left and right
    assert out[:, 15:45].min() == 200                          # the picture is not stretched
    assert fit(np.zeros((30, 60, 3), np.uint8), (40, 20)).shape == (20, 40, 3)
