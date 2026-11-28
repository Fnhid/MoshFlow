"""End-to-end: real video files in, a real video file out."""

from __future__ import annotations

import os

import numpy as np
import pytest

from moshflow.api import EffectInfo
from moshflow.app.render_job import JobProgress, render_project
from moshflow.app.store import load_project, save_project
from moshflow.cli import main as cli_main
from moshflow.core.project import Clip, OutputSettings, Project, Transition
from moshflow.core.renderer import EffectError, RenderCancelled, Renderer
from moshflow.core.timeline import ClipNode, RenderGraph, TransitionNode
from moshflow.videoio import OpenCVReader, OpenCVWriter, probe


def make_video(path, value, frames=30, size=(64, 48), fps=10.0):
    with OpenCVWriter(str(path), size, fps) as writer:
        for _ in range(frames):
            writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    return str(path)


@pytest.fixture
def clips(tmp_path):
    return make_video(tmp_path / "dark.mp4", 40), make_video(tmp_path / "bright.mp4", 200)


def two_clip_project(clips, output, effect="crossfade", **kwargs):
    return (Project(output=OutputSettings(path=str(output), **kwargs))
            .with_clip(Clip(clips[0]))
            .with_clip(Clip(clips[1]))
            .with_transition(0, Transition(effect, 1.0, "center")))


def test_probe_reads_video_properties(clips):
    info = probe(clips[0])
    assert (info.width, info.height, info.fps, info.frame_count) == (64, 48, 10.0, 30)


def test_crossfade_end_to_end(clips, tmp_path):
    output = tmp_path / "out.mp4"
    result = render_project(two_clip_project(clips, output))
    assert result.frames == 60
    info = probe(str(output))
    assert (info.width, info.height, info.frame_count) == (64, 48, 60)
    with OpenCVReader(str(output)) as reader:
        means = [float(reader.read(i).mean()) for i in (0, 29, 59)]
    # mp4v is lossy: these clips were encoded twice, which shifts bright values by up to ~5.
    assert means[0] == pytest.approx(40, abs=6)      # still clip A
    assert 40 + 20 < means[1] < 200 - 20             # mid-transition mixes both
    assert means[2] == pytest.approx(200, abs=6)     # clip B


def test_clips_are_conformed_to_the_output_format(tmp_path):
    a = make_video(tmp_path / "a.mp4", 50, frames=20, size=(64, 48), fps=10.0)
    b = make_video(tmp_path / "b.mp4", 90, frames=40, size=(32, 32), fps=20.0)
    output = tmp_path / "out.mp4"
    result = render_project(two_clip_project((a, b), output, width=40, height=30, fps=10.0))
    assert result.size == (40, 30)
    assert result.frames == 40                       # 2 s of each at 10 fps
    assert probe(str(output)).frame_count == 40


def test_trim_shortens_a_clip(clips, tmp_path):
    project = two_clip_project(clips, tmp_path / "out.mp4", effect="hard_cut")
    project = (project.with_clip_replaced(0, Clip(clips[0], start=1.0))
               .with_clip_replaced(1, Clip(clips[1], end=2.0)))
    assert render_project(project).frames == 20 + 20


def test_cancel_removes_the_partial_file(clips, tmp_path):
    output = tmp_path / "out.mp4"
    progress = JobProgress()
    progress.cancel()
    with pytest.raises(RenderCancelled):
        render_project(two_clip_project(clips, output), progress)
    assert not os.path.exists(output)


def test_missing_input_is_an_error(tmp_path):
    project = two_clip_project((str(tmp_path / "nope.mp4"), str(tmp_path / "nope2.mp4")), tmp_path / "o.mp4")
    with pytest.raises(IOError):
        render_project(project)


def test_project_file_round_trip_with_relative_paths(clips, tmp_path):
    project = two_clip_project((os.path.basename(clips[0]), os.path.basename(clips[1])), "out.mp4")
    path = tmp_path / "demo.moshproj"
    save_project(project, str(path))
    loaded = load_project(str(path))
    assert loaded.clips[0].path == clips[0]
    assert render_project(loaded).path == str(tmp_path / "out.mp4")


def test_cli_try_and_effects(clips, tmp_path, capsys):
    assert cli_main(["effects"]) == 0
    assert "crossfade" in capsys.readouterr().out
    output = tmp_path / "t.mp4"
    assert cli_main(["try", "--effect", "crossfade", "--param", "ease=2", clips[0], clips[1], "-o", str(output)]) == 0
    assert output.exists()
    assert cli_main(["try", "--effect", "no_such", clips[0], clips[1], "-o", str(tmp_path / "x.mp4")]) == 1
    assert cli_main(["try", "--effect", "crossfade", "--param", "ease=99", clips[0], clips[1],
                     "-o", str(tmp_path / "y.mp4")]) == 1


class _BadEffect:
    """Stands in for a broken plug-in."""
    info = EffectInfo("bad", "Bad", "test", "test")
    frames = 3
    shape = (4, 4, 3)
    dtype = np.uint8

    @classmethod
    def param_specs(cls):
        return []

    def render(self, ctx, progress):
        for _ in range(self.frames):
            yield np.zeros(self.shape, dtype=self.dtype)


class _Source:
    fps, size = 10.0, (4, 4)

    def __len__(self):
        return 4

    def frame(self, index):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class _Sink:
    def write(self, frame):
        pass

    def close(self):
        pass


@pytest.mark.parametrize("attrs, message", [({"frames": 1}, "yielded 1 frames"),
                                            ({"frames": 9}, "more than"),
                                            ({"shape": (2, 2, 3)}, "expected"),
                                            ({"dtype": np.float32}, "expected")])
def test_renderer_catches_contract_violations(attrs, message):
    effect = type("Broken", (_BadEffect,), attrs)
    node = TransitionNode(0, 0, 1, range(2, 4), range(0, 1), Transition("bad"))
    renderer = Renderer(RenderGraph([ClipNode(0, range(0, 2)), node]), [_Source(), _Source()], lambda _: effect)
    with pytest.raises(EffectError, match=message):
        renderer.render(_Sink())
