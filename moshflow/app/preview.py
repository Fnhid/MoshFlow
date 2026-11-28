"""Transition preview rendering at a reduced size for interactive playback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moshflow.app.render_job import JobProgress
from moshflow.core.conform import ConformedSource
from moshflow.core.discovery import discover_effects
from moshflow.core.project import OutputSettings, Project
from moshflow.core.renderer import Renderer
from moshflow.core.timeline import TimelineError, resolve
from moshflow.videoio import OpenCVReader

PREVIEW_MAX_WIDTH = 480
PREVIEW_MAX_HEIGHT = 270
LEAD_SECONDS = 0.25


@dataclass(frozen=True)
class PreviewClip:
    frames: tuple[np.ndarray, ...]
    fps: float
    size: tuple[int, int]


class _FrameCollector:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())


def _preview_size(width: int, height: int) -> tuple[int, int]:
    scale = min(PREVIEW_MAX_WIDTH / max(width, 1), PREVIEW_MAX_HEIGHT / max(height, 1), 1.0)
    out_w = max(2, int(round(width * scale)) & ~1)
    out_h = max(2, int(round(height * scale)) & ~1)
    return out_w, out_h


def render_transition_preview(
    project: Project,
    cut_index: int,
    progress: JobProgress | None = None,
) -> PreviewClip:
    """Renders a short preview around transition `cut_index` at preview resolution."""
    if not (0 <= cut_index < len(project.transitions)):
        raise IndexError(f"transition index out of range: {cut_index}")

    clip_a = project.clips[cut_index]
    clip_b = project.clips[cut_index + 1]
    transition = project.transitions[cut_index]

    discover_effects()
    reader_a = OpenCVReader(clip_a.path)
    reader_b = None
    try:
        reader_b = OpenCVReader(clip_b.path)
        out = project.output
        base_w = out.width or reader_a.info.width
        base_h = out.height or reader_a.info.height
        fps = out.fps or reader_a.info.fps
        size = _preview_size(base_w, base_h)

        # Keep enough frames around the cut for the transition plus a brief lead-in/out.
        a_total = reader_a.info.duration if clip_a.end is None else min(clip_a.end, reader_a.info.duration)
        b_total = reader_b.info.duration if clip_b.end is None else min(clip_b.end, reader_b.info.duration)
        a_start = max(clip_a.start, a_total - (transition.duration + LEAD_SECONDS))
        b_end = min(b_total, clip_b.start + transition.duration + LEAD_SECONDS)

        mini = Project(
            clips=(
                type(clip_a)(clip_a.path, start=a_start, end=a_total),
                type(clip_b)(clip_b.path, start=clip_b.start, end=b_end),
            ),
            transitions=(transition,),
            output=OutputSettings(path="preview.mp4", width=size[0], height=size[1], fps=fps),
        )
        sources = [
            ConformedSource(reader_a, size, fps, a_start, a_total),
            ConformedSource(reader_b, size, fps, clip_b.start, b_end),
        ]
        graph = resolve(mini, sources)
        collector = _FrameCollector()
        Renderer(graph, sources).render(collector, progress)
        if not collector.frames:
            raise TimelineError("preview produced no frames")
        return PreviewClip(tuple(collector.frames), fps, size)
    finally:
        reader_a.close()
        if reader_b is not None:
            reader_b.close()
