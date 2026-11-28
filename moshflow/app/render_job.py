"""RenderJob: renders a whole project to a video file.

Used by the CLI directly and by the UI from a background thread.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

from moshflow.core.conform import ConformedSource
from moshflow.core.discovery import discover_effects
from moshflow.core.project import Project
from moshflow.core.renderer import Renderer
from moshflow.core.timeline import TimelineError, resolve
from moshflow.videoio import OpenCVReader, OpenCVWriter


class JobProgress:
    """Thread-safe ProgressReporter with a cancel switch and a callback."""

    def __init__(self, on_progress: Callable[[int, int], None] | None = None):
        self._cancel = threading.Event()
        self._on_progress = on_progress

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def report(self, done: int, total: int) -> None:
        if self._on_progress:
            self._on_progress(done, total)


@dataclass(frozen=True)
class RenderResult:
    path: str
    frames: int
    size: tuple[int, int]
    fps: float
    seconds: float


def render_project(project: Project, progress: JobProgress | None = None,
                   output_path: str | None = None) -> RenderResult:
    """Renders `project` to `output_path` (default: project.output.path).

    The video is written to a temporary file next to the target and only
    replaces it on success, so an error or Cancel never leaves a broken file
    or destroys an earlier export.
    """
    project.validate()
    if not project.clips:
        raise TimelineError("the project has no clips")
    path = os.path.abspath(output_path or project.output.path)
    if any(os.path.abspath(clip.path) == path for clip in project.clips):
        raise ValueError(f"the output file would overwrite an input clip: {path}")
    discover_effects()
    started = time.perf_counter()

    readers = []
    try:
        for clip in project.clips:
            readers.append(OpenCVReader(clip.path))
        first, out = readers[0].info, project.output
        size = (out.width or first.width, out.height or first.height)
        fps = out.fps or first.fps
        sources = [ConformedSource(r, size, fps, c.start, c.end) for r, c in zip(readers, project.clips)]
        graph = resolve(project, sources)

        root, ext = os.path.splitext(path)
        temp = f"{root}.part{ext or '.mp4'}"
        try:
            with OpenCVWriter(temp, size, fps) as writer:
                frames = Renderer(graph, sources).render(writer, progress)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.remove(temp)
    finally:
        for reader in readers:
            reader.close()

    return RenderResult(path, frames, size, fps, time.perf_counter() - started)
