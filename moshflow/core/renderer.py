"""Renderer: walks a RenderGraph and streams the frames into a FrameSink."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from moshflow.api import (
    FrameSink,
    FrameSource,
    ProgressReporter,
    TransitionContext,
    TransitionEffect,
    get_effect,
    validate_params,
)
from moshflow.core.timeline import ClipNode, RenderGraph, TransitionNode


class RenderCancelled(Exception):
    pass


class EffectError(RuntimeError):
    """An effect broke the contract (wrong frame count, shape or type)."""


class NullProgress:
    cancelled = False

    def report(self, done: int, total: int) -> None:
        pass


class _EffectProgress:
    """Maps an effect's own progress onto the progress of the whole render."""

    def __init__(self, outer: ProgressReporter, offset: int, length: int, total: int):
        self._outer, self._offset, self._length, self._total = outer, offset, length, total

    @property
    def cancelled(self) -> bool:
        return self._outer.cancelled

    def report(self, done: int, total: int) -> None:
        if total > 0:
            self._outer.report(self._offset + int(self._length * min(done, total) / total), self._total)


class Renderer:
    def __init__(self, graph: RenderGraph, sources: Sequence[FrameSource],
                 effect_lookup: Callable[[str], type[TransitionEffect]] = get_effect):
        self.graph = graph
        self.sources = sources
        self._lookup = effect_lookup
        width, height = sources[0].size
        self._shape = (height, width, 3)

    def render(self, sink: FrameSink, progress: ProgressReporter | None = None) -> int:
        """Writes every frame of the graph to `sink`; returns the frame count."""
        progress = progress or NullProgress()
        total = self.graph.total_frames
        done = 0
        for node in self.graph.nodes:
            if isinstance(node, ClipNode):
                frames = (self.sources[node.clip].frame(i) for i in node.frames)
            else:
                frames = self._transition_frames(node, _EffectProgress(progress, done, node.length, total))
            for frame in frames:
                if progress.cancelled:
                    raise RenderCancelled()
                sink.write(frame)
                done += 1
                progress.report(done, total)
        return done

    def _transition_frames(self, node: TransitionNode, progress: ProgressReporter):
        t = node.transition
        effect_cls = self._lookup(t.effect)
        try:
            params = validate_params(effect_cls.param_specs(), t.params)
        except ValueError as e:
            raise ValueError(f"transition {node.cut + 1} ({effect_cls.info.name}): {e}") from None
        ctx = TransitionContext(
            outgoing=self.sources[node.a],
            incoming=self.sources[node.b],
            a_window=node.a_window,
            b_window=node.b_window,
            params=params,
            seed=t.seed,
        )
        name = effect_cls.info.name
        count = 0
        for frame in effect_cls().render(ctx, progress):
            if progress.cancelled:
                raise RenderCancelled()
            if count >= ctx.length:
                raise EffectError(f"{name} yielded more than {ctx.length} frames")
            if not isinstance(frame, np.ndarray) or frame.shape != self._shape or frame.dtype != np.uint8:
                got = f"{frame.shape} {frame.dtype}" if isinstance(frame, np.ndarray) else type(frame).__name__
                raise EffectError(f"{name} frame {count}: expected {self._shape} uint8, got {got}")
            count += 1
            yield frame
        if progress.cancelled:
            raise RenderCancelled()
        if count != ctx.length:
            raise EffectError(f"{name} yielded {count} frames, expected {ctx.length}")
