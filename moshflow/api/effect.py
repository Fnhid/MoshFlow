"""The transition-effect contract (API v1).

An effect is one TransitionEffect subclass in its own file under
moshflow/effects/, decorated with @register. See README.md, "Writing an
effect", and tests/test_effect_contract.py for the rules every effect follows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from moshflow.api.frames import Frame, FrameSource, ProgressReporter
from moshflow.api.params import ParamSpec


@dataclass(frozen=True)
class EffectInfo:
    id: str           # "crossfade"; used in project files
    name: str         # "Crossfade"; shown in the UI
    author: str
    description: str


@dataclass(frozen=True)
class TransitionContext:
    """Everything an effect may use to render one transition.

    The output replaces the frames of A in a_window followed by the frames of
    B in b_window. Effects may still read any other frame of either clip,
    e.g. the last clean frame of A before the window.
    """

    outgoing: FrameSource          # clip A
    incoming: FrameSource          # clip B
    a_window: range                # frames of A covered by the transition
    b_window: range                # frames of B covered by the transition
    params: Mapping[str, object]   # validated against param_specs()
    seed: int                      # use for any randomness

    @property
    def length(self) -> int:
        """Number of frames the effect must yield."""
        return len(self.a_window) + len(self.b_window)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of every frame, input and output."""
        return self.outgoing.size

    @property
    def fps(self) -> float:
        return self.outgoing.fps


class TransitionEffect(ABC):
    info: EffectInfo

    @classmethod
    @abstractmethod
    def param_specs(cls) -> Sequence[ParamSpec]:
        """Declares tunable parameters; the UI builds widgets from it."""

    @abstractmethod
    def render(self, ctx: TransitionContext, progress: ProgressReporter) -> Iterator[Frame]:
        """Yields exactly ctx.length frames (H x W x 3, uint8)."""
