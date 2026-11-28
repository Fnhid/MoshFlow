"""Frame types and the small interfaces effects and the renderer talk through."""

from __future__ import annotations

try:
    from typing import Protocol
except ImportError:            # Python 3.7: only used as a type hint
    Protocol = object

import numpy as np

# H x W x 3, uint8, BGR. Frames handed to effects are read-only.
Frame = np.ndarray


class FrameSource(Protocol):
    """Random-access, already conformed frames of one clip."""

    fps: float
    size: tuple[int, int]  # (width, height)

    def __len__(self) -> int: ...

    def frame(self, index: int) -> Frame: ...


class FrameSink(Protocol):
    """Receives finished frames in order (e.g. a video file writer)."""

    def write(self, frame: Frame) -> None: ...

    def close(self) -> None: ...


class ProgressReporter(Protocol):
    """Lets long work report progress and notice a cancel request."""

    @property
    def cancelled(self) -> bool: ...

    def report(self, done: int, total: int) -> None: ...
