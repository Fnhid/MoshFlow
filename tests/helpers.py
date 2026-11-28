"""Synthetic clips for the tests."""

from __future__ import annotations

import numpy as np


class ArraySource:
    """An in-memory FrameSource whose frames are read-only, like the real one."""

    def __init__(self, frames: list[np.ndarray], fps: float = 10.0):
        self._frames = frames
        for frame in frames:
            frame.flags.writeable = False
        self.fps = fps
        self.size = (frames[0].shape[1], frames[0].shape[0])

    def __len__(self) -> int:
        return len(self._frames)

    def frame(self, index: int) -> np.ndarray:
        return self._frames[index]


def moving_clip(length: int, size: tuple[int, int] = (64, 48), seed: int = 1) -> list[np.ndarray]:
    """A textured image that shifts right by 2 pixels per frame."""
    width, height = size
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 256, (height, width + 2 * length, 3), dtype=np.uint8)
    return [texture[:, 2 * (length - i):2 * (length - i) + width].copy() for i in range(length)]


class FakeProgress:
    def __init__(self, cancelled: bool = False):
        self.cancelled = cancelled
        self.reports: list[tuple[int, int]] = []

    def report(self, done: int, total: int) -> None:
        self.reports.append((done, total))
