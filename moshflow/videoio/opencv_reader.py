"""Decoding with cv2.VideoCapture (I/O only)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

_FALLBACK_FPS = 30.0


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class OpenCVReader:
    """Random access to the decoded frames of one video file.

    Sequential reads are cheap; jumping backwards or far ahead seeks.
    """

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = _FALLBACK_FPS
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            count = self._count_frames()
        self.info = VideoInfo(
            path=path,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            frame_count=count,
        )
        if count <= 0:
            raise IOError(f"video has no frames: {path}")
        self._next = 0
        self._last: np.ndarray | None = None

    def read(self, index: int) -> np.ndarray:
        """Frame `index` as an H x W x 3 uint8 BGR array."""
        if not 0 <= index < self.info.frame_count:
            raise IndexError(f"frame {index} out of range 0..{self.info.frame_count - 1}")
        if index != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index
        ok, frame = self._cap.read()
        if not ok:
            # Containers sometimes report a few more frames than they hold.
            if self._last is None:
                raise IOError(f"cannot decode frame {index} of {self.info.path}")
            return self._last
        self._next = index + 1
        self._last = frame
        return frame

    def close(self) -> None:
        self._cap.release()

    def _count_frames(self) -> int:
        count = 0
        while self._cap.grab():
            count += 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return count

    def __enter__(self) -> "OpenCVReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def probe(path: str) -> VideoInfo:
    with OpenCVReader(path) as reader:
        return reader.info
