"""Encoding with cv2.VideoWriter (I/O only). Implements the FrameSink interface."""

from __future__ import annotations

import cv2
import numpy as np


class OpenCVWriter:
    def __init__(self, path: str, size: tuple[int, int], fps: float, fourcc: str = "mp4v"):
        self.path = path
        self.size = size
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*fourcc), fps, size)
        if not self._writer.isOpened():
            raise IOError(f"cannot create video file: {path}")

    def write(self, frame: np.ndarray) -> None:
        width, height = self.size
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected a {width}x{height} uint8 BGR frame, got {frame.shape} {frame.dtype}")
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "OpenCVWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
