"""ConformedSource: one clip seen at the project's size and frame rate.

It implements the FrameSource interface on top of any reader that offers
`info` (fps, frame_count) and `read(index)`.
"""

from typing import Protocol

import numpy as np

from moshflow.core.cache import LRUCache
from moshflow.toolkit import resize


class FrameReader(Protocol):
    info: object  # needs .fps and .frame_count

    def read(self, index: int) -> np.ndarray: ...


class ConformedSource:
    def __init__(self, reader: FrameReader, size: tuple[int, int], fps: float,
                 start: float = 0.0, end: float | None = None, cache_size: int = 48):
        self.size = size
        self.fps = fps
        self._reader = reader
        src_fps = reader.info.fps
        total = reader.info.frame_count
        self._first = min(int(round(start * src_fps)), total - 1)
        last = total if end is None else min(total, int(round(end * src_fps)))
        self._src_len = max(1, last - self._first)
        self._step = src_fps / fps
        self._len = max(1, int(round(self._src_len / self._step)))
        self._cache: LRUCache[np.ndarray] = LRUCache(cache_size)

    def __len__(self) -> int:
        return self._len

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self._len:
            raise IndexError(f"frame {index} out of range 0..{self._len - 1}")
        # Nearest-frame resampling from the source frame rate.
        src = self._first + min(int(index * self._step + 0.5), self._src_len - 1)
        return self._cache.get_or_create(src, lambda: self._load(src))

    def _load(self, src: int) -> np.ndarray:
        frame = fit(self._reader.read(src), self.size)
        frame = np.ascontiguousarray(frame)
        frame.flags.writeable = False
        return frame


def fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resizes a frame into size = (width, height) without stretching it.

    A clip with another aspect ratio (a 4:3 clip in a 16:9 project, a vertical phone
    clip) is scaled to fit and centred on black bars, as a video editor does.
    """
    width, height = size
    src_h, src_w = frame.shape[:2]
    if src_w * height == src_h * width:
        return resize.bilinear(frame, size)
    scale = min(width / src_w, height / src_h)
    fit_w = max(1, min(width, int(round(src_w * scale))))
    fit_h = max(1, min(height, int(round(src_h * scale))))
    out = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    left, top = (width - fit_w) // 2, (height - fit_h) // 2
    out[top:top + fit_h, left:left + fit_w] = resize.bilinear(frame, (fit_w, fit_h))
    return out
