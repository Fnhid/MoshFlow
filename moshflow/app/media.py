"""Reading video properties and preview frames for the UI, without depending on OpenCV."""

from __future__ import annotations

import numpy as np

from moshflow.videoio import OpenCVReader, VideoInfo, probe


def read_frame(path: str, index: int = 0) -> np.ndarray | None:
    """Reads a single BGR frame from `path`, clamping `index` to valid bounds."""
    try:
        with OpenCVReader(path) as reader:
            total = reader.info.frame_count
            if total <= 0:
                return None
            clamped = max(0, min(index, total - 1))
            return reader.read(clamped)
    except (IOError, IndexError):
        return None


__all__ = ["VideoInfo", "probe", "read_frame"]
