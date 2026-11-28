"""Video input and output. This is the only package that uses OpenCV,
and it uses only cv2.VideoCapture and cv2.VideoWriter."""

from __future__ import annotations

from moshflow.videoio.opencv_reader import OpenCVReader, VideoInfo, probe
from moshflow.videoio.opencv_writer import OpenCVWriter

__all__ = ["OpenCVReader", "OpenCVWriter", "VideoInfo", "probe"]
