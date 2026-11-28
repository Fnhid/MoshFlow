"""PreviewPanel: center monitor for inspecting clips and playing transition previews."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from moshflow.app.catalog import EffectCatalog
from moshflow.app.media import read_frame
from moshflow.app.preview import PreviewClip, render_transition_preview
from moshflow.app.render_job import JobProgress
from moshflow.core.project import Project
from moshflow.core.renderer import RenderCancelled
from moshflow.ui.frame_display import frame_to_pixmap
from moshflow.ui.model import CLIP, TRANSITION, ProjectModel
from moshflow.ui.theme import BG_CANVAS, TEXT_MUTED

WORKER_WAIT_MS = 5_000


class PreviewWorker(QThread):
    progressed = Signal(int, int)
    ready = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project: Project, cut_index: int, parent=None):
        super().__init__(parent)
        self._project = project
        self._cut_index = cut_index
        self._job = JobProgress(self.progressed.emit)

    def cancel(self) -> None:
        self._job.cancel()

    def run(self) -> None:
        try:
            clip = render_transition_preview(self._project, self._cut_index, self._job)
            self.ready.emit(clip)
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class _FrameCanvas(QWidget):
    """Draws a BGR frame letterboxed inside the monitor using hand-written resizing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._placeholder = "Import clips and add them to the timeline to preview."
        self._cached_size: tuple[int, int] | None = None
        self._cached_pixmap: QPixmap | None = None
        self.setMinimumHeight(200)

    def set_frame(self, frame: np.ndarray | None, placeholder: str | None = None) -> None:
        self._frame = frame
        if placeholder is not None:
            self._placeholder = placeholder
        self._cached_pixmap = None
        self._cached_size = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._cached_pixmap = None
        self._cached_size = None
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG_CANVAS))
        if self._frame is None:
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(self.rect().adjusted(24, 16, -24, -16), Qt.AlignCenter | Qt.TextWordWrap, self._placeholder)
            return
        max_w = max(1, self.width() - 16)
        max_h = max(1, self.height() - 16)
        if self._cached_pixmap is None or self._cached_size != (max_w, max_h):
            self._cached_pixmap = frame_to_pixmap(self._frame, (max_w, max_h))
            self._cached_size = (max_w, max_h)
        pix = self._cached_pixmap
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        painter.drawPixmap(x, y, pix)


def _format_timecode(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


class PreviewPanel(QFrame):
    def __init__(self, model: ProjectModel, catalog: EffectCatalog, parent=None):
        super().__init__(parent)
        self.setProperty("panel", True)
        self._model = model
        self._catalog = catalog
        self._worker: PreviewWorker | None = None
        self._rendered: PreviewClip | None = None
        self._rendered_signature: tuple | None = None
        self._clip_frames_cache: dict[tuple[str, int], np.ndarray] = {}
        self._clip_range: tuple[str, int, int, float] | None = None   # (path, start_f, end_f, fps)
        self._current_index = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        # Monitor frame
        monitor_box = QFrame()
        monitor_box.setProperty("monitor", True)
        monitor_layout = QVBoxLayout(monitor_box)
        monitor_layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = _FrameCanvas()
        monitor_layout.addWidget(self.canvas, 1)

        # Transport controls: Play automatically renders transitions on demand.
        self.play_button = QPushButton("Play")
        self.play_button.setProperty("compact", True)
        self.play_button.setFixedWidth(72)
        self.play_button.clicked.connect(self.toggle_play)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.valueChanged.connect(self._on_scrub)

        self.timecode = QLabel("00:00.00 / 00:00.00")
        self.timecode.setProperty("role", "timecode")

        # Trim the selected clip at the frame shown (also the I and O keys).
        self.start_button = QPushButton("Set start")
        self.end_button = QPushButton("Set end")
        for button, slot, tip in ((self.start_button, self.set_start_here, "Start the clip at this frame (I)"),
                                  (self.end_button, self.set_end_here, "End the clip at this frame (O)")):
            button.setProperty("compact", True)
            button.setToolTip(tip)
            button.clicked.connect(slot)

        transport = QHBoxLayout()
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setSpacing(10)
        transport.addWidget(self.play_button)
        transport.addWidget(self.scrubber, 1)
        transport.addWidget(self.timecode)
        transport.addWidget(self.start_button)
        transport.addWidget(self.end_button)

        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        self.status_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(monitor_box, 1)
        layout.addLayout(transport)
        layout.addWidget(self.status_label)

        model.changed.connect(self.refresh)
        model.selectionChanged.connect(self.refresh)
        self.refresh()

    # --- trimming at the frame shown --------------------------------------
    def playhead_time(self) -> float | None:
        """Time in the selected clip's file of the frame shown, or None if no clip is shown."""
        if self._clip_range is None or self._rendered is not None:
            return None
        _, start_f, _, fps = self._clip_range
        return (start_f + self._current_index) / (fps or 30.0)

    def set_start_here(self) -> None:
        self._trim_here("start")

    def set_end_here(self) -> None:
        self._trim_here("end")

    def _trim_here(self, side: str) -> None:
        selection = self._model.selection
        time = self.playhead_time()
        if time is None or not selection or selection[0] != CLIP:
            return
        index = selection[1]
        clip = self._model.project.clips[index]
        info = self._model.info(clip.path)
        total = info.duration if info else time
        end = total if clip.end is None else min(clip.end, total)
        time = round(time, 2)
        if side == "start" and time < end - 0.05:
            self._model.set_clip(index, replace(clip, start=time), "trim start", object())
        elif side == "end" and time > clip.start + 0.05:
            # The frame shown stays in: the clip ends after it.
            stop = round(time + 1.0 / (self._clip_range[3] or 30.0), 2)
            self._model.set_clip(index, replace(clip, end=None if stop >= total - 0.005 else stop), "trim end", object())
        self._model.select(CLIP, index)

    def _set_status(self, message: str = "") -> None:
        self.status_label.setText(message)
        self.status_label.setVisible(bool(message))

    def _read_cached_frame(self, path: str, index: int) -> np.ndarray | None:
        key = (path, index)
        if key in self._clip_frames_cache:
            return self._clip_frames_cache[key]
        frame = read_frame(path, index)
        if frame is not None:
            if len(self._clip_frames_cache) >= 48:
                self._clip_frames_cache.pop(next(iter(self._clip_frames_cache)))
            self._clip_frames_cache[key] = frame
        return frame

    def _transition_signature(self, index: int) -> tuple:
        project = self._model.project
        return (index, project.clips[index], project.clips[index + 1], project.transitions[index], project.output)

    def refresh(self) -> None:
        self._stop_playback()
        project = self._model.project
        selection = self._model.selection

        if not project.clips:
            self.stop_worker()
            self._rendered = None
            self._rendered_signature = None
            self._clip_range = None
            self.canvas.set_frame(None, "Add clips to the timeline to preview.")
            self._configure_scrubber(0, 0, 0.0, 0.0)
            self.play_button.setEnabled(False)
            self._set_status("")
            return

        if selection and selection[0] == TRANSITION and selection[1] < len(project.transitions):
            self._show_transition(selection[1])
            return

        clip_index = selection[1] if (selection and selection[0] == CLIP and selection[1] < len(project.clips)) else 0
        self._show_clip(clip_index)

    def _show_clip(self, index: int) -> None:
        self.stop_worker()
        self._rendered = None
        self._rendered_signature = None
        clip = self._model.project.clips[index]
        info = self._model.info(clip.path)

        if not info or info.frame_count <= 0:
            self._clip_range = None
            self.canvas.set_frame(None, "Unable to read frames from this clip.")
            self._configure_scrubber(0, 0, 0.0, 0.0)
            self.play_button.setEnabled(False)
            self._set_status("Clip media cannot be read.")
            return

        fps = info.fps or 30.0
        start_f = max(0, min(info.frame_count - 1, int(round(clip.start * fps))))
        end_sec = info.duration if clip.end is None else min(clip.end, info.duration)
        end_f = max(start_f, min(info.frame_count - 1, int(round(end_sec * fps)) - 1))
        self._clip_range = (clip.path, start_f, end_f, fps)
        total = max(1, end_f - start_f + 1)
        self._current_index = 0

        frame = self._read_cached_frame(clip.path, start_f)
        self.canvas.set_frame(frame)
        self._configure_scrubber(0, total - 1, 0.0, total / fps)
        self.play_button.setEnabled(total > 1)
        self._set_status("")

    def _show_transition(self, index: int) -> None:
        self._clip_range = None
        transition = self._model.project.transitions[index]
        signature = self._transition_signature(index)
        if self._rendered is not None and self._rendered_signature == signature:
            self._show_rendered_frame(0)
            return

        self.stop_worker()
        self._rendered = None
        self._rendered_signature = None

        clip_a = self._model.project.clips[index]
        info_a = self._model.info(clip_a.path)
        frame = None
        if info_a and info_a.frame_count > 0:
            end_sec = info_a.duration if clip_a.end is None else min(clip_a.end, info_a.duration)
            cut_frame = max(0, min(info_a.frame_count - 1, int(round(end_sec * info_a.fps)) - 1))
            frame = self._read_cached_frame(clip_a.path, cut_frame)
        self.canvas.set_frame(frame, "Press Play to preview this transition.")
        self._configure_scrubber(0, 0, 0.0, transition.duration)
        self.play_button.setEnabled(True)
        self._set_status("")

    def _show_rendered_frame(self, index: int) -> None:
        if self._rendered is None or not self._rendered.frames:
            return
        total = len(self._rendered.frames)
        self._current_index = max(0, min(index, total - 1))
        self.canvas.set_frame(self._rendered.frames[self._current_index])
        fps = self._rendered.fps or 30.0
        self._configure_scrubber(self._current_index, total - 1, self._current_index / fps, total / fps)
        self.play_button.setEnabled(total > 1)
        self._set_status("")

    def _configure_scrubber(self, value: int, maximum: int, current_s: float, total_s: float) -> None:
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(0, maximum))
        self.scrubber.setValue(max(0, min(value, max(0, maximum))))
        self.scrubber.setEnabled(maximum > 0)
        self.scrubber.blockSignals(False)
        self.timecode.setText(f"{_format_timecode(current_s)} / {_format_timecode(total_s)}")

    # --- playback and scrubbing ------------------------------------------
    def toggle_play(self) -> None:
        if self._worker is not None:
            self.stop_worker()
            self.play_button.setText("Play")
            self._set_status("")
            return
        if self._timer.isActive():
            self._stop_playback()
            return
        selection = self._model.selection
        if selection and selection[0] == TRANSITION and self._rendered is None:
            self.start_transition_preview()
            return
        fps = 30.0
        if self._rendered is not None:
            fps = self._rendered.fps or 30.0
        elif self._clip_range is not None:
            fps = self._clip_range[3] or 30.0
        else:
            return
        interval_ms = max(15, int(round(1000.0 / fps)))
        self._timer.start(interval_ms)
        self.play_button.setText("Pause")

    def _stop_playback(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self.play_button.setText("Play")

    def _advance_frame(self) -> None:
        maximum = self.scrubber.maximum()
        if maximum <= 0:
            self._stop_playback()
            return
        next_idx = (self._current_index + 1) % (maximum + 1)
        self.scrubber.setValue(next_idx)

    def _on_scrub(self, value: int) -> None:
        self._current_index = value
        if self._rendered is not None and self._rendered.frames:
            total = len(self._rendered.frames)
            idx = max(0, min(value, total - 1))
            self.canvas.set_frame(self._rendered.frames[idx])
            fps = self._rendered.fps or 30.0
            self.timecode.setText(f"{_format_timecode(idx / fps)} / {_format_timecode(total / fps)}")
            return
        if self._clip_range is not None:
            path, start_f, end_f, fps = self._clip_range
            total = max(1, end_f - start_f + 1)
            frame = self._read_cached_frame(path, start_f + value)
            if frame is not None:
                self.canvas.set_frame(frame)
            self.timecode.setText(f"{_format_timecode(value / fps)} / {_format_timecode(total / fps)}")

    # --- background transition preview -----------------------------------
    def start_transition_preview(self) -> None:
        selection = self._model.selection
        if not selection or selection[0] != TRANSITION:
            return
        cut_index = selection[1]
        if cut_index >= len(self._model.project.transitions):
            return
        self.stop_worker()
        self._stop_playback()
        signature = self._transition_signature(cut_index)
        self._worker = PreviewWorker(self._model.project, cut_index, self)
        self._worker.progressed.connect(self._on_preview_progress)
        self._worker.ready.connect(lambda clip, sig=signature: self._on_preview_ready(clip, sig))
        self._worker.failed.connect(self._on_preview_failed)
        self._worker.cancelled.connect(self._on_preview_cancelled)
        self._worker.finished.connect(self._on_preview_finished)
        self.play_button.setText("Stop")
        self.play_button.setEnabled(True)
        self._set_status("Rendering preview…")
        self._worker.start()

    def stop_worker(self) -> None:
        self._stop_playback()
        worker = self._worker
        if worker is not None:
            self._worker = None
            worker.cancel()
            worker.wait(WORKER_WAIT_MS)

    def _on_preview_progress(self, done: int, total: int) -> None:
        self._set_status(f"Rendering preview… {done}/{max(total, 1)}")

    def _on_preview_ready(self, clip: PreviewClip, signature: tuple) -> None:
        selection = self._model.selection
        if not selection or selection[0] != TRANSITION:
            return
        if selection[1] >= len(self._model.project.transitions):
            return
        if self._transition_signature(selection[1]) != signature:
            return
        self._rendered = clip
        self._rendered_signature = signature
        self._show_rendered_frame(0)
        self.toggle_play()

    def _on_preview_failed(self, message: str) -> None:
        self.play_button.setText("Play")
        self.play_button.setEnabled(True)
        self._set_status(f"Preview error: {message}")

    def _on_preview_cancelled(self) -> None:
        self.play_button.setText("Play")
        self.play_button.setEnabled(True)
        self._set_status("")

    def _on_preview_finished(self) -> None:
        self._worker = None
