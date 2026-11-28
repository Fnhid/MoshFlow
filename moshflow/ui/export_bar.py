"""ExportBar: output settings, the Export button, progress and Cancel.

The render runs in a QThread, so the UI stays responsive and can keep editing.
Projects are immutable, so the render keeps using the version from when Export
was pressed.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from moshflow.app.render_job import JobProgress, RenderResult, render_project
from moshflow.core.project import OutputSettings, Project
from moshflow.core.renderer import RenderCancelled
from moshflow.ui.model import ProjectModel

MAX_DIMENSION = 7680
MAX_FPS = 240.0

SIZE_PRESETS: tuple[tuple[str, str, tuple[int, int] | None], ...] = (
    ("Auto", "auto", None),
    ("1080p", "1920x1080", (1920, 1080)),
    ("720p", "1280x720", (1280, 720)),
    ("360p", "640x360", (640, 360)),
)
PRESET_BY_KEY = {key: size for _, key, size in SIZE_PRESETS}
KEY_BY_SIZE = {size: key for _, key, size in SIZE_PRESETS}


class RenderWorker(QThread):
    progressed = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project: Project, output: str, parent=None):
        super().__init__(parent)
        self._project = project
        self._output = output
        self._job = JobProgress(self.progressed.emit)

    def cancel(self) -> None:
        self._job.cancel()

    def run(self) -> None:
        try:
            self.succeeded.emit(render_project(self._project, self._job, self._output))
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))


def _auto_spin(maximum: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, maximum)
    spin.setSpecialValueText("auto")
    spin.setFixedWidth(72)
    return spin


class ExportBar(QFrame):
    def __init__(self, model: ProjectModel, parent=None):
        super().__init__(parent)
        self.setProperty("panel", True)
        self._model = model
        self.worker: RenderWorker | None = None

        output_label = QLabel("EXPORT")
        output_label.setProperty("role", "panelTitle")

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("output.mp4")
        self.path_edit.setMinimumWidth(180)
        browse = QPushButton("…")
        browse.setProperty("compact", True)
        browse.setFixedWidth(30)
        browse.setToolTip("Choose output file")
        browse.clicked.connect(self._browse)

        self.settings_button = QPushButton("Settings ▾")
        self.settings_button.setProperty("compact", True)
        self.settings_button.setCheckable(True)
        self.settings_button.toggled.connect(self._toggle_settings)

        self.preset_box = QComboBox()
        for label, key, _ in SIZE_PRESETS:
            self.preset_box.addItem(label, key)
        self.preset_box.addItem("Custom", "custom")
        self.preset_box.setFixedWidth(86)

        self.width_spin = _auto_spin(MAX_DIMENSION)
        self.height_spin = _auto_spin(MAX_DIMENSION)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.0, MAX_FPS)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSpecialValueText("auto")
        self.fps_spin.setFixedWidth(74)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setVisible(False)
        self.status = QLabel("Ready")
        self.status.setProperty("role", "muted")

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_export)
        self.export_button = QPushButton("Export")
        self.export_button.setProperty("accent", True)
        self.export_button.clicked.connect(self.start_export)

        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(8)
        main_row.addWidget(output_label)
        main_row.addWidget(self.path_edit, 2)
        main_row.addWidget(browse)
        main_row.addWidget(self.settings_button)
        main_row.addSpacing(8)
        main_row.addWidget(self.progress)
        main_row.addWidget(self.status, 1)
        main_row.addWidget(self.cancel_button)
        main_row.addWidget(self.export_button)

        # Collapsible output settings drawer: out of the way until needed.
        self._settings_drawer = QWidget()
        drawer_layout = QHBoxLayout(self._settings_drawer)
        drawer_layout.setContentsMargins(0, 4, 0, 0)
        drawer_layout.setSpacing(8)
        for text, widget in (
            ("Preset", self.preset_box),
            ("Width", self.width_spin),
            ("Height", self.height_spin),
            ("FPS", self.fps_spin),
        ):
            lbl = QLabel(text)
            lbl.setProperty("role", "muted")
            drawer_layout.addWidget(lbl)
            drawer_layout.addWidget(widget)
        drawer_layout.addStretch(1)
        self._settings_drawer.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addLayout(main_row)
        layout.addWidget(self._settings_drawer)

        self.path_edit.textEdited.connect(self._store_settings)
        self.preset_box.currentIndexChanged.connect(self._apply_preset)
        self.width_spin.valueChanged.connect(self._store_settings)
        self.height_spin.valueChanged.connect(self._store_settings)
        self.fps_spin.valueChanged.connect(self._store_settings)
        self._storing = False
        model.changed.connect(self._load_settings)      # also after Undo / Redo and Open
        self._load_settings()

    # --- settings --------------------------------------------------------
    def _toggle_settings(self, opened: bool) -> None:
        self._settings_drawer.setVisible(opened)
        self.settings_button.setText("Settings ▴" if opened else "Settings ▾")

    def _sync_preset_box(self, width: int | None, height: int | None) -> None:
        if not width and not height:
            key = "auto"
        elif width and height:
            key = KEY_BY_SIZE.get((width, height), "custom")
        else:
            key = "custom"
        idx = self.preset_box.findData(key)
        self.preset_box.blockSignals(True)
        self.preset_box.setCurrentIndex(idx)
        self.preset_box.blockSignals(False)

    def _apply_preset(self) -> None:
        key = self.preset_box.currentData()
        if key == "custom":
            return
        size = PRESET_BY_KEY.get(key)
        width, height = (0, 0) if size is None else size
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        self._store_settings()

    def _load_settings(self) -> None:
        if self._storing:
            return      # the change came from these widgets; leave the text and cursor alone
        out = self._model.project.output
        for widget in (self.path_edit, self.width_spin, self.height_spin, self.fps_spin):
            widget.blockSignals(True)
        self.path_edit.setText(out.path)
        self.width_spin.setValue(out.width or 0)
        self.height_spin.setValue(out.height or 0)
        self.fps_spin.setValue(out.fps or 0.0)
        for widget in (self.path_edit, self.width_spin, self.height_spin, self.fps_spin):
            widget.blockSignals(False)
        self._sync_preset_box(out.width, out.height)

    def _settings_from_widgets(self) -> OutputSettings:
        return OutputSettings(
            path=self.path_edit.text().strip() or "output.mp4",
            width=self.width_spin.value() or None,
            height=self.height_spin.value() or None,
            fps=self.fps_spin.value() or None,
        )

    def _store_settings(self, *_) -> None:
        self._storing = True
        try:
            settings = self._settings_from_widgets()
            self._sync_preset_box(settings.width, settings.height)
            self._model.set_output(settings)
        finally:
            self._storing = False

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export video", self.resolved_output(), "MP4 video (*.mp4)")
        if path:
            self.path_edit.setText(path)
            self._store_settings()

    def resolved_output(self) -> str:
        """A relative output path is placed next to the first clip."""
        path = self._model.project.output.path
        clips = self._model.project.clips
        if os.path.isabs(path) or not clips:
            return path
        return os.path.join(os.path.dirname(clips[0].path), path)

    # --- export ----------------------------------------------------------
    def start_export(self) -> None:
        project = self._model.project
        if not project.clips:
            QMessageBox.information(self, "Export", "Add at least one clip to the timeline first.")
            return
        if bool(project.output.width) != bool(project.output.height):
            QMessageBox.information(self, "Export", "Set both width and height, or leave both on auto.")
            return
        output = self.resolved_output()
        if os.path.exists(output):
            answer = QMessageBox.question(self, "Export", f"{output}\nalready exists. Replace it?")
            if answer != QMessageBox.Yes:
                return
        self.run_worker(output)

    def run_worker(self, output: str) -> None:
        self.worker = RenderWorker(self._model.project, output, self)
        self.worker.progressed.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_finished)
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("Rendering…")
        self.worker.start()

    def cancel_export(self) -> None:
        if self.worker:
            self.status.setText("Cancelling…")
            self.worker.cancel()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)

    def _on_success(self, result: RenderResult) -> None:
        self.status.setText(f"Done in {result.seconds:.1f} s: {os.path.basename(result.path)}")

    def _on_failure(self, message: str) -> None:
        self.progress.setVisible(False)
        self.status.setText("Export failed")
        QMessageBox.critical(self, "Export failed", message)

    def _on_cancelled(self) -> None:
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.status.setText("Cancelled")

    def _on_finished(self) -> None:
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.worker = None
