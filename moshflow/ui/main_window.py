"""MainWindow: puts the views together and handles the File and Edit menus."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from moshflow.app.catalog import EffectCatalog
from moshflow.app.store import EXTENSION
from moshflow.ui.export_bar import ExportBar
from moshflow.ui.inspector import Inspector
from moshflow.ui.media_bin import MediaBin
from moshflow.ui.model import ProjectModel
from moshflow.ui.preview_panel import PreviewPanel
from moshflow.ui.theme import STYLESHEET, apply_theme, build_palette
from moshflow.ui.timeline_view import TimelineView

PROJECT_FILTER = f"MoshFlow project (*{EXTENSION})"
CLOSE_WAIT_MS = 10_000


class MainWindow(QMainWindow):
    def __init__(self, catalog: EffectCatalog | None = None):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            app.setPalette(build_palette())
        self.setStyleSheet(STYLESHEET)
        self.catalog = catalog or EffectCatalog()
        self.model = ProjectModel(self)
        self.media_bin = MediaBin(self.model)
        self.preview = PreviewPanel(self.model, self.catalog)
        self.inspector = Inspector(self.model, self.catalog)
        self.timeline = TimelineView(self.model, self.catalog)
        self.export_bar = ExportBar(self.model)

        self.media_bin.setMinimumWidth(220)
        self.preview.setMinimumWidth(340)
        self.inspector.setMinimumWidth(280)

        top = QSplitter(Qt.Horizontal)
        top.setChildrenCollapsible(False)
        top.addWidget(self.media_bin)
        top.addWidget(self.preview)
        top.addWidget(self.inspector)
        top.setStretchFactor(0, 2)
        top.setStretchFactor(1, 5)
        top.setStretchFactor(2, 3)

        timeline_panel = self._build_timeline_panel()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(top, 1)
        layout.addWidget(timeline_panel)
        layout.addWidget(self.export_bar)
        self.setCentralWidget(central)
        self.resize(1280, 760)

        self._build_menu()
        self.media_bin.errors.connect(self._show_errors)
        self.timeline.errors.connect(self._show_errors)
        self.model.changed.connect(self._update_title)
        self.model.changed.connect(self._update_timeline_summary)
        self._update_title()
        self._update_timeline_summary()

    def _build_timeline_panel(self) -> QFrame:
        panel = QFrame()
        panel.setProperty("panel", True)

        title = QLabel("TIMELINE")
        title.setProperty("role", "panelTitle")
        self._timeline_summary = QLabel("0.0 s")
        self._timeline_summary.setProperty("role", "muted")

        zoom_out = QPushButton("-")
        zoom_out.setProperty("compact", True)
        zoom_out.setFixedWidth(26)
        zoom_out.setToolTip("Zoom out timeline")
        zoom_out.clicked.connect(self.timeline.zoom_out)

        zoom_in = QPushButton("+")
        zoom_in.setProperty("compact", True)
        zoom_in.setFixedWidth(26)
        zoom_in.setToolTip("Zoom in timeline")
        zoom_in.clicked.connect(self.timeline.zoom_in)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(title)
        header.addWidget(self._timeline_summary)
        header.addStretch(1)
        header.addWidget(zoom_out)
        header.addWidget(zoom_in)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self.timeline)
        return panel

    def _update_timeline_summary(self) -> None:
        clips = self.model.project.clips
        total = sum(self.model.clip_duration(i) for i in range(len(clips)))
        self._timeline_summary.setText(f"{total:.1f} s" if clips else "")

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        for text, shortcut, handler in [
            ("&New", QKeySequence.New, self.new_project),
            ("&Open…", QKeySequence.Open, self.open_project),
            ("&Save", QKeySequence.Save, self.save_project),
            ("Save &As…", QKeySequence.SaveAs, self.save_project_as),
            ("&Quit", QKeySequence.Quit, self.close),
        ]:
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _=False, h=handler: h())
            menu.addAction(action)
        edit = self.menuBar().addMenu("&Edit")
        undo = self.model.undo_stack.createUndoAction(self, "&Undo")
        undo.setShortcut(QKeySequence.Undo)
        redo = self.model.undo_stack.createRedoAction(self, "&Redo")
        redo.setShortcut(QKeySequence.Redo)
        edit.addActions([undo, redo])
        edit.addSeparator()
        for text, key, handler in [("Set Clip &Start Here", "I", self.preview.set_start_here),
                                   ("Set Clip &End Here", "O", self.preview.set_end_here)]:
            action = QAction(text, self)
            action.setShortcut(key)        # text fields still take the letters when focused
            action.triggered.connect(lambda _=False, h=handler: h())
            edit.addAction(action)
        render = self.menuBar().addMenu("&Render")
        export = QAction("&Export", self)
        export.setShortcut("Ctrl+E")
        export.triggered.connect(lambda _=False: self.export_bar.start_export())
        render.addAction(export)

    def show_load_warnings(self) -> None:
        """Tells the user about effect files that could not be loaded."""
        if self.catalog.failures:
            details = "\n\n".join(f"{f.module}:\n{f.error}" for f in self.catalog.failures)
            QMessageBox.warning(self, "Some effects could not be loaded", details)

    def _show_errors(self, errors: list[str]) -> None:
        QMessageBox.warning(self, "MoshFlow", "\n".join(errors))

    def _update_title(self) -> None:
        name = self.model.path or "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.model.dirty else ''} - MoshFlow")

    # --- project files ---------------------------------------------------
    def new_project(self) -> None:
        if self._confirm_discard():
            self.model.new_project()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if not path:
            return
        try:
            errors = self.model.open_project(path)
        except (OSError, ValueError, TypeError, KeyError) as e:
            QMessageBox.critical(self, "Open project", f"Cannot open {path}:\n{e}")
            return
        if errors:
            self._show_errors(errors)

    def save_project(self) -> bool:
        if not self.model.path:
            return self.save_project_as()
        return self._save_to(self.model.path)

    def save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.model.path or f"project{EXTENSION}", PROJECT_FILTER)
        return self._save_to(path) if path else False

    def _save_to(self, path: str) -> bool:
        try:
            self.model.save_project(path)
        except OSError as e:
            QMessageBox.critical(self, "Save project", f"Cannot save {path}:\n{e}")
            return False
        self._update_title()
        return True

    def _confirm_discard(self) -> bool:
        if not self.model.dirty:
            return True
        answer = QMessageBox.question(self, "MoshFlow", "Save changes to the current project?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save_project()
        return answer == QMessageBox.Discard

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self.preview.stop_worker()
        worker = self.export_bar.worker
        if worker:
            worker.cancel()
            worker.wait(CLOSE_WAIT_MS)
        event.accept()


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MoshFlow")
    apply_theme(app)
    window = MainWindow()
    window.show()
    window.show_load_warnings()
    return app.exec()
