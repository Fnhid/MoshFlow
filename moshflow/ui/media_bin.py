"""MediaBin: the imported videos. Clips are dragged from here onto the timeline."""

from __future__ import annotations

from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from moshflow.app.media import read_frame
from moshflow.ui.frame_display import frame_to_pixmap
from moshflow.ui.model import ProjectModel

CLIP_MIME = "application/x-moshflow-clip"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)"
THUMB_SIZE = (76, 44)


def local_paths(mime: QMimeData) -> list[str]:
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]


class _MediaList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.ElideMiddle)
        self.setIconSize(QSize(*THUMB_SIZE))
        self.setSpacing(2)

    def mimeTypes(self):
        return [CLIP_MIME]

    def mimeData(self, items):
        mime = QMimeData()
        if items:
            mime.setData(CLIP_MIME, items[0].data(Qt.UserRole).encode("utf-8"))
        return mime


class MediaBin(QFrame):
    errors = Signal(list)

    def __init__(self, model: ProjectModel, parent=None):
        super().__init__(parent)
        self.setProperty("panel", True)
        self._model = model
        self._icons: dict[str, QIcon] = {}
        self.setAcceptDrops(True)

        title = QLabel("MEDIA")
        title.setProperty("role", "panelTitle")

        self.list = _MediaList()
        self.list.itemDoubleClicked.connect(lambda item: model.add_clip(item.data(Qt.UserRole)))

        self._empty_hint = QLabel("Drop videos here or click + Import")
        self._empty_hint.setProperty("role", "muted")
        self._empty_hint.setAlignment(Qt.AlignCenter)
        self._empty_hint.setWordWrap(True)

        import_button = QPushButton("+ Import")
        import_button.clicked.connect(self._choose_files)
        add_button = QPushButton("+ Timeline")
        add_button.clicked.connect(self._add_selected)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addWidget(import_button)
        buttons.addWidget(add_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self._empty_hint)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)

        model.mediaChanged.connect(self._refresh)
        self._refresh()

    def _thumbnail_icon(self, path: str) -> QIcon | None:
        if path in self._icons:
            return self._icons[path]
        frame = read_frame(path, 0)
        if frame is None:
            return None
        icon = QIcon(frame_to_pixmap(frame, THUMB_SIZE))
        self._icons[path] = icon
        return icon

    def _refresh(self) -> None:
        self.list.clear()
        self._empty_hint.setVisible(len(self._model.media) == 0)
        for info in self._model.media:
            item = QListWidgetItem(f"{info.name}\n{info.duration:.1f} s")
            item.setData(Qt.UserRole, info.path)
            item.setToolTip(f"{info.path}\n{info.width}×{info.height} · {info.fps:.2g} fps · {info.duration:.2f} s")
            icon = self._thumbnail_icon(info.path)
            if icon is not None:
                item.setIcon(icon)
            self.list.addItem(item)

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import videos", "", VIDEO_FILTER)
        self._import(paths)

    def _import(self, paths: list[str]) -> None:
        errors = self._model.import_media(paths)
        if errors:
            self.errors.emit(errors)

    def _add_selected(self) -> None:
        item = self.list.currentItem()
        if item:
            self._model.add_clip(item.data(Qt.UserRole))

    def dragEnterEvent(self, event) -> None:
        if local_paths(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self._import(local_paths(event.mimeData()))
        event.acceptProposedAction()
