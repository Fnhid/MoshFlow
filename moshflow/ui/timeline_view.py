"""TimelineView: clips from left to right with a timecode ruler and transition chips.

Click selects a clip or a chip, dragging a clip reorders it, dragging a clip's left
or right edge trims it, Delete removes it, and clips can be dropped in from the
media bin or the file manager.
"""

from __future__ import annotations

import os
from dataclasses import replace
from itertools import count

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

from moshflow.app.catalog import EffectCatalog
from moshflow.app.media import read_frame
from moshflow.ui.frame_display import frame_to_pixmap
from moshflow.ui.media_bin import CLIP_MIME, local_paths
from moshflow.ui.model import CLIP, TRANSITION, ProjectModel
from moshflow.ui.theme import (
    CHIP_BORDER,
    CHIP_FILL,
    CLIP_BORDER,
    CLIP_FILL,
    CLIP_STRIPE,
    HARD_CUT_BORDER,
    HARD_CUT_FILL,
    RULER_BG,
    RULER_LINE,
    RULER_TEXT,
    SELECTED_RING,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TIMELINE_BG,
)

DEFAULT_PIXELS_PER_SECOND = 36.0
MIN_PIXELS_PER_SECOND = 16.0
MAX_PIXELS_PER_SECOND = 96.0
MIN_CLIP_WIDTH = 152
MAX_CLIP_WIDTH = 440
CHIP_WIDTH = 114
CHIP_GAP = 8
RULER_HEIGHT = 24
CLIP_TOP = RULER_HEIGHT + 14
CLIP_HEIGHT = 68
CHIP_HEIGHT = 46
THUMB_SIZE = (54, 34)
EDGE = 7                 # pixels on each side of a clip edge that grab it for trimming
MIN_CLIP_SECONDS = 0.1   # a trimmed clip keeps at least this length


class TimelineView(QGraphicsView):
    errors = Signal(list)

    def __init__(self, model: ProjectModel, catalog: EffectCatalog, parent=None):
        super().__init__(parent)
        self._model = model
        self._catalog = catalog
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setRenderHint(QPainter.Antialiasing)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFixedHeight(134)
        self.setBackgroundBrush(QBrush(TIMELINE_BG))

        self._pixels_per_second = DEFAULT_PIXELS_PER_SECOND
        self._thumbs: dict[str, QPixmap] = {}
        self.regions: list[tuple[QRectF, str, int]] = []
        self._clip_rects: list[QRectF] = []
        self._press_pos: QPointF | None = None
        self._drag_clip: int | None = None
        self._marker = None
        self._trim: tuple[int, str, float, float, int] | None = None   # index, side, press x, value, step
        self._drags = count()
        self.viewport().setMouseTracking(True)

        model.changed.connect(self.rebuild)
        model.selectionChanged.connect(self.rebuild)
        self.rebuild()

    # --- zoom ------------------------------------------------------------
    def zoom_in(self) -> None:
        self._pixels_per_second = min(MAX_PIXELS_PER_SECOND, self._pixels_per_second * 1.25)
        self.rebuild()

    def zoom_out(self) -> None:
        self._pixels_per_second = max(MIN_PIXELS_PER_SECOND, self._pixels_per_second / 1.25)
        self.rebuild()

    # --- drawing ---------------------------------------------------------
    def _thumbnail(self, path: str) -> QPixmap | None:
        if path in self._thumbs:
            return self._thumbs[path]
        frame = read_frame(path, 0)
        if frame is None:
            return None
        pixmap = frame_to_pixmap(frame, THUMB_SIZE)
        self._thumbs[path] = pixmap
        return pixmap

    def rebuild(self) -> None:
        self._scene.clear()
        self._marker = None
        self.regions, self._clip_rects = [], []
        project = self._model.project
        if not project.clips:
            self._draw_empty_state()
            return

        x = 12.0
        elapsed = 0.0
        ruler_marks: list[tuple[float, float]] = [(x, 0.0)]
        for i, clip in enumerate(project.clips):
            seconds = self._model.clip_duration(i)
            width = min(MAX_CLIP_WIDTH, max(MIN_CLIP_WIDTH, seconds * self._pixels_per_second))
            rect = QRectF(x, CLIP_TOP, width, CLIP_HEIGHT)
            self._clip_rects.append(rect)
            self._add_clip_card(rect, i, clip.path, seconds)
            elapsed += seconds
            x += width + CHIP_GAP
            ruler_marks.append((x - CHIP_GAP / 2, elapsed))

            if i < len(project.transitions):
                transition = project.transitions[i]
                chip_top = CLIP_TOP + (CLIP_HEIGHT - CHIP_HEIGHT) / 2
                chip = QRectF(x, chip_top, CHIP_WIDTH, CHIP_HEIGHT)
                self._add_transition_chip(chip, i, transition.effect, transition.duration)
                x += CHIP_WIDTH + CHIP_GAP

        total_width = max(x + 24.0, float(self.viewport().width() - 4))
        self._draw_ruler(total_width, ruler_marks)
        self._scene.setSceneRect(0.0, 0.0, total_width, CLIP_TOP + CLIP_HEIGHT + 16.0)

    def _draw_empty_state(self) -> None:
        width = max(520.0, float(self.viewport().width() - 28))
        box = QRectF(12.0, 18.0, width, 86.0)
        path = QPainterPath()
        path.addRoundedRect(box, 8, 8)
        pen = QPen(RULER_LINE, 1.5, Qt.DashLine)
        self._scene.addPath(path, pen, QBrush(RULER_BG))
        hint = QGraphicsSimpleTextItem("Drop video clips here from the Media Bin or your file manager")
        hint.setBrush(QBrush(RULER_TEXT))
        bounds = hint.boundingRect()
        hint.setPos(box.center().x() - bounds.width() / 2, box.center().y() - bounds.height() / 2)
        self._scene.addItem(hint)
        self._scene.setSceneRect(0.0, 0.0, width + 24.0, 120.0)

    def _draw_ruler(self, total_width: float, marks: list[tuple[float, float]]) -> None:
        self._scene.addRect(QRectF(0, 0, total_width, RULER_HEIGHT), QPen(Qt.NoPen), QBrush(RULER_BG))
        self._scene.addLine(0, RULER_HEIGHT, total_width, RULER_HEIGHT, QPen(RULER_LINE, 1))
        font = QFont(self.font())
        font.setPointSize(max(8, font.pointSize() - 1))
        for x_pos, sec in marks:
            self._scene.addLine(x_pos, RULER_HEIGHT - 7, x_pos, RULER_HEIGHT, QPen(RULER_TEXT, 1))
            mins = int(sec // 60)
            rem = sec - mins * 60
            label = QGraphicsSimpleTextItem(f"{mins}:{rem:04.1f}")
            label.setFont(font)
            label.setBrush(QBrush(RULER_TEXT))
            label.setPos(x_pos + 4, 4)
            self._scene.addItem(label)

    def _add_clip_card(self, rect: QRectF, index: int, clip_path: str, seconds: float) -> None:
        selected = self._model.selection == (CLIP, index)
        card_path = QPainterPath()
        card_path.addRoundedRect(rect, 7, 7)
        pen = QPen(SELECTED_RING, 2.2) if selected else QPen(CLIP_BORDER, 1.2)
        card = self._scene.addPath(card_path, pen, QBrush(CLIP_FILL))

        # Top accent stripe
        stripe_rect = QRectF(rect.left() + 1, rect.top() + 1, rect.width() - 2, 4)
        stripe_path = QPainterPath()
        stripe_path.addRoundedRect(stripe_rect, 2, 2)
        self._scene.addPath(stripe_path, QPen(Qt.NoPen), QBrush(SELECTED_RING if selected else CLIP_STRIPE))

        text_left = rect.left() + 10
        thumb = self._thumbnail(clip_path)
        if thumb is not None and rect.width() >= MIN_CLIP_WIDTH:
            pix_item = self._scene.addPixmap(thumb)
            pix_item.setParentItem(card)
            pix_y = rect.top() + 8 + (rect.height() - 12 - thumb.height()) / 2
            pix_item.setPos(rect.left() + 8, pix_y)
            text_left = rect.left() + 8 + thumb.width() + 8

        avail = max(36, int(rect.right() - text_left - 8))
        bold_font = QFont(self.font())
        bold_font.setWeight(QFont.DemiBold)
        name_text = QFontMetrics(bold_font).elidedText(os.path.basename(clip_path), Qt.ElideMiddle, avail)
        name_item = QGraphicsSimpleTextItem(name_text, card)
        name_item.setFont(bold_font)
        name_item.setBrush(QBrush(QColor(TEXT_PRIMARY)))
        name_item.setPos(text_left, rect.top() + 15)

        sub_font = QFont(self.font())
        sub_font.setPointSize(max(8, sub_font.pointSize() - 1))
        sub_text = QFontMetrics(sub_font).elidedText(f"{seconds:.1f} s", Qt.ElideRight, avail)
        sub_item = QGraphicsSimpleTextItem(sub_text, card)
        sub_item.setFont(sub_font)
        sub_item.setBrush(QBrush(QColor(TEXT_SECONDARY)))
        sub_item.setPos(text_left, rect.top() + 36)

        self.regions.append((rect, CLIP, index))

    def _add_transition_chip(self, rect: QRectF, index: int, effect_id: str, duration: float) -> None:
        selected = self._model.selection == (TRANSITION, index)
        is_cut = effect_id == "hard_cut"
        fill = HARD_CUT_FILL if is_cut else CHIP_FILL
        border = SELECTED_RING if selected else (HARD_CUT_BORDER if is_cut else CHIP_BORDER)

        chip_path = QPainterPath()
        chip_path.addRoundedRect(rect, 9, 9)
        chip = self._scene.addPath(chip_path, QPen(border, 2.2 if selected else 1.2), QBrush(fill))

        avail = max(24, int(rect.width() - 16))
        title_font = QFont(self.font())
        title_font.setWeight(QFont.DemiBold)
        title_text = QFontMetrics(title_font).elidedText(self._effect_name(effect_id), Qt.ElideRight, avail)
        title_item = QGraphicsSimpleTextItem(title_text, chip)
        title_item.setFont(title_font)
        title_item.setBrush(QBrush(QColor("#FFFFFF")))
        title_box = title_item.boundingRect()
        title_item.setPos(rect.center().x() - title_box.width() / 2, rect.top() + 7)

        sub_font = QFont(self.font())
        sub_font.setPointSize(max(8, sub_font.pointSize() - 1))
        sub_item = QGraphicsSimpleTextItem(f"{duration:g} s", chip)
        sub_item.setFont(sub_font)
        sub_item.setBrush(QBrush(QColor("#FFE1EE" if not is_cut else TEXT_SECONDARY)))
        sub_box = sub_item.boundingRect()
        sub_item.setPos(rect.center().x() - sub_box.width() / 2, rect.top() + 25)

        self.regions.append((rect, TRANSITION, index))

    def _effect_name(self, effect_id: str) -> str:
        try:
            return self._catalog.get(effect_id).name
        except KeyError:
            return f"{effect_id}?"

    # --- hit testing -----------------------------------------------------
    def _hit(self, scene_pos: QPointF) -> tuple[str, int] | None:
        for rect, kind, index in self.regions:
            if rect.contains(scene_pos):
                return kind, index
        return None

    def _edge(self, scene_pos: QPointF) -> tuple[int, str] | None:
        """(clip index, "start" or "end") when the point is on a clip's left or right edge."""
        for index, rect in enumerate(self._clip_rects):
            if rect.top() <= scene_pos.y() <= rect.bottom():
                if abs(scene_pos.x() - rect.left()) <= EDGE:
                    return index, "start"
                if abs(scene_pos.x() - rect.right()) <= EDGE:
                    return index, "end"
        return None

    def _trimmed(self, index: int, side: str, value: float, dx: float):
        """The clip with its start or end moved by dx pixels, kept inside the file."""
        clip = self._model.project.clips[index]
        info = self._model.info(clip.path)
        total = info.duration if info else value
        end = total if clip.end is None else min(clip.end, total)
        seconds = value + dx / self._pixels_per_second
        if side == "start":
            return replace(clip, start=round(min(max(seconds, 0.0), end - MIN_CLIP_SECONDS), 2))
        seconds = round(min(max(seconds, clip.start + MIN_CLIP_SECONDS), total), 2)
        return replace(clip, end=None if seconds >= total - 0.005 else seconds)

    def _insert_index(self, x: float) -> int:
        return sum(1 for rect in self._clip_rects if rect.center().x() < x)

    def _marker_x(self, index: int) -> float:
        if not self._clip_rects:
            return 12.0
        if index >= len(self._clip_rects):
            return self._clip_rects[-1].right() + CHIP_GAP / 2
        return self._clip_rects[index].left() - CHIP_GAP / 2

    def _show_marker(self, index: int) -> None:
        self._hide_marker()
        x = self._marker_x(index)
        path = QPainterPath()
        path.moveTo(x - 6, RULER_HEIGHT)
        path.lineTo(x + 6, RULER_HEIGHT)
        path.lineTo(x, RULER_HEIGHT + 8)
        path.closeSubpath()
        path.moveTo(x, RULER_HEIGHT + 8)
        path.lineTo(x, CLIP_TOP + CLIP_HEIGHT + 6)
        self._marker = self._scene.addPath(path, QPen(SELECTED_RING, 2.5), QBrush(SELECTED_RING))

    def _hide_marker(self) -> None:
        if self._marker is not None:
            self._scene.removeItem(self._marker)
            self._marker = None

    # --- mouse and keyboard ----------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        self.setFocus()
        scene_pos = self.mapToScene(event.position().toPoint())
        edge = self._edge(scene_pos)
        if edge is not None:                      # start trimming that edge
            index, side = edge
            clip = self._model.project.clips[index]
            info = self._model.info(clip.path)
            total = info.duration if info else 0.0
            value = clip.start if side == "start" else (total if clip.end is None else clip.end)
            self._trim = (index, side, scene_pos.x(), value, next(self._drags))
            self._model.select(CLIP, index)
            return
        hit = self._hit(scene_pos)
        self._press_pos = event.position()
        self._drag_clip = hit[1] if hit and hit[0] == CLIP else None
        if hit:
            self._model.select(*hit)
        else:
            self._model.select(None)

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._trim is not None:
            index, side, press_x, value, step = self._trim
            clip = self._trimmed(index, side, value, scene_pos.x() - press_x)
            self._model.set_clip(index, clip, f"trim {side}", step)
            return
        if self._drag_clip is None:
            self.viewport().setCursor(Qt.SizeHorCursor if self._edge(scene_pos) else Qt.ArrowCursor)
            return
        if self._press_pos is None:
            return
        if (event.position() - self._press_pos).manhattanLength() < QApplication.startDragDistance():
            return
        scene_x = self.mapToScene(event.position().toPoint()).x()
        self._show_marker(self._insert_index(scene_x))

    def mouseReleaseEvent(self, event) -> None:
        if self._trim is not None:
            index = self._trim[0]
            self._trim = None
            self._model.select(CLIP, index)       # the Inspector shows the new start and end
            return
        moving = self._marker is not None and self._drag_clip is not None
        self._hide_marker()
        src, self._drag_clip, self._press_pos = self._drag_clip, None, None
        if moving:
            target = self._insert_index(self.mapToScene(event.position().toPoint()).x())
            if target > src:
                target -= 1
            if target != src:
                self._model.move_clip(src, target)

    def keyPressEvent(self, event) -> None:
        selection = self._model.selection
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and selection and selection[0] == CLIP:
            self._model.remove_clip(selection[1])
        else:
            super().keyPressEvent(event)

    # --- drops -----------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(CLIP_MIME) or local_paths(event.mimeData()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        self._show_marker(self._insert_index(self.mapToScene(event.position().toPoint()).x()))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._hide_marker()

    def dropEvent(self, event) -> None:
        self._hide_marker()
        index = self._insert_index(self.mapToScene(event.position().toPoint()).x())
        mime = event.mimeData()
        if mime.hasFormat(CLIP_MIME):
            paths = [bytes(mime.data(CLIP_MIME)).decode("utf-8")]
        else:
            paths = local_paths(mime)
            errors = self._model.import_media(paths)
            if errors:
                self.errors.emit(errors)
            paths = [p for p in paths if self._model.info(p)]
        for offset, path in enumerate(paths):
            self._model.add_clip(path, index + offset)
        event.acceptProposedAction()
