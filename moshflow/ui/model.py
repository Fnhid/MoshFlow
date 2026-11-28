"""ProjectModel: the state all views share. Views edit it and listen to its signals.

Every change to the project goes through the undo stack. A Project is
immutable, so a history entry only keeps the project from before and after.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from moshflow.app.media import VideoInfo, probe
from moshflow.app.store import load_project, save_project
from moshflow.core.project import Clip, OutputSettings, Project, Transition

CLIP = "clip"
TRANSITION = "transition"

Selection = Optional[Tuple[str, int]]


class ProjectModel(QObject):
    changed = Signal()            # the project's contents changed
    selectionChanged = Signal()
    mediaChanged = Signal()
    projectReplaced = Signal()    # New / Open

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.media: list[VideoInfo] = []
        self.path: str | None = None
        self.selection: Selection = None
        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(lambda _: self.changed.emit())

    @property
    def dirty(self) -> bool:
        """True when the project differs from the last saved or opened version."""
        return not self.undo_stack.isClean()

    # --- media -----------------------------------------------------------
    def import_media(self, paths: list[str]) -> list[str]:
        """Adds videos to the media bin; returns error messages for bad files."""
        errors = []
        known = {info.path for info in self.media}
        for path in paths:
            if path in known:
                continue
            try:
                self.media.append(probe(path))
                known.add(path)
            except IOError as e:
                errors.append(str(e))
        self.mediaChanged.emit()
        return errors

    def info(self, path: str) -> VideoInfo | None:
        for info in self.media:
            if info.path == path:
                return info
        return None

    # --- edits -----------------------------------------------------------
    def add_clip(self, path: str, index: int | None = None) -> None:
        project = self.project.with_clip(Clip(path), index)
        self._edit("Add clip", project, (CLIP, len(project.clips) - 1 if index is None else index))

    def remove_clip(self, index: int) -> None:
        self._edit("Remove clip", self.project.without_clip(index), None)

    def move_clip(self, src: int, dst: int) -> None:
        self._edit("Move clip", self.project.with_clip_moved(src, dst), (CLIP, dst))

    def set_clip(self, index: int, clip: Clip, what: str, step: object = None) -> None:
        """Changes one clip. Repeated changes of the same `what` (and `step`) become one
        undo step; give each mouse drag its own `step` so two drags stay two steps."""
        self._edit(f"Change {what}", self.project.with_clip_replaced(index, clip), self.selection,
                   merge_key=(CLIP, index, what, step))

    def set_transition(self, index: int, transition: Transition, what: str) -> None:
        """Changes one transition. Repeated changes of the same `what` become one undo step."""
        self._edit(f"Change {what}", self.project.with_transition(index, transition), self.selection,
                   merge_key=(TRANSITION, index, what))

    def set_output(self, output: OutputSettings) -> None:
        self._edit("Change export settings", self.project.with_output(output), self.selection,
                   merge_key=("output",))

    def clip_duration(self, index: int) -> float:
        """Length of a clip on the timeline in seconds, after trimming."""
        clip = self.project.clips[index]
        info = self.info(clip.path)
        total = info.duration if info else 0.0
        end = total if clip.end is None else min(clip.end, total)
        return max(0.0, end - clip.start)

    def _edit(self, text: str, project: Project, selection: Selection, merge_key: tuple | None = None) -> None:
        if project != self.project:
            self.undo_stack.push(_ProjectChange(self, text, project, selection, merge_key))

    def _show(self, project: Project, selection: Selection, refresh_views: bool) -> None:
        """Makes `project` current. `refresh_views` rebuilds views that show their own copy of values."""
        self.project = project
        self.changed.emit()
        if refresh_views or selection != self.selection:
            self.select(*(selection or (None,)))

    # --- selection -------------------------------------------------------
    def select(self, kind: str | None, index: int = -1) -> None:
        self.selection = (kind, index) if kind else None
        self.selectionChanged.emit()

    # --- project files ---------------------------------------------------
    def new_project(self) -> None:
        self._replace(Project(), None)

    def open_project(self, path: str) -> list[str]:
        """Loads a project file; returns error messages for clips that cannot be read."""
        project = load_project(path)
        self._replace(project, path)
        return self.import_media(list(dict.fromkeys(clip.path for clip in project.clips)))

    def save_project(self, path: str) -> None:
        save_project(self.project, path)
        self.path = path
        self.undo_stack.setClean()

    def _replace(self, project: Project, path: str | None) -> None:
        self.project, self.path, self.selection = project, path, None
        self.undo_stack.clear()
        self.media = []
        self.mediaChanged.emit()
        self.projectReplaced.emit()
        self.changed.emit()
        self.selectionChanged.emit()


class _ProjectChange(QUndoCommand):
    """One entry in the undo history: the project before and after a change."""

    MERGEABLE_ID = 1

    def __init__(self, model: ProjectModel, text: str, after: Project, selection: Selection,
                 merge_key: tuple | None):
        super().__init__(text)
        self._model = model
        self._before, self._selection_before = model.project, model.selection
        self._after, self._selection_after = after, selection
        self._merge_key = merge_key
        self._applied_once = False

    def id(self) -> int:
        return self.MERGEABLE_ID if self._merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, _ProjectChange) or other._merge_key != self._merge_key:
            return False
        self._after, self._selection_after = other._after, other._selection_after
        return True

    def redo(self) -> None:
        # The first time, the view that made the change already shows the new value;
        # rebuilding it would interrupt the user, e.g. in the middle of a slider drag.
        self._model._show(self._after, self._selection_after, refresh_views=self._applied_once)
        self._applied_once = True

    def undo(self) -> None:
        self._model._show(self._before, self._selection_before, refresh_views=True)
