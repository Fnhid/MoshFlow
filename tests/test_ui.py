"""UI tests, run without a visible window."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox

from moshflow.api import BoolParam, ChoiceParam, FloatParam, IntParam
from moshflow.core.project import Transition
from moshflow.ui.inspector import make_param_widget
from moshflow.ui.main_window import MainWindow
from moshflow.ui.model import CLIP, TRANSITION
from test_render import make_video


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path):
    win = MainWindow()
    win.show()
    paths = [make_video(tmp_path / f"{name}.mp4", value) for name, value in (("a", 40), ("b", 120), ("c", 200))]
    assert win.model.import_media(paths) == []
    win.paths = paths
    yield win
    win.model.undo_stack.setClean()     # no "Save changes?" question on close
    win.close()


def click(win, target):
    for rect, kind, index in win.timeline.regions:
        if (kind, index) == target:
            pos = win.timeline.mapFromScene(rect.center())
            QTest.mouseClick(win.timeline.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
            return
    raise AssertionError(f"{target} is not on the timeline")


def wait_for_export(app, bar, timeout=30.0):
    deadline = time.monotonic() + timeout
    while bar.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert bar.worker is None, "export did not finish"


def test_media_bin_lists_imported_videos(window):
    assert window.media_bin.list.count() == 3
    assert "a.mp4" in window.media_bin.list.item(0).text()


def test_bad_file_is_reported_not_raised(window, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    assert len(window.model.import_media([str(bad)])) == 1
    assert window.media_bin.list.count() == 3


def test_timeline_shows_clips_and_chips(window):
    for path in window.paths:
        window.model.add_clip(path)
    kinds = [(kind, index) for _, kind, index in window.timeline.regions]
    assert kinds.count((CLIP, 0)) == 1 and (CLIP, 2) in kinds
    assert (TRANSITION, 0) in kinds and (TRANSITION, 1) in kinds


def test_clicking_selects_and_fills_the_inspector(window):
    window.model.add_clip(window.paths[0])
    window.model.add_clip(window.paths[1])
    click(window, (TRANSITION, 0))
    assert window.model.selection == (TRANSITION, 0)
    combos = window.inspector.findChildren(QComboBox)
    assert combos and combos[0].findData("crossfade") >= 0
    click(window, (CLIP, 1))
    assert window.model.selection == (CLIP, 1)


def test_changing_the_effect_shows_its_parameters(window):
    window.model.add_clip(window.paths[0])
    window.model.add_clip(window.paths[1])
    click(window, (TRANSITION, 0))
    effect_box = window.inspector.findChildren(QComboBox)[0]
    effect_box.setCurrentIndex(effect_box.findData("crossfade"))
    transition = window.model.project.transitions[0]
    assert transition.effect == "crossfade"
    labels = [w.value() for w in window.inspector.findChildren(QDoubleSpinBox)]
    assert 1.0 in labels                                    # the default of "ease"
    assert window.model.dirty


def test_dragging_a_clip_reorders_it(window):
    for path in window.paths:
        window.model.add_clip(path)
    timeline, viewport = window.timeline, window.timeline.viewport()
    first = timeline._clip_rects[0]
    last = timeline._clip_rects[2]
    start = timeline.mapFromScene(first.center())
    end = timeline.mapFromScene(QPointF(last.right() - 2, last.center().y()))
    QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(viewport, end)
    QApplication.processEvents()
    assert timeline._marker is not None
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, end)
    assert [os.path.basename(c.path) for c in window.model.project.clips] == ["b.mp4", "c.mp4", "a.mp4"]


def test_delete_key_removes_the_selected_clip(window):
    window.model.add_clip(window.paths[0])
    window.model.add_clip(window.paths[1])
    click(window, (CLIP, 0))
    QTest.keyClick(window.timeline, Qt.Key_Delete)
    assert [os.path.basename(c.path) for c in window.model.project.clips] == ["b.mp4"]
    assert window.model.project.transitions == ()


def test_param_widget_factory():
    results = []
    assert isinstance(make_param_widget(IntParam("i", "I", 3, 1, 9), 3, results.append), QSpinBox)
    assert isinstance(make_param_widget(BoolParam("b", "B", True), True, results.append), QCheckBox)
    assert isinstance(make_param_widget(ChoiceParam("c", "C", "x", ("x", "y")), "x", results.append), QComboBox)
    row = make_param_widget(FloatParam("f", "F", 0.5, 0.0, 1.0), 0.5, results.append)
    row.findChild(QDoubleSpinBox).setValue(0.25)
    assert results[-1] == 0.25


def test_export_runs_in_the_background(app, window, tmp_path):
    window.model.add_clip(window.paths[0])
    window.model.add_clip(window.paths[1])
    window.model.set_transition(0, Transition("crossfade"), "effect")
    output = tmp_path / "out.mp4"
    bar = window.export_bar
    bar.run_worker(str(output))
    assert not bar.export_button.isEnabled()
    wait_for_export(app, bar)
    assert output.exists()
    assert bar.export_button.isEnabled() and bar.status.text().startswith("Done")
    assert bar.progress.value() == bar.progress.maximum()


def test_cancel_export_leaves_no_file(app, window, tmp_path):
    window.model.add_clip(window.paths[0])
    output = tmp_path / "cancelled.mp4"
    bar = window.export_bar
    bar.run_worker(str(output))
    bar.cancel_export()
    wait_for_export(app, bar)
    assert not output.exists()


def test_project_save_and_open(window, tmp_path):
    window.model.add_clip(window.paths[0])
    window.model.add_clip(window.paths[1])
    window.model.set_transition(0, Transition(duration=2.5), "duration")
    path = str(tmp_path / "demo.moshproj")
    window.model.save_project(path)
    assert not window.model.dirty
    window.model.new_project()
    assert window.model.project.clips == () and window.media_bin.list.count() == 0
    assert window.model.open_project(path) == []
    assert window.model.project.transitions[0].duration == 2.5
    assert window.media_bin.list.count() == 2
    assert len(window.timeline._clip_rects) == 2


def clip_names(window):
    return [os.path.basename(c.path) for c in window.model.project.clips]


def test_undo_and_redo_timeline_edits(window):
    stack = window.model.undo_stack
    for path in window.paths:
        window.model.add_clip(path)
    window.model.move_clip(0, 2)
    window.model.remove_clip(0)
    assert clip_names(window) == ["c.mp4", "a.mp4"]
    stack.undo()
    assert clip_names(window) == ["b.mp4", "c.mp4", "a.mp4"]
    stack.undo()
    assert clip_names(window) == ["a.mp4", "b.mp4", "c.mp4"]
    stack.redo()
    assert clip_names(window) == ["b.mp4", "c.mp4", "a.mp4"]
    while stack.canUndo():
        stack.undo()
    assert clip_names(window) == [] and not window.model.dirty


def test_one_slider_drag_is_one_undo_step(window):
    window.model.add_clip(window.paths[0])
    window.model.add_clip(window.paths[1])
    click(window, (TRANSITION, 0))
    effect_box = window.inspector.findChildren(QComboBox)[0]
    effect_box.setCurrentIndex(effect_box.findData("crossfade"))
    ease = window.inspector.findChildren(QDoubleSpinBox)[-1]     # Duration first, then the effect's "ease"
    for value in (1.5, 2.0, 2.5):
        ease.setValue(value)
    assert window.model.project.transitions[0].params == {"ease": 2.5}
    window.model.undo_stack.undo()
    assert window.model.project.transitions[0].effect == "crossfade"
    assert window.model.project.transitions[0].params == {}
    assert window.inspector.findChildren(QDoubleSpinBox)[-1].value() == 1.0   # the Inspector shows it too


def test_undo_restores_export_settings(window):
    bar = window.export_bar
    bar.width_spin.setValue(320)
    bar.height_spin.setValue(180)
    assert (window.model.project.output.width, window.model.project.output.height) == (320, 180)
    window.model.undo_stack.undo()
    assert window.model.project.output.width is None
    assert (bar.width_spin.value(), bar.height_spin.value()) == (0, 0)


def test_saving_marks_the_project_clean(window, tmp_path):
    window.model.add_clip(window.paths[0])
    assert window.model.dirty and window.windowTitle().startswith("Untitled *")
    window.model.save_project(str(tmp_path / "p.moshproj"))
    assert not window.model.dirty and "*" not in window.windowTitle()
    window.model.add_clip(window.paths[1])
    window.model.undo_stack.undo()
    assert not window.model.dirty and "*" not in window.windowTitle()


def test_edit_menu_has_undo_and_redo(window):
    actions = {a.text().replace("&", ""): a for a in window.menuBar().actions()}
    edit = {a.text().replace("&", "").split(" ")[0]: a for a in actions["Edit"].menu().actions()}
    assert edit["Undo"].shortcut() == QKeySequence(QKeySequence.Undo)
    window.model.add_clip(window.paths[0])
    assert edit["Undo"].isEnabled() and "Add clip" in edit["Undo"].text()
    edit["Undo"].trigger()
    assert window.model.project.clips == ()


def test_relative_output_goes_next_to_the_first_clip(window):
    window.model.add_clip(window.paths[0])
    assert window.export_bar.resolved_output() == os.path.join(os.path.dirname(window.paths[0]), "output.mp4")


def test_frame_to_pixmap_preserves_aspect_ratio():
    import numpy as np
    from moshflow.ui.frame_display import frame_to_pixmap

    frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    pixmap = frame_to_pixmap(frame, (80, 80))
    assert not pixmap.isNull()
    assert (pixmap.width(), pixmap.height()) == (80, 45)


def test_preview_panel_renders_selected_transition(app, window):
    window.model.add_clip(window.paths[0])
    window.model.add_clip(window.paths[1])
    window.model.set_transition(0, Transition("crossfade", duration=0.4), "effect")
    window.model.select(TRANSITION, 0)

    window.preview.start_transition_preview()
    deadline = time.monotonic() + 15.0
    while window.preview._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()

    assert window.preview._worker is None
    assert window.preview._rendered is not None
    assert len(window.preview._rendered.frames) > 0
    assert window.preview.scrubber.maximum() == len(window.preview._rendered.frames) - 1
    window.preview.stop_worker()


def test_export_preset_updates_dimensions_and_undoes(window):
    bar = window.export_bar
    idx_720 = bar.preset_box.findData("1280x720")
    assert idx_720 >= 0
    bar.preset_box.setCurrentIndex(idx_720)
    assert (window.model.project.output.width, window.model.project.output.height) == (1280, 720)
    assert bar.preset_box.currentData() == "1280x720"
    window.model.undo_stack.undo()
    assert window.model.project.output.width is None
    assert bar.preset_box.currentData() == "auto"


def test_timeline_zoom_scales_clip_widths(window):
    window.model.add_clip(window.paths[0])
    # Add a longer video width test by checking zoom ratio changes pixels per second.
    before = window.timeline._pixels_per_second
    window.timeline.zoom_in()
    assert window.timeline._pixels_per_second > before
    window.timeline.zoom_out()
    assert abs(window.timeline._pixels_per_second - before) < 1e-6


def test_launcher_falls_back_to_local_venv(monkeypatch, tmp_path):
    import run as launcher

    fake_venv = tmp_path / ".venv" / "Scripts" / "python.exe"
    fake_venv.parent.mkdir(parents=True)
    fake_venv.write_text("", encoding="utf-8")

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_missing_packages", lambda: ["PySide6"])
    monkeypatch.setattr(launcher, "_python_is_ready", lambda p: p == fake_venv)
    monkeypatch.delenv(launcher.REEXEC_GUARD, raising=False)
    called = []
    monkeypatch.setattr(launcher, "_reexec", lambda exe, args: called.append((exe, args)) or 0)

    assert launcher.launch(["--flag"]) == 0
    assert called == [(fake_venv, ["--flag"])]


def test_launcher_prints_guide_when_python_too_old(monkeypatch, capsys):
    import run as launcher

    monkeypatch.setattr(launcher.sys, "version_info", (3, 9, 13, "final", 0))
    monkeypatch.setattr(launcher, "_missing_packages", lambda: [])
    monkeypatch.setattr(launcher, "_find_ready_venv", lambda root: None)

    assert launcher.launch([]) == 1
    err = capsys.readouterr().err
    assert "Python 3.10" in err
    assert "requirements.txt" in err


def test_clip_end_shows_the_clip_length_by_default(window):
    window.model.add_clip(window.paths[0])                 # 30 frames at 10 fps: 3 s
    click(window, (CLIP, 0))
    start_box, end_box = window.inspector.findChildren(QDoubleSpinBox)[:2]
    assert end_box.value() == pytest.approx(3.0) and window.model.project.clips[0].end is None
    end_box.setValue(2.0)
    assert window.model.project.clips[0].end == pytest.approx(2.0)
    end_box.setValue(end_box.maximum())                   # back to the end: plays to the end again
    assert window.model.project.clips[0].end is None


def drag(window, start, end):
    viewport = window.timeline.viewport()
    QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(viewport, end)
    QApplication.processEvents()
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, end)


def test_dragging_a_clip_edge_trims_it(window):
    window.model.add_clip(window.paths[0])                 # 3 s
    window.model.add_clip(window.paths[1])
    timeline = window.timeline
    rect = timeline._clip_rects[0]
    pps = timeline._pixels_per_second
    y = rect.center().y()
    # Right edge 1 s to the left: the clip ends at 2 s.
    drag(window, timeline.mapFromScene(QPointF(rect.right(), y)), timeline.mapFromScene(QPointF(rect.right() - pps, y)))
    assert window.model.project.clips[0].end == pytest.approx(2.0, abs=0.05)
    # Left edge 0.5 s to the right: the clip starts at 0.5 s.
    rect = timeline._clip_rects[0]
    drag(window, timeline.mapFromScene(QPointF(rect.left(), y)), timeline.mapFromScene(QPointF(rect.left() + pps / 2, y)))
    assert window.model.project.clips[0].start == pytest.approx(0.5, abs=0.05)
    assert [os.path.basename(c.path) for c in window.model.project.clips] == ["a.mp4", "b.mp4"]   # not reordered
    window.model.undo_stack.undo()                         # each drag is one undo step
    assert window.model.project.clips[0].start == 0.0
    assert window.model.project.clips[0].end == pytest.approx(2.0, abs=0.05)


def test_set_start_and_end_at_the_frame_shown(window):
    window.model.add_clip(window.paths[0])                 # 30 frames at 10 fps
    window.model.select(CLIP, 0)
    preview = window.preview
    preview.scrubber.setValue(5)                           # 0.5 s
    preview.set_start_here()
    assert window.model.project.clips[0].start == pytest.approx(0.5)
    preview.scrubber.setValue(10)                          # 1.0 s after the new start: 1.5 s in the file
    preview.set_end_here()
    assert window.model.project.clips[0].end == pytest.approx(1.6)   # the frame shown stays in


def test_i_and_o_keys_trim_the_selected_clip(window):
    window.model.add_clip(window.paths[0])
    window.model.select(CLIP, 0)
    window.preview.scrubber.setValue(5)
    edit = {a.text().replace("&", ""): a for a in window.menuBar().actions()}["Edit"].menu()
    start = next(a for a in edit.actions() if a.shortcut() == QKeySequence("I"))
    start.trigger()
    assert window.model.project.clips[0].start == pytest.approx(0.5)
