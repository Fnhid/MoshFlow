"""Inspector: settings of the selected transition or clip.

The controls of an effect are generated from its param_specs(), so a new effect
gets its UI without any change here.
"""

from __future__ import annotations

import os
import random
from dataclasses import replace
from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from moshflow.api import BoolParam, ChoiceParam, FloatParam, IntParam, ParamSpec, validate_params
from moshflow.app.catalog import EffectCatalog
from moshflow.core.project import ALIGNMENTS, Transition
from moshflow.ui.model import CLIP, TRANSITION, ProjectModel

ALIGNMENT_LABELS = {"before": "Before the cut", "center": "Centred on the cut", "after": "After the cut"}
SLIDER_STEPS = 1000
MAX_DURATION = 60.0
MAX_SEED = 999_999


def make_param_widget(spec: ParamSpec, value: object, on_change: Callable[[object], None]) -> QWidget:
    """Creates one control for a parameter specification."""
    if isinstance(spec, BoolParam):
        box = QCheckBox()
        box.setChecked(bool(value))
        box.toggled.connect(on_change)
        return box
    if isinstance(spec, ChoiceParam):
        combo = QComboBox()
        combo.addItems(spec.options)
        combo.setCurrentText(str(value))
        combo.currentTextChanged.connect(on_change)
        return combo
    if isinstance(spec, IntParam):
        spin = QSpinBox()
        spin.setRange(spec.min, spec.max)
        spin.setValue(int(value))
        spin.valueChanged.connect(on_change)
        return spin
    if isinstance(spec, FloatParam):
        return _make_float_widget(spec, float(value), on_change)
    raise TypeError(f"unknown parameter type: {spec!r}")


def _make_float_widget(spec: FloatParam, value: float, on_change: Callable[[object], None]) -> QWidget:
    span = (spec.max - spec.min) or 1.0
    spin = QDoubleSpinBox()
    spin.setDecimals(3)
    spin.setRange(spec.min, spec.max)
    spin.setSingleStep(span / 100)
    spin.setFixedWidth(74)
    spin.setValue(value)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, SLIDER_STEPS)
    slider.setValue(round((value - spec.min) / span * SLIDER_STEPS))

    def from_slider(position: int) -> None:
        spin.setValue(spec.min + position / SLIDER_STEPS * span)

    def from_spin(new_value: float) -> None:
        slider.blockSignals(True)
        slider.setValue(round((new_value - spec.min) / span * SLIDER_STEPS))
        slider.blockSignals(False)
        on_change(new_value)

    slider.valueChanged.connect(from_slider)
    spin.valueChanged.connect(from_spin)
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(slider, 1)
    layout.addWidget(spin)
    return row


def _make_card(title: str | None = None, action: QWidget | None = None) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setProperty("card", True)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    if title or action:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if title:
            lbl = QLabel(title)
            lbl.setProperty("role", "panelTitle")
            row.addWidget(lbl)
        row.addStretch(1)
        if action:
            row.addWidget(action)
        layout.addLayout(row)
    return card, layout


class Inspector(QFrame):
    def __init__(self, model: ProjectModel, catalog: EffectCatalog, parent=None):
        super().__init__(parent)
        self.setProperty("panel", True)
        self._model = model
        self._catalog = catalog
        self._advanced_open = False

        self._header_title = QLabel("INSPECTOR")
        self._header_title.setProperty("role", "panelTitle")

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addWidget(self._header_title)
        layout.addWidget(self._scroll, 1)

        model.selectionChanged.connect(self.rebuild)
        self.rebuild()

    def rebuild(self) -> None:
        selection = self._model.selection
        if selection and selection[0] == TRANSITION:
            self._header_title.setText("TRANSITION")
            content = self._transition_page(selection[1])
        elif selection and selection[0] == CLIP:
            self._header_title.setText("CLIP")
            content = self._clip_page(selection[1])
        else:
            self._header_title.setText("INSPECTOR")
            content = QLabel("Select a clip or transition on the timeline.")
            content.setProperty("role", "muted")
            content.setAlignment(Qt.AlignCenter)
            content.setWordWrap(True)
        self._show_page(content)

    def _show_page(self, page: QWidget) -> None:
        # The old page may be sending the signal that got us here (its combo box
        # or Remove button), so it must not be deleted until that signal returns.
        old = self._scroll.takeWidget()
        self._scroll.setWidget(page)
        if old is not None:
            old.deleteLater()

    # --- transition ------------------------------------------------------
    def _transition_page(self, index: int) -> QWidget:
        transition = self._model.project.transitions[index]
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 2, 2, 0)
        root.setSpacing(10)

        # Primary controls: Effect + Duration + Effect parameters right at the top.
        main_form = QFormLayout()
        main_form.setContentsMargins(0, 0, 0, 0)
        main_form.setSpacing(10)
        main_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        effect_box = QComboBox()
        for entry in self._catalog.entries():
            effect_box.addItem(entry.name, entry.id)
        position = effect_box.findData(transition.effect)
        if position < 0:
            effect_box.addItem(f"{transition.effect} (not available)", transition.effect)
            position = effect_box.count() - 1
        effect_box.setCurrentIndex(position)
        effect_box.currentIndexChanged.connect(lambda _: self._set_effect(index, effect_box.currentData()))
        main_form.addRow("Effect", effect_box)

        duration = QDoubleSpinBox()
        duration.setRange(0.0, MAX_DURATION)
        duration.setDecimals(2)
        duration.setSingleStep(0.1)
        duration.setSuffix(" s")
        duration.setValue(transition.duration)
        duration.valueChanged.connect(lambda v: self._update_transition(index, duration=v))
        main_form.addRow("Duration", duration)
        root.addLayout(main_form)

        entry = None
        try:
            entry = self._catalog.get(transition.effect)
        except KeyError:
            pass

        if entry is not None and entry.description:
            note = QLabel(entry.description)
            note.setProperty("role", "muted")
            note.setWordWrap(True)
            root.addWidget(note)

        # Effect-specific parameters immediately visible (keep float params after Duration).
        if entry is not None and entry.specs:
            reset_button = QPushButton("Reset")
            reset_button.setProperty("compact", True)
            reset_button.clicked.connect(lambda: self._reset_params(index))
            params_card, params_layout = _make_card("PARAMETERS", reset_button)
            params_form = QFormLayout()
            params_form.setContentsMargins(0, 0, 0, 0)
            params_form.setSpacing(10)
            values = self._current_params(transition, entry.specs)
            for spec in entry.specs:
                setter = self._param_setter(index, spec, entry.specs)
                params_form.addRow(spec.label, make_param_widget(spec, values[spec.id], setter))
            params_layout.addLayout(params_form)
            root.addWidget(params_card)

        # Secondary options tucked behind a 1-click toggle.
        adv_toggle = QPushButton("▾ Advanced" if self._advanced_open else "▸ Advanced")
        adv_toggle.setProperty("flatToggle", True)

        adv_card, adv_layout = _make_card()
        adv_form = QFormLayout()
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_form.setSpacing(8)

        alignment = QComboBox()
        for value in ALIGNMENTS:
            alignment.addItem(ALIGNMENT_LABELS[value], value)
        alignment.setCurrentIndex(ALIGNMENTS.index(transition.alignment))
        alignment.currentIndexChanged.connect(lambda _: self._update_transition(index, alignment=alignment.currentData()))
        adv_form.addRow("Alignment", alignment)

        seed = QSpinBox()
        seed.setRange(0, MAX_SEED)
        seed.setValue(transition.seed)
        seed.valueChanged.connect(lambda v: self._update_transition(index, seed=v))
        shuffle = QPushButton("Random")
        shuffle.setProperty("compact", True)
        shuffle.clicked.connect(lambda: seed.setValue(random.randint(0, MAX_SEED)))
        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(6)
        seed_layout.addWidget(seed, 1)
        seed_layout.addWidget(shuffle)
        adv_form.addRow("Seed", seed_row)

        adv_layout.addLayout(adv_form)
        adv_card.setVisible(self._advanced_open)

        def toggle_advanced() -> None:
            self._advanced_open = not self._advanced_open
            adv_card.setVisible(self._advanced_open)
            adv_toggle.setText("▾ Advanced" if self._advanced_open else "▸ Advanced")

        adv_toggle.clicked.connect(toggle_advanced)
        root.addWidget(adv_toggle)
        root.addWidget(adv_card)
        root.addStretch(1)
        return page

    @staticmethod
    def _current_params(transition: Transition, specs: Sequence[ParamSpec]) -> dict[str, object]:
        """The transition's parameter values, or the defaults if they do not fit its effect."""
        try:
            return validate_params(specs, transition.params)
        except ValueError:
            return validate_params(specs, {})

    def _param_setter(self, index: int, spec: ParamSpec, specs: Sequence[ParamSpec]) -> Callable[[object], None]:
        def setter(value: object) -> None:
            transition = self._model.project.transitions[index]
            params = {**self._current_params(transition, specs), spec.id: value}
            self._model.set_transition(index, replace(transition, params=params), spec.label)
        return setter

    def _update_transition(self, index: int, **changes: object) -> None:
        transition = self._model.project.transitions[index]
        self._model.set_transition(index, replace(transition, **changes), next(iter(changes)))

    def _set_effect(self, index: int, effect_id: str) -> None:
        self._update_transition(index, effect=effect_id, params={})
        self._show_page(self._transition_page(index))   # controls of the new effect

    def _reset_params(self, index: int) -> None:
        self._update_transition(index, params={})
        self._show_page(self._transition_page(index))

    # --- clip ------------------------------------------------------------
    def _clip_page(self, index: int) -> QWidget:
        clip = self._model.project.clips[index]
        info = self._model.info(clip.path)
        total = info.duration if info else 0.0
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 2, 2, 0)
        root.setSpacing(10)

        name_label = QLabel(os.path.basename(clip.path))
        name_label.setProperty("role", "sectionTitle")
        name_label.setWordWrap(True)
        root.addWidget(name_label)
        if not info:
            missing = QLabel("The file cannot be read.")
            missing.setProperty("role", "muted")
            root.addWidget(missing)

        trim_card, trim_layout = _make_card("TRIM")
        trim_form = QFormLayout()
        trim_form.setContentsMargins(0, 0, 0, 0)
        trim_form.setSpacing(10)

        start = QDoubleSpinBox()
        start.setRange(0.0, max(total, clip.start))
        start.setDecimals(2)
        start.setSuffix(" s")
        start.setValue(clip.start)
        start.valueChanged.connect(lambda v: self._update_clip(index, "trim start", start=v))
        trim_form.addRow("Start", start)

        # End shows the clip's real end time; at the maximum the clip plays to its end
        # (stored as None, so it still does if the file is replaced by a longer one).
        last = max(total, clip.end or 0.0)
        end = QDoubleSpinBox()
        end.setRange(0.0, last)
        end.setDecimals(2)
        end.setSuffix(" s")
        end.setValue(clip.end if clip.end is not None else last)
        end.valueChanged.connect(
            lambda v: self._update_clip(index, "trim end", end=None if v >= last - 0.005 else v))
        trim_form.addRow("End", end)
        trim_layout.addLayout(trim_form)
        root.addWidget(trim_card)

        remove = QPushButton("Remove from Timeline")
        remove.setProperty("danger", True)
        remove.clicked.connect(lambda: self._model.remove_clip(index))
        root.addWidget(remove)
        root.addStretch(1)
        return page

    def _update_clip(self, index: int, what: str, **changes: object) -> None:
        clip = self._model.project.clips[index]
        self._model.set_clip(index, replace(clip, **changes), what)
