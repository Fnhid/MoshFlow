"""Visual theme constants and Qt stylesheet for MoshFlow."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

BG_APP = "#111017"
BG_PANEL = "#181622"
BG_SURFACE = "#211E2E"
BG_ELEVATED = "#2A263A"
BG_CANVAS = "#0B0A10"
BORDER_SUBTLE = "#2D293E"
BORDER_STRONG = "#3F3957"

TEXT_PRIMARY = "#F2F0F8"
TEXT_SECONDARY = "#A39DB8"
TEXT_MUTED = "#726C86"

ACCENT_PRIMARY = "#DC2F78"
ACCENT_PRIMARY_HOVER = "#EE428B"
ACCENT_PRIMARY_PRESSED = "#B92261"
ACCENT_CYAN = "#2DE2E6"
ACCENT_WARN = "#E5A443"

CLIP_FILL = QColor("#2A263C")
CLIP_STRIPE = QColor("#6E59B5")
CLIP_BORDER = QColor("#433C5E")
CHIP_FILL = QColor("#C92A6E")
CHIP_BORDER = QColor("#EC5996")
HARD_CUT_FILL = QColor("#3A354C")
HARD_CUT_BORDER = QColor("#595272")
SELECTED_RING = QColor("#2DE2E6")
RULER_BG = QColor("#15131E")
RULER_LINE = QColor("#2E2A40")
RULER_TEXT = QColor("#8D87A3")
TIMELINE_BG = QColor("#13111B")


STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_APP};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", "Malgun Gothic", sans-serif;
    font-size: 12px;
}}

QMenuBar {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER_SUBTLE};
    padding: 2px 6px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background-color: {BG_ELEVATED};
}}
QMenu {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {ACCENT_PRIMARY};
    color: #FFFFFF;
}}
QMenu::item:disabled {{
    color: {TEXT_MUTED};
}}

QFrame[panel="true"] {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}

QFrame[card="true"] {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
}}

QFrame[monitor="true"] {{
    background-color: {BG_CANVAS};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
}}

QLabel {{
    background: transparent;
    color: {TEXT_PRIMARY};
}}
QLabel[role="panelTitle"] {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
}}
QLabel[role="sectionTitle"] {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: 600;
}}
QLabel[role="muted"] {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}
QLabel[role="badge"] {{
    background-color: {BG_ELEVATED};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel[role="accentBadge"] {{
    background-color: rgba(220, 47, 120, 48);
    color: #FF88BA;
    border: 1px solid rgba(220, 47, 120, 140);
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel[role="cyanBadge"] {{
    background-color: rgba(45, 226, 230, 32);
    color: {ACCENT_CYAN};
    border: 1px solid rgba(45, 226, 230, 120);
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel[role="draftBadge"] {{
    background-color: rgba(229, 164, 67, 36);
    color: {ACCENT_WARN};
    border: 1px solid rgba(229, 164, 67, 130);
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}}
QLabel[role="timecode"] {{
    color: {ACCENT_CYAN};
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 12px;
    font-weight: 600;
}}

QSplitter::handle {{
    background-color: {BG_APP};
}}
QSplitter::handle:horizontal {{
    width: 6px;
}}
QSplitter::handle:vertical {{
    height: 6px;
}}
QSplitter::handle:hover {{
    background-color: {BORDER_STRONG};
}}

QListWidget {{
    background-color: {BG_CANVAS};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    outline: none;
    padding: 4px;
}}
QListWidget::item {{
    background-color: {BG_SURFACE};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px;
    margin: 2px 0;
}}
QListWidget::item:hover {{
    background-color: {BG_ELEVATED};
    border-color: {BORDER_STRONG};
}}
QListWidget::item:selected {{
    background-color: #2E2844;
    border-color: {ACCENT_CYAN};
    color: {TEXT_PRIMARY};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {ACCENT_PRIMARY};
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: #5A527A;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_CYAN};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_PRIMARY};
    selection-color: #FFFFFF;
    outline: none;
}}

QPushButton {{
    background-color: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 5px 12px;
    font-weight: 500;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: #35304A;
    border-color: #5A527A;
}}
QPushButton:pressed {{
    background-color: {BG_SURFACE};
}}
QPushButton:disabled {{
    background-color: {BG_SURFACE};
    color: {TEXT_MUTED};
    border-color: {BORDER_SUBTLE};
}}
QPushButton[accent="true"] {{
    background-color: {ACCENT_PRIMARY};
    color: #FFFFFF;
    border: 1px solid #F05497;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background-color: {ACCENT_PRIMARY_HOVER};
}}
QPushButton[accent="true"]:pressed {{
    background-color: {ACCENT_PRIMARY_PRESSED};
}}
QPushButton[accent="true"]:disabled {{
    background-color: #57253B;
    color: #A88496;
    border-color: #6B2D49;
}}
QPushButton[danger="true"] {{
    background-color: rgba(201, 52, 74, 38);
    color: #FF94A3;
    border: 1px solid rgba(201, 52, 74, 130);
}}
QPushButton[danger="true"]:hover {{
    background-color: rgba(201, 52, 74, 72);
}}
QPushButton[compact="true"] {{
    padding: 3px 8px;
    min-height: 18px;
}}

QPushButton[flatToggle="true"] {{
    background: transparent;
    border: none;
    color: {TEXT_SECONDARY};
    text-align: left;
    padding: 4px 2px;
    font-size: 11px;
    font-weight: 600;
}}
QPushButton[flatToggle="true"]:hover {{
    color: {TEXT_PRIMARY};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_ELEVATED};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_CYAN};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {TEXT_PRIMARY};
    border: 2px solid {ACCENT_CYAN};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_CYAN};
}}

QCheckBox {{
    spacing: 8px;
    background: transparent;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER_STRONG};
    background-color: {BG_SURFACE};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT_PRIMARY};
    border-color: #F05497;
}}

QProgressBar {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-size: 11px;
    font-weight: 600;
    min-height: 18px;
}}
QProgressBar::chunk {{
    background-color: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {ACCENT_PRIMARY},
        stop: 1 {ACCENT_CYAN}
    );
    border-radius: 4px;
}}

QGraphicsView {{
    background-color: {TIMELINE_BG.name()};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: #5E5680;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #5E5680;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
"""


def build_palette() -> QPalette:
    """Dark palette used for Fusion's native icons and arrows."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_APP))
    palette.setColor(QPalette.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(BG_SURFACE))
    palette.setColor(QPalette.AlternateBase, QColor(BG_PANEL))
    palette.setColor(QPalette.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(BG_ELEVATED))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_SECONDARY))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_MUTED))
    return palette


def apply_theme(app: QApplication) -> None:
    """Applies the dark studio palette and stylesheet to `app`."""
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(STYLESHEET)
