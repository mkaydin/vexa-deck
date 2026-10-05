"""Dark cyberpunk terminal typography and ASCII framing for the Qt desktop."""

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF

BACKGROUND = "#080c10"
PANEL = "#0d151b"
EDGE = "#28434b"
CYAN = "#63e6d2"
PINK = "#b194df"
AMBER = "#abc78a"
TEXT = "#c9dcd8"
MUTED = "#748f96"
FAMILY = "DejaVu Sans Mono"


def install_font():
    """Select an installed terminal font, including Unicode fallback for theme input."""
    global FAMILY
    available = QFontDatabase.families()
    FAMILY = next(
        (
            name
            for name in ("DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono")
            if name in available
        ),
        QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family(),
    )
    return FAMILY


def font(size=10, weight=500):
    result = QFont(FAMILY)
    result.setStyleHint(QFont.StyleHint.Monospace)
    result.setFixedPitch(True)
    result.setPixelSize(max(11, round(size * 1.18)))
    result.setWeight(QFont.Weight.Bold if weight >= 600 else QFont.Weight.Normal)
    return result


def ascii_frame(painter, rect, color=None):
    """Draw an actual +---+ / | character border without decorating the interior."""
    painter.save()
    painter.setFont(font(9))
    painter.setPen(QColor(EDGE if color is None else color))
    metrics = QFontMetricsF(painter.font())
    cell, line = metrics.horizontalAdvance("M"), metrics.height()
    columns = max(2, int((rect.width() - 4) // cell))
    x, top = rect.x() + 2, rect.y() + 2 + metrics.ascent()
    bottom = rect.bottom() - metrics.descent() - 1
    edge = "+" + "-" * (columns - 2) + "+"
    painter.drawText(QPointF(x, top), edge)
    painter.drawText(QPointF(x, bottom), edge)
    y = top + line
    while y < bottom - line / 2:
        painter.drawText(QPointF(x, y), "|")
        painter.drawText(QPointF(x + (columns - 1) * cell, y), "|")
        y += line
    painter.restore()


STYLE = """
QWidget#root, QDialog { background: #080c10; color: #c9dcd8; }
QFrame#panel { background: #0d151b; border: 0; border-radius: 0; }
QFrame#titleBar { background: #0d151b; border: 0; border-bottom: 1px solid #28434b; }
QFrame#titleBar QLabel { color: #63e6d2; background: transparent; }
QPushButton#captionButton, QPushButton#captionClose { background: transparent;
    border: 1px solid transparent; padding: 0; }
QPushButton#captionButton:hover, QPushButton#captionButton[keyboardFocus="true"]:focus {
    background: #102a2d; border-color: #3a9b8e; }
QPushButton#captionClose:hover, QPushButton#captionClose[keyboardFocus="true"]:focus {
    background: #342034; border-color: #b194df; }
QPushButton#captionButton:pressed { background: #17393d; }
QPushButton#captionClose:pressed { background: #4a2948; }
QLineEdit, QPlainTextEdit, QTableWidget { background: #080f14; color: #c9dcd8;
    border: 1px solid #28434b; border-radius: 0; selection-background-color: #193e40;
    selection-color: #a9ffee; }
QLineEdit { padding: 9px; font-size: 13px; }
QLineEdit:focus, QPlainTextEdit:focus { border-color: #63e6d2; }
QPlainTextEdit { padding: 8px; }
QPushButton { color: #90b2b7; background: #0c171d; border: 1px solid #28434b;
    border-radius: 0; text-align: left; padding: 7px 9px; }
QPushButton:hover { color: #a9ffee; background: #10272c; border-color: #63e6d2; }
QPushButton:pressed { background: #17393d; color: #c9fff0; }
QPushButton:checked { background: #132d31; color: #63e6d2; border-color: #63e6d2; }
QPushButton:disabled { color: #40555d; background: #0c1319; border-color: #1b2b33; }
QPushButton#primary { background: #102a2d; border-color: #3a9b8e; color: #84f5d7; }
QPushButton#danger { background: #201521; border-color: #66455f; color: #d99db8; }
QPushButton#chip { background: #0b141c; }
QHeaderView::section { background: #122029; color: #91b5b8; padding: 8px;
    border: 0; border-bottom: 1px solid #28434b; }
QTableWidget::item { padding: 6px; }
QScrollArea { background: transparent; border: 0; }
QWidget#videoSettingsBody { background: #0d151b; color: #c9dcd8; }
QComboBox, QSpinBox, QDoubleSpinBox, QListWidget { background: #080f14; color: #c9dcd8;
    border: 1px solid #28434b; padding: 5px; selection-background-color: #193e40; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #63e6d2; }
QComboBox QAbstractItemView { background: #0d151b; color: #c9dcd8;
    selection-background-color: #193e40; }
QTabWidget::pane { border: 1px solid #28434b; }
QTabBar::tab { background: #0d151b; color: #748f96; padding: 8px 16px; }
QTabBar::tab:selected { background: #102a2d; color: #63e6d2; }
QCheckBox { color: #c9dcd8; spacing: 8px; }
QCheckBox::indicator { width: 12px; height: 12px; background: #080c10;
    border: 1px solid #31575d; }
QCheckBox::indicator:checked { background: #63e6d2; border-color: #63e6d2; }
QSlider::groove:horizontal { height: 4px; background: #28434b; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; background: #63e6d2; }
QSlider::groove:vertical { width: 4px; background: #28434b; }
QSlider::handle:vertical { height: 12px; margin: 0 -5px; background: #63e6d2; }
QWidget:disabled { color: #40555d; }
QLabel#mutedLabel { color: #748f96; }
QDialog QLabel { color: #90b2b7; background: transparent; }
QScrollBar:vertical { background: #0b1219; width: 9px; }
QScrollBar::handle:vertical { background: #31575d; min-height: 24px; border-radius: 0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""

# Appearance controls affect console chrome, not the user's Pixel/ASCII video palettes.
THEMES = {
    "terminal": ("Vexa terminal", {}),
    "violet": ("Neon violet", {
        "#080c10": "#0c0912", "#0d151b": "#15101f", "#28434b": "#49345e",
        "#63e6d2": "#c694ff", "#b194df": "#ff85c8", "#abc78a": "#87dacd",
        "#c9dcd8": "#e1d6ed", "#748f96": "#9c88ae", "#90b2b7": "#bda8cf",
        "#3a9b8e": "#8d60b8", "#a9ffee": "#ead4ff", "#84f5d7": "#d5afff",
    }),
    "amber": ("Amber terminal", {
        "#080c10": "#100d08", "#0d151b": "#1a150d", "#28434b": "#53452b",
        "#63e6d2": "#f0c56c", "#b194df": "#e39870", "#abc78a": "#b5cc82",
        "#c9dcd8": "#e5dcc8", "#748f96": "#a09578", "#90b2b7": "#c5b18a",
        "#3a9b8e": "#a38842", "#a9ffee": "#ffe1a3", "#84f5d7": "#ffdc87",
    }),
    "ice": ("Ice blue", {
        "#080c10": "#080c15", "#0d151b": "#0d1624", "#28434b": "#2d4667",
        "#63e6d2": "#78caff", "#b194df": "#afadff", "#abc78a": "#8ee0c0",
        "#c9dcd8": "#d1e0ef", "#748f96": "#8198b2", "#90b2b7": "#9db9d5",
        "#3a9b8e": "#458dc0", "#a9ffee": "#c2eaff", "#84f5d7": "#a5ddff",
    }),
}
BASE_COLORS = {name: globals()[name] for name in
               ("BACKGROUND", "PANEL", "EDGE", "CYAN", "PINK", "AMBER", "TEXT", "MUTED")}
CURRENT_THEME = "terminal"


def themed_css(css):
    import re
    # Derived surfaces also follow the palette's hue; keep their original luminance.
    return re.sub(r"#[0-9a-fA-F]{6}", lambda match: theme_color(match.group()), css)


def theme_color(color):
    import colorsys
    mapping = THEMES[CURRENT_THEME][1]
    color = color.lower()
    if color in mapping:
        return mapping[color]
    if not mapping or color in {"#ffffff", "#000000"}:
        return color
    original = QColor(color)
    h, s, v = colorsys.rgb_to_hsv(original.redF(), original.greenF(), original.blueF())
    if s < 0.15:
        return color
    # Purple warning surfaces retain their relationship to the secondary accent.
    anchor = QColor(mapping["#b194df"] if 0.7 < h < 0.98 else mapping["#63e6d2"])
    hue = colorsys.rgb_to_hsv(anchor.redF(), anchor.greenF(), anchor.blueF())[0]
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return QColor.fromRgbF(r, g, b).name()


def apply_theme(app, name):
    global CURRENT_THEME
    if name not in THEMES:
        name = "terminal"
    CURRENT_THEME = name
    for role, base in BASE_COLORS.items():
        globals()[role] = theme_color(base)
    app.setStyleSheet(themed_css(STYLE))
    for widget in app.allWidgets():
        original = widget.property("vexaSourceStyle")
        if original:
            # Call the Qt base implementation, preserving the unmodified source colors.
            from PySide6.QtWidgets import QWidget
            QWidget.setStyleSheet(widget, themed_css(original))
        widget.update()
