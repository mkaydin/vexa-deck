"""Blank center and ASCII terminal audio readouts."""

import pixel_theme as theme
from pixel_theme import ascii_frame, font
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QWidget


class NeonPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        ascii_frame(painter, self.rect())


class StageWidget(QWidget):
    """An intentionally blank bordered box; no artwork or animation resources."""

    def __init__(self):
        super().__init__()
        self.setAccessibleName("Empty video panel")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BACKGROUND))
        ascii_frame(painter, self.rect())


class VideoViewport(QWidget):
    """Fill the available video area without adding empty side gutters."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.stage = StageWidget()
        self.stage.setParent(self)

    def resizeEvent(self, event):
        self.stage.setGeometry(self.rect())
        super().resizeEvent(event)


class SpectrumWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(160)
        self.setAccessibleName("32-band audio spectrum")
        self.target = [0.0] * 32
        self.values = [0.0] * 32
        self.reduced_motion = False
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.animate)
        self._timer.start()

    def set_levels(self, values):
        if len(values) == 32:
            self.target = [max(0.0, min(1.0, float(value))) for value in values]

    def animate(self):
        amount = 1 if self.reduced_motion else 0.35
        self.values = [
            value + (target - value) * amount
            for value, target in zip(self.values, self.target, strict=True)
        ]
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        ascii_frame(p, self.rect())
        p.setFont(font(9))
        p.setPen(QColor(theme.MUTED))
        p.drawText(16, 27, "FFT::32 / AUDIO ENERGY")
        band_width = (self.width() - 32) / 32
        segments = max(1, (self.height() - 52) // 12)
        glyph_width = p.fontMetrics().horizontalAdvance("##")
        for index, value in enumerate(self.values):
            color = theme.CYAN if index < 12 else theme.AMBER if index < 24 else theme.PINK
            x = 16 + (index + 0.5) * band_width - glyph_width / 2
            for segment in range(segments):
                active = segment < round(value * segments)
                p.setPen(QColor(color if active else theme.EDGE))
                p.drawText(QPointF(x, self.height() - 24 - segment * 12), "##" if active else "..")


class DeckCard(QWidget):
    def __init__(self, name):
        super().__init__()
        self.setFixedHeight(96)
        self.name, self.asset_id = name, "NO TRACK LOADED"
        self.title = "NO TRACK LOADED"
        self.active = self.playing = self.loaded = self.fading = False
        self.gain = self.position_s = self.duration_s = 0.0
        self.reduced_motion = False

    def set_status(self, data, asset_id, *, active, duration_s, title=None):
        self.asset_id = asset_id or "NO TRACK LOADED"
        self.title = title or self.asset_id
        self.setToolTip(f"{self.title}\nID: {self.asset_id}" if asset_id else self.title)
        self.active, self.playing = active, bool(data.get("playing"))
        self.loaded, self.fading = bool(data.get("loaded")), bool(data.get("fading"))
        self.gain = float(data.get("gain") or 0)
        self.position_s = float(data.get("position") or 0) / 44100
        self.duration_s = float(duration_s or 0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        accent = theme.CYAN if self.name.endswith("A") else theme.PINK
        w = self.width()
        p.fillRect(self.rect(), QColor(theme.PANEL))
        ascii_frame(p, self.rect(), accent if self.active else theme.EDGE)
        p.setFont(font(10))
        p.setPen(QColor(accent))
        p.drawText(16, 27, f"[ {self.name} ]")
        state = (
            "MIX" if self.fading else "PLAY" if self.playing else "CUED" if self.loaded else "EMPTY"
        )
        state = f"[ {state} ]"
        p.drawText(w - 16 - p.fontMetrics().horizontalAdvance(state), 27, state)
        p.setPen(QColor(theme.TEXT))
        p.drawText(
            16,
            46,
            p.fontMetrics().elidedText(self.title.upper(), Qt.TextElideMode.ElideRight, w - 32),
        )
        p.setPen(QColor(theme.MUTED))
        position, duration = int(self.position_s), int(self.duration_s)
        p.drawText(
            16, 63, f"{position // 60}:{position % 60:02d} / {duration // 60}:{duration % 60:02d}"
        )
        cells = max(8, (w - 32) // p.fontMetrics().horizontalAdvance("M") - 2)
        completed = int(cells * min(1, self.position_s / self.duration_s)) if self.duration_s else 0
        p.setPen(QColor(accent if self.playing else theme.MUTED))
        p.drawText(16, 79, "[" + "=" * completed + "." * (cells - completed) + "]")
