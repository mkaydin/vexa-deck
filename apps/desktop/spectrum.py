"""Responsive post-master spectrum styles, independent of video and console palettes."""

import math
import time

import pixel_theme as theme
from pixel_theme import ascii_frame, font
from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

DEFAULTS = dict(
    style="ascii",
    palette="console",
    bands=32,
    response="balanced",
    peaks=True,
    labels=False,
    gain=0,
    height="default",
)
CHOICES = dict(
    style=("ascii", "led", "bars", "mirror"),
    palette=("console", "neon", "amber", "mono"),
    bands=(16, 32, 64),
    response=("balanced", "quick", "relaxed"),
    height=("compact", "default", "tall"),
)


def validate(raw):
    result = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return result
    for key, value in (raw or {}).items():
        if (key in CHOICES and value in CHOICES[key]) or (
            key in ("peaks", "labels") and type(value) is bool
        ):
            result[key] = value
        elif key == "gain" and type(value) in (float, int) and math.isfinite(value):
            result[key] = max(-20, min(20, value))
    return result


class SpectrumWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(160)
        self.setAccessibleName("Post-master audio spectrum")
        self.options = dict(DEFAULTS)
        self.target = [0.0] * 64
        self.values = [0.0] * 64
        self.peak_values = [0.0] * 64
        self.reduced_motion = False
        self._received = 0.0
        self._animated = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.animate)
        self._timer.start()

    def configure(self, options):
        self.options = validate(options)
        self.update()

    def set_levels(self, values):
        if len(values) not in (16, 32, 64):
            return
        try:
            values = [max(0, min(1, float(v))) if math.isfinite(float(v)) else 0 for v in values]
        except (ValueError, TypeError):
            return
        self.target = [v for v in values for _ in range(64 // len(values))]
        self._received = time.monotonic()

    def animate(self):
        now = time.monotonic()
        dt = min(0.1, now - self._animated)
        self._animated = now
        if now - self._received > 0.5:
            self.target = [0.0] * 64
        if not self.isVisible() or self.window().isMinimized():
            return
        attack, release = {"balanced": (0.04, 0.18), "quick": (0.02, 0.1), "relaxed": (0.12, 0.35)}[
            self.options["response"]
        ]
        for index, target in enumerate(self.target):
            current = self.values[index]
            amount = (
                1
                if self.reduced_motion
                else 1 - math.exp(-dt / (attack if target > current else release))
            )
            self.values[index] += (target - current) * amount
            self.peak_values[index] = max(self.values[index], self.peak_values[index] - dt * 0.35)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        ascii_frame(p, self.rect())
        p.setFont(font(9))
        p.setPen(QColor(theme.MUTED))
        p.drawText(16, 27, "SPECTRUM / MASTER OUTPUT")
        p.setFont(font(8))
        available = max(1, self.width() - 32)
        count = min(self.options["bands"], max(1, available // 10))
        # Reduce to divisors of 64, preserving all source bins.
        count = next(n for n in (64, 32, 16, 8, 4, 2, 1) if n <= count)
        palettes = {
            "console": (theme.CYAN, theme.AMBER, theme.PINK),
            "neon": ("#63e6d2", "#8486ff", "#ff85c8"),
            "amber": ("#b5cc82", "#f0c56c", "#e39870"),
            "mono": (theme.TEXT,) * 3,
        }
        colors = palettes[self.options["palette"]]
        bottom = self.height() - (30 if self.options["labels"] else 18)
        height = max(1, bottom - 40)
        segments = max(1, height // 11)
        width = available / count
        for index in range(count):
            group = slice(index * 64 // count, (index + 1) * 64 // count)
            level = min(1, max(0, max(self.values[group]) + self.options["gain"] / 70))
            peak = min(1, max(0, max(self.peak_values[group]) + self.options["gain"] / 70))
            color = colors[min(2, index * 3 // count)]
            x = 16 + index * width
            p.setPen(QColor(color))
            style = self.options["style"]
            if style == "ascii":
                glyph = "##" if p.fontMetrics().horizontalAdvance("##") <= width - 2 else "#"
                gx = x + (width - p.fontMetrics().horizontalAdvance(glyph)) / 2
                for segment in range(segments):
                    p.setPen(QColor(color if segment < round(level * segments) else theme.EDGE))
                    p.drawText(
                        QPointF(gx, bottom - segment * 11),
                        glyph if segment < round(level * segments) else ".",
                    )
            elif style == "led":
                for segment in range(segments):
                    p.fillRect(
                        int(x + 1),
                        bottom - (segment + 1) * 11,
                        max(1, int(width - 3)),
                        7,
                        QColor(color if segment < round(level * segments) else theme.EDGE),
                    )
            else:
                y = bottom - height * level
                if style == "mirror":
                    y = bottom - height / 2 - height * level / 2
                p.fillRect(
                    int(x + 1),
                    int(y),
                    max(1, int(width - 3)),
                    max(0, int(height * level)),
                    QColor(color),
                )
            if self.options["peaks"] and peak > 0.01 and style != "mirror":
                p.setPen(QColor(theme.TEXT))
                p.drawLine(
                    int(x + 1),
                    int(bottom - peak * height),
                    int(x + width - 3),
                    int(bottom - peak * height),
                )
        if self.options["labels"]:
            p.setPen(QColor(theme.MUTED))
            p.drawText(16, self.height() - 12, "BASS")
            p.drawText(self.width() // 2 - 12, self.height() - 12, "MID")
            p.drawText(self.width() - 52, self.height() - 12, "HIGH")
