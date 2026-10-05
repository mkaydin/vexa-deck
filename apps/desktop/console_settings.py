"""Master audio equalizer and persisted console appearance preferences."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

import pixel_theme as theme
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from window_chrome import window_layout

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "Flat": ([0, 0, 0, 0, 0], 0),
    "Bass lift": ([5, 2, 0, -1, 1], -5),
    "Warm": ([3, 2, 0, -2, -3], -3),
    "Clarity": ([-2, -1, 1, 3, 2], -3),
    "Club": ([4, 1, -2, 2, 4], -4),
}


class ConsoleSettingsDialog(QDialog):
    message = Signal(str)

    def __init__(self, window, *, persist=True, config_path=None):
        super().__init__(window)
        self.owner = window
        self.persist = persist
        self.config_path = config_path or ROOT / "var/config/console.json"
        self._syncing = False
        self._inflight = False
        self._revision = 0
        self._ready = bool(window.preview)
        self._audio_running = False
        self.setWindowTitle("VEXA / CONSOLE SETTINGS")
        self.setWindowIcon(window.windowIcon())
        self.resize(580, 470)
        self.setMinimumSize(520, 420)
        layout = window_layout(self, self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        self.audio_page = QWidget()
        audio = QVBoxLayout(self.audio_page)
        description = QLabel("MASTER EQ // DJ OUTPUT | 5 BANDS / ±12 dB")
        description.setWordWrap(True)
        audio.addWidget(description)
        row = QHBoxLayout()
        self.enabled = QCheckBox("Enable equalizer")
        self.enabled.setAccessibleName("Enable master equalizer")
        self.enabled.toggled.connect(self._edited)
        row.addWidget(self.enabled)
        row.addStretch()
        self.preset = QComboBox()
        self.preset.setAccessibleName("Equalizer preset")
        self.preset.addItem("Custom")
        self.preset.addItems(list(PRESETS))
        self.preset.activated.connect(self._preset_selected)
        row.addWidget(self.preset)
        audio.addLayout(row)
        bands = QHBoxLayout()
        self.sliders, self.readouts = [], []
        for name in ("60 Hz", "250 Hz", "1 kHz", "4 kHz", "12 kHz"):
            column = QVBoxLayout()
            heading = QLabel(name)
            heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(heading)
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setTickInterval(3)
            slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
            slider.setMinimumHeight(160)
            slider.setAccessibleName(f"Equalizer {name} gain in dB")
            slider.valueChanged.connect(self._edited)
            column.addWidget(slider, 1, Qt.AlignmentFlag.AlignHCenter)
            readout = QLabel("+0 dB")
            readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(readout)
            self.sliders.append(slider)
            self.readouts.append(readout)
            bands.addLayout(column, 1)
        audio.addLayout(bands, 1)
        preamp = QHBoxLayout()
        preamp.addWidget(QLabel("PREAMP"))
        self.preamp = QSlider(Qt.Orientation.Horizontal)
        self.preamp.setRange(-12, 0)
        self.preamp.setAccessibleName("Equalizer preamp in dB")
        self.preamp.valueChanged.connect(self._edited)
        preamp.addWidget(self.preamp, 1)
        self.preamp_value = QLabel("+0 dB")
        self.preamp_value.setMinimumWidth(65)
        preamp.addWidget(self.preamp_value)
        audio.addLayout(preamp)
        self.audio_status = QLabel("Waiting for audio backend")
        self.audio_status.setWordWrap(True)
        audio.addWidget(self.audio_status)
        reset = QPushButton("[ RESET TO FLAT ]")
        reset.clicked.connect(self.reset_equalizer)
        audio.addWidget(reset)
        tabs.addTab(self.audio_page, "Equalizer")

        appearance_page = QWidget()
        appearance = QVBoxLayout(appearance_page)
        appearance.addWidget(QLabel("CONSOLE PALETTE // apply instantly"))
        self.theme_selector = QComboBox()
        self.theme_selector.setAccessibleName("GUI appearance theme")
        for key, (name, _) in theme.THEMES.items():
            self.theme_selector.addItem(name, key)
        saved = "terminal"
        if persist and self.config_path.exists():
            with suppress(OSError, ValueError, AttributeError):
                saved = json.loads(self.config_path.read_text()).get("theme", "terminal")
        index = self.theme_selector.findData(saved)
        self.theme_selector.setCurrentIndex(max(0, index))
        appearance.addWidget(self.theme_selector)
        note = QLabel(
            "Changes console panels, spectrum and window controls.\n"
            "Choose the music's vibe in SESSION. "
            "Video effect colors have their own settings."
        )
        note.setWordWrap(True)
        appearance.addWidget(note)
        appearance.addStretch()
        tabs.addTab(appearance_page, "Appearance")
        self.theme_selector.currentIndexChanged.connect(self._theme_selected)
        theme.apply_theme(QApplication.instance(), self.theme_selector.currentData())
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._send)
        self.set_online(window.preview)

    def _theme_selected(self):
        selected = self.theme_selector.currentData()
        theme.apply_theme(QApplication.instance(), selected)
        if self.persist:
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.config_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"theme": selected}, indent=2) + "\n")
                temporary.replace(self.config_path)
            except OSError as exc:
                self.message.emit(f"Could not save appearance: {exc}")
        self.message.emit(f"Console theme: {selected}")

    def set_online(self, online):
        self.audio_page.setEnabled(online and self._ready)
        if not online:
            self.audio_status.setText(
                "Audio backend offline; EQ will be available after connection."
            )
        elif self.owner.preview:
            self.audio_status.setText("Design preview // controls do not change live audio")

    def payload(self):
        return {
            "enabled": self.enabled.isChecked(),
            "gains_db": [slider.value() for slider in self.sliders],
            "preamp_db": self.preamp.value(),
        }

    def _readouts(self):
        for readout, slider in zip(self.readouts, self.sliders, strict=True):
            readout.setText(f"{slider.value():+d} dB")
        self.preamp_value.setText(f"{self.preamp.value():+d} dB")

    def _edited(self, *_args):
        if self._syncing:
            return
        self._readouts()
        self.preset.setCurrentIndex(0)
        self._revision += 1
        self.audio_status.setText(
            "Applying master EQ..."
            if not self.owner.preview
            else "Design preview // controls do not change live audio"
        )
        self._debounce.start()

    def _load_values(self, values):
        self._syncing = True
        self.enabled.setChecked(bool(values.get("enabled")))
        for slider, value in zip(self.sliders, values.get("gains_db", [0] * 5), strict=True):
            slider.setValue(round(value))
        self.preamp.setValue(round(values.get("preamp_db", 0)))
        self._readouts()
        self._syncing = False

    def sync_equalizer(self, values, *, running=None):
        if running is not None:
            self._audio_running = running
        if not values or self._inflight or self._debounce.isActive():
            return
        self._ready = True
        self.audio_page.setEnabled(True)
        # Avoid changing a slider during a drag when a poll delivers an older value.
        if any(slider.isSliderDown() for slider in [*self.sliders, self.preamp]):
            return
        self._load_values(values)
        if not self.owner.preview:
            self.audio_status.setText(
                "Saved // applies when playback starts"
                if values.get("pending") and not self._audio_running
                else "EQ smoothing..."
                if values.get("pending")
                else "Master EQ active // saved"
                if values.get("enabled")
                else "Bypassed // saved"
            )

    def _preset_selected(self, index):
        name = self.preset.itemText(index)
        if name not in PRESETS:
            return
        gains, preamp = PRESETS[name]
        self._load_values({"enabled": True, "gains_db": gains, "preamp_db": preamp})
        self._edited()
        self.preset.setCurrentIndex(index)

    def reset_equalizer(self):
        self._load_values({"enabled": False, "gains_db": [0] * 5, "preamp_db": 0})
        self._edited()
        self.preset.setCurrentIndex(self.preset.findText("Flat"))

    def _send(self):
        if self.owner.preview or self.owner.client is None or self._inflight:
            return
        self._inflight = True
        revision = self._revision
        payload = self.payload()

        def received(values, error):
            self._inflight = False
            if error:
                self.audio_status.setText(f"EQ update failed: {error}")
                self.message.emit(f"EQ update failed: {error}")
            else:
                self.message.emit(f"Master EQ saved: {payload}")
            if revision != self._revision:
                self._debounce.start()
            elif not error:
                self.sync_equalizer(values)

        self.owner.client.configure_equalizer(payload, received)
