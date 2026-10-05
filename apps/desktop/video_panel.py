"""Local looping videos with native Qt controls and a bundled WebGL/ASCII renderer."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pixel_theme as theme
from pixel_stage import VideoViewport
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from window_chrome import window_layout

# Initialize WebEngine's shared graphics support before QApplication creates
# native Wayland surfaces. Views and renderer processes are still created lazily.
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
try:
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).parent / "assets/video"
PALETTES = json.loads((ASSETS / "palettes.json").read_text())
PALETTES["vexa"] = [
    [int(color[start : start + 2], 16) / 255 for start in (1, 3, 5)]
    for color in (
        "#080c10",
        "#0d151b",
        "#28434b",
        "#31575d",
        "#748f96",
        "#63e6d2",
        "#b194df",
        "#66455f",
        "#d99db8",
        "#abc78a",
        "#c9dcd8",
        "#a9ffee",
    )
]
ASCII_DEFAULTS_VERSION = 2
DEFAULTS = {
    "mode": "normal",
    "fit": "fill",
    "fps": 24,
    "pixel": {
        "size": 6,
        "palette": "vexa",
        "dither": 0.2,
        "edgeThreshold": 0.3,
        "edgeIntensity": 0.5,
        "edgeColor": "#63e6d2",
    },
    "ascii": {
        "resolution": 70,
        "fontSizeFactor": 3,
        "threshold": 30,
        "invert": False,
        "color": "#c7205b",
        "color2": "#0032ff",
        "background": "#080c37",
        "gradient": True,
        "saturation": 60,
        "randomness": 15,
        "textMode": "ramp",
        "text": "wavesand",
        "effectWidth": 1.0,
    },
}
# bounds also validate preferences read from disk, independently of the GUI controls
BOUNDS = {
    "fps": (10, 60),
    "pixel.size": (1, 32),
    "pixel.dither": (0, 1),
    "pixel.edgeThreshold": (0.01, 0.5),
    "pixel.edgeIntensity": (0, 1),
    "ascii.resolution": (10, 200),
    "ascii.fontSizeFactor": (0, 10),
    "ascii.threshold": (0, 95),
    "ascii.saturation": (0, 100),
    "ascii.randomness": (0, 100),
    "ascii.effectWidth": (0, 1),
}
CHOICES = {
    "mode": ("normal", "pixel", "ascii"),
    "fit": ("fill", "fit"),
    "pixel.palette": tuple(PALETTES),
    "ascii.textMode": ("ramp", "user"),
}


def value_at(options, key):
    parts = key.split(".")
    return options[parts[0]][parts[1]] if len(parts) == 2 else options[key]


def assign(options, key, value):
    parts = key.split(".")
    if len(parts) == 2:
        options[parts[0]][parts[1]] = value
    else:
        options[key] = value


def validated_options(raw):
    options = copy.deepcopy(DEFAULTS)
    for group, default in DEFAULTS.items():
        entries = default.items() if isinstance(default, dict) else [(None, default)]
        for field, expected in entries:
            key = f"{group}.{field}" if field else group
            try:
                value = value_at(raw, key)
            except (KeyError, TypeError):
                continue
            if key in BOUNDS and isinstance(value, (int, float)) and not isinstance(value, bool):
                low, high = BOUNDS[key]
                value = max(low, min(high, value))
                if isinstance(expected, int):
                    value = int(value)
            elif key in CHOICES:
                if value not in CHOICES[key]:
                    continue
            elif isinstance(expected, bool):
                if not isinstance(value, bool):
                    continue
            elif isinstance(expected, str):
                if not isinstance(value, str):
                    continue
                if key.endswith(("Color", "color", "color2", "background")) and (
                    len(value) != 7 or not value.startswith("#") or not QColor(value).isValid()
                ):
                    continue
                value = value[:256]
            else:
                continue
            assign(options, key, value)
    return options


def control(text, callback=None):
    button = QPushButton(f"[ {text} ]")
    button.setMinimumHeight(30)
    if callback:
        button.clicked.connect(callback)
    return button


class VideoSettings(QDialog):
    def __init__(self, panel):
        super().__init__(panel.window())
        self.panel = panel
        self.setWindowTitle("VEXA / VIDEO SETTINGS")
        self.setWindowIcon(panel.window().windowIcon())
        palette = self.palette()
        for role, color in (
            (QPalette.ColorRole.Window, "#080c10"),
            (QPalette.ColorRole.Base, "#0d151b"),
            (QPalette.ColorRole.Button, "#0d151b"),
            (QPalette.ColorRole.WindowText, "#c9dcd8"),
            (QPalette.ColorRole.Text, "#c9dcd8"),
            (QPalette.ColorRole.ButtonText, "#c9dcd8"),
        ):
            palette.setColor(role, QColor(theme.theme_color(color)))
        self.setPalette(palette)
        self.resize(580, 590)
        self.setMinimumSize(440, 400)
        layout = window_layout(self, self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        self.editors = {}
        self.playback = QWidget()
        self.playback.setObjectName("videoSettingsBody")
        playback_layout = QVBoxLayout(self.playback)
        playback_form = QFormLayout()
        playback_layout.addLayout(playback_form)
        self.combo(
            playback_form, "Scale", "fit", {"fill": "Fill / crop", "fit": "Fit / full frame"}
        )
        self.number(playback_form, "Effect FPS limit", "fps", 1)
        note = QLabel("Videos stay muted. The list repeats in order; a single video loops itself.")
        note.setWordWrap(True)
        playback_layout.addWidget(note)
        self.list = QListWidget()
        self.list.setAccessibleName("Uploaded video loop list")
        self.list.itemDoubleClicked.connect(self.play_selected)
        playback_layout.addWidget(self.list, 1)
        actions = QHBoxLayout()
        self.pause_button = control("PAUSE", self.toggle_pause)
        actions.addWidget(self.pause_button)
        actions.addWidget(control("NEXT", panel.next_video))
        actions.addWidget(control("REMOVE", self.remove_selected))
        actions.addWidget(control("CLEAR", panel.clear_videos))
        playback_layout.addLayout(actions)
        playback_layout.addWidget(
            QLabel("Double-click a video to play it. Remove only changes this list.")
        )
        tabs.addTab(self.playback, "Videos")
        pixel_form = self.tab(tabs, "Pixel art")
        self.number(pixel_form, "Pixel size", "pixel.size", 1)
        self.combo(
            pixel_form, "Palette", "pixel.palette", {name: name.title() for name in PALETTES}
        )
        self.number(pixel_form, "Dithering strength", "pixel.dither", 0.01)
        self.number(pixel_form, "Edge threshold", "pixel.edgeThreshold", 0.01)
        self.number(pixel_form, "Edge intensity", "pixel.edgeIntensity", 0.01)
        self.color(pixel_form, "Edge color", "pixel.edgeColor")
        ascii_form = self.tab(tabs, "ASCII art")
        self.number(ascii_form, "Resolution", "ascii.resolution", 1)
        self.number(ascii_form, "Font Size Factor", "ascii.fontSizeFactor", 1)
        self.number(ascii_form, "Threshold (%)", "ascii.threshold", 1)
        self.check(ascii_form, "Invert brightness", "ascii.invert")
        self.color(ascii_form, "Text color / shadows", "ascii.color")
        self.color(ascii_form, "Text color / highlights", "ascii.color2")
        self.color(ascii_form, "Background color", "ascii.background")
        self.check(ascii_form, "Background gradient", "ascii.gradient")
        self.number(ascii_form, "Background saturation", "ascii.saturation", 1)
        self.number(ascii_form, "Randomness (%)", "ascii.randomness", 1)
        self.combo(
            ascii_form,
            "Text mode",
            "ascii.textMode",
            {"ramp": "Random Text", "user": "User Text"},
        )
        text = QLineEdit(panel.options["ascii"]["text"])
        text.setMaxLength(256)
        text.textChanged.connect(lambda value: panel.set_option("ascii.text", value))
        ascii_form.addRow("Custom text", text)
        self.editors["ascii.text"] = text
        self.number(ascii_form, "Effect width (1 = full)", "ascii.effectWidth", 0.01)
        tabs.setCurrentIndex({"normal": 0, "pixel": 1, "ascii": 2}[panel.options["mode"]])
        footer = QHBoxLayout()
        footer.addWidget(control("RESET STYLES", self.reset))
        footer.addStretch()
        footer.addWidget(control("DONE", self.close))
        layout.addLayout(footer)
        panel.playlist_changed.connect(self.refresh_list)
        self.refresh_list()

    def tab(self, tabs, title):
        area = QScrollArea()
        area.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName("videoSettingsBody")
        form = QFormLayout(body)
        form.setSpacing(10)
        area.setWidget(body)
        tabs.addTab(area, title)
        return form

    def combo(self, form, title, key, choices):
        widget = QComboBox()
        for value, text in choices.items():
            widget.addItem(text, value)
        widget.setCurrentIndex(widget.findData(value_at(self.panel.options, key)))
        widget.currentIndexChanged.connect(
            lambda _: self.panel.set_option(key, widget.currentData())
        )
        widget.setAccessibleName(title)
        form.addRow(title, widget)
        self.editors[key] = widget

    def number(self, form, title, key, step):
        widget = QSpinBox() if isinstance(value_at(DEFAULTS, key), int) else QDoubleSpinBox()
        widget.setRange(*BOUNDS[key])
        widget.setSingleStep(step)
        widget.setValue(value_at(self.panel.options, key))
        widget.valueChanged.connect(lambda value: self.panel.set_option(key, value))
        widget.setAccessibleName(title)
        form.addRow(title, widget)
        self.editors[key] = widget

    def check(self, form, title, key):
        widget = QCheckBox()
        widget.setChecked(value_at(self.panel.options, key))
        widget.toggled.connect(lambda value: self.panel.set_option(key, value))
        widget.setAccessibleName(title)
        form.addRow(title, widget)
        self.editors[key] = widget

    def color(self, form, title, key):
        widget = control(value_at(self.panel.options, key))

        def choose():
            color = QColorDialog.getColor(QColor(value_at(self.panel.options, key)), self, title)
            if color.isValid():
                self.panel.set_option(key, color.name())
                widget.setText(f"[ {color.name()} ]")

        widget.clicked.connect(choose)
        widget.setAccessibleName(title)
        form.addRow(title, widget)
        self.editors[key] = widget

    def reset(self):
        for group in ("pixel", "ascii"):
            self.panel.options[group] = copy.deepcopy(DEFAULTS[group])
        self.panel.apply_options()
        for key, widget in self.editors.items():
            value = value_at(self.panel.options, key)
            widget.blockSignals(True)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.findData(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(value)
            else:
                widget.setText(f"[ {value} ]")
            widget.blockSignals(False)

    def refresh_list(self):
        self.pause_button.setText("[ RESUME ]" if self.panel.paused else "[ PAUSE ]")
        self.list.clear()
        for index, path in enumerate(self.panel.paths):
            self.list.addItem(f"{'>' if index == self.panel.current_index else ' '} {path.name}")
            self.list.item(index).setToolTip(str(path))
        if self.panel.paths:
            self.list.setCurrentRow(self.panel.current_index)

    def play_selected(self, _item):
        self.panel.select_video(self.list.currentRow())

    def remove_selected(self):
        self.panel.remove_video(self.list.currentRow())

    def toggle_pause(self):
        self.panel.set_paused(not self.panel.paused)
        self.pause_button.setText("[ RESUME ]" if self.panel.paused else "[ PAUSE ]")


class VideoPanel(QWidget):
    message = Signal(str)
    playlist_changed = Signal()

    def __init__(self, parent=None, *, persist=True, config_path=None):
        super().__init__(parent)
        self.persist = persist
        self.config_path = config_path or ROOT / "var/config/video-panel.json"
        self.options = copy.deepcopy(DEFAULTS)
        self.paths = []
        self.current_index = 0
        self.paused = False
        self.reduced_motion = False
        self.view = None
        self.settings_dialog = None
        self.ready = False
        self.pending_restore = False
        self.window_handle = None
        migrate_ascii_defaults = False
        if persist:
            try:
                saved = json.loads(self.config_path.read_text())
                self.options = validated_options(saved.get("options", {}))
                if saved.get("ascii_defaults_version") != ASCII_DEFAULTS_VERSION:
                    # Install the requested screenshot preset once, preserving
                    # video paths, pixel options and playback preferences.
                    self.options["ascii"] = copy.deepcopy(DEFAULTS["ascii"])
                    migrate_ascii_defaults = True
                self.paths = [
                    Path(path)
                    for path in saved.get("videos", [])
                    if isinstance(path, str) and Path(path).is_file()
                ]
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        upload = control("UPLOAD", self.upload)
        upload.setAccessibleName("Upload videos")
        toolbar.addWidget(upload)
        self.mode = QComboBox()
        self.mode.setAccessibleName("Video style")
        for text, mode in (
            ("Normal video", "normal"),
            ("Pixel art", "pixel"),
            ("ASCII art", "ascii"),
        ):
            self.mode.addItem(text, mode)
        self.mode.setCurrentIndex(self.mode.findData(self.options["mode"]))
        self.mode.currentIndexChanged.connect(
            lambda _: self.set_option("mode", self.mode.currentData())
        )
        toolbar.addWidget(self.mode, 1)
        toolbar.addWidget(control("SETTINGS", self.open_settings))
        layout.addLayout(toolbar)
        self.status = QLabel("LOCAL VIDEO / NO VIDEO LOADED")
        self.status.setMinimumWidth(0)
        self.status.setFixedHeight(18)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.status.setObjectName("mutedLabel")
        layout.addWidget(self.status)
        self.viewport = VideoViewport()
        self.viewport.setMinimumSize(320, 160)
        self.viewport.stage.installEventFilter(self)
        layout.addWidget(self.viewport, 1)
        # WebEngine is started lazily: idle consoles do not launch a browser renderer.
        if self.paths:
            self.pending_restore = True
        if migrate_ascii_defaults:
            self.save()

    def eventFilter(self, watched, event):
        if watched is self.viewport.stage and event.type() == QEvent.Type.Resize and self.view:
            self.view.setGeometry(self.viewport.stage.rect())
        if watched is self.window_handle and event.type() == QEvent.Type.Expose:
            QTimer.singleShot(0, self.sync_suspension)
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        handle = self.window().windowHandle()
        if handle is not None and handle is not self.window_handle:
            self.window_handle = handle
            handle.installEventFilter(self)
        if self.pending_restore:
            self.pending_restore = False
            self.ensure_view()
        self.sync_suspension()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.sync_suspension()

    def ensure_view(self):
        if self.view:
            return True
        if QWebEngineView is None:
            self.report(
                "QtWebEngine is missing. Install the PySide6 QtWebEngine package for video effects."
            )
            return False
        host = self

        class LocalPage(QWebEnginePage):
            def javaScriptConsoleMessage(self, level, message, line, source):
                if message.startswith("VEXA_VIDEO:"):
                    try:
                        host.on_renderer_event(json.loads(message.split(":", 1)[1]))
                    except (ValueError, TypeError, KeyError):
                        host.report("Invalid video renderer response")
                elif level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
                    host.report(f"Video renderer error: {message}")

        self.profile = QWebEngineProfile(self)
        self.view = QWebEngineView(self.viewport.stage)
        self.page = LocalPage(self.profile, self.view)
        self.page.setBackgroundColor(QColor(theme.BACKGROUND))
        self.page.setAudioMuted(True)
        settings = self.page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        self.view.setPage(self.page)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.view.setGeometry(self.viewport.stage.rect())
        self.view.show()
        self.view.loadFinished.connect(self.page_loaded)
        self.page.renderProcessTerminated.connect(self.renderer_stopped)
        self.view.load(QUrl.fromLocalFile(str(ASSETS / "index.html")))
        return True

    def renderer_stopped(self, *_):
        self.ready = False
        if self.view:
            self.view.hide()
            self.view.deleteLater()
            self.view = None
            self.profile.deleteLater()
        self.report("Video renderer stopped. Upload a video to restart it.")

    def page_loaded(self, success):
        if not success:
            self.report("Could not load the local video renderer")
            return
        self.ready = True
        self.run(
            "initialize", {"shader": (ASSETS / "pixel.frag").read_text(), "palettes": PALETTES}
        )
        self.apply_options()
        self.sync_playlist()
        self.sync_suspension()

    def run(self, method, *args):
        if self.view and self.ready:
            self.page.runJavaScript(
                f"window.VexaVideo.{method}({','.join(json.dumps(arg) for arg in args)});"
            )

    def report(self, message):
        self.status.setText(message)
        self.status.setToolTip(message)
        self.message.emit(message)

    def on_renderer_event(self, event):
        kind = event["kind"]
        if kind == "source":
            self.current_index = event["index"]
            self.report(
                f"VIDEO {self.current_index + 1}/{len(self.paths)} / {event['name']} / MUTED LOOP"
            )
            self.playlist_changed.emit()
        elif kind == "error":
            self.report(event["message"])
        elif kind in ("ready", "recovered"):
            self.message.emit(f"Local video renderer {kind}")

    def upload(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Upload videos for VEXA",
            str(Path.home() / "Videos"),
            "Videos (*.mp4 *.webm *.mkv *.mov *.m4v *.avi);;All files (*)",
        )
        if paths:
            self.add_videos(paths)

    def add_videos(self, paths):
        valid = [Path(path).resolve() for path in paths if Path(path).is_file()]
        if not valid:
            self.report("No readable video files selected")
            return
        if not self.ensure_view():
            return
        for path in valid:
            if path not in self.paths:
                self.paths.append(path)
        self.paused = False
        self.run("setPaused", False)
        self.apply_options()
        self.sync_playlist(preserve=True)
        self.save()
        self.message.emit(f"Video upload: {len(valid)} selected; {len(self.paths)} in loop")

    def sync_playlist(self, *, preserve=False):
        self.current_index = min(self.current_index, max(0, len(self.paths) - 1))
        self.run(
            "setPlaylist",
            [
                {"url": QUrl.fromLocalFile(str(path)).toString(), "name": path.name}
                for path in self.paths
            ],
            self.current_index,
            preserve,
        )
        self.playlist_changed.emit()
        if not self.paths:
            self.report("LOCAL VIDEO / NO VIDEO LOADED")

    def select_video(self, index):
        if 0 <= index < len(self.paths):
            self.current_index = index
            self.sync_playlist()

    def next_video(self):
        if self.paths:
            self.select_video((self.current_index + 1) % len(self.paths))

    def remove_video(self, index):
        if 0 <= index < len(self.paths):
            del self.paths[index]
            if index < self.current_index:
                self.current_index -= 1
            self.sync_playlist(preserve=True)
            self.save()

    def clear_videos(self):
        self.paths.clear()
        self.sync_playlist()
        self.save()
        if self.view:
            self.view.hide()

    def set_option(self, key, value):
        assign(self.options, key, value)
        self.options = validated_options(self.options)
        self.apply_options()

    def apply_options(self):
        if self.paths and self.view:
            self.view.show()
        self.run("configure", self.options)
        self.save()

    def set_paused(self, paused):
        self.paused = paused
        self.run("setPaused", paused)

    def sync_suspension(self):
        handle = self.window().windowHandle()
        self.run(
            "setSuspended",
            self.reduced_motion
            or not self.isVisible()
            or self.window().isMinimized()
            or (handle is not None and not handle.isExposed()),
        )

    def set_reduced_motion(self, reduced):
        self.reduced_motion = reduced
        self.sync_suspension()

    def open_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = VideoSettings(self)
        self.settings_dialog.show()
        self.settings_dialog.raise_()

    def save(self):
        if not self.persist:
            return
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.config_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "ascii_defaults_version": ASCII_DEFAULTS_VERSION,
                        "options": self.options,
                        "videos": [str(path) for path in self.paths],
                    },
                    indent=2,
                )
                + "\n"
            )
            temporary.replace(self.config_path)
        except OSError as error:
            self.message.emit(f"Cannot save video preferences: {error}")

    def shutdown(self):
        if self.settings_dialog:
            self.settings_dialog.close()
        self.run("shutdown")
        if self.view:
            self.view.deleteLater()
            self.view = None
            self.profile.deleteLater()
        self.ready = False
