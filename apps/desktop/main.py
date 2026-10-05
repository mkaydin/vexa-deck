"""VEXA//DECK Linux desktop console (Qt Widgets, Wayland compatible).

Run with the host's PySide6 installation:

    /usr/bin/python3 apps/desktop/main.py

The GUI launches the audio API in a separate Python 3.12 process. Set VEXA_API_URL to connect to
an already running host backend. --preview renders a static design preview without opening audio.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pixel_theme as console_theme
from backend_runtime import workspace_pythonpath
from console_settings import ConsoleSettingsDialog
from pixel_stage import DeckCard, NeonPanel, SpectrumWidget
from pixel_theme import font, install_font
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from video_panel import VideoPanel
from window_chrome import window_layout

ROOT = Path(__file__).resolve().parents[2]


def application_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        icon.addFile(str(ROOT / "assets/icons" / f"vexa-icon-{size}.png"))
    return icon


class ThemedLabel(QLabel):
    def setStyleSheet(self, css):
        self.setProperty("vexaSourceStyle", css)
        super().setStyleSheet(console_theme.themed_css(css))


def label(text: str, *, size: int = 11, color: str = "#c9dcd8", weight: int = 500) -> QLabel:
    result = ThemedLabel(text)
    result.setFont(font(size, weight))
    result.setStyleSheet(f"color: {color}; background: transparent;")
    result.setWordWrap(True)
    return result


def button(text: str, *, role: str = "secondary") -> QPushButton:
    result = QPushButton(f"[ {text} ]")
    result.setObjectName(role)
    result.setCursor(Qt.CursorShape.PointingHandCursor)
    result.setFont(font(10, 700))
    result.setMinimumHeight(34)
    return result


def panel() -> QFrame:
    result = NeonPanel()
    result.setObjectName("panel")
    return result


class BackendClient(QWidget):
    status_received = Signal(dict)
    health_changed = Signal(bool)
    action_finished = Signal(str, dict)
    problem = Signal(str)

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.network = QNetworkAccessManager(self)
        self.session_id: str | None = None
        self._pending_status = False
        self._poll = QTimer(self)
        self._poll.setInterval(400)
        self._poll.timeout.connect(self.poll)
        self._poll.start()

    def _request(self, method: str, path: str, payload: dict | None, callback) -> None:
        request = QNetworkRequest(QUrl(self.base_url + path))
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        if method == "POST":
            data = json.dumps(payload or {}).encode("utf-8")
            reply = self.network.post(request, data)
        elif method == "DELETE":
            reply = self.network.deleteResource(request)
        else:
            reply = self.network.get(request)

        def finished() -> None:
            raw = bytes(reply.readAll())
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = {}
            failed = reply.error() != QNetworkReply.NetworkError.NoError
            if failed:
                message = body.get("detail") or reply.errorString()
                callback(None, str(message))
            else:
                callback(body, None)
            reply.deleteLater()

        reply.finished.connect(finished)

    def poll(self) -> None:
        if self._pending_status:
            return
        self._pending_status = True

        def received(body: dict | None, error: str | None) -> None:
            self._pending_status = False
            self.health_changed.emit(error is None)
            if body is not None:
                self.status_received.emit(body)

        self._request("GET", "/live", None, received)

    def start_or_steer(self, theme: str, *, instrumental: bool | None = None) -> None:
        theme = theme.strip()
        if not theme:
            self.problem.emit("Enter a theme first.")
            return
        if self.session_id:
            path = f"/sessions/{self.session_id}/theme"
            action = "theme"
        else:
            path = "/sessions"
            action = "start"

        def received(body: dict | None, error: str | None) -> None:
            if error:
                self.problem.emit(f"Could not set the theme: {error}")
                return
            if body and body.get("session_id"):
                self.session_id = body["session_id"]
            self.action_finished.emit(action, body or {})
            self.poll()

        self._request("POST", path, {"theme": theme, "instrumental": instrumental}, received)

    def stop(self) -> None:
        def received(body: dict | None, error: str | None) -> None:
            if error:
                self.problem.emit(f"Could not stop playback: {error}")
                return
            self.session_id = None
            self.action_finished.emit("stop", body or {})
            self.poll()

        self._request("POST", "/live/stop", {}, received)

    def feedback(self, rating: str) -> None:
        if not self.session_id:
            self.problem.emit("Start a set before sending feedback.")
            return

        def received(body: dict | None, error: str | None) -> None:
            if error:
                self.problem.emit(f"Could not save feedback: {error}")
            else:
                self.action_finished.emit("feedback", body or {})

        self._request("POST", f"/sessions/{self.session_id}/feedback", {"rating": rating}, received)

    def configure_equalizer(self, payload, callback):
        self._request("POST", "/audio/equalizer", payload, callback)

    def list_tracks(self, callback) -> None:
        self._request("GET", "/depot/tracks", None, callback)

    def delete_track(self, asset_id: str, callback) -> None:
        self._request("DELETE", f"/depot/tracks/{asset_id}", None, callback)


class BackendProcess:
    def __init__(self, on_output=None) -> None:
        self.process: QProcess | None = None
        self.output = ""
        self.on_output = on_output
        log_dir = Path(os.environ.get("VEXA_LOG_DIR", ROOT / "var/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("VEXA_RUN_ID", datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
        self.log_path = log_dir / f"{run_id}-backend.log"
        self.url = os.environ.get("VEXA_API_URL", "").strip()
        if self.url:
            return
        backend_python = ROOT / ".venv/bin/python"
        if not backend_python.exists():
            raise RuntimeError("Backend environment missing. Run 'uv sync --extra core' first.")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._capture)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert(
            "PYTHONPATH", workspace_pythonpath(ROOT, environment.value("PYTHONPATH"))
        )
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            environment.insert(name, os.environ.get("VEXA_CPU_THREADS", "2"))
        environment.insert("VEXA_HOST", "127.0.0.1")
        environment.insert("VEXA_PORT", str(port))
        environment.insert("VEXA_LIVE", "1")
        environment.insert("VEXA_LIBRARY", str(ROOT / "assets/library"))
        self.process.setProcessEnvironment(environment)
        self.process.setProgram(str(backend_python))
        self.process.setArguments(["-m", "vexa_orchestrator"])
        self.process.start()

    def _capture(self) -> None:
        if self.process is not None:
            chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self.output = (self.output + chunk)[-8000:]
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(chunk)
            if self.on_output is not None:
                self.on_output(chunk)

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.process.terminate()
        if not self.process.waitForFinished(2000):
            self.process.kill()
            self.process.waitForFinished(2000)


class DepotDialog(QDialog):
    """Browse admitted YuE2 renders without touching the live audio engine."""

    def __init__(self, client: BackendClient, parent: QWidget) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("VEXA // GENERATED DEPOT")
        self.setWindowIcon(parent.windowIcon())
        self.resize(900, 520)
        layout = window_layout(self, self)
        layout.addWidget(
            label(
                "GENERATED TRACKS / SELECT ONE TO PREVIEW OR DELETE",
                size=11,
                color="#63e6d2",
                weight=700,
            )
        )
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["TITLE", "THEME", "LENGTH", "BPM", "STATE", "ASSET"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        self.listen_button = button("PLAY / LISTEN", role="primary")
        self.listen_button.clicked.connect(self.listen)
        controls.addWidget(self.listen_button)
        stop_button = button("STOP PREVIEW")
        stop_button.clicked.connect(self.stop_preview)
        controls.addWidget(stop_button)
        delete_button = button("DELETE", role="danger")
        delete_button.clicked.connect(self.delete_selected)
        controls.addWidget(delete_button)
        refresh_button = button("REFRESH")
        refresh_button.clicked.connect(self.refresh)
        controls.addWidget(refresh_button)
        layout.addLayout(controls)
        self.message = label(
            "Preview plays separately from the live decks.", size=9, color="#90b2b7"
        )
        layout.addWidget(self.message)
        self.audio_output: QAudioOutput | None = None
        self.player: QMediaPlayer | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def selected_id(self) -> str | None:
        row = self.table.currentRow()
        item = self.table.item(row, 5) if row >= 0 else None
        return item.text() if item is not None else None

    def selected_title(self) -> str:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text() if item else (self.selected_id() or "track")

    def refresh(self) -> None:
        selected = self.selected_id()

        def received(body, error) -> None:
            if error:
                self.message.setText(f"Depot unavailable: {error}")
                return
            tracks = body if isinstance(body, list) else []
            self.table.setRowCount(len(tracks))
            for row, track in enumerate(tracks):
                length = int(float(track.get("duration_s") or 0))
                state = (
                    "ON AIR"
                    if track.get("playing")
                    else "PREPARED"
                    if track.get("prepared")
                    else "READY"
                )
                values = [
                    str(track.get("title") or track.get("asset_id") or "--"),
                    str(track.get("theme") or "--"),
                    f"{length // 60}:{length % 60:02d}",
                    f"{float(track.get('bpm') or 0):.1f}",
                    state,
                    str(track.get("asset_id") or ""),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(
                        f"{track.get('title') or ''}\nID: {track.get('asset_id')}\n"
                        f"{track.get('style') or ''}"
                    )
                    self.table.setItem(row, column, item)
                if track.get("asset_id") == selected:
                    self.table.selectRow(row)
            self.message.setText(f"{len(tracks)} admitted generated tracks in the depot.")

        self.client.list_tracks(received)

    def listen(self) -> None:
        asset_id = self.selected_id()
        if asset_id is None:
            self.message.setText("Select a track first.")
            return
        if self.player is None:
            self.audio_output = QAudioOutput(self)
            self.audio_output.setVolume(0.7)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.audio_output)
            self.player.errorOccurred.connect(
                lambda _error, message: self.message.setText(f"Preview error: {message}")
            )
        self.player.stop()
        self.player.setSource(QUrl(f"{self.client.base_url}/depot/tracks/{asset_id}/audio"))
        self.player.play()
        self.message.setText(f"Listening to {self.selected_title()}")

    def stop_preview(self) -> None:
        if self.player is not None:
            self.player.stop()
        self.message.setText("Preview stopped.")

    def delete_selected(self) -> None:
        asset_id = self.selected_id()
        if asset_id is None:
            self.message.setText("Select a track first.")
            return
        title = self.selected_title()
        if QMessageBox.question(
            self, "Delete generated track", f"Move {title} and its reports to recoverable trash?"
        ) != (QMessageBox.StandardButton.Yes):
            return
        self.stop_preview()

        def received(_body, error) -> None:
            self.message.setText(
                f"Delete failed: {error}" if error else f"{title} moved to trash."
            )
            if not error:
                self.refresh()

        self.client.delete_track(asset_id, received)

    def closeEvent(self, event) -> None:
        self.stop_preview()
        super().closeEvent(event)


class VexaWindow(QMainWindow):
    def __init__(self, *, preview: bool = False) -> None:
        super().__init__()
        self.preview = preview
        self.backend: BackendProcess | None = None
        self.client: BackendClient | None = None
        self.started_at: float | None = None
        self._closing = False
        self.depot_dialog: DepotDialog | None = None
        log_dir = Path(os.environ.get("VEXA_LOG_DIR", ROOT / "var/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("VEXA_RUN_ID", datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
        self.log_path = log_dir / f"{run_id}-gui.log"
        self._last_status_signature = None
        self._last_status_error: str | None = None
        self._last_audio_underruns = 0
        self.setWindowTitle("VEXA//DECK -- DJ Vexa")
        self.setWindowIcon(application_icon())
        self.resize(1440, 900)
        self.setMinimumSize(720, 700)
        self.console_settings = None
        self._build()
        self.console_settings = ConsoleSettingsDialog(self, persist=not preview)
        self.console_settings.message.connect(self._log)
        self.console_settings.set_online(preview)
        self._log(f"GUI ready; run={run_id}")
        self._log("Center panel: local looping video, Normal / Pixel / ASCII styles")
        if preview:
            self._preview_status()
        else:
            try:
                self.backend = BackendProcess(self._backend_output)
                if self.backend.process is not None:
                    self.backend.process.finished.connect(self.on_backend_exit)
                self.client = BackendClient(self.backend.url)
                self.client.status_received.connect(self.on_status)
                self.client.health_changed.connect(self.on_health)
                self.client.action_finished.connect(self.on_action)
                self.client.problem.connect(self.on_problem)
                self.api_label.setText("CONNECTING / HOST AUDIO")
                self._log(f"Backend address {self.backend.url}")
            except RuntimeError as exc:
                self.on_problem(str(exc))

    def _log(self, message: str) -> None:
        line = f"{datetime.now(UTC).strftime('%H:%M:%S')} {message}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if hasattr(self, "log_view"):
            self.log_view.appendPlainText(line)

    def _backend_output(self, chunk: str) -> None:
        for line in chunk.splitlines():
            if line.strip():
                self.log_view.appendPlainText(line.strip())

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = window_layout(self, root)

        header = QHBoxLayout()
        masthead = label("V E X A // D E C K", size=20, color="#c9dcd8")
        masthead.setWordWrap(False)
        header.addWidget(masthead)
        self.subtitle = label(":: AFTER_HOURS / RADIO ::", color="#b194df")
        self.subtitle.setWordWrap(False)
        header.addWidget(self.subtitle)
        header.addStretch()
        self.api_label = label("CONNECTING / HOST AUDIO", size=9, color="#748f96")
        self.api_label.setMaximumWidth(210)
        header.addWidget(self.api_label)
        self.live_label = label("OFF AIR", color="#748f96")
        header.addWidget(self.live_label)
        outer.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(8)
        outer.addLayout(content, 1)
        left = panel()
        left.setFixedWidth(240)
        controls = QVBoxLayout(left)
        self.controls_layout = controls
        controls.setContentsMargins(18, 22, 18, 22)
        controls.setSpacing(8)
        controls.addWidget(label("[01] // SESSION", color="#63e6d2"))
        self.session_title = label("SET THE\nATMOSPHERE_", size=16, color="#c9dcd8")
        self.session_title.setFixedHeight(56)
        controls.addWidget(self.session_title)
        self.theme_input = QLineEdit()
        self.theme_input.setPlaceholderText("dark melodic techno...")
        self.theme_input.setMinimumHeight(36)
        self.theme_input.setAccessibleName("Theme or vibe")
        self.theme_input.returnPressed.connect(self.submit_theme)
        controls.addWidget(self.theme_input)
        self.vocal_mode = QComboBox()
        self.vocal_mode.setAccessibleName("Music vocal mode")
        self.vocal_mode.setMinimumHeight(30)
        self.vocal_mode.addItem("AUTO / FOLLOW THEME", None)
        self.vocal_mode.addItem("NO LYRICS / INSTRUMENTAL", True)
        self.vocal_mode.addItem("WITH VOCALS", False)
        self.vocal_mode.setToolTip(
            "Auto follows your theme. No lyrics generates instrumental tracks and selects "
            "instrumental depot matches. With vocals requests original lyrics. "
            "Select the mode before starting or shifting the vibe."
        )
        controls.addWidget(self.vocal_mode)
        self.start_button = button("START THE SET", role="primary")
        self.start_button.clicked.connect(self.submit_theme)
        controls.addWidget(self.start_button)
        quick_select_title = label("// QUICK SELECT", color="#748f96")
        self.quick_select_widgets = [quick_select_title]
        controls.addWidget(quick_select_title)
        for theme in ("rain-soaked jazz", "dark melodic techno", "warm house"):
            chip = button(
                {
                    "rain-soaked jazz": "RAIN-SOAKED JAZZ",
                    "dark melodic techno": "MELODIC TECHNO",
                    "warm house": "WARM HOUSE",
                }[theme],
                role="chip",
            )
            chip.clicked.connect(lambda checked=False, value=theme: self.choose_theme(value))
            self.quick_select_widgets.append(chip)
            controls.addWidget(chip)
        controls.addSpacing(4)
        controls.addWidget(label("// CURRENT VIBE", color="#abc78a"))
        self.theme_label = label("Awaiting your direction", color="#c9dcd8")
        self.theme_label.setMaximumHeight(62)
        controls.addWidget(self.theme_label)
        self.message = label("Vexa is ready. Enter a vibe to prepare your set.", color="#748f96")
        self.message.setMaximumHeight(98)
        controls.addWidget(self.message)
        controls.addWidget(label("// YUE2 TRACK BATCH", color="#b194df"))
        self.generation_label = label("0 / 10 READY\nAwaiting a theme", color="#748f96")
        controls.addWidget(self.generation_label)
        controls.addStretch()
        self.motion_button = button("REDUCE MOTION", role="chip")
        self.motion_button.setCheckable(True)
        self.motion_button.toggled.connect(self.set_reduced_motion)
        controls.addWidget(self.motion_button)
        content.addWidget(left)

        center = QVBoxLayout()
        center.setSpacing(8)
        content.addLayout(center, 1)
        self.video_panel = VideoPanel(self, persist=not self.preview)
        self.video_panel.message.connect(self._log)
        self.video_slot = self.video_panel.viewport
        self.stage = self.video_slot.stage
        center.addWidget(self.video_panel, 1)
        deck_row = QHBoxLayout()
        deck_row.setSpacing(8)
        self.deck_a, self.deck_b = DeckCard("DECK A"), DeckCard("DECK B")
        deck_row.addWidget(self.deck_a)
        deck_row.addWidget(self.deck_b)
        center.addLayout(deck_row)

        # Secondary session information collapses at small widths. Core controls stay visible.
        self.right_panel = panel()
        self.right_panel.setFixedWidth(240)
        details = QVBoxLayout(self.right_panel)
        details.setContentsMargins(18, 22, 18, 22)
        details.setSpacing(12)
        details.addWidget(label("[02] // NEXT RECORD", color="#b194df"))
        self.next_asset = label("No transition prepared", size=14, color="#c9dcd8")
        details.addWidget(self.next_asset)
        self.next_detail = label(
            "Cue and compatibility checks prepare the next record.", color="#748f96"
        )
        details.addWidget(self.next_detail)
        details.addSpacing(8)
        details.addWidget(label("// EVENT HISTORY", color="#abc78a"))
        self.history_layout = QVBoxLayout()
        self.history_layout.setSpacing(10)
        details.addLayout(self.history_layout)
        self._history_widgets = []
        for _ in range(4):
            widget = label("--", color="#748f96")
            widget.setMaximumHeight(62)
            self.history_layout.addWidget(widget)
            self._history_widgets.append(widget)
        details.addStretch()
        details.addWidget(label("// USER SIGNAL", color="#63e6d2"))
        details.addWidget(label("Like a track? Your feedback is saved for Vexa.", color="#748f96"))
        content.addWidget(self.right_panel)

        # Keep the spectrum across the console beneath the video and deck area.
        self.spectrum = SpectrumWidget()
        outer.addWidget(self.spectrum)

        self.diagnostics = QDialog(self)
        self.diagnostics.setWindowTitle("VEXA / DIAGNOSTICS")
        self.diagnostics.setWindowIcon(self.windowIcon())
        self.diagnostics.resize(800, 480)
        diagnostics_layout = window_layout(self.diagnostics, self.diagnostics)
        diagnostics_layout.addWidget(label("SYSTEM / EVENT LOG", color="#63e6d2"))
        self.summary_label = label("No session", color="#abc78a")
        diagnostics_layout.addWidget(self.summary_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(1000)
        self.log_view.setAccessibleName("Detailed event log")
        self.log_view.setFont(font(10))
        diagnostics_layout.addWidget(self.log_view)
        diagnostics_layout.addWidget(label(str(self.log_path), color="#748f96"))

        self.footer = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.footer.setSpacing(8)
        footer_status = QHBoxLayout()
        footer_status.setSpacing(8)
        self.transport = label("TRANSPORT / STANDBY", color="#abc78a")
        self.transport.setWordWrap(False)
        footer_status.addWidget(self.transport)
        footer_status.addStretch()
        self.health = label("AUDIO / WAITING", color="#748f96")
        self.health.setWordWrap(False)
        footer_status.addWidget(self.health)
        self.footer.addLayout(footer_status, 1)
        footer_actions = QHBoxLayout()
        footer_actions.setSpacing(8)
        footer_actions.addStretch()
        self.like_button, self.dislike_button = button("LIKE"), button("DISLIKE")
        self.like_button.clicked.connect(lambda: self.send_feedback("like"))
        self.dislike_button.clicked.connect(lambda: self.send_feedback("dislike"))
        footer_actions.addWidget(self.like_button)
        footer_actions.addWidget(self.dislike_button)
        self.depot_button = button("DEPOT", role="chip")
        self.depot_button.clicked.connect(self.open_depot)
        footer_actions.addWidget(self.depot_button)
        diagnostics_button = button("INFO", role="chip")
        diagnostics_button.clicked.connect(self.show_diagnostics)
        footer_actions.addWidget(diagnostics_button)
        self.settings_button = button("EQ / THEME", role="chip")
        self.settings_button.clicked.connect(self.show_console_settings)
        footer_actions.addWidget(self.settings_button)
        self.stop_button = button("STOP", role="danger")
        self.stop_button.clicked.connect(self.stop)
        footer_actions.addWidget(self.stop_button)
        self.footer.addLayout(footer_actions)
        outer.addLayout(self.footer)
        self._set_controls(False)
        self.start_button.setEnabled(self.preview)

    def show_console_settings(self):
        self.console_settings.show()
        self.console_settings.raise_()

    def show_diagnostics(self):
        self.diagnostics.show()
        self.diagnostics.raise_()

    def resizeEvent(self, event):
        if hasattr(self, "controls_layout"):
            compact = self.height() < 760
            self.controls_layout.setSpacing(6 if compact else 8)
            self.controls_layout.setContentsMargins(
                18, 18 if compact else 22, 18, 18 if compact else 22
            )
            self.session_title.setText("SET THE VIBE_" if compact else "SET THE\nATMOSPHERE_")
            self.session_title.setFixedHeight(30 if compact else 56)
        if hasattr(self, "footer"):
            self.footer.setDirection(
                QBoxLayout.Direction.TopToBottom
                if self.width() < 1000
                else QBoxLayout.Direction.LeftToRight
            )
            self.subtitle.setVisible(self.width() >= 1000)
        if hasattr(self, "right_panel"):
            self.right_panel.setVisible(self.width() >= 1280)
        if hasattr(self, "quick_select_widgets"):
            for widget in self.quick_select_widgets:
                widget.setVisible(self.height() >= 820)
        super().resizeEvent(event)

    def _set_controls(self, running: bool) -> None:
        self.like_button.setEnabled(running)
        self.dislike_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.start_button.setText("[ SHIFT THE VIBE ]" if running else "[ START THE SET ]")

    def choose_theme(self, theme: str) -> None:
        self.theme_input.setText(theme)
        self.theme_input.setFocus()

    def submit_theme(self) -> None:
        if self.preview:
            self.message.setText("Preview mode: audio backend is not running.")
        elif self.client is not None:
            self.client.start_or_steer(
                self.theme_input.text(), instrumental=self.vocal_mode.currentData()
            )

    def stop(self) -> None:
        if self.client is not None:
            self.client.stop()

    def open_depot(self) -> None:
        if self.client is None:
            self._log("Depot browser needs the audio backend")
            return
        if self.depot_dialog is None:
            self.depot_dialog = DepotDialog(self.client, self)
        self.depot_dialog.refresh()
        self.depot_dialog.show()
        self.depot_dialog.raise_()

    def send_feedback(self, rating: str) -> None:
        if self.client is not None:
            self.client.feedback(rating)

    def set_reduced_motion(self, enabled: bool) -> None:
        self.video_panel.set_reduced_motion(enabled)
        self.spectrum.reduced_motion = enabled
        self.deck_a.reduced_motion = enabled
        self.deck_b.reduced_motion = enabled
        self.motion_button.setText("[ MOTION OFF ]" if enabled else "[ REDUCE MOTION ]")

    def on_health(self, online: bool) -> None:
        self.console_settings.set_online(online)
        self.start_button.setEnabled(online)
        self.api_label.setText("API / CONNECTED" if online else "API / CONNECTING")
        self.api_label.setStyleSheet(
            "color: #63e6d2; background: transparent;"
            if online
            else "color: #748f96; background: transparent;"
        )

    def on_problem(self, message: str) -> None:
        self._log(f"ERROR {message}")
        self.message.setText(message)
        self.message.setStyleSheet("color: #d99db8; background: transparent;")

    def on_backend_exit(self, code: int, _status: QProcess.ExitStatus) -> None:
        if self.backend is None or self._closing:
            return
        detail = self.backend.output.strip().splitlines()
        self.on_problem(
            f"Audio backend exited ({code}). {detail[-1] if detail else 'See terminal output.'}"
        )
        self.api_label.setText("AUDIO BACKEND OFFLINE")
        self.start_button.setEnabled(False)

    def on_action(self, action: str, data: dict) -> None:
        self._log(f"ACTION {action} {json.dumps(data, default=str)[:500]}")
        if action == "stop":
            self.started_at = None
            self.message.setText("Audio stopped. Vexa is standing by.")
        elif action == "feedback":
            self.message.setText("Feedback saved for later Laya training.")
        elif action in {"start", "theme"}:
            if self.started_at is None:
                self.started_at = time.monotonic()
            self.message.setText(
                "Vexa is finding full-length matches and starting a 10-track YuE2 batch."
                if action == "start"
                else "New direction received. The current track keeps playing."
            )
        self.message.setStyleSheet("color: #90b2b7; background: transparent;")

    def on_status(self, status: dict) -> None:
        self.console_settings.sync_equalizer(
            (status.get("audio") or {}).get("equalizer"), running=bool(status.get("running"))
        )
        signature = (
            status.get("current_asset_id"),
            status.get("prepared_asset_id"),
            status.get("prepared_loaded"),
            status.get("generating"),
            (status.get("generation") or {}).get("current"),
            status.get("error"),
        )
        if signature != self._last_status_signature:
            self._log(
                f"STATUS current={signature[0]} next={signature[1]} "
                f"loaded={signature[2]} generating={signature[3]} "
                f"track={signature[4]} error={signature[5]}"
            )
            self._last_status_signature = signature
        running = bool(status.get("running"))
        generating = bool(status.get("generating"))
        generation = status.get("generation") or {}
        completed = int(generation.get("completed") or 0)
        failed = int(generation.get("failed") or 0)
        current = int(generation.get("current") or 0)
        origin = generation.get("prompt_origin") or "planning"
        self.generation_label.setText(
            f"{completed} / {generation.get('target', 10)} READY  /  "
            f"{failed} FAILED\n"
            + (
                f"Rendering track {current}  /  {origin}"
                if current
                else "Writing music plan..."
                if generating
                else "Batch complete"
                if completed or failed
                else "Planning failed / retry set"
                if status.get("error")
                else "Awaiting a theme"
            )
        )
        if self.client is not None:
            self.client.session_id = status.get("session_id")
        self.spectrum.set_levels(status.get("spectrum") or [0.0] * 32)
        self._set_controls(running)
        self.live_label.setText("LIVE" if running else "GENERATING" if generating else "OFF AIR")
        self.live_label.setStyleSheet(
            "color: #63e6d2; background: transparent;"
            if running
            else "color: #b194df; background: transparent;"
            if generating
            else "color: #748f96; background: transparent;"
        )
        self.theme_label.setText(status.get("theme") or "Awaiting your first direction")
        upcoming = status.get("prepared_asset_id")
        self.next_asset.setText(
            status.get("prepared_title") or "Next track ready"
            if upcoming
            else "Creating your set"
            if generating
            else "Waiting for a direction"
        )
        self.next_detail.setText(
            "Compatible transition prepared"
            if upcoming
            else "Generation runs in the background while music continues."
            if generating
            else "Vexa will hold the current track until a safe match is ready."
        )
        self.summary_label.setText(
            f"{self.theme_label.text()}\n{self.next_asset.text()} / {self.next_detail.text()}\n"
            f"{self.generation_label.text()}"
        )
        audio = status.get("audio") or {}
        titles = status.get("asset_titles") or {}
        active = int(status.get("active_deck") or 0)
        duration = status.get("current_duration_s")
        prepared_duration = status.get("prepared_duration_s")
        self.deck_a.set_status(
            audio.get("deck_a") or {},
            status.get("deck_a_asset_id"),
            title=titles.get(status.get("deck_a_asset_id"))
            or (status.get("current_title") if active == 0 else status.get("prepared_title")),
            active=running and active == 0,
            duration_s=duration if active == 0 else prepared_duration,
        )
        self.deck_b.set_status(
            audio.get("deck_b") or {},
            status.get("deck_b_asset_id"),
            title=titles.get(status.get("deck_b_asset_id"))
            or (status.get("current_title") if active == 1 else status.get("prepared_title")),
            active=running and active == 1,
            duration_s=duration if active == 1 else prepared_duration,
        )
        events = [
            event
            for event in status.get("events") or []
            if event.get("event") in {"start", "transition", "generated", "generation_failed"}
        ]
        for widget, event in zip(self._history_widgets, reversed(events[-4:]), strict=False):
            name = event.get("title") or titles.get(event.get("asset_id"))
            name = name or event.get("asset_id") or event.get("reason") or "unknown"
            widget.setText(f"{event.get('event', '').upper()} / {name}")
        for widget in self._history_widgets[len(events[-4:]) :]:
            widget.setText("--")
        error = status.get("error")
        if error:
            if error != self._last_status_error:
                self.on_problem(error)
        elif self._last_status_error:
            self.message.setText(
                "Preparing your music plan and tracks..." if generating
                else "Vexa is playing your set." if running
                else "Vexa is ready. Enter a vibe to prepare your set."
            )
            self.message.setStyleSheet("color: #90b2b7; background: transparent;")
        self._last_status_error = error
        elapsed = int(time.monotonic() - self.started_at) if self.started_at else 0
        self.transport.setText(
            f"TRANSPORT / {elapsed // 60:02d}:{elapsed % 60:02d}"
            + (f"  ::  {status['bpm']:.1f} BPM" if status.get("bpm") else "")
        )
        stats = audio.get("stats") or {}
        underruns = int(stats.get("underruns", 0))
        device = audio.get("device") or {}
        if underruns > self._last_audio_underruns:
            self._log(
                f"AUDIO underruns={underruns} "
                f"device_underflows={stats.get('output_underflows', 0)} "
                f"callback_ms={stats.get('worst_callback_ms', 0)} "
                f"cpu_load={device.get('cpu_load', 0):.3f} "
                f"generating={generating}"
            )
        self._last_audio_underruns = underruns
        self.health.setText(f"AUDIO / {underruns} XRUN")
        self.health.setToolTip(
            f"{underruns} underruns / {stats.get('queue_dropped', 0)} dropped / "
            f"{float(device.get('latency_s', 0)) * 1000:.0f} ms buffer"
        )

    @staticmethod
    def display_asset(asset_id: str) -> str:
        parts = asset_id.split("-")
        if len(parts) >= 3 and parts[0] == "depot" and parts[1].isdigit():
            return f"{parts[2].replace('_', ' ').upper()} / {parts[1]} BPM"
        return asset_id.replace("-", " ").replace("_", " ").upper()

    def _preview_status(self) -> None:
        self.api_label.setText("PREVIEW / NO AUDIO")
        self.message.setText("Design preview. Audio backend is offline.")
        self.theme_input.setText("dark melodic techno")
        sample = {
            "running": True,
            "generating": False,
            "theme": "dark melodic techno",
            "generation": {
                "target": 10,
                "completed": 10,
                "failed": 0,
                "current": 0,
                "prompt_origin": "llm:local",
            },
            "current_title": "Neon Circuit Reverie",
            "prepared_title": "Afterhours Voltage Drift",
            "asset_titles": {"live-92efb7f7f5ea": "Neon Circuit Reverie",
                             "live-a4e26849bf03": "Afterhours Voltage Drift"},
            "current_asset_id": "live-92efb7f7f5ea",
            "prepared_asset_id": "live-a4e26849bf03",
            "prepared_loaded": True,
            "deck_a_asset_id": "live-92efb7f7f5ea",
            "deck_b_asset_id": "live-a4e26849bf03",
            "active_deck": 0,
            "bpm": 128.0,
            "current_duration_s": 165.0,
            "prepared_duration_s": 168.0,
            "spectrum": [0.35 + 0.25 * abs(math.sin(index * 0.71)) for index in range(32)],
            "audio": {
                "deck_a": {"playing": True, "loaded": True, "gain": 1.0, "position": 400000},
                "deck_b": {"playing": False, "loaded": True, "gain": 0.0, "position": 0},
                "stats": {"underruns": 0, "queue_dropped": 0},
            },
            "events": [{"event": "start", "asset_id": "live-92efb7f7f5ea"}],
        }
        self.on_status(sample)
        self.live_label.setText("PREVIEW")

    def closeEvent(self, event) -> None:
        self._closing = True
        self.video_panel.shutdown()
        if self.depot_dialog is not None:
            self.depot_dialog.close()
        if self.backend is not None:
            self.backend.close()
        self._log("GUI closed")
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true", help="show design without backend or audio"
    )
    parser.add_argument("--screenshot", type=Path, help="save a PNG after the window is painted")
    args = parser.parse_args()
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    app = QApplication(sys.argv[:1])
    app.setApplicationName("vexa-deck")
    app.setApplicationDisplayName("VEXA//DECK")
    app.setDesktopFileName("vexa-deck")
    app.setWindowIcon(application_icon())
    install_font()
    app.setFont(font())
    console_theme.apply_theme(app, "terminal")
    window = VexaWindow(preview=args.preview)
    window.show()
    if args.screenshot:

        def capture() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            app.quit()

        QTimer.singleShot(500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
