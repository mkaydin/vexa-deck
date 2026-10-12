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
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import pixel_theme as console_theme
from backend_runtime import workspace_pythonpath
from console_settings import ConsoleSettingsDialog
from generation_controls import GenerationControlsDialog, GenerationReportDialog
from model_manager import ModelManager
from pixel_stage import DeckCard, NeonPanel, SpectrumWidget
from pixel_theme import font, install_font
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
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
    QProgressBar,
    QPushButton,
    QScrollArea,
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
    telemetry_received = Signal(dict)
    status_received = Signal(dict)
    models_received = Signal(dict)
    batches_received = Signal(dict)
    writers_received = Signal(dict)
    writer_pending = Signal(bool)
    generation_settings_received = Signal(dict)
    intent_received = Signal(dict)
    theme_pending = Signal(bool)
    playback_pending = Signal(bool)
    health_changed = Signal(bool)
    action_finished = Signal(str, dict)
    problem = Signal(str)

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.network = QNetworkAccessManager(self)
        self.session_id: str | None = None
        self._telemetry_reply = None
        self._telemetry_buffer = b""
        self._pending_status = False
        self._pending_theme = False
        self._pending_playback_start = False
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
                self.start_telemetry()

        self._request("GET", "/live", None, received)

    def shutdown(self):
        self._poll.stop()
        if self._telemetry_reply is not None:
            self._telemetry_reply.abort()

    def start_telemetry(self):
        if self._telemetry_reply is not None:
            return
        reply = self.network.get(QNetworkRequest(QUrl(self.base_url + "/audio/telemetry")))
        self._telemetry_reply = reply
        self._telemetry_buffer = b""

        def received():
            self._telemetry_buffer += bytes(reply.readAll())
            while b"\n\n" in self._telemetry_buffer:
                frame, self._telemetry_buffer = self._telemetry_buffer.split(b"\n\n", 1)
                if frame.startswith(b"data: "):
                    with suppress(ValueError, UnicodeDecodeError):
                        self.telemetry_received.emit(json.loads(frame[6:]))
            if len(self._telemetry_buffer) > 65536:
                self._telemetry_buffer = b""

        def finished():
            self._telemetry_reply = None
            reply.deleteLater()

        reply.readyRead.connect(received)
        reply.finished.connect(finished)

    def refresh_models(self) -> None:
        def received(body, error):
            if error:
                self.problem.emit(f"Could not read music generators: {error}")
            elif body:
                self.models_received.emit(body)

        self._request("GET", "/generation/models", None, received)

    def refresh_batches(self) -> None:
        def received(body, error):
            if error:
                self.problem.emit(f"Could not read saved batches: {error}")
            elif body is not None:
                self.batches_received.emit(body)

        self._request("GET", "/generation/batches", None, received)

    def refresh_writers(self) -> None:
        def received(body, error):
            if error:
                self.problem.emit(f"Could not read prompt writers: {error}")
            elif body is not None:
                self.writers_received.emit(body)

        self._request("GET", "/generation/writers", None, received)

    def refresh_generation_settings(self) -> None:
        def received(body, error):
            if error:
                self.problem.emit(f"Could not read generation settings: {error}")
            elif body is not None:
                self.generation_settings_received.emit(body)

        self._request("GET", "/generation/settings", None, received)

    def configure_writer(self, profile: dict) -> None:
        def received(body, error):
            self.writer_pending.emit(False)
            if error:
                self.problem.emit(f"Could not select prompt writer: {error}")
                self.refresh_writers()
                return
            self.writers_received.emit(body or {})
            self.action_finished.emit("writer", body or {})

        self.writer_pending.emit(True)
        self._request("POST", "/generation/writer", profile, received)

    def resume_batch(self, batch_id: str) -> None:
        def received(body, error):
            if error:
                self.problem.emit(f"Could not resume batch: {error}")
                return
            self.action_finished.emit("resume", body or {})
            self.poll()

        self._request(
            "POST", f"/generation/batches/{batch_id}/resume", {"retry_failed": True}, received
        )

    def start_or_steer(
        self,
        theme: str,
        *,
        instrumental: bool | None = None,
        generator: str | None = None,
        generator_preset: str | None = None,
        generation_settings: dict | None = None,
        generation_enabled: bool = True,
    ) -> None:
        theme = theme.strip()
        if self._pending_theme:
            return
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
            self._pending_theme = False
            self.theme_pending.emit(False)
            if error:
                self.problem.emit(f"Could not set the theme: {error}")
                return
            if body and body.get("session_id"):
                self.session_id = body["session_id"]
            self.action_finished.emit(action, body or {})
            self.poll()

        payload = {
            "theme": theme,
            "instrumental": instrumental,
            "generator": generator,
            "generator_preset": generator_preset,
            "generation_settings": generation_settings,
            "generation_enabled": generation_enabled,
        }

        def previewed(body, error):
            if error:
                self._pending_theme = False
                self.theme_pending.emit(False)
                self.problem.emit(f"Check generation settings: {error}")
                return
            self.intent_received.emit(body)
            self._request("POST", path, payload, received)

        self._pending_theme = True
        self.theme_pending.emit(True)
        if generation_settings is not None:
            self._request("POST", "/generation/preview", payload, previewed)
        else:
            self._request("POST", path, payload, received)

    def start_playback(self) -> None:
        if self._pending_playback_start:
            return
        self._pending_playback_start = True
        self.playback_pending.emit(True)

        def received(body: dict | None, error: str | None) -> None:
            if error:
                self.problem.emit(f"Could not start playback: {error}")
            else:
                self.status_received.emit(body or {})
                self.action_finished.emit("playback_start", body or {})
            self._pending_playback_start = False
            self.playback_pending.emit(False)
            self.poll()

        self._request("POST", "/live/start", {}, received)

    def stop(self) -> None:
        def received(body: dict | None, error: str | None) -> None:
            if error:
                self.problem.emit(f"Could not stop playback: {error}")
                return
            self.status_received.emit(body or {})
            self.action_finished.emit("stop", body or {})
            self.poll()

        self._request("POST", "/live/stop", {}, received)

    def skip(self):
        def received(body, error):
            if error:
                self.problem.emit("Could not skip: " + error)
            else:
                self.status_received.emit(body or {})
                self.action_finished.emit("skip", body or {})
            self.poll()

        self._request("POST", "/live/skip", {}, received)

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
        if not self.process.waitForFinished(8000):
            self.process.kill()
            self.process.waitForFinished(2000)


class DepotDialog(QDialog):
    """Browse admitted generated tracks without touching the live audio engine."""

    def __init__(self, client: BackendClient, parent: QWidget) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("VEXA / MUSIC LIBRARY")
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
        self.table.setColumnHidden(5, True)
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
            self.message.setText(f"Delete failed: {error}" if error else f"{title} moved to trash.")
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
        self._model_catalog = {}
        self._models_requested = False
        self._resumable_batch_id = None
        self._writer_catalog = []
        self._selected_writer = None
        self._writer_pending = False
        self._theme_pending = False
        self._playback_pending = False
        self._api_online = preview
        self._generation_preferences = None
        self.model_manager = None
        self._setup_prompted = False
        self._generation_auto_disabled = False
        self.generation_dialog = None
        self.report_dialog = None
        self._last_telemetry = 0.0
        self.last_status = {}
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
                self.client.models_received.connect(self.on_models)
                self.client.batches_received.connect(self.on_batches)
                self.client.writers_received.connect(self.on_writers)
                self.client.writer_pending.connect(self.on_writer_pending)
                self.client.generation_settings_received.connect(self.on_generation_settings)
                self.client.intent_received.connect(self.on_intent)
                self.client.theme_pending.connect(self.on_theme_pending)
                self.client.playback_pending.connect(self.on_playback_pending)
                self.client.telemetry_received.connect(self.on_telemetry)
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
        header.setSpacing(8)
        self.masthead = label("VEXA // DECK", size=18, color="#c9dcd8", weight=700)
        self.masthead.setWordWrap(False)
        header.addWidget(self.masthead)
        self.subtitle = label("AFTER HOURS / RADIO", size=9, color="#b194df")
        self.subtitle.setWordWrap(False)
        header.addWidget(self.subtitle)
        header.addStretch()
        self.depot_button = button("LIBRARY", role="chip")
        self.depot_button.setAccessibleName("Music library")
        self.depot_button.setToolTip("Listen to and manage generated tracks")
        self.depot_button.clicked.connect(self.open_depot)
        header.addWidget(self.depot_button)
        self.settings_button = button("EQ / THEME", role="chip")
        self.settings_button.clicked.connect(self.show_console_settings)
        header.addWidget(self.settings_button)
        self.diagnostics_button = button("LOGS", role="chip")
        self.diagnostics_button.clicked.connect(self.show_diagnostics)
        header.addWidget(self.diagnostics_button)
        self.api_label = label("CONNECTING / HOST AUDIO", size=9, color="#748f96")
        self.api_label.setMaximumWidth(160)
        header.addWidget(self.api_label)
        self.live_label = label("OFF AIR", size=10, color="#748f96", weight=700)
        self.live_label.setWordWrap(False)
        header.addWidget(self.live_label)
        outer.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(12)
        outer.addLayout(content, 1)
        self.session_panel = panel()
        self.session_scroll = QScrollArea()
        self.session_scroll.setFixedWidth(300)
        self.session_scroll.setWidgetResizable(True)
        self.session_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.session_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.session_scroll.setWidget(self.session_panel)
        controls = QVBoxLayout(self.session_panel)
        self.controls_layout = controls
        controls.setContentsMargins(16, 18, 16, 18)
        controls.setSpacing(10)
        controls.addWidget(label("01 / YOUR SESSION", size=9, color="#63e6d2"))
        self.session_title = label("Set the atmosphere", size=16, weight=700)
        controls.addWidget(self.session_title)
        self.theme_input = QLineEdit()
        self.theme_input.setPlaceholderText("Describe a sound or mood…")
        self.theme_input.setMinimumHeight(40)
        self.theme_input.setAccessibleName("Theme or vibe")
        self.theme_input.textChanged.connect(self.theme_input.setToolTip)
        self.theme_input.returnPressed.connect(self.submit_theme)
        controls.addWidget(self.theme_input)
        self.vocal_mode = QComboBox()
        self.vocal_mode.setAccessibleName("Music vocal mode")
        self.vocal_mode.setMinimumHeight(34)
        self.vocal_mode.addItem("Vocals: follow the theme", None)
        self.vocal_mode.addItem("Instrumental / no lyrics", True)
        self.vocal_mode.addItem("Sung vocals", "singing")
        self.vocal_mode.addItem("Spoken rap", "rap")
        self.vocal_mode.setToolTip("Choose the vocal mode before starting or changing the vibe.")
        controls.addWidget(self.vocal_mode)
        self.start_button = button("START THE SET", role="primary")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.submit_theme)
        controls.addWidget(self.start_button)
        self.quick_select = QComboBox()
        self.quick_select.setAccessibleName("Quick mood selection")
        self.quick_select.addItem("Try a mood…", None)
        for name, value in (
            ("Rain-soaked jazz", "rain-soaked jazz"),
            ("Dark melodic techno", "dark melodic techno"),
            ("Warm house", "warm house"),
        ):
            self.quick_select.addItem(name, value)
        self.quick_select.activated.connect(self._choose_quick_mood)
        controls.addWidget(self.quick_select)
        self.music_setup_button = button("MUSIC SETUP", role="chip")
        self.music_setup_button.setToolTip("Models, prompt writer and detailed generation controls")
        self.music_setup_button.clicked.connect(self.show_music_setup)
        controls.addWidget(self.music_setup_button)
        self.intent_label = label("10 tracks / full length", size=9, color="#748f96")
        controls.addWidget(self.intent_label)
        controls.addSpacing(6)
        controls.addWidget(label("CURRENT DIRECTION", size=9, color="#abc78a"))
        self.theme_label = label("Awaiting your direction", color="#c9dcd8")
        self.theme_label.setMaximumHeight(48)
        controls.addWidget(self.theme_label)
        self.message = label("Enter a vibe to prepare your set.", size=10, color="#748f96")
        self.message.setMaximumHeight(90)
        controls.addWidget(self.message)
        controls.addSpacing(6)
        controls.addWidget(label("02 / GENERATION", size=9, color="#b194df"))
        self.batch_progress = QProgressBar()
        self.batch_progress.setAccessibleName("Tracks ready in current batch")
        self.batch_progress.setRange(0, 10)
        self.batch_progress.setValue(0)
        self.batch_progress.setTextVisible(False)
        self.batch_progress.setFixedHeight(8)
        controls.addWidget(self.batch_progress)
        self.generation_label = label("0 / 10 READY / Awaiting a theme", size=9, color="#748f96")
        controls.addWidget(self.generation_label)
        recovery = QHBoxLayout()
        recovery.setSpacing(6)
        self.resume_button = button("RETRY", role="chip")
        self.resume_button.setToolTip("Resume the saved batch and retry failed slots")
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(self.resume_batch)
        recovery.addWidget(self.resume_button)
        self.track_reports_button = button("REPORTS", role="chip")
        self.track_reports_button.setToolTip(
            "Track plans, measured quality, failures and cancellation"
        )
        self.track_reports_button.clicked.connect(self.open_track_reports)
        recovery.addWidget(self.track_reports_button)
        controls.addLayout(recovery)
        controls.addSpacing(6)
        controls.addWidget(label("03 / UP NEXT", size=9, color="#63e6d2"))
        self.next_asset = label("No transition prepared", size=12, weight=700)
        self.next_asset.setMaximumHeight(52)
        controls.addWidget(self.next_asset)
        self.next_detail = label(
            "Vexa prepares the next compatible record.", size=9, color="#748f96"
        )
        self.next_detail.setMaximumHeight(100)
        controls.addWidget(self.next_detail)
        controls.addStretch()
        content.addWidget(self.session_scroll)

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
        self.spectrum = SpectrumWidget()
        center.addWidget(self.spectrum)

        self._build_music_setup()
        self.diagnostics = QDialog(self)
        self.diagnostics.setWindowTitle("VEXA / LOGS & ACTIVITY")
        self.diagnostics.setWindowIcon(self.windowIcon())
        self.diagnostics.resize(800, 600)
        diagnostics_layout = window_layout(self.diagnostics, self.diagnostics)
        self.summary_label = label("No session", size=10, color="#abc78a")
        diagnostics_layout.addWidget(self.summary_label)
        diagnostics_layout.addWidget(label("RECENT DECK MOVES", size=9, color="#b194df"))
        self.history_layout = QVBoxLayout()
        self.history_layout.setSpacing(6)
        diagnostics_layout.addLayout(self.history_layout)
        self._history_widgets = []
        for _ in range(4):
            widget = label("--", size=9, color="#748f96")
            self.history_layout.addWidget(widget)
            self._history_widgets.append(widget)
        diagnostics_layout.addWidget(label("SYSTEM / EVENT LOG", size=9, color="#63e6d2"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(1000)
        self.log_view.setAccessibleName("Detailed event log")
        self.log_view.setFont(font(10))
        diagnostics_layout.addWidget(self.log_view, 1)
        diagnostics_layout.addWidget(label(str(self.log_path), size=9, color="#748f96"))

        self.footer = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.footer.setSpacing(8)
        footer_status = QHBoxLayout()
        self.transport = label("TRANSPORT / STANDBY", size=9, color="#abc78a")
        self.transport.setWordWrap(False)
        footer_status.addWidget(self.transport)
        footer_status.addStretch()
        self.health = label("AUDIO / WAITING", size=9, color="#748f96")
        self.health.setWordWrap(False)
        footer_status.addWidget(self.health)
        self.footer.addLayout(footer_status, 1)
        self.footer_actions = QHBoxLayout()
        self.footer_actions.setSpacing(8)
        self.like_button, self.dislike_button = button("LIKE"), button("DISLIKE")
        self.like_button.setToolTip("Save positive feedback on the current track")
        self.dislike_button.setToolTip("Save negative feedback on the current track")
        self.like_button.clicked.connect(lambda: self.send_feedback("like"))
        self.dislike_button.clicked.connect(lambda: self.send_feedback("dislike"))
        self.footer_actions.addWidget(self.like_button)
        self.footer_actions.addWidget(self.dislike_button)
        self.footer_actions.addStretch()
        self.play_button = button("START", role="primary")
        self.play_button.setAccessibleName("Start playback")
        self.play_button.setToolTip("Start the set, restart stopped playback, or play ready music.")
        self.play_button.clicked.connect(self.start_playback)
        self.footer_actions.addWidget(self.play_button)
        self.stop_button = button("STOP", role="danger")
        self.stop_button.setToolTip("Stop playback only. Music generation continues.")
        self.stop_button.clicked.connect(self.stop)
        self.skip_button = button("SKIP", role="chip")
        self.skip_button.setAccessibleName("Skip current track")
        self.skip_button.setToolTip("Advance once; generation continues")
        self.skip_button.clicked.connect(lambda: self.client.skip() if self.client else None)
        self.footer_actions.addWidget(self.skip_button)
        self.footer_actions.addWidget(self.stop_button)
        self.footer.addLayout(self.footer_actions)
        outer.addLayout(self.footer)
        self._set_controls(False)
        self.start_button.setEnabled(self.preview)
        self.setTabOrder(self.theme_input, self.vocal_mode)
        self.setTabOrder(self.vocal_mode, self.start_button)
        self.setTabOrder(self.start_button, self.quick_select)
        self.setTabOrder(self.quick_select, self.music_setup_button)

    def _build_music_setup(self):
        self.music_setup = QDialog(self)
        self.music_setup.setWindowTitle("VEXA / MUSIC SETUP")
        self.music_setup.setWindowIcon(self.windowIcon())
        self.music_setup.resize(570, 570)
        self.music_setup.setMinimumSize(440, 450)
        outer = window_layout(self.music_setup, self.music_setup)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("generationSettingsBody")
        scroll.setWidget(body)
        settings = QVBoxLayout(body)
        settings.setContentsMargins(18, 18, 18, 18)
        settings.setSpacing(12)
        settings.addWidget(label("SHAPE YOUR NEXT BATCH", size=16, weight=700))
        settings.addWidget(
            label(
                "Choose the generator and prompt writer here. For length, tempo, lyrics, "
                "arrangement and quality rules, open Detailed controls.",
                size=10,
                color="#748f96",
            )
        )
        self.generate_enabled = QCheckBox("Generate new music (off = depot only)")
        self.generate_enabled.setChecked(True)
        self.generate_enabled.clicked.connect(
            lambda: setattr(self, "_generation_auto_disabled", False)
        )
        self.generate_enabled.setToolTip(
            "Existing generation keeps running; this controls new generation requests."
        )
        settings.addWidget(self.generate_enabled)
        settings.addWidget(label("MUSIC GENERATOR", size=9, color="#63e6d2"))
        self.generator_select = QComboBox()
        self.generator_select.setAccessibleName("Music generator")
        self.generator_select.setMinimumHeight(34)
        for identity, name in (
            ("yue2", "YuE2"),
            ("acestep", "ACE-Step 1.5 Turbo"),
            ("minimax", "MiniMax Music 3"),
        ):
            self.generator_select.addItem(name, identity)
        settings.addWidget(self.generator_select)
        settings.addWidget(label("MODEL PRESET", size=9, color="#748f96"))
        self.generator_preset = QComboBox()
        self.generator_preset.setAccessibleName("Generator quality preset")
        self.generator_preset.setMinimumHeight(34)
        settings.addWidget(self.generator_preset)
        self.model_status = label("Checking installed models", size=9, color="#748f96")
        settings.addWidget(self.model_status)
        settings.addWidget(label("PROMPT WRITER", size=9, color="#b194df"))
        self.writer_select = QComboBox()
        self.writer_select.setAccessibleName("Prompt writer for new batches")
        self.writer_select.setMinimumHeight(34)
        self.writer_select.addItem("Checking writers...", None)
        self.writer_select.setEnabled(False)
        self.writer_select.setToolTip(
            "Changes apply to new batches; existing batches retain their writer."
        )
        self.writer_select.currentIndexChanged.connect(self.on_writer_choice)
        settings.addWidget(self.writer_select)
        self.generation_controls_button = button("DETAILED CONTROLS", role="primary")
        self.generation_controls_button.clicked.connect(self.open_generation_controls)
        settings.addWidget(self.generation_controls_button)
        self.model_manager_button = button("MODEL MANAGER / DOWNLOAD", role="chip")
        self.model_manager_button.clicked.connect(self.show_model_manager)
        settings.addWidget(self.model_manager_button)
        self.motion_button = button("REDUCE MOTION", role="chip")
        self.motion_button.setCheckable(True)
        self.motion_button.toggled.connect(self.set_reduced_motion)
        settings.addWidget(self.motion_button)
        settings.addStretch()
        outer.addWidget(scroll, 1)
        done = button("DONE", role="chip")
        done.clicked.connect(self.music_setup.hide)
        outer.addWidget(done)
        self.generator_select.currentIndexChanged.connect(self.on_generator_choice)
        self.generator_preset.currentIndexChanged.connect(self.update_model_status)
        self.on_generator_choice()

    def show_model_manager(self):
        if self.model_manager is None:
            self.model_manager = ModelManager(self)
        self.model_manager.resize(min(680, self.width() - 40), min(620, self.height() - 40))
        self.model_manager.show()
        self.model_manager.raise_()

    def _choose_quick_mood(self, index):
        value = self.quick_select.itemData(index)
        if value:
            self.choose_theme(value)
        self.quick_select.setCurrentIndex(0)

    def show_music_setup(self):
        self.music_setup.resize(min(570, self.width() - 40), min(570, self.height() - 40))
        self.music_setup.show()
        self.music_setup.raise_()

    def show_console_settings(self):
        self.console_settings.show()
        self.console_settings.raise_()

    def show_diagnostics(self):
        self.diagnostics.resize(min(800, self.width() - 40), min(600, self.height() - 40))
        self.diagnostics.show()
        self.diagnostics.raise_()

    def resizeEvent(self, event):
        compact = self.width() < 1000
        if hasattr(self, "session_scroll"):
            self.session_scroll.setFixedWidth(250 if compact else 300)
            short = self.height() < 780
            self.controls_layout.setSpacing(6 if short else 10)
            self.controls_layout.setContentsMargins(
                16, 14 if short else 18, 16, 14 if short else 18
            )
            self.message.setMaximumHeight(54 if short else 90)
            self.next_detail.setMaximumHeight(56 if short else 100)
        if hasattr(self, "footer"):
            self.footer.setDirection(
                QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight
            )
            self.subtitle.setVisible(self.width() >= 1200)
            self.api_label.setVisible(self.width() >= 1000)
        if hasattr(self, "spectrum"):
            preferred = {"compact": 110, "default": 160, "tall": 220}[
                self.spectrum.options["height"]
            ]
            self.spectrum.setFixedHeight(min(110, preferred) if self.height() < 780 else preferred)
        super().resizeEvent(event)

    def _set_controls(self, running: bool) -> None:
        self.like_button.setEnabled(running)
        self.dislike_button.setEnabled(running)
        continuity = self.last_status.get("continuity") or {}
        fading = any(
            (self.last_status.get("audio") or {}).get(key, {}).get("fading")
            for key in ("deck_a", "deck_b")
        )
        self.skip_button.setEnabled(
            running
            and not fading
            and not any(continuity.get(k) for k in ("waiting", "stopped", "skipping"))
        )
        continuity = self.last_status.get("continuity") or {}
        self.stop_button.setEnabled(
            (running or bool(self.last_status.get("startup"))) and not continuity.get("stopped")
        )
        self.start_button.setText(
            "[ SHIFT THE VIBE ]" if self.last_status.get("session_id") else "[ START THE SET ]"
        )
        self._update_play_button()

    def _update_play_button(self) -> None:
        waiting = bool((self.last_status.get("continuity") or {}).get("waiting"))
        idle = not any(self.last_status.get(key) for key in ("running", "generating", "startup"))
        self.play_button.setEnabled(
            self._api_online
            and not self._playback_pending
            and not self._theme_pending
            and not self._writer_pending
            and (waiting or idle)
        )

    def on_playback_pending(self, pending: bool) -> None:
        self._playback_pending = pending
        self._update_play_button()

    def choose_theme(self, theme: str) -> None:
        self.theme_input.setText(theme)
        self.theme_input.setFocus()

    def on_models(self, body: dict) -> None:
        self._model_catalog = {row["id"]: row for row in body.get("models", [])}
        selected = body.get("selected", {})
        self.generator_select.blockSignals(True)
        index = self.generator_select.findData(selected.get("generator", "yue2"))
        if index >= 0:
            self.generator_select.setCurrentIndex(index)
        self.generator_select.blockSignals(False)
        self.on_generator_choice()
        index = self.generator_preset.findData(selected.get("preset"))
        if index >= 0:
            self.generator_preset.setCurrentIndex(index)
        rows = list(self._model_catalog.values())
        selected_row = self._model_catalog.get(self.generator_select.currentData(), {})
        current_available = any(
            p.get("available") and p["id"] == self.generator_preset.currentData()
            for p in selected_row.get("presets", [])
        )
        alternatives = [
            (r["id"], p["id"]) for r in rows for p in r.get("presets", []) if p.get("available")
        ]
        if not current_available and alternatives:
            generator, preset = alternatives[0]
            self.generator_select.setCurrentIndex(self.generator_select.findData(generator))
            self.generator_preset.setCurrentIndex(self.generator_preset.findData(preset))
        if not alternatives:
            if self.generate_enabled.isChecked():
                self._generation_auto_disabled = True
            self.generate_enabled.setChecked(False)
        elif self._generation_auto_disabled:
            self.generate_enabled.setChecked(True)
            self._generation_auto_disabled = False
        self._log("Music generator catalog loaded")
        usable = any(
            preset.get("available")
            for row in self._model_catalog.values()
            for preset in row.get("presets", [])
        )
        marker = self.log_path.parent.parent / "config/model-setup-dismissed"
        if not self.preview and not usable and not self._setup_prompted and not marker.exists():
            self._setup_prompted = True
            QTimer.singleShot(0, self.show_model_manager)

    def on_writers(self, body: dict) -> None:
        self._writer_catalog = body.get("writers", self._writer_catalog)
        self._selected_writer = body.get("selected", self._selected_writer)
        self.writer_select.blockSignals(True)
        self.writer_select.clear()
        for row in self._writer_catalog:
            suffix = " / INSTALL FIRST" if row.get("available") is False else ""
            self.writer_select.addItem(row["name"] + suffix, row["model"])
        self.writer_select.addItem("Template / instrumental only", "none")
        selected = (self._selected_writer or {}).get("model")
        index = self.writer_select.findData(selected)
        if index < 0 and selected:
            self.writer_select.addItem(selected, selected)
            index = self.writer_select.count() - 1
        self.writer_select.setCurrentIndex(max(0, index))
        self.writer_select.blockSignals(False)
        self.writer_select.setEnabled(self._api_online and not self._writer_pending)

    def on_writer_pending(self, pending: bool) -> None:
        self._writer_pending = pending
        self._update_play_button()
        if not pending:
            self.on_writers({})
        self.writer_select.setEnabled(self._api_online and not pending)
        self.start_button.setEnabled(self._api_online and not pending and not self._theme_pending)

    def on_theme_pending(self, pending: bool) -> None:
        self._theme_pending = pending
        self._update_play_button()
        self.start_button.setEnabled(self._api_online and not pending and not self._writer_pending)

    def on_writer_choice(self, *_args) -> None:
        if self.client is None or not self._selected_writer:
            return
        model = self.writer_select.currentData()
        row = next((r for r in self._writer_catalog if r["model"] == model), None)
        if row and row.get("available") is False:
            self.on_problem("Install this writer in the private model depot first.")
            self.on_writers({})
            return
        profile = {**self._selected_writer, "model": model, "digest": None}
        profile.update(
            mode=row["mode"] if row else "template",
            provider="ollama",
            base="http://127.0.0.1:11434" if row and row["mode"] == "remote" else "",
        )
        if model == "none":
            profile.update(provider="none", mode="template")
        self.client.configure_writer(profile)

    def on_generator_choice(self, *_args) -> None:
        identity = self.generator_select.currentData()
        variants = {
            "yue2": (("Q8", "q8"),),
            "acestep": (("Turbo", "turbo"),),
            "minimax": (
                ("Balanced Q6 / Q8", "balanced"),
                ("Quality Q8 / Q8", "quality"),
                ("Compact Q5 / Q5", "compact"),
            ),
        }
        self.generator_preset.blockSignals(True)
        self.generator_preset.clear()
        for name, preset in variants.get(identity, ()):
            self.generator_preset.addItem(name, preset)
        self.generator_preset.blockSignals(False)
        self.update_model_status()

    def update_model_status(self, *_args) -> None:
        row = self._model_catalog.get(self.generator_select.currentData())
        if row is None:
            self.model_status.setText(
                "Preview / checking installation" if self.preview else "Checking installed models"
            )
            return
        preset = next(
            (item for item in row["presets"] if item["id"] == self.generator_preset.currentData()),
            None,
        )
        ready = preset is not None and preset["available"]
        self.model_status.setText(
            ("EXPERIMENTAL / READY" if row.get("experimental") else "READY · next batch")
            if ready
            else "NEEDS SETUP"
        )
        self.model_status.setToolTip(
            (preset.get("reason") if preset else "Preset missing")
            or "Current music keeps playing when the generator changes."
        )

    def submit_theme(self) -> None:
        if self._writer_pending:
            self.on_problem("Wait for the writer selection to be saved.")
            return
        if self.preview:
            self.message.setText("Preview mode: audio backend is not running.")
        elif self.client is not None:
            controls = (
                dict(self._generation_preferences or {}) if self._generation_preferences else None
            )
            mode = self.vocal_mode.currentData()
            if controls is not None:
                controls["vocal_mode"] = (
                    "instrumental" if mode is True else mode if isinstance(mode, str) else "auto"
                )
            self.client.start_or_steer(
                self.theme_input.text(),
                instrumental=mode if isinstance(mode, bool) else None,
                generator=self.generator_select.currentData(),
                generator_preset=self.generator_preset.currentData(),
                generation_settings=controls,
                generation_enabled=self.generate_enabled.isChecked(),
            )

    def on_generation_settings(self, body: dict) -> None:
        self._generation_preferences = body.get("selected", self._generation_preferences)
        controls = self._generation_preferences or {}
        mode = {"auto": None, "instrumental": True, "singing": "singing", "rap": "rap"}
        self.vocal_mode.setCurrentIndex(
            max(0, self.vocal_mode.findData(mode.get(controls.get("vocal_mode"))))
        )
        duration = controls.get("duration_s")
        length = f"{duration:g}s" if duration else "165-178.5s"
        self.intent_label.setText(
            f"{controls.get('track_count', 10)} tracks / {length}"
            f" / {controls.get('start_policy', 'depot')}"
        )

    def on_intent(self, body: dict) -> None:
        plan = body["plan"]
        intent = plan["intent"]
        self.intent_label.setText(
            f"{intent['genre']} / {intent['vocal_mode']}\n"
            f"{body['controls']['track_count']} x {plan['duration_s']:g}s / {intent['bpm']:g} BPM"
        )
        self.intent_label.setToolTip(
            "\n".join(body.get("warnings", [])) or body["semantic_evidence"]
        )

    def open_generation_controls(self) -> None:
        if self.client is None or not self._selected_writer or self._generation_preferences is None:
            self.on_problem("Generation controls need the connected backend and writer settings.")
            return
        if self.generation_dialog is None:
            self.generation_dialog = GenerationControlsDialog(self)
        self.generation_dialog.sync(self._generation_preferences, self._selected_writer)
        self.generation_dialog.resize(min(650, self.width() - 40), min(730, self.height() - 40))
        self.generation_dialog.show()
        self.generation_dialog.raise_()

    def open_track_reports(self) -> None:
        if self.client is None:
            self.on_problem("Track reports need the connected backend.")
            return
        if self.report_dialog is None:
            self.report_dialog = GenerationReportDialog(self)
        self.report_dialog.refresh()
        self.report_dialog.resize(min(830, self.width() - 40), min(710, self.height() - 40))
        self.report_dialog.show()
        self.report_dialog.raise_()

    def start_playback(self) -> None:
        if self.client is None:
            return
        if (self.last_status.get("continuity") or {}).get("waiting"):
            self.client.start_playback()
        else:
            self.submit_theme()

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
        self.depot_dialog.resize(min(900, self.width() - 40), min(520, self.height() - 40))
        self.depot_dialog.show()
        self.depot_dialog.raise_()

    def send_feedback(self, rating: str) -> None:
        if self.client is not None:
            self.client.feedback(rating)

    def set_reduced_motion(self, enabled: bool) -> None:
        self.video_panel.set_reduced_motion(enabled)
        self.spectrum.reduced_motion = enabled
        self.spectrum._timer.setInterval(100 if enabled else 33)
        self.deck_a.reduced_motion = enabled
        self.deck_b.reduced_motion = enabled
        self.motion_button.setText("[ MOTION OFF ]" if enabled else "[ REDUCE MOTION ]")

    def on_health(self, online: bool) -> None:
        self._api_online = online
        self.console_settings.set_online(online)
        if online and not self._models_requested and self.client is not None:
            self._models_requested = True
            self.client.refresh_models()
            self.client.refresh_batches()
            self.client.refresh_writers()
            self.client.refresh_generation_settings()
        if not online:
            self._models_requested = False
        self.start_button.setEnabled(
            online and not self._writer_pending and not self._theme_pending
        )
        self._update_play_button()
        self.writer_select.setEnabled(
            online and not self._writer_pending and self._selected_writer is not None
        )
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
        self._api_online = False
        self._update_play_button()

    def on_action(self, action: str, data: dict) -> None:
        self._log(f"ACTION {action} {json.dumps(data, default=str)[:500]}")
        if action == "stop":
            self.started_at = None
            self.message.setText(
                "Playback stopped. Generation continues. Press START to play again."
            )
        elif action == "skip":
            self.message.setText("Skipping this track. Generation continues.")
        elif action == "feedback":
            self.message.setText("Feedback saved for later Laya training.")
        elif action == "resume":
            self.message.setText("Resuming saved batch. Accepted tracks are reused.")
        elif action == "playback_start":
            if self.started_at is None:
                self.started_at = time.monotonic()
            self.message.setText("Playing generated music. Your batch continues in the background.")
        elif action == "writer":
            self.message.setText("Prompt writer saved for new batches. Current playback continues.")
        elif action in {"start", "theme"}:
            if self.started_at is None:
                self.started_at = time.monotonic()
            self.message.setText(
                "Vexa is preparing your set with the selected generation settings."
                if action == "start"
                else "New direction received. The current track keeps playing."
            )
        if action in {"start", "theme"} and not self.generate_enabled.isChecked():
            self.message.setText("Depot-only mode. Vexa uses compatible ready music.")
        self.message.setStyleSheet("color: #90b2b7; background: transparent;")

    def on_batches(self, data: dict) -> None:
        batch = next((row for row in data.get("batches", []) if row["state"] != "completed"), None)
        if batch:
            self._resumable_batch_id = batch["id"]
            self.resume_button.setToolTip(
                f"Saved batch: {batch['theme']}\n{batch['completed']} / {batch['target']} ready"
            )
            self.resume_button.setEnabled(True)

    def resume_batch(self) -> None:
        if self.client is not None and self._resumable_batch_id:
            self.resume_button.setEnabled(False)
            self.client.resume_batch(self._resumable_batch_id)

    def on_telemetry(self, body):
        self._last_telemetry = time.monotonic()
        self.spectrum.set_levels([0.0] * 64 if body.get("stale") else body.get("levels", []))
        self.spectrum.setToolTip(
            f"Post-master stereo / RMS {body.get('rms_dbfs', -120):.1f} dBFS / "
            f"Peak {body.get('peak_dbfs', -120):.1f} dBFS"
        )

    def on_status(self, status: dict) -> None:
        self.last_status = status
        if self.report_dialog is not None and self.report_dialog.isVisible():
            self.report_dialog.update_actions()
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
        target = max(1, int(generation.get("target") or 10))
        self.batch_progress.setRange(0, target)
        self.batch_progress.setValue(min(completed, target))
        self.batch_progress.setToolTip(f"{completed} of {target} generated tracks are ready")
        origin = generation.get("prompt_origin") or "planning"
        rows = generation.get("tracks") or []
        if generation.get("batch_id"):
            resumable = completed < generation.get("target", 10)
            self._resumable_batch_id = generation["batch_id"] if resumable else None
        self.resume_button.setEnabled(bool(self._resumable_batch_id) and not generating)
        stage = str(generation.get("stage") or "planning").replace("_", " ")
        self.generation_label.setText(
            f"{completed} / {generation.get('target', 10)} READY  /  "
            f"{generation.get('rejected', 0)} REJECTED\n{failed} FAILED ATTEMPTS\n"
            + (
                f"{stage.upper()} {current or ''}  /  {origin}"
                if generating
                else "Batch complete"
                if completed == generation.get("target", 10)
                else "Batch incomplete / resume available"
                if completed or failed
                else "Generation cancelled / start or resume"
                if stage == "interrupted"
                else "Planning failed / retry set"
                if status.get("error")
                else "Awaiting a theme"
            )
        )
        self.generation_label.setToolTip(
            "\n".join(
                f"{row['slot']:02} {row['state']}: {row.get('title') or 'Untitled'}"
                + (f" — {row['error']}" if row.get("error") else "")
                for row in rows
            )
        )
        if self.client is not None:
            self.client.session_id = status.get("session_id")
        if time.monotonic() - self._last_telemetry > 0.6:
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
        self.theme_label.setToolTip(self.theme_label.text())
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
            else "Vexa will finish the current track and wait for a safe match."
        )
        continuity = status.get("continuity") or {}
        if continuity:
            secured = int(continuity.get("secured_s") or 0)
            self.next_detail.setText(
                self.next_detail.text()
                + f"\nBUFFER {secured // 60:02d}:{secured % 60:02d} / secured"
                + (
                    "\nSupply slower than buffer / preparing music"
                    if continuity.get("shortfall")
                    else ""
                )
            )
            self.next_detail.setToolTip(
                f"Conditional inventory: {continuity.get('conditional_inventory_s', 0):.0f}s. "
                f"{continuity.get('assumption', '')}\n"
                f"Preparation misses: {continuity.get('preparation_misses', 0)}. "
                "Repeat policy: never within this set."
            )
            if continuity.get("stopped"):
                self.live_label.setText("STOPPED")
                self.next_asset.setText("Playback stopped")
                self.next_detail.setText(
                    "Generation continues. Press START to restart the current track."
                    if status.get("current_asset_id")
                    else "Generation continues. Press START when a generated track is ready."
                )
            elif continuity.get("skip_waiting"):
                self.live_label.setText("WAITING")
                self.next_asset.setText("Waiting for next track")
                self.next_detail.setText(
                    "Current track skipped. Playback resumes when an eligible track is ready."
                )
            elif continuity.get("waiting"):
                self.live_label.setText("WAITING")
                self.next_asset.setText("Waiting for verified music")
                self.next_detail.setText(
                    "Playback is silent. Press START to play ready generated music."
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
            if event.get("event")
            in {"start", "resume", "transition", "generated", "generation_failed"}
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
                if status.get("error_kind") == "batch_warning":
                    self._log(f"BATCH WARNING {error}")
                    self.message.setText("Batch paused or incomplete: " + error)
                    self.message.setStyleSheet("color: #abc78a; background: transparent;")
                else:
                    self.on_problem(error)
        elif self._last_status_error:
            self.message.setText(
                "Preparing your music plan and tracks..."
                if generating
                else "Vexa is playing your set."
                if running
                else "Vexa is ready. Enter a vibe to prepare your set."
            )
            self.message.setStyleSheet("color: #90b2b7; background: transparent;")
        self._last_status_error = error
        startup = status.get("startup")
        if startup and generating and not error and not continuity.get("stopped"):
            self.message.setText(
                f"Waiting for {startup['policy']}: {startup['ready']} / "
                f"{startup['required']} prepared tracks."
            )
            eta = continuity.get("startup_eta_s")
            self.message.setText(
                self.message.text()
                + (
                    f" Estimated wait: {int(eta) // 60}m {int(eta) % 60}s (measured history)."
                    if eta is not None
                    else " ETA unknown: measuring this generator."
                )
            )
        self.message.setToolTip(self.message.text())
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
        self.health.setText(
            "AUDIO / STOPPED"
            if continuity.get("stopped")
            else "AUDIO / WAITING"
            if continuity.get("waiting")
            else f"AUDIO / {underruns} XRUN"
        )
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
            "asset_titles": {
                "live-92efb7f7f5ea": "Neon Circuit Reverie",
                "live-a4e26849bf03": "Afterhours Voltage Drift",
            },
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
        if self.client is not None:
            self.client.shutdown()
        for dialog in (
            self.console_settings,
            self.generation_dialog,
            self.report_dialog,
            self.music_setup,
            self.model_manager,
            self.diagnostics,
        ):
            if dialog is not None:
                dialog.close()
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
