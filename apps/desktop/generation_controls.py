"""Generation preferences and stored track reports; no model imports in the GUI."""

from __future__ import annotations

import json

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QWidget,
)
from window_chrome import window_layout


def combo(options):
    widget = QComboBox()
    for text, value in options:
        widget.addItem(text, value)
    return widget


def number(low, high, value, *, decimal=False):
    widget = QDoubleSpinBox() if decimal else QSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    if decimal:
        widget.setDecimals(2)
        widget.setSingleStep(0.05)
    return widget


def evidence_summary(evidence):
    if not evidence:
        return None
    result = {key: evidence.get(key) for key in ("evidence_version", "status", "corrections")}
    for stage in ("raw", "final"):
        data = evidence.get(stage)
        if not data:
            result[stage] = None
            continue
        result[stage] = {key: value for key, value in data.items() if key != "level_curve"}
        result[stage]["level_curve_windows"] = len(data.get("level_curve", []))
    observations = evidence.get("observations") or {}
    result["observations"] = {
        key: value
        for key, value in observations.items()
        if key not in {"audio_embedding", "windows"}
    }
    result["observations"]["windows"] = [
        {key: window.get(key) for key in ("start_s", "end_s", "voice_margin")}
        for window in observations.get("windows", [])
    ]
    result["observations"]["cached_embedding_dimensions"] = len(
        observations.get("audio_embedding", [])
    )
    return result


class GenerationControlsDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.writer = {}
        self._busy = False
        self.setWindowTitle("VEXA / GENERATION CONTROLS")
        self.setWindowIcon(owner.windowIcon())
        self.setModal(True)
        self.resize(650, min(730, owner.height() - 40))
        self.setMinimumSize(510, 470)
        layout = window_layout(self, self)
        note = QLabel("NEW BATCHES // Active and resumed batches keep their saved settings.")
        note.setWordWrap(True)
        layout.addWidget(note)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        self.fields = {}

        def page(title):
            content = QWidget()
            content.setObjectName("generationSettingsBody")
            form = QFormLayout(content)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            tabs.addTab(scroll, title)
            return form

        def field(form, key, title, widget):
            self.fields[key] = widget
            widget.setAccessibleName(title)
            form.addRow(title, widget)
            return widget

        basic = page("SET")
        field(basic, "track_count", "Tracks", number(1, 10, 10))
        self.duration_auto = QCheckBox("Use the default 165-178.5 second track variation")
        self.duration_auto.setChecked(True)
        basic.addRow(self.duration_auto)
        duration = field(basic, "duration_s", "Duration / seconds", number(150, 180, 165))
        duration.setEnabled(False)
        self.duration_auto.toggled.connect(lambda checked: duration.setEnabled(not checked))
        field(basic, "genre", "Genre / subgenre override", QLineEdit())
        field(
            basic,
            "vocal_mode",
            "Vocals",
            combo(
                [
                    ("Auto / follow theme", "auto"),
                    ("Instrumental / no voice", "instrumental"),
                    ("Sung vocals", "singing"),
                    ("Spoken rap", "rap"),
                ]
            ),
        )
        field(
            basic,
            "language",
            "Vocal language",
            combo(
                [
                    ("Auto / follow theme", "auto"),
                    ("English", "en"),
                    ("Turkish", "tr"),
                    ("Mandarin Chinese", "zh"),
                    ("Japanese", "ja"),
                    ("Korean", "ko"),
                    ("Spanish", "es"),
                    ("French", "fr"),
                    ("German", "de"),
                ]
            ),
        )
        field(
            basic,
            "start_policy",
            "Start playback",
            combo(
                [
                    ("Compatible depot / then generated tracks", "depot"),
                    ("Wait for generated buffer", "buffer"),
                    ("Wait for complete generated batch", "complete"),
                ]
            ),
        )
        field(basic, "buffer_tracks", "Prepared buffer / tracks", number(1, 10, 2))
        speed = field(
            basic,
            "render_speed",
            "MiniMax render speed",
            combo(
                [
                    ("Fast / 12 steps (experimental)", "fast"),
                    ("Balanced / 20 steps", "balanced"),
                    ("Quality / 30 steps", "quality"),
                ]
            ),
        )
        speed.setToolTip(
            "Fewer denoising steps reduce MiniMax rendering time and may affect sound quality. "
            "Full track length and audio/cue gates remain active. "
            "Other generators ignore this setting."
        )
        speed.setCurrentIndex(speed.findData("balanced"))
        field(basic, "seed", "Base seed / reproducible requests", number(0, 2147483000, 42))

        music = page("MUSIC")
        self.bpm_auto = QCheckBox("Resolve tempo from the theme")
        self.bpm_auto.setChecked(True)
        music.addRow(self.bpm_auto)
        bpm = field(music, "bpm", "Tempo / BPM", number(40, 220, 128))
        bpm.setEnabled(False)
        self.bpm_auto.toggled.connect(lambda checked: bpm.setEnabled(not checked))
        field(
            music,
            "tempo_policy",
            "Measured tempo policy",
            combo(
                [
                    ("Hint / existing mix rules", "hint"),
                    ("Strict / target ±3 BPM", "strict"),
                ]
            ),
        )
        field(
            music,
            "energy_arc",
            "Energy progression",
            combo(
                [
                    ("DJ rise and release", "dj"),
                    ("Steady", "steady"),
                    ("Build", "build"),
                    ("Wind down", "wind_down"),
                ]
            ),
        )
        field(music, "energy", "Steady energy target / 0-1", number(0, 1, 0.55, decimal=True))
        for key, title in [
            ("texture", "Era / production texture hint"),
            ("groove", "Groove / density hint"),
            ("include_instruments", "Include instrument hints / comma separated"),
            ("exclude_instruments", "Exclude instrument hints / comma separated"),
        ]:
            field(music, key, title, QLineEdit())
        hint = QLabel(
            "Genre, instruments, energy and texture are conditioning targets. "
            "Their audible presence is not verified yet. Duration, loudness and cue "
            "gates remain mandatory. Strict tempo accepts half/double-time equivalents."
        )
        hint.setWordWrap(True)
        music.addRow(hint)

        lyrics = page("LYRICS")
        field(lyrics, "lyric_topic", "Topic override", QLineEdit())
        field(lyrics, "lyric_direction", "Narrator / delivery direction", QLineEdit())
        field(lyrics, "custom_lyrics", "Custom original lyrics / optional", QPlainTextEdit())
        instructions = QLabel(
            "Use [Verse 1], [Refrain] (or [Chorus]) and [Verse 2]. "
            "A single refrain is reused. Intro, outro and instrumental break "
            "stay empty. Rap needs two distinct 16-line verses and a 4-line "
            "refrain. Custom lyrics skip the lyric-writing request."
        )
        instructions.setWordWrap(True)
        lyrics.addRow(instructions)

        writer = page("WRITER / BUDGETS")
        self.writer_name = QLabel("Loading selected writer...")
        writer.addRow(self.writer_name)
        self.writer_fields = {}
        for key, title, low, high, value, decimal in [
            ("context_length", "Context tokens", 4096, 32768, 8192, False),
            ("max_tokens", "Output token budget", 512, 6000, 4000, False),
            ("creativity", "Music writing creativity", 0, 1, 0.35, True),
            ("lyric_creativity", "Lyric writing creativity", 0, 1, 0.6, True),
        ]:
            widget = number(low, high, value, decimal=decimal)
            self.writer_fields[key] = widget
            writer.addRow(title, widget)
        for key, title, low, high, value in [
            ("plan_attempts", "Planning attempts / slot", 1, 5, 3),
            ("render_attempts", "Rendering attempts / slot", 1, 5, 3),
            ("planning_timeout_s", "Planning deadline / seconds", 60, 1800, 600),
            ("rendering_timeout_s", "Rendering deadline / seconds", 600, 14400, 6000),
        ]:
            field(writer, key, title, number(low, high, value))
        unsupported = QLabel(
            "Stem analysis, voice-sample-only mode and arbitrary arrangement "
            "editing are not available. Seed records make requests reproducible; "
            "identical audio across model/runtime changes is not guaranteed."
        )
        unsupported.setWordWrap(True)
        writer.addRow(unsupported)

        self.resolved = QPlainTextEdit()
        self.resolved.setReadOnly(True)
        self.resolved.setMaximumHeight(145)
        self.resolved.setPlaceholderText(
            "Preview your current theme to see resolved intent and overrides."
        )
        layout.addWidget(self.resolved)
        actions = QHBoxLayout()
        self.preview_button = QPushButton("[ PREVIEW INTENT ]")
        self.save_button = QPushButton("[ SAVE FOR NEW BATCHES ]")
        self.preview_button.clicked.connect(self.preview)
        self.save_button.clicked.connect(self.save)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)

    def sync(self, controls, writer):
        self.writer = dict(writer or {})
        self.writer_name.setText("SELECTED WRITER // " + self.writer.get("model", "Unavailable"))
        for key, widget in self.fields.items():
            value = controls.get(key)
            if value is None:
                continue
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(value)))
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(", ".join(value) if isinstance(value, list) else value)
            else:
                widget.setValue(value)
        self.duration_auto.setChecked(controls.get("duration_s") is None)
        self.bpm_auto.setChecked(controls.get("bpm") is None)
        for key, widget in self.writer_fields.items():
            widget.setEnabled(self.writer.get("mode") != "template")
            if key in self.writer:
                widget.setValue(self.writer[key])

    def values(self):
        result = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QComboBox):
                result[key] = widget.currentData()
            elif isinstance(widget, QPlainTextEdit):
                result[key] = widget.toPlainText().strip()
            elif isinstance(widget, QLineEdit):
                result[key] = widget.text().strip()
                if key.endswith("instruments"):
                    result[key] = [v.strip() for v in result[key].split(",") if v.strip()]
            else:
                result[key] = widget.value()
        result["duration_s"] = None if self.duration_auto.isChecked() else result["duration_s"]
        result["bpm"] = None if self.bpm_auto.isChecked() else result["bpm"]
        return result

    def preview(self):
        payload = {
            "theme": self.owner.theme_input.text(),
            "generation_settings": self.values(),
            "generator": self.owner.generator_select.currentData(),
        }

        def received(body, error):
            if error:
                self.resolved.setPlainText("CHECK SETTINGS // " + error)
                return
            plan = body["plan"]
            intent = plan["intent"]
            self.resolved.setPlainText(
                f"{intent['genre']} // {intent['vocal_mode']} / {intent['language']}\n"
                f"{self.values()['track_count']} tracks / {plan['duration_s']:g}s / "
                f"{intent['bpm']:g} BPM / energy {plan['energy']:.2f}\n"
                f"Overrides: {', '.join(body['overrides']) or 'none'}\n"
                + "\n".join(body.get("warnings", []))
                + "\n"
                + body["semantic_evidence"]
            )

        self.owner.client._request("POST", "/generation/preview", payload, received)

    def save(self):
        if self._busy or not self.writer:
            return
        self._busy = True
        self.save_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.owner.on_writer_pending(True)
        controls = self.values()
        writer = {**self.writer, **{key: w.value() for key, w in self.writer_fields.items()}}
        writer["seed"] = controls["seed"]

        def finish(body, error):
            self._busy = False
            self.save_button.setEnabled(True)
            self.preview_button.setEnabled(True)
            self.owner.on_writer_pending(False)
            if error:
                self.resolved.setPlainText("SAVE FAILED // " + error)
                self.owner.client.refresh_writers()
                return
            self.owner.on_generation_settings(body)
            self.resolved.setPlainText(
                "Saved for new batches. Active/resumed batches and playback continue."
            )

        def writer_saved(body, error):
            if error:
                finish(None, error)
                return
            self.owner.on_writers(body)
            self.writer = body["selected"]
            self.owner.client._request("POST", "/generation/settings", controls, finish)

        def validated(body, error):
            if error:
                finish(None, error)
                return
            self.owner.client._request("POST", "/generation/writer", writer, writer_saved)

        self.owner.client._request("POST", "/generation/settings/validate", controls, validated)


class GenerationReportDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.batch = None
        self._detail_key = None
        self.setWindowTitle("VEXA / TRACK REPORTS")
        self.setWindowIcon(owner.windowIcon())
        self.resize(min(830, owner.width() - 40), min(710, owner.height() - 40))
        self.setMinimumSize(520, 440)
        layout = window_layout(self, self)
        self.batches = QComboBox()
        self.batches.currentIndexChanged.connect(self.load_batch)
        layout.addWidget(self.batches)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "TITLE", "STAGE", "PLAN", "RENDER"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 250)
        self.table.itemSelectionChanged.connect(self.load_report)
        layout.addWidget(self.table, 1)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        layout.addWidget(self.detail, 1)
        actions = QGridLayout()
        self.refresh_button = QPushButton("[ REFRESH ]")
        self.retry_button = QPushButton("[ RETRY SLOT ]")
        self.replace_button = QPushButton("[ REPLACE SLOT ]")
        self.cancel_button = QPushButton("[ CANCEL GENERATION ]")
        for index, widget in enumerate(
            (
                self.refresh_button,
                self.retry_button,
                self.replace_button,
                self.cancel_button,
            )
        ):
            actions.addWidget(widget, index // 2, index % 2)
        self.refresh_button.clicked.connect(self.refresh)
        self.retry_button.clicked.connect(lambda: self.action("retry"))
        self.replace_button.clicked.connect(lambda: self.action("replace"))
        self.cancel_button.clicked.connect(lambda: self.action("cancel"))
        layout.addLayout(actions)
        note = QLabel(
            "Replace creates new audio; the original depot file remains available. "
            "Cancel generation keeps DJ playback running. Retry/replace require an idle queue."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.timer = QTimer(self)
        self.timer.setInterval(2500)
        self.timer.timeout.connect(
            lambda: (
                self.refresh()
                if self.isVisible() and self.owner.last_status.get("generating")
                else None
            )
        )
        self.timer.start()

    def refresh(self):
        selected = self.batches.currentData()

        def received(body, error):
            if error:
                self.detail.setPlainText(error)
                return
            self.batches.blockSignals(True)
            self.batches.clear()
            for batch in body.get("batches", []):
                self.batches.addItem(
                    f"{batch['state']} / {batch['completed']}/{batch['target']} / "
                    f"{batch['theme'][:60]}",
                    batch["id"],
                )
            index = self.batches.findData(selected)
            self.batches.setCurrentIndex(max(0, index))
            self.batches.blockSignals(False)
            self.load_batch()

        self.owner.client._request("GET", "/generation/batches", None, received)

    def load_batch(self, *_args):
        batch_id = self.batches.currentData()
        selected = self.selected()
        selected_slot = selected["slot"] if selected else 1
        self._detail_key = None
        self.batch = None
        self.table.setRowCount(0)
        if not batch_id:
            self.detail.setPlainText("No saved batches yet.")
            self.update_actions()
            return

        def received(body, error):
            if batch_id != self.batches.currentData():
                return
            if error:
                self.detail.setPlainText(error)
                return
            self.batch = body
            self.table.setRowCount(len(body["tracks"]))
            for row, track in enumerate(body["tracks"]):
                for col, value in enumerate(
                    (
                        track["slot"],
                        track.get("title") or "Untitled",
                        track["state"],
                        track["plan_attempts"],
                        track["render_attempts"],
                    )
                ):
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))
            self.table.selectRow(min(selected_slot - 1, len(body["tracks"]) - 1))
            self.update_actions()

        self.owner.client._request("GET", f"/generation/batches/{batch_id}", None, received)

    def selected(self):
        row = self.table.currentRow()
        return (
            self.batch["tracks"][row]
            if self.batch and 0 <= row < len(self.batch["tracks"])
            else None
        )

    def update_actions(self):
        running = bool((self.owner.last_status or {}).get("generating"))
        track = self.selected()
        self.retry_button.setEnabled(
            bool(track and track["state"] in {"rejected", "unavailable"}) and not running
        )
        self.replace_button.setEnabled(bool(track) and not running)
        self.cancel_button.setEnabled(running)

    def load_report(self):
        track = self.selected()
        self.update_actions()
        if not track:
            return
        key = (self.batch["id"], track["slot"])
        self._detail_key = key

        def received(body, error):
            if key != self._detail_key:
                return
            if error:
                self.detail.setPlainText(error)
                return
            lines = [
                f"{body.get('title') or 'Untitled'} // {body['state']}",
                "REQUESTED → MEASURED (unknown means not analyzed)",
            ]
            for name, requested in body["requested"].items():
                value = body["measured"].get(name)
                lines.append(f"{name}: {requested} → {value if value is not None else 'unknown'}")
            lines += [
                "",
                "PLAYBACK / MIXING CAPABILITIES",
                json.dumps(body.get("admission"), indent=2),
                "MEASURED AUDIO",
                json.dumps(
                    {key: value for key, value in body["measured"].items() if key != "level_curve"},
                    indent=2,
                ),
                "QUALITY GATES",
                json.dumps(body["quality"], indent=2),
                "RAW / FINAL AUDIO AND OBSERVATIONS",
                json.dumps(evidence_summary(body.get("audio_evidence")), indent=2),
                "PREVIOUS RENDER EVIDENCE",
                json.dumps(body.get("evidence_history", []), indent=2),
                "ATTEMPTS / CORRECTIONS",
                json.dumps(body["attempts"], indent=2),
                "ERROR: " + (body.get("error") or "none"),
                body["semantic_evidence"],
                "",
                "PLANNING / WRITER DETAILS",
                json.dumps(body["brief"], indent=2, ensure_ascii=False),
            ]
            self.detail.setPlainText("\n".join(lines))

        self.owner.client._request(
            "GET", f"/generation/batches/{key[0]}/tracks/{key[1]}", None, received
        )

    def action(self, action):
        track = self.selected()
        if action != "cancel" and not track:
            return
        path = (
            "/generation/cancel"
            if action == "cancel"
            else f"/generation/batches/{self.batch['id']}/tracks/{track['slot']}/{action}"
        )

        def received(body, error):
            if error:
                self.detail.setPlainText(error)
                return
            self.owner.client.poll()
            self.refresh()

        self.owner.client._request("POST", path, {}, received)
