"""First-run discovery and persistent model installation controls."""

from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
)
from window_chrome import window_layout


class ModelManager(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.models = []
        self.jobs = []
        self._pending = False
        self._inspecting = False
        self._ready_ids = set()
        self.setWindowTitle("VEXA / MODEL MANAGER")
        self.setWindowIcon(owner.windowIcon())
        self.resize(630, 570)
        layout = window_layout(self, self)
        intro = QLabel(
            "SET UP MUSIC GENERATION\nDownload a supported generator or continue "
            "with your videos and existing music depot.\n"
            "Choose a prompt writer in Music setup; Template supports instrumental music "
            "without a downloaded writer."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.selector = QComboBox()
        self.selector.setAccessibleName("Model to install")
        self.selector.currentIndexChanged.connect(self.show_model)
        layout.addWidget(self.selector)
        self.requirements = QLabel("Checking installed models…")
        self.requirements.setWordWrap(True)
        layout.addWidget(self.requirements)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QLabel("No installation active")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.install_button = QPushButton("[ INSTALL ]")
        self.install_button.clicked.connect(self.install)
        self.terms_button = QPushButton("[ MODEL TERMS ]")
        self.terms_button.clicked.connect(self.terms)
        self.cancel_button = QPushButton("[ CANCEL DOWNLOAD ]")
        self.cancel_button.clicked.connect(self.cancel)
        self.resume_button = QPushButton("[ RESUME / RETRY ]")
        self.resume_button.clicked.connect(self.resume)
        for widget in (
            self.install_button,
            self.terms_button,
            self.cancel_button,
            self.resume_button,
        ):
            row.addWidget(widget)
        layout.addLayout(row)
        self.job_select = QComboBox()
        self.job_select.setAccessibleName("Installation job")
        self.job_select.currentIndexChanged.connect(self.show_job)
        layout.addWidget(self.job_select)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 1)
        done = QPushButton("[ CONTINUE / CLOSE ]")
        done.clicked.connect(self.close)
        layout.addWidget(done)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)

    def showEvent(self, event):
        self.refresh()
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        if not self.owner.preview:
            marker = Path(self.owner.log_path).parent.parent / "config/model-setup-dismissed"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        super().closeEvent(event)

    def refresh(self):
        if self.owner.client is None or self._pending:
            return
        self._pending = True

        def received(body, error):
            self._pending = False
            if error:
                self.status.setText("Model manager unavailable: " + error)
                return
            selected = self.selector.currentData()
            self.models = body.get("models", [])
            self.selector.blockSignals(True)
            self.selector.clear()
            for model in self.models:
                identity = model["generator"] + ":" + model["preset"]
                self.selector.addItem(
                    model["name"]
                    + " / "
                    + model["preset"]
                    + (" / INSTALLED" if model["installed"] else ""),
                    identity,
                )
            self.selector.setCurrentIndex(max(0, self.selector.findData(selected)))
            self.selector.blockSignals(False)
            self.show_model()
            self.owner.client._request("GET", "/models/installations", None, self.received_jobs)

        self.owner.client._request("GET", "/models/catalog", None, received)

    def received_jobs(self, body, error):
        if error:
            self.status.setText("Cannot read installation jobs: " + error)
            return
        selected = self.job_select.currentData()
        self.jobs = body.get("jobs", [])
        self.job_select.blockSignals(True)
        self.job_select.clear()
        for job in reversed(self.jobs):
            self.job_select.addItem(
                job["spec"]["generator"] + " / " + job["spec"]["preset"] + " / " + job["state"],
                job["id"],
            )
        self.job_select.setCurrentIndex(max(0, self.job_select.findData(selected)))
        self.job_select.blockSignals(False)
        newly_ready = {j["id"] for j in self.jobs if j["state"] == "ready"} - self._ready_ids
        if newly_ready:
            self.owner.client.refresh_models()
        self._ready_ids |= newly_ready
        self.show_job()
        job = self.job()
        if job:

            def detail_received(body, error):
                if not error and body and self.job_select.currentData() == body.get("id"):
                    for item in self.jobs:
                        if item["id"] == body["id"]:
                            item.update(body)
                    self.show_job()

            self.owner.client._request(
                "GET", f"/models/installations/{job['id']}", None, detail_received
            )

    def model(self):
        index = self.selector.currentIndex()
        return self.models[index] if 0 <= index < len(self.models) else None

    def job(self):
        return next((j for j in self.jobs if j["id"] == self.job_select.currentData()), None)

    def show_model(self):
        model = self.model()
        if model:
            self.requirements.setText(
                ("Installed / " + model["location"] if model["installed"] else "Download required")
                + "\nRequirements: "
                + model["requirements"]
                + "\nPinned revision: "
                + model["model_revision"][:12]
                + "\nPreflight checks file sizes and disk space. "
                "Runtime setup waits while music is active."
            )
        active = any(
            j["state"] not in ("ready", "failed", "cancelled", "interrupted") for j in self.jobs
        )
        self.install_button.setEnabled(bool(model and not active and not self._inspecting))
        self.install_button.setText(
            "[ VERIFY / REPAIR ]" if model and model["installed"] else "[ INSTALL ]"
        )
        self.terms_button.setEnabled(model is not None)

    def show_job(self):
        job = self.job()
        active = job and job["state"] not in ("ready", "failed", "cancelled", "interrupted")
        self.cancel_button.setEnabled(bool(active))
        self.resume_button.setEnabled(
            bool(job and job["state"] in ("failed", "cancelled", "interrupted"))
        )
        if job:
            total, done = job.get("total_bytes"), job.get("completed_bytes", 0)
            self.progress.setRange(0, 100 if total else 0)
            self.progress.setValue(round(done / total * 100) if total else 0)
            detail = (
                f"{done / 1024**3:.2f} / {total / 1024**3:.2f} GiB"
                if total
                else "Size pending preflight"
            )
            speed, eta = job.get("speed_bps"), job.get("eta_s")
            if job["state"] == "downloading" and speed:
                detail += f" / {speed / 1024**2:.1f} MiB/s"
            if job["state"] == "downloading" and eta is not None:
                detail += f" / ~{int(eta) // 60}m remaining"
            self.status.setText(
                job["state"].replace("_", " ").upper()
                + " / "
                + detail
                + "\n"
                + job.get("message", "")
            )
            self.details.setPlainText(
                "Current file: "
                + str(job.get("file") or "—")
                + "\n"
                + job.get("message", "")
                + "\n\nInstallation ID: "
                + job["id"]
                + "\nDetailed log: var/model-installs/"
                + job["id"]
                + ".log"
                + "\n\n"
                + job.get("log_tail", "")
            )
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        self.show_model()

    def terms(self):
        model = self.model()
        if model:
            QDesktopServices.openUrl(QUrl(model["license_url"]))

    def install(self):
        model = self.model()
        if not model:
            return
        self._inspecting = True
        self.install_button.setEnabled(False)
        self.status.setText("Checking download size, disk space and prerequisites…")

        def inspected(body, error):
            self._inspecting = False
            if error:
                self.status.setText(error)
                self.show_model()
                return
            if body["missing_tools"]:
                self.status.setText(
                    "Install prerequisites first: " + ", ".join(body["missing_tools"])
                )
                self.show_model()
                return
            if body["free_bytes"] < body["required_free_bytes"]:
                self.status.setText("Not enough free disk space for this installation")
                self.show_model()
                return
            summary = (
                f"{model['name']} / {model['preset']}\n"
                f"Model files: {body['total_bytes'] / 1024**3:.2f} GiB\n"
                f"New/missing files: {body['incremental_bytes'] / 1024**3:.2f} GiB\n"
                f"Free disk: {body['free_bytes'] / 1024**3:.1f} GiB / "
                f"minimum reserve: {body['required_free_bytes'] / 1024**3:.1f} GiB\n\n"
                "Review MODEL TERMS first. Install and use this verified managed copy "
                "for new batches under those terms?"
            )
            answer = QMessageBox.question(self, "Install music model", summary)
            self.show_model()
            if answer == QMessageBox.StandardButton.Yes:
                self.owner.client._request(
                    "POST",
                    "/models/installations",
                    dict(generator=model["generator"], preset=model["preset"], accept_terms=True),
                    self.action,
                )

        self.owner.client._request(
            "GET",
            f"/models/catalog/{model['generator']}/{model['preset']}/preflight",
            None,
            inspected,
        )

    def action(self, body, error):
        if error:
            self.status.setText(error)
        self.refresh()

    def cancel(self):
        job = self.job()
        if job:
            self.owner.client._request(
                "POST", f"/models/installations/{job['id']}/cancel", {}, self.action
            )

    def resume(self):
        job = self.job()
        if job:
            self.owner.client._request(
                "POST", f"/models/installations/{job['id']}/resume", {}, self.action
            )
