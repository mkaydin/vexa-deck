"""Exercise v0.2 spectrum appearances, model setup and skip routes without music or downloads."""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
import main  # noqa: E402
from pixel_theme import apply_theme, font, install_font  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402


def check():
    app = QApplication([])
    install_font()
    app.setFont(font())
    apply_theme(app, "terminal")
    window = main.VexaWindow(preview=True)
    client = main.BackendClient("http://127.0.0.1:9")
    client._poll.stop()
    client.poll = lambda: None
    window.client = client
    client.status_received.connect(window.on_status)
    client.problem.connect(window.on_problem)
    client.telemetry_received.connect(window.on_telemetry)
    output = ROOT / "var/reports/v02-gui"
    output.mkdir(parents=True, exist_ok=True)
    model = dict(
        generator="minimax",
        preset="balanced",
        name="MiniMax Music 3",
        installed=False,
        location="/fixture",
        requirements="git, cmake, CUDA toolkit",
        model_revision="d4f7a6402a4ca10c8b3ca2f648c8a46c103558bc",
        license_url="https://huggingface.co/fixture/model",
    )
    job = dict(
        id="fixture-job",
        spec=model,
        state="downloading",
        total_bytes=10743976525,
        completed_bytes=4297590610,
        speed_bps=100 * 1024**2,
        eta_s=61,
        message="Downloading fixture weights",
        file="MiniMax-Music3-language_model-Q6_K.gguf",
    )
    calls = []
    status = copy.deepcopy(window.last_status)

    def request(method, path, body, callback):
        calls.append((method, path, body))
        if path == "/models/catalog":
            callback({"models": [model]}, None)
        elif path == "/models/installations" and method == "GET":
            callback({"jobs": [job]}, None)
        elif path.endswith("/preflight"):
            callback(
                dict(
                    total_bytes=1000,
                    incremental_bytes=1000,
                    free_bytes=10000,
                    required_free_bytes=2000,
                    missing_tools=[],
                ),
                None,
            )
        elif method == "POST":
            callback(status if path == "/live/skip" else job, None)

    client._request = request
    report = {"audio_played": False, "models_downloaded": False, "styles": []}
    for width, height in [(1440, 900), (720, 700)]:
        job["state"] = "downloading"
        window.resize(width, height)
        window.show()
        app.processEvents()
        assert window.skip_button.isVisible() and window.skip_button.isEnabled()
        assert (
            window.skip_button.mapTo(window, window.skip_button.rect().topLeft()).x()
            + window.skip_button.width()
            <= width
        )
        for style in ("ascii", "led", "bars", "mirror"):
            fields = window.console_settings.spectrum_fields
            fields["style"].setCurrentIndex(fields["style"].findData(style))
            fields["palette"].setCurrentIndex(fields["palette"].findData("neon"))
            fields["bands"].setCurrentIndex(fields["bands"].findData(64))
            window.on_telemetry(
                dict(
                    levels=[(i % 12) / 12 for i in range(64)],
                    rms_dbfs=-18,
                    peak_dbfs=-3,
                    stale=False,
                )
            )
            window.spectrum.values = window.spectrum.target.copy()
            window.spectrum.peak_values = window.spectrum.target.copy()
            app.processEvents()
            assert window.spectrum.options["style"] == style
            window.grab().save(str(output / f"{style}-{width}.png"))
        window.show_model_manager()
        app.processEvents()
        manager = window.model_manager
        assert manager.progress.value() == 40 and manager.cancel_button.isEnabled()
        assert not manager.resume_button.isEnabled()
        manager.grab().save(str(output / f"model-manager-{width}.png"))
        assert manager.width() <= width
        for control in (
            manager.selector,
            manager.install_button,
            manager.cancel_button,
            manager.resume_button,
        ):
            assert control.isVisible()
            point = control.mapTo(manager, control.rect().topLeft())
            assert point.x() + control.width() <= manager.width()
        manager.cancel_button.click()
        assert any(call[1].endswith("/cancel") for call in calls)
        job["state"] = "interrupted"
        manager.refresh()
        app.processEvents()
        assert manager.resume_button.isEnabled()
        manager.resume_button.click()
        assert any(call[1].endswith("/resume") for call in calls)
        old_question = QMessageBox.question
        try:
            QMessageBox.question = lambda *a: QMessageBox.StandardButton.Yes
            manager.install_button.click()
        finally:
            QMessageBox.question = old_question
        assert any(
            call[0] == "POST" and call[1] == "/models/installations" and call[2]["accept_terms"]
            for call in calls
        )
        manager.hide()
        window.skip_button.click()
        assert calls[-1][1] == "/live/skip"
        report["styles"].append(dict(size=[width, height], styles=4, model_manager_fits=True))
    stopped = copy.deepcopy(status)
    stopped["continuity"] = dict(waiting=True, stopped=True)
    window.on_status(stopped)
    assert not window.skip_button.isEnabled()
    waiting = copy.deepcopy(status)
    waiting["continuity"] = dict(waiting=True, skip_waiting=True)
    window.on_status(waiting)
    assert not window.skip_button.isEnabled()
    assert "Waiting for next track" in window.next_asset.text()
    # Use a temporary dismissal marker for first-launch checks, never the user's configuration.
    with tempfile.TemporaryDirectory() as directory:
        window.log_path = Path(directory) / "var/logs/gui.log"
        window.log_path.parent.mkdir(parents=True)
        window.preview = False
        window._setup_prompted = False
        window.on_models({"models": [], "selected": {}})
        app.processEvents()
        assert manager.isVisible() and not window.generate_enabled.isChecked()
        manager.close()
        assert (Path(directory) / "var/config/model-setup-dismissed").exists()
        window._setup_prompted = False
        window.on_models({"models": [], "selected": {}})
        app.processEvents()
        assert not manager.isVisible()
        window.on_models(
            {
                "models": [dict(id="minimax", presets=[dict(id="balanced", available=True)])],
                "selected": dict(generator="minimax", preset="balanced"),
            }
        )
        assert window.generate_enabled.isChecked()
        window.close()
    report.update(
        skip_routing=True,
        skip_guards=True,
        installation_routing=True,
        first_launch_dismissal=True,
        first_ready_model_enables_generation=True,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    check()
