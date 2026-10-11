"""Verify model/preset controls and submitted API payloads without generation or sound."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
from backend_runtime import workspace_pythonpath  # noqa: E402

for path in workspace_pythonpath(ROOT).split(os.pathsep):
    sys.path.insert(0, path)


def main():
    from main import BackendClient, VexaWindow
    from pixel_theme import font, install_font
    from PySide6.QtWidgets import QApplication
    from vexa_orchestrator.generators import catalog

    app = QApplication([])
    install_font()
    app.setFont(font())
    window = VexaWindow(preview=True)
    window.on_models({"models": catalog(), "selected": {"generator": "yue2", "preset": "q8"}})
    client = BackendClient("http://127.0.0.1:9")
    client._poll.stop()
    sent = []

    def request(method, path, payload, callback):
        sent.append((method, path, payload))
        if path == "/sessions" or path.endswith("/theme"):
            callback({"session_id": "gui-test"}, None)

    client._request = request
    client.poll = lambda: None
    window.client = client
    window.preview = False
    window.show()
    report = {"audio_played": False, "tracks_generated": False, "models": [], "layouts": []}
    for generator, preset in [
        ("yue2", "q8"),
        ("acestep", "turbo"),
        ("minimax", "balanced"),
        ("minimax", "quality"),
        ("minimax", "compact"),
    ]:
        window.generator_select.setCurrentIndex(window.generator_select.findData(generator))
        window.generator_preset.setCurrentIndex(window.generator_preset.findData(preset))
        window.theme_input.setText("instrumental techno")
        window.submit_theme()
        assert sent[-1][2]["generator"] == generator
        assert sent[-1][2]["generator_preset"] == preset
        report["models"].append({"generator": generator, "preset": preset, "payload": "pass"})
    window.on_batches(
        {
            "batches": [
                {
                    "id": "saved-batch",
                    "state": "interrupted",
                    "theme": "dark techno",
                    "completed": 3,
                    "target": 10,
                }
            ]
        }
    )
    assert window.resume_button.isEnabled()
    window.resume_button.click()
    assert sent[-1] == ("POST", "/generation/batches/saved-batch/resume", {"retry_failed": True})
    window.on_status(
        {
            "generating": True,
            "generation": {
                "batch_id": "saved-batch",
                "target": 10,
                "completed": 3,
                "failed": 2,
                "rejected": 1,
                "current": 5,
                "stage": "planning",
                "tracks": [{"slot": 5, "state": "planning", "title": "Night", "error": None}],
            },
        }
    )
    assert not window.resume_button.isEnabled()
    assert "PLANNING" in window.generation_label.text()
    assert "FAILED ATTEMPTS" in window.generation_label.text()
    report["recovery_controls"] = "pass"
    window.on_writers(
        {
            "writers": [
                {"model": "qwen3.5:9b", "name": "Qwen", "mode": "local", "available": True},
                {"model": "gemma4:e4b", "name": "Gemma", "mode": "local", "available": True},
                {"model": "nemotron-3-super:cloud", "name": "Nemotron", "mode": "remote"},
            ],
            "selected": {"mode": "local", "model": "gemma4:e4b", "base": "", "provider": "ollama"},
        }
    )
    window.writer_select.setCurrentIndex(window.writer_select.findData("qwen3.5:9b"))
    assert sent[-1][0:2] == ("POST", "/generation/writer")
    assert sent[-1][2]["model"] == "qwen3.5:9b" and sent[-1][2]["base"] == ""
    window.writer_select.setCurrentIndex(window.writer_select.findData("nemotron-3-super:cloud"))
    assert sent[-1][2]["mode"] == "remote" and sent[-1][2]["provider"] == "ollama"
    window.writer_select.setCurrentIndex(window.writer_select.findData("none"))
    assert sent[-1][2]["mode"] == "template"
    report["writer_controls"] = "pass"
    window.on_writer_pending(True)
    assert not window.start_button.isEnabled() and not window.writer_select.isEnabled()
    window.on_writer_pending(False)
    assert window.start_button.isEnabled() and window.writer_select.isEnabled()
    report_dir = ROOT / "var/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for width, height in [(1440, 900), (720, 700), (960, 1040)]:
        window.resize(width, height)
        app.processEvents()
        assert window.width() == width and window.height() == height
        assert window.start_button.isVisible() and window.resume_button.isVisible()
        window.show_music_setup()
        app.processEvents()
        for control in (
            window.generator_select,
            window.generator_preset,
            window.motion_button,
            window.writer_select,
        ):
            assert control.isVisible() and control.height() >= 25
        window.music_setup.grab().save(str(report_dir / f"music-setup-{width}x{height}.png"))
        window.music_setup.hide()
        window.grab().save(str(report_dir / f"generator-gui-{width}x{height}.png"))
        report["layouts"].append(f"{width}x{height}")
    (report_dir / "generator-gui-qa.json").write_text(json.dumps(report, indent=2) + "\n")
    window.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    main()
