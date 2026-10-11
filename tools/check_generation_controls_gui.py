"""Offscreen controls, preview submission and track-report QA with no generation or audio."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))


def main():
    from main import BackendClient, VexaWindow
    from pixel_theme import apply_theme, font, install_font
    from PySide6.QtWidgets import QApplication, QTabWidget

    app = QApplication([])
    install_font()
    app.setFont(font())
    apply_theme(app, "terminal")
    window = VexaWindow(preview=True)
    client = BackendClient("http://127.0.0.1:9")
    client._poll.stop()
    client.poll = lambda: None
    window.client = client
    window.preview = False
    writer = {
        "mode": "local",
        "provider": "ollama",
        "model": "gemma4:e4b",
        "base": "",
        "digest": None,
        "context_length": 8192,
        "max_tokens": 4000,
        "creativity": 0.35,
        "lyric_creativity": 0.6,
        "seed": 42,
    }
    window.on_writers({"selected": writer})
    window.on_generation_settings({"selected": {"track_count": 10, "vocal_mode": "auto"}})
    window.theme_input.setText("dark melodic techno instrumental")
    window.open_generation_controls()
    dialog = window.generation_dialog
    sent = []
    preview = {
        "plan": {
            "duration_s": 180,
            "energy": 0.4,
            "intent": {
                "genre": "techno",
                "vocal_mode": "instrumental",
                "language": "unknown",
                "bpm": 128,
            },
        },
        "controls": {"track_count": 3},
        "overrides": ["duration_s", "track_count"],
        "warnings": [],
        "semantic_evidence": "targets and hints",
    }
    batch = {
        "id": "qa-batch",
        "state": "partial",
        "theme": "techno",
        "completed": 1,
        "target": 2,
        "tracks": [
            {
                "slot": 1,
                "state": "ready",
                "title": "Night Circuit",
                "plan_attempts": 1,
                "render_attempts": 1,
            },
            {
                "slot": 2,
                "state": "rejected",
                "title": "Late Signal",
                "plan_attempts": 3,
                "render_attempts": 0,
            },
        ],
    }

    def request(method, path, payload, callback):
        sent.append((method, path, payload))
        if path.endswith("/validate") or path in {"/generation/settings", "/generation/writer"}:
            callback({"selected": payload}, None)
        elif path == "/generation/preview":
            callback(preview, None)
        elif path == "/sessions" or path.endswith("/theme"):
            callback({"session_id": "qa-session"}, None)
        elif path == "/generation/batches":
            callback({"batches": [batch]}, None)
        elif path.endswith("/qa-batch"):
            callback(batch, None)
        elif "/tracks/" in path:
            failed = path.endswith("/2")
            callback(
                {
                    "title": "Late Signal" if failed else "Night Circuit",
                    "state": "rejected" if failed else "ready",
                    "requested": {"duration_s": 180, "vocal_mode": "instrumental"},
                    "measured": {"duration_s": None if failed else 179.8, "vocal_mode": None},
                    "quality": {"decode_ok": True},
                    "admission": {
                        "playable": not failed,
                        "mixable": not failed,
                        "constraints_satisfied": None,
                    },
                    "audio_evidence": {
                        "evidence_version": 1,
                        "status": "rejected" if failed else "admitted",
                        "raw": {"integrated_lufs": -16.2},
                        "final": {"integrated_lufs": -14.1},
                        "observations": {"status": "unavailable"},
                        "corrections": [{"operation": "loudness_and_peak"}],
                    },
                    "attempts": [],
                    "error": "Verse 2 repeats Verse 1" if failed else None,
                    "brief": {},
                    "semantic_evidence": "unknown",
                },
                None,
            )

    client._request = request
    dialog.fields["track_count"].setValue(3)
    dialog.duration_auto.setChecked(False)
    dialog.fields["duration_s"].setValue(180)
    dialog.fields["vocal_mode"].setCurrentIndex(
        dialog.fields["vocal_mode"].findData("instrumental")
    )
    dialog.fields["seed"].setValue(654)
    dialog.fields["render_speed"].setCurrentIndex(dialog.fields["render_speed"].findData("fast"))
    dialog.writer_fields["creativity"].setValue(0.45)
    dialog.preview()
    assert sent[-1][1] == "/generation/preview"
    assert "180s" in dialog.resolved.toPlainText()
    dialog.save()
    assert sent[-1][1] == "/generation/settings" and sent[-1][2]["track_count"] == 3
    assert sent[-2][2]["seed"] == 654 and sent[-2][2]["creativity"] == 0.45
    assert window._generation_preferences["duration_s"] == 180
    assert window._generation_preferences["render_speed"] == "fast"
    assert not window._writer_pending
    dialog.hide()
    window.submit_theme()
    assert sent[-1][1] == "/sessions"
    assert sent[-1][2]["generation_settings"]["duration_s"] == 180
    assert sent[-2][1] == "/generation/preview"
    window.open_track_reports()
    reports = window.report_dialog
    assert "179.8" in reports.detail.toPlainText() and "unknown" in reports.detail.toPlainText()
    reports.table.selectRow(1)
    assert reports.retry_button.isEnabled()
    window.last_status = {"generating": True}
    reports.update_actions()
    assert not reports.retry_button.isEnabled() and reports.cancel_button.isEnabled()
    window.last_status = {}
    reports.hide()
    directory = ROOT / "var/reports/generation-controls"
    directory.mkdir(parents=True, exist_ok=True)
    sizes = []
    for width, height in [(1440, 900), (720, 700), (960, 1040)]:
        window.resize(width, height)
        window.show()
        window.open_generation_controls()
        app.processEvents()
        assert dialog.width() <= width and dialog.height() <= height
        for index in range(4):
            dialog.findChild(QTabWidget).setCurrentIndex(index)
            app.processEvents()
            dialog.grab().save(str(directory / f"controls-{width}x{height}-{index}.png"))
        dialog.hide()
        window.open_track_reports()
        app.processEvents()
        assert reports.width() <= width and reports.height() <= height
        reports.grab().save(str(directory / f"reports-{width}x{height}.png"))
        reports.hide()
        sizes.append(f"{width}x{height}")
    result = {
        "settings_and_writer_payloads": "pass",
        "preview_before_submit": "pass",
        "track_report_and_action_guards": "pass",
        "layouts": sizes,
        "tracks_generated": False,
        "audio_played": False,
    }
    (directory / "gui-qa.json").write_text(json.dumps(result, indent=2) + "\n")
    window.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    main()
