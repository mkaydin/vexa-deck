"""Check main controls, detail access and responsive geometry without audio or generation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))

import main  # noqa: E402
from pixel_theme import apply_theme, font, install_font  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def contained(widget, owner):
    origin = widget.mapTo(owner, QPoint())
    assert widget.isVisible(), widget.objectName()
    assert origin.x() >= 0 and origin.y() >= 0, (type(widget).__name__, origin)
    assert origin.x() + widget.width() <= owner.width(), (type(widget).__name__, origin)
    assert origin.y() + widget.height() <= owner.height(), (type(widget).__name__, origin)


def check():
    app = QApplication([])
    install_font()
    app.setFont(font())
    apply_theme(app, "terminal")
    window = main.VexaWindow(preview=True)
    client = main.BackendClient("http://127.0.0.1:9")
    client._poll.stop()
    requests = []
    client._request = lambda method, path, body, callback: requests.append((method, path))
    window.client = client
    window.on_writers({"selected": {"mode": "template", "model": "none"}})
    window.on_generation_settings({"selected": {"track_count": 10, "vocal_mode": "auto"}})
    directory = ROOT / "var/reports/console-layout"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"audio_played": False, "tracks_generated": False, "layouts": []}
    for width, height in [(1440, 900), (1050, 740), (960, 1040), (720, 700)]:
        window.resize(width, height)
        window.show()
        app.processEvents()
        assert window.size().toTuple() == (width, height)
        assert window.stage.geometry() == window.video_slot.rect()
        assert window.spectrum.width() == window.video_panel.width()
        assert window.deck_a.width() + window.deck_b.width() + 8 == window.video_panel.width()
        for widget in (
            window.theme_input,
            window.vocal_mode,
            window.start_button,
            window.music_setup_button,
            window.quick_select,
            window.depot_button,
            window.settings_button,
            window.diagnostics_button,
            window.play_button,
            window.stop_button,
            window.like_button,
            window.dislike_button,
            window.video_panel,
            window.spectrum,
        ):
            contained(widget, window)
        window.grab().save(str(directory / f"console-{width}x{height}.png"))
        window.music_setup_button.click()
        app.processEvents()
        for widget in (
            window.generator_select,
            window.generator_preset,
            window.writer_select,
            window.generation_controls_button,
            window.motion_button,
        ):
            contained(widget, window.music_setup)
        window.music_setup.grab().save(str(directory / f"setup-{width}.png"))
        window.generation_controls_button.click()
        app.processEvents()
        assert window.generation_dialog.isVisible()
        assert window.generation_dialog.width() <= width
        window.generation_dialog.hide()
        window.music_setup.hide()
        window.settings_button.click()
        app.processEvents()
        assert window.console_settings.isVisible()
        window.console_settings.hide()
        window.depot_button.click()
        app.processEvents()
        assert window.depot_dialog.isVisible()
        assert window.depot_dialog.width() <= width
        assert window.depot_dialog.table.isColumnHidden(5)
        window.depot_dialog.hide()
        window.diagnostics_button.click()
        app.processEvents()
        assert window.diagnostics.isVisible()
        assert window.diagnostics.width() <= width
        window.diagnostics.hide()
        report["layouts"].append(
            {
                "window": [width, height],
                "video": [window.stage.width(), window.stage.height()],
                "spectrum": [window.spectrum.width(), window.spectrum.height()],
                "main_controls_fit": True,
                "detail_controls_reachable": True,
            }
        )
    # Choosing a mood fills the input; it must not generate or play anything.
    requests.clear()
    window.quick_select.activated.emit(2)
    assert window.theme_input.text() == "dark melodic techno" and not requests
    window.theme_input.returnPressed.emit()
    assert not requests and "Preview mode" in window.message.text()
    window.on_status(
        {
            "generating": True,
            "generation": {"target": 3, "completed": 1, "current": 2, "stage": "rendering"},
            "continuity": {"waiting": True, "stopped": True},
        }
    )
    assert window.batch_progress.maximum() == 3 and window.batch_progress.value() == 1
    assert window.play_button.isEnabled() and not window.stop_button.isEnabled()
    assert window.live_label.text() == "STOPPED"
    report.update(quick_mood_does_not_start=True, stopped_generation_preserved=True)
    window.close()
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    check()
