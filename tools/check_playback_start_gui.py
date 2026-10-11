"""Check silent-gap recovery controls with a simulated backend and no audio device."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))

import main  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def check():
    app = QApplication([])
    window = main.VexaWindow(preview=True)
    window._models_requested = True
    client = main.BackendClient("http://127.0.0.1:9999")
    client._poll.stop()
    window.client = client
    client.status_received.connect(window.on_status)
    client.playback_pending.connect(window.on_playback_pending)
    client.action_finished.connect(window.on_action)
    client.problem.connect(window.on_problem)
    calls = []
    pending = []

    def request(method, path, payload, callback):
        calls.append((method, path, payload))
        if method == "POST":
            pending.append(callback)

    client._request = request
    playing = copy.deepcopy(window.last_status)
    playing["session_id"] = "test"
    playing["continuity"] = {"waiting": False}
    window.on_status(playing)
    assert not window.play_button.isEnabled()
    waiting = copy.deepcopy(playing)
    waiting["continuity"] = {"waiting": True}
    waiting["prepared_asset_id"] = None
    waiting["audio"]["deck_a"]["playing"] = False
    waiting["audio"]["deck_b"]["playing"] = False
    window.on_status(waiting)
    assert window.play_button.isEnabled()
    assert "Press START" in window.next_detail.text()
    assert window.footer_actions.itemAt(3).widget() is window.play_button
    assert window.footer_actions.itemAt(4).widget() is window.stop_button
    window.show()
    app.processEvents()
    report_dir = ROOT / "var/reports/playback-start"
    report_dir.mkdir(parents=True, exist_ok=True)
    for width, height in [(1440, 900), (720, 700)]:
        window.resize(width, height)
        app.processEvents()
        assert window.play_button.isVisible() and window.stop_button.isVisible()
        assert window.play_button.geometry().right() < window.width()
        assert window.stop_button.geometry().right() < window.width()
        window.grab().save(str(report_dir / f"waiting-{width}.png"))
    window.play_button.click()
    assert not window.play_button.isEnabled()
    client.start_playback()
    assert calls == [("POST", "/live/start", {})]
    # Failure clears the pending guard and permits retry once new music is ready.
    pending.pop()(None, "No unplayed generated track is ready.")
    assert window.play_button.isEnabled()
    assert "No unplayed generated track" in window.message.text()
    window.play_button.click()
    resumed = copy.deepcopy(playing)
    resumed["current_title"] = "New generated music"
    pending.pop()(resumed, None)
    assert not window.play_button.isEnabled()
    assert "Playing generated music" in window.message.text()
    stopped = copy.deepcopy(waiting)
    stopped.update(running=False, generating=True)
    stopped["continuity"]["stopped"] = True
    client.stop()
    assert calls[-1] == ("POST", "/live/stop", {})
    pending.pop()(stopped, None)
    assert client.session_id == "test"
    assert window.play_button.isEnabled() and not window.stop_button.isEnabled()
    assert window.live_label.text() == "STOPPED"
    assert "Generation continues" in window.message.text()
    assert "SHIFT THE VIBE" in window.start_button.text()
    window.play_button.click()
    assert calls[-1] == ("POST", "/live/start", {})
    pending.pop()(resumed, None)
    # STOP is also reachable while waiting for the first generated track.
    starting = copy.deepcopy(playing)
    starting.update(
        running=False,
        generating=True,
        startup={"policy": "depot", "ready": 0, "required": 1},
    )
    window.on_status(starting)
    assert window.stop_button.isEnabled()
    starting["continuity"] = {"waiting": True, "stopped": True}
    window.on_status(starting)
    assert window.live_label.text() == "STOPPED" and window.play_button.isEnabled()
    assert window.health.text() == "AUDIO / STOPPED"
    assert "Generation continues" in window.next_detail.text()
    window.on_status(waiting)
    window.on_health(False)
    assert not window.play_button.isEnabled()
    window.on_health(True)
    assert window.play_button.isEnabled()
    window.on_status({"running": False})
    started = []
    window.submit_theme = lambda: started.append(True)
    window.play_button.click()
    assert started == [True]
    client._poll.stop()
    window.close()
    report = {
        "button_next_to_stop": True,
        "silent_gap_request": "POST /live/start",
        "playing_disabled": True,
        "duplicate_request_blocked": True,
        "failed_request_retry": True,
        "offline_disabled": True,
        "idle_starts_set": True,
        "sizes": [1440, 720],
        "audio_played": False,
    }
    report.update(
        stop_preserves_session=True,
        stop_keeps_generation_visible=True,
        stopped_start_uses_playback_endpoint=True,
        startup_can_be_stopped=True,
    )
    (report_dir / "gui-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    check()
