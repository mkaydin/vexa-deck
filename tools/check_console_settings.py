"""Check GUI EQ → real HTTP API, palettes, persistence and compact layouts without sound.

Run with the system Python/PySide6; launches a temporary backend in the project venv.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))

import main  # noqa: E402
import pixel_theme as theme  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def check():
    app = QApplication([])
    theme.install_font()
    app.setFont(theme.font())
    report = {"platform": app.platformName(), "audio_device_opened": False, "palettes": []}
    output = ROOT / "var/reports"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vexa-console-") as directory:
        temporary = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        # This uses production API, persistence and mixer construction; opens no device.
        server_code = """import sys
from pathlib import Path
import uvicorn
from vexa_orchestrator.api import create_app
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.live import LiveSet
root = Path(sys.argv[1])
uvicorn.run(create_app(live=LiveSet(Depot(root / "library"),
            preview_dir=root / "previews")), host="127.0.0.1", port=int(sys.argv[2]),
            log_level="warning")
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = main.workspace_pythonpath(ROOT)
        with (temporary / "backend.log").open("w") as log:
            server = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-c", server_code, directory, str(port)],
                env=environment,
                stdout=log,
                stderr=log,
            )
            window = None
            try:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        with urlopen(url + "/audio/equalizer", timeout=1) as response:
                            assert response.status == 200
                        break
                    except HTTPError as exc:
                        raise RuntimeError((temporary / "backend.log").read_text()[-5000:]) from exc
                    except (OSError, URLError):
                        if server.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(
                                (temporary / "backend.log").read_text()[-5000:]
                            ) from None
                        time.sleep(0.05)
                window = main.VexaWindow(preview=True)
                window.preview = False
                window.client = main.BackendClient(url)
                window.client.status_received.connect(window.on_status)
                window.client.health_changed.connect(window.on_health)
                settings = window.console_settings
                settings.persist = True
                settings.config_path = temporary / "appearance.json"
                window.resize(720, 700)
                window.show()

                def wait(predicate):
                    until = time.monotonic() + 5
                    while not predicate() and time.monotonic() < until:
                        app.processEvents()
                        time.sleep(0.01)
                    assert predicate()

                def saved():
                    path = temporary / "config/equalizer.json"
                    return json.loads(path.read_text()) if path.exists() else None

                wait(lambda: settings.audio_page.isEnabled())
                QTest.mouseClick(window.settings_button, Qt.MouseButton.LeftButton)
                wait(settings.isVisible)
                QTest.mouseClick(settings.enabled, Qt.MouseButton.LeftButton)
                settings.sliders[0].setFocus()
                for _ in range(4):
                    QTest.keyClick(settings.sliders[0], Qt.Key.Key_Up)
                wait(lambda: saved() and saved()["gains_db"][0] == 4)
                settings._preset_selected(settings.preset.findText("Club"))
                wait(lambda: saved() and saved()["preamp_db"] == -4)
                assert saved()["gains_db"] == [4, 1, -2, 2, 4]
                settings.grab().save(str(output / "console-equalizer.png"))
                settings.reset_equalizer()
                wait(lambda: saved() and saved()["enabled"] is False)
                assert saved()["gains_db"] == [0] * 5
                report["equalizer_http_updates"] = "pass"
                settings.hide()
                for key in theme.THEMES:
                    settings.theme_selector.setCurrentIndex(settings.theme_selector.findData(key))
                    app.processEvents()
                    assert key == theme.CURRENT_THEME
                    stage = window.stage.grab().toImage()
                    assert stage.pixelColor(stage.width() // 2, stage.height() // 2) == QColor(
                        theme.BACKGROUND
                    )
                    assert QColor(theme.CYAN).name() in window.api_label.styleSheet()
                    window.on_health(True)  # dynamic status updates must retain selected palette
                    assert QColor(theme.CYAN).name() in window.api_label.styleSheet()
                    capture = output / f"console-theme-{key}.png"
                    window.grab().save(str(capture))
                    assert (
                        window.stop_button.mapTo(window, QPoint()).x()
                        + (window.stop_button.width())
                        <= window.width()
                    )
                    report["palettes"].append({"theme": key, "capture": str(capture)})
                assert json.loads(settings.config_path.read_text())["theme"] == "ice"
                # Restart the dialog with the same preference file; use a clean preview parent.
                from console_settings import ConsoleSettingsDialog

                fresh = ConsoleSettingsDialog(window, config_path=settings.config_path)
                assert fresh.theme_selector.currentData() == "ice"
                fresh.close()
                report["appearance_persistence"] = "pass"
                settings.set_online(False)
                assert not settings.audio_page.isEnabled()
                assert settings.theme_selector.isEnabled()
                report["offline_behavior"] = "pass"
            finally:
                if window is not None:
                    window.close()
                    app.processEvents()
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    path = output / "console-settings-qa.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    check()
