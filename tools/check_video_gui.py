"""Exercise local video modes, settings and looping without opening the music backend.

Run with the desktop Python on Wayland. FFmpeg creates short, silent local test clips.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))

import main  # noqa: E402
from pixel_theme import STYLE, font, install_font  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from video_panel import VideoPanel  # noqa: E402


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        QTest.qWait(30)
    raise AssertionError("Video condition timed out")


def state(panel):
    result = []
    panel.page.runJavaScript("JSON.stringify(window.VexaVideo.state())", result.append)
    wait_for(lambda: bool(result))
    return json.loads(result[0])


def main_check():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=Path, help="also verify decoding of an existing local video"
    )
    args = parser.parse_args()
    app = QApplication([])
    install_font()
    app.setFont(font())
    app.setStyleSheet(STYLE)
    app.setQuitOnLastWindowClosed(False)
    output = ROOT / "var/reports"
    output.mkdir(parents=True, exist_ok=True)
    report = {"platform": app.platformName(), "audio_backend_started": False, "modes": {}}
    with tempfile.TemporaryDirectory(prefix="vexa-video-qa-") as directory:
        directory = Path(directory)
        first, second = directory / "first # clip.webm", directory / "second clip.webm"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=360x640:rate=24",
                "-t",
                "1.2",
                "-an",
                "-c:v",
                "libvpx",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-y",
                str(first),
            ],
            check=True,
        )
        second.write_bytes(first.read_bytes())
        window = main.VexaWindow(preview=True)
        window.show()
        panel = window.video_panel
        errors = []
        panel.message.connect(
            lambda message: (
                errors.append(message)
                if "error" in message.lower() or "cannot decode" in message.lower()
                else None
            )
        )
        panel.add_videos([first])
        wait_for(lambda: panel.ready)
        wait_for(lambda: state(panel)["ready"] >= 2)
        wait_for(lambda: state(panel)["loops"] >= 1)
        for mode in ("normal", "pixel", "ascii"):
            panel.mode.setCurrentIndex(panel.mode.findData(mode))
            before = state(panel)["frames"]
            wait_for(lambda previous=before: state(panel)["frames"] > previous + 6)
            current = state(panel)
            assert current["muted"] and current["volume"] == 0
            assert current["mode"] == mode and not current["error"], current
            capture = output / f"video-gui-{mode}.png"
            window.grab().save(str(capture))
            report["modes"][mode] = {"state": current, "capture": str(capture)}
        panel.open_settings()
        QTest.qWait(100)
        dialog = panel.settings_dialog
        dialog.editors["pixel.size"].setValue(12)
        dialog.editors["ascii.resolution"].setValue(64)
        dialog.editors["ascii.text"].setText("VEXA // DJ")
        dialog.editors["ascii.textMode"].setCurrentIndex(1)
        wait_for(lambda: state(panel)["resolution"] == 64 and state(panel)["pixelSize"] == 12)
        assert panel.options["ascii"]["text"] == "VEXA // DJ"
        dialog.close()
        panel.set_paused(True)
        QTest.qWait(80)
        paused = state(panel)["time"]
        QTest.qWait(250)
        assert abs(state(panel)["time"] - paused) < 0.05
        panel.set_paused(False)
        wait_for(lambda: not state(panel)["paused"])
        panel.set_reduced_motion(True)
        QTest.qWait(100)
        assert state(panel)["paused"]
        panel.set_reduced_motion(False)
        wait_for(lambda: not state(panel)["paused"])
        window.hide()
        wait_for(lambda: state(panel)["paused"])
        window.show()
        wait_for(lambda: not state(panel)["paused"])
        sources = []
        panel.playlist_changed.connect(lambda: sources.append(panel.current_index))
        panel.add_videos([second])
        wait_for(lambda: 1 in sources)
        wait_for(lambda: sources[-1] == 0)
        report["playlist_sequence"] = sources
        if args.video:
            panel.clear_videos()
            panel.add_videos([args.video])
            wait_for(lambda: state(panel)["ready"] >= 2)
            dialog.reset()
            panel.set_option("fit", "fit")
            report["reference_decode"] = {}
            for mode in ("normal", "pixel", "ascii"):
                panel.mode.setCurrentIndex(panel.mode.findData(mode))
                QTest.qWait(350)
                actual = state(panel)
                assert actual["mode"] == mode and not actual["error"], actual
                report["reference_decode"][mode] = actual
                window.grab().save(str(output / f"video-gui-reference-{mode}.png"))
        panel.remove_video(0)
        assert not panel.paths
        panel.add_videos([first])
        wait_for(lambda: state(panel)["ready"] >= 2)
        panel.clear_videos()
        assert not panel.paths and not panel.view.isVisible()
        # Invalid media is reported and skipped; a healthy loop continues.
        broken = directory / "invalid.mp4"
        broken.write_bytes(b"not a video")
        expected_errors = []
        panel.message.connect(
            lambda message: expected_errors.append(message) if "Cannot decode" in message else None
        )
        panel.add_videos([broken, first])
        wait_for(lambda: bool(expected_errors))
        wait_for(lambda: state(panel)["index"] == 1 and state(panel)["ready"] >= 2)
        panel.clear_videos()
        # Persistence of preferences and local paths, without writing user configuration.
        config = directory / "preferences.json"
        saved = VideoPanel(persist=True, config_path=config)
        saved.paths = [first]
        saved.set_option("mode", "ascii")
        saved.set_option("ascii.resolution", 80)
        restored = VideoPanel(persist=True, config_path=config)
        assert restored.paths == [first]
        assert restored.options["mode"] == "ascii" and restored.options["ascii"]["resolution"] == 80
        saved.deleteLater()
        restored.deleteLater()
        assert not [error for error in errors if error not in expected_errors], errors
        report.update(
            looping="pass",
            settings="pass",
            pause="pass",
            reduced_motion="pass",
            clear="pass",
            persistence="pass",
            hidden_suspension="pass",
            invalid_video_skip="pass",
        )
        window.close()
        QTest.qWait(100)
    path = output / "video-gui-qa.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_check()
