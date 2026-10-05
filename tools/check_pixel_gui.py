"""Verify blank center and usable desktop layouts without opening audio."""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))

import main  # noqa: E402
from pixel_theme import BACKGROUND, STYLE, font, install_font  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def check_blank(stage):
    image = stage.grab().toImage()
    expected = QColor(BACKGROUND)
    samples = 0
    margin = round(22 * stage.devicePixelRatioF())
    for x in range(margin, image.width() - margin, max(1, image.width() // 20)):
        for y in range(margin, image.height() - margin, max(1, image.height() // 20)):
            assert image.pixelColor(x, y) == expected, (x, y)
            samples += 1
    assert not hasattr(stage, "controller") and not hasattr(stage, "store")
    assert not hasattr(stage, "_timer")
    return samples


def main_check():
    app = QApplication([])
    install_font()
    app.setFont(font())
    app.setStyleSheet(STYLE)
    report = {"qt_platform": app.platformName(), "layouts": [], "audio_tested": False}
    output = ROOT / "var/reports"
    output.mkdir(parents=True, exist_ok=True)
    for width, height in ((1440, 900), (1050, 740), (960, 1020), (720, 700)):
        # Request each initial size on a fresh Wayland surface. The compositor
        # can retain the current size when a mapped window requests a resize.
        window = main.VexaWindow(preview=True)
        window.resize(width, height)
        window.right_panel.setVisible(width >= 1280)
        window.show()
        window.spectrum.values = window.spectrum.target.copy()
        app.processEvents()
        # Wayland configures window sizes asynchronously after a resize request.
        deadline = time.monotonic() + 2
        while window.size().toTuple() != (width, height) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        capture = output / f"pixel-gui-{width}-{window.devicePixelRatioF():g}x.png"
        window.grab().save(str(capture))
        assert window.size().toTuple() == (width, height), window.size().toTuple()
        assert window.stop_button.isVisible() and window.theme_input.isVisible()
        assert window.theme_input.height() >= 36
        assert window.depot_button.isVisible() and window.like_button.isVisible()
        assert window.right_panel.isVisible() == (width >= 1280)
        assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
        assert window.title_bar.isVisible()
        assert window.title_bar.close_button.isVisible()
        assert window.stage.geometry() == window.video_slot.rect()
        assert window.stage.mapTo(window, QPoint()).x() == window.deck_a.mapTo(window, QPoint()).x()
        assert window.stop_button.mapTo(window, QPoint()).x() + window.stop_button.width() <= width
        assert window.spectrum.height() == 160
        assert window.spectrum.width() == window.centralWidget().width() - 24
        spectrum_top = window.spectrum.mapTo(window, QPoint()).y()
        assert spectrum_top > window.deck_a.mapTo(window, QPoint()).y() + window.deck_a.height()
        report["layouts"].append(
            {
                "size": [width, height],
                "capture": str(capture),
                "blank_samples": check_blank(window.stage),
                "video_size": [window.stage.width(), window.stage.height()],
                "video_side_gutters": [
                    window.stage.x(),
                    window.video_slot.width() - window.stage.width(),
                ],
                "spectrum_size": [window.spectrum.width(), window.spectrum.height()],
            }
        )
        window.set_reduced_motion(True)
        app.processEvents()
        check_blank(window.stage)
        window.show_diagnostics()
        app.processEvents()
        assert window.diagnostics.isVisible()
        window.diagnostics.close()
        window.close()
        app.processEvents()
    # Exercise caption actions on a separate preview; never start an audio session.
    window = main.VexaWindow(preview=True)
    window.show()
    app.processEvents()

    def await_state(predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        assert predicate()
        # Window state flags precede Wayland's configure/layout round trip.
        QTest.qWait(150)

    await_state(lambda: window.windowHandle().isExposed())
    QTest.mouseClick(window.title_bar.maximize_button, Qt.MouseButton.LeftButton)
    await_state(window.isMaximized)
    assert window.title_bar.maximize_button.toolTip() == "Restore"
    assert not window.resize_controller.edges_at(QPoint(1, 1))
    QTest.mouseClick(window.title_bar.maximize_button, Qt.MouseButton.LeftButton)
    await_state(lambda: not window.isMaximized())
    assert window.resize_controller.edges_at(QPoint(1, 1)) == (Qt.Edge.LeftEdge | Qt.Edge.TopEdge)
    QTest.mouseClick(window.title_bar.close_button, Qt.MouseButton.LeftButton)
    await_state(lambda: not window.isVisible())
    assert window._closing
    # Test minimize last; returning from it belongs to the desktop task switcher.
    window = main.VexaWindow(preview=True)
    window.show()
    await_state(lambda: window.windowHandle().isExposed())
    QTest.mouseClick(window.title_bar.minimize_button, Qt.MouseButton.LeftButton)
    # Wayland may clear Qt's minimized flag while the surface stays unexposed.
    await_state(lambda: not window.windowHandle().isExposed())
    window.close()
    report.update(blank_center="pass", diagnostics_check="pass", character_animation_loaded=False)
    report["caption_controls"] = "pass"
    path = output / f"pixel-gui-qa-{os.environ.get('QT_SCALE_FACTOR', '1')}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_check()
