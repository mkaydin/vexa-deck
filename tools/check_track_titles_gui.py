"""Verify visible titles and ID-based depot actions without playing or deleting real tracks."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))

import main  # noqa: E402
import pixel_theme as theme  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402


class PreviewPlayer:
    def stop(self):
        pass

    def setSource(self, url):
        self.url = url.toString()

    def play(self):
        pass


def check():
    app = QApplication([])
    theme.install_font()
    app.setFont(theme.font())
    window = main.VexaWindow(preview=True)
    window.show()
    app.processEvents()
    assert window.deck_a.title == "Neon Circuit Reverie"
    assert window.deck_b.title == "Afterhours Voltage Drift"
    assert window.deck_a.asset_id == "live-92efb7f7f5ea"
    assert window.next_asset.text() == "Afterhours Voltage Drift"
    assert "Neon Circuit Reverie" in window._history_widgets[0].text()
    assert "live-92efb7f7f5ea" in window.deck_a.toolTip()
    rows = [{"asset_id": "live-012345abcdef", "title": "Concrete Memory",
             "theme": "90s underground hip hop", "style": "Dusty swung drums and warm bass",
             "duration_s": 165, "bpm": 90},
            {"asset_id": "live-fedcba543210", "title": "Redline Ritual", "theme": "JDM phonk",
             "duration_s": 172, "bpm": 150}]
    deleted = []
    client = SimpleNamespace(base_url="http://127.0.0.1:9999",
                             list_tracks=lambda callback: callback(rows, None),
                             delete_track=lambda identity, callback:
                             (deleted.append(identity), callback({}, None)))
    depot = main.DepotDialog(client, window)
    depot.show()
    app.processEvents()
    assert depot.table.columnCount() == 6
    depot.table.selectRow(0)
    assert depot.selected_title() == "Concrete Memory"
    assert depot.selected_id() == "live-012345abcdef"
    depot.player = PreviewPlayer()
    depot.listen()
    assert depot.message.text() == "Listening to Concrete Memory"
    assert depot.player.url.endswith("/depot/tracks/live-012345abcdef/audio")
    confirmation = []
    question = QMessageBox.question
    try:
        def confirm(parent, heading, text):
            confirmation.append(text)
            return QMessageBox.StandardButton.Yes
        QMessageBox.question = confirm
        depot.delete_selected()
    finally:
        QMessageBox.question = question
    assert "Concrete Memory" in confirmation[0]
    assert deleted == ["live-012345abcdef"]
    report_dir = ROOT / "var/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    depot.grab().save(str(report_dir / "track-titles-depot.png"))
    depot.close()
    app.processEvents()
    window.grab().save(str(report_dir / "track-titles-playback.png"))
    report = {"platform": app.platformName(), "deck_titles": "pass", "up_next": "pass",
              "history": "pass", "depot": "pass", "listen_and_delete_use_ids": "pass",
              "audio_played": False, "real_tracks_deleted": False}
    (report_dir / "track-titles-gui-qa.json").write_text(json.dumps(report, indent=2) + "\n")
    window.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    check()
