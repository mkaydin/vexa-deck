"""Offscreen listening-page/export QA. Synthetic labels never count as human listening."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    from PySide6.QtCore import QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    QApplication(sys.argv)
    view = QWebEngineView()
    profile = QWebEngineProfile(view)
    page = QWebEnginePage(profile, view)
    view.setPage(page)
    view.resize(1040, 1000)
    loaded, downloads = [], []
    view.loadFinished.connect(loaded.append)

    def wait_for(predicate):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if predicate():
                return
            QTest.qWait(30)
        raise AssertionError("listening page QA timed out")

    def javascript(source):
        output = []
        page.runJavaScript(source, output.append)
        wait_for(lambda: bool(output))
        return json.loads(output[0])

    def download(request):
        request.setDownloadDirectory(str(args.out.resolve()))
        request.setDownloadFileName("synthetic-gui-annotations.json")
        downloads.append(request)
        request.accept()

    profile.downloadRequested.connect(download)
    view.show()
    view.load(QUrl.fromLocalFile(str((args.bundle / "index.html").resolve())))
    wait_for(lambda: bool(loaded))
    assert loaded[-1]
    result = javascript("""JSON.stringify({
        audio:document.querySelectorAll('audio').length,
        anonymous:!document.body.innerText.match(/minimax|acestep|yue2/i),
        autoplay:!!document.querySelector('audio[autoplay]'),
        ratingFields:document.querySelectorAll('[data-field="taste"]').length
    })""")
    expected = json.loads((args.bundle / "bundle.json").read_text())
    count = sum(bool(row.get("review_audio")) for row in expected["items"])
    assert result == {"audio": count, "anonymous": True, "autoplay": False, "ratingFields": count}
    javascript("""(() => {
        const audio = document.querySelector('audio');
        audio.preload = 'metadata'; audio.load();
        return JSON.stringify(true);
    })()""")
    wait_for(
        lambda: javascript("""JSON.stringify(
        Number.isFinite(document.querySelector('audio').duration)
        && document.querySelector('audio').duration > 0)""")
    )
    result["media_decodes"] = True
    view.grab().save(str(args.out / "listening-review.png"))
    # Exercise export only; this reviewer and output are explicitly synthetic.
    javascript("""(() => {
        document.querySelector('#reviewer').value = 'gui-qa-synthetic';
        const card = document.querySelector('section[data-index]');
        card.querySelector('[data-field="taste"]').value = '4';
        card.querySelector('[data-field="vocals_present"]').value = 'false';
        card.querySelector('[data-field="heard_genre"]').value = 'techno';
        card.querySelector('[data-field="listened_full"]').checked = true;
        document.querySelector('#export').click();
        return JSON.stringify(true);
    })()""")
    wait_for(lambda: downloads and downloads[-1].isFinished())
    labels = json.loads((args.out / "synthetic-gui-annotations.json").read_text())
    assert labels["bundle_id"] == expected["bundle_id"] and len(labels["reviews"]) == 1
    label = labels["reviews"][0]
    assert label["taste"] == 4 and label["vocals_present"] is False
    assert label["heard_genre"] == "techno" and label["id"].startswith("blind-")
    result.update(
        export_valid=True,
        human_review_performed=False,
        synthetic_labels=True,
        bundle_id=expected["bundle_id"],
    )
    (args.out / "gui-report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    view.close()


if __name__ == "__main__":
    main()
