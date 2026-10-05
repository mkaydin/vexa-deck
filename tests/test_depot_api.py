"""Generated depot management must never remove audio held by the live set."""

import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from vexa_orchestrator.api import create_app


def test_generated_depot_lists_streams_and_guards_delete(tmp_path):
    asset_id = "live-test-track"
    wav = tmp_path / f"{asset_id}.wav"
    wav.write_bytes(b"RIFFtest-audio")
    wav.with_suffix(".brief.json").write_text(
        '{"theme":"warm house","style":"warm piano house"}')
    manifest = SimpleNamespace(
        asset_id=asset_id,
        audio=SimpleNamespace(duration_s=165.0),
        beat_grid=SimpleNamespace(bpm=122.0),
        provenance=SimpleNamespace(source_prompt="warm piano house"),
    )
    asset = SimpleNamespace(manifest=manifest, path=wav)

    class Depot:
        def __init__(self):
            self.assets = {asset_id: asset}

        def reload(self):
            self.assets = {key: value for key, value in self.assets.items()
                           if value.path.exists()}

    live = SimpleNamespace(depot=Depot(), _lock=threading.RLock(), state=None,
                           current=asset, prepared=None)
    app = create_app(live=live)

    def route(path, method):
        return next(item.endpoint for item in app.routes
                    if getattr(item, "path", "") == path and method in item.methods)

    listed = route("/depot/tracks", "GET")
    audio = route("/depot/tracks/{asset_id}/audio", "GET")
    delete = route("/depot/tracks/{asset_id}", "DELETE")
    assert listed()[0]["duration_s"] == 165.0
    assert audio(asset_id).path == wav
    with pytest.raises(HTTPException) as blocked:
        delete(asset_id)
    assert blocked.value.status_code == 409 and wav.exists()
    live.current = None
    assert delete(asset_id)["asset_id"] == asset_id
    assert not wav.exists()
    assert not (tmp_path / f"{asset_id}.brief.json").exists()
    assert listed() == []
