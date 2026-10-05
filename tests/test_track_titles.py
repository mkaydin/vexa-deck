"""Names survive planning, metadata reload and playback; storage identity stays intact."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from vexa_contracts import AssetManifest, DeckId, MusicalClock, SessionState
from vexa_orchestrator.api import create_app
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.generation import (
    _parse_track_plan,
    _repair_instrumental_plan,
    isolated_generate_theme,
    plan_theme,
)
from vexa_orchestrator.live import LiveSet
from vexa_orchestrator.track_titles import clean_title, fallback_title

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return AssetManifest.model_validate_json(
        (ROOT / "data/fixtures/asset_manifest.sample.json").read_text()
    )


def test_writer_titles_are_cleaned_unique_and_not_audio_conditioning():
    rows = [
        {
            "title": "  夜雨回声  ",
            "style": f"Dark phonk, gritty cowbell riff {i}, "
            "deep 808 bass and rolling hi-hats with evolving instrumental motifs.",
            "lyrics": "",
        }
        for i in range(2)
    ]
    plans = _parse_track_plan(json.dumps({"tracks": rows}), "phonk instrumental", 2, "test")
    assert plans[0].title == "夜雨回声"
    assert plans[1].title == "夜雨回声 / 02"
    assert all("夜雨回声" not in plan.style for plan in plans)
    assert clean_title("Night\nShift\x00") == "Night Shift"
    assert len(clean_title("x" * 100)) == 80
    assert clean_title("Track 01") == ""


def test_missing_or_invalid_titles_do_not_abort_generation(monkeypatch):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    plans = plan_theme("rain-soaked jazz")
    assert all(plan.title for plan in plans)
    assert len({plan.title.casefold() for plan in plans}) == 10
    assert fallback_title("phonk", "808", "abc") == fallback_title("phonk", "808", "abc")
    for invalid in (None, 42, "", "Untitled"):
        plan = _parse_track_plan(
            json.dumps(
                {
                    "tracks": [
                        {
                            "title": invalid,
                            "style": "Phonk with crunchy cowbell riffs, rolling drums "
                            "and ominous synth pads.",
                            "lyrics": "",
                        }
                    ]
                }
            ),
            "phonk instrumental",
            1,
            "test",
        )[0]
        assert len(plan.title.split()) >= 2


def test_instrumental_repair_keeps_the_title():
    raw = json.dumps(
        {
            "tracks": [
                {
                    "title": "Redline Ritual",
                    "style": "Phonk with ominous cowbells and rolling hi-hats. "
                    "Chopped vocal samples.",
                    "lyrics": "Unwanted lyrics.",
                }
            ]
        }
    )
    repaired, _ = _repair_instrumental_plan(raw, "phonk instrumental", 1)
    assert json.loads(repaired)["tracks"][0]["title"] == "Redline Ritual"


def test_titles_cross_the_isolated_worker_boundary(monkeypatch, tmp_path):
    manifest = fixture()
    manifest.title = "Concrete Memory"
    captured = {}

    def run_job(action, payload, **kwargs):
        captured.update(payload)
        return {"manifest": manifest.model_dump(mode="json"), "path": str(tmp_path / "a.wav")}

    monkeypatch.setattr("vexa_orchestrator.background.run_job", run_job)
    asset = isolated_generate_theme("underground rap", tmp_path, title="Concrete Memory")
    assert captured["title"] == asset.manifest.title == "Concrete Memory"


def test_legacy_names_persist_without_changing_audio_and_are_visible_to_api(tmp_path):
    from tools.name_depot_tracks import name_tracks

    manifest = fixture()
    manifest.asset_id = "live-0123456789ab"
    manifest.provenance.source_prompt = "Dusty underground boom bap hip hop, vinyl, swung drums."
    source = tmp_path / f"{manifest.asset_id}.wav"
    source.write_bytes(b"original audio bytes")
    metadata = source.with_suffix(".json")
    old = manifest.model_dump(mode="json")
    old.pop("title")  # pre-title manifests remain supported
    metadata.write_text(json.dumps(old))
    brief = {"theme": "90s underground rap", "lyrics": "original words"}
    source.with_suffix(".brief.json").write_text(json.dumps(brief))
    before = Depot(tmp_path).assets[manifest.asset_id].manifest.title
    changed = name_tracks(tmp_path)
    assert len(changed) == 1 and changed[0]["title"] == before
    assert name_tracks(tmp_path) == []
    new = json.loads(metadata.read_text())
    assert {key: value for key, value in new.items() if key != "title"} == old
    assert source.read_bytes() == b"original audio bytes"
    assert json.loads(source.with_suffix(".brief.json").read_text()) == brief
    depot = Depot(tmp_path)
    live = LiveSet(depot, preview_dir=tmp_path / "previews")
    live.current = depot.assets[manifest.asset_id]
    live.state = SessionState(session_id="titles", theme=brief["theme"], clock=MusicalClock(bpm=90))
    live.state.deck(DeckId.A).asset_id = manifest.asset_id
    live._event({"event": "start", "asset_id": manifest.asset_id})
    client = TestClient(create_app(live=live))
    listed = client.get("/depot/tracks").json()[0]
    assert listed["asset_id"] == manifest.asset_id and listed["title"] == before
    status = client.get("/live").json()
    assert status["current_title"] == status["asset_titles"][manifest.asset_id] == before
    assert status["events"][-1]["title"] == before
    assert live.feedback("like")["title"] == before
