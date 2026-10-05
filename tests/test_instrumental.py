"""No-lyrics intent survives planning, API theme changes and depot selection."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from vexa_contracts import AssetManifest, Estimate
from vexa_orchestrator.ace_planner import plan_in_worker
from vexa_orchestrator.api import CreateSessionBody, ThemeBody, create_app
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.generation import _parse_track_plan, _theme_profile
from vexa_orchestrator.vocal_intent import apply_instrumental_mode, has_vocal_delivery, vocal_intent


@pytest.mark.parametrize("theme", [
    "underground theme music with no lyrics",
    "90s underground rap with no lyrics",
    "trap without lyrics",
    "a sad song without vocals",
    "house with no singing",
    "lyric-free jazz", "vocal free rap", "sözsüz rap", "vokalsiz techno", "sozsuz rap",
])
def test_negated_lyrics_never_turn_into_a_vocal_request(theme):
    assert vocal_intent(theme) is False
    assert not _theme_profile(theme)[2]


def test_explicit_mode_overrides_conflicting_theme_text():
    instrumental = apply_instrumental_mode("90s underground rap with vocals", True)
    assert vocal_intent(instrumental) is False and not _theme_profile(instrumental)[2]
    assert "underground rap" in instrumental
    vocal = apply_instrumental_mode("90s underground rap with no lyrics instrumental", False)
    assert vocal_intent(vocal) is True and _theme_profile(vocal)[2]
    assert "no lyrics" not in vocal and "instrumental" not in vocal
    theme = "underground theme music with no lyrics"
    assert apply_instrumental_mode(theme, None) == theme
    assert _theme_profile("not instrumental rap")[2]


def test_instrumental_prompt_accepts_empty_lyrics_and_rejects_vocal_drift():
    theme = "underground theme music with no lyrics"
    style = "Underground instrumental beat with dusty swung drums and dark bass. No vocals."
    plan = {"tracks": [{"style": style, "lyrics": ""}]}
    brief = _parse_track_plan(json.dumps(plan), theme, 1, "local-test")[0]
    assert brief.lyrics == "" and brief.duration_s == 165
    assert "Instrumental, no vocals" in brief.style
    plan["tracks"][0]["lyrics"] = "These words should not be sent to YuE2."
    with pytest.raises(ValueError, match="empty lyrics"):
        _parse_track_plan(json.dumps(plan), theme, 1, "local-test")
    plan["tracks"][0].update(lyrics="", style=style + " Female vocals and sung refrain.")
    with pytest.raises(ValueError, match="vocal delivery"):
        _parse_track_plan(json.dumps(plan), theme, 1, "local-test")
    assert not has_vocal_delivery("Instrumental drums, no singing, without vocals.")
    assert has_vocal_delivery("No lyrics, female vocal humming over the drums.")


def test_local_instrumental_plan_uses_no_lyric_sections_and_discards_vocal_refinement(
    monkeypatch, tmp_path
):
    theme = "90s underground rap with no lyrics"
    style = "Underground boom bap hip hop, dusty jazz chops and swung drums, no vocals."
    monkeypatch.setattr("vexa_orchestrator.ace_planner.write_local_briefs", lambda *a: [
        SimpleNamespace(style=style, lyrics="", origin="local-test")
    ])
    calls = []

    def sample(*a, **kwargs):
        calls.append(kwargs)
        return {"caption": "Boom bap hip hop with a dry MC delivery and rap verses.",
                "lyrics": "", "bpm": "92", "keyscale": "D minor",
                "timesignature": "4", "language": "unknown", "duration": "165"}

    monkeypatch.setattr("vexa_orchestrator.ace_planner.TextPlanner",
                        lambda: SimpleNamespace(sample=sample))
    monkeypatch.setattr("vexa_orchestrator.ace_planner.ROOT", tmp_path)
    row = plan_in_worker(theme, 1)["tracks"][0]
    assert row["lyrics"] == "" and row["music_plan"]["instrumental"] is True
    assert row["music_plan"]["language"] == "unknown"
    assert calls[0]["instrumental"] is True and calls[0]["lyrics"] == ""
    assert not row["music_plan"]["refinement_accepted"]
    assert not any("Verse" in part["section"] for part in row["music_plan"]["sections"])


def test_instrumental_depot_excludes_vocal_and_unknown_legacy_tracks(tmp_path):
    fixture = Path(__file__).parents[1] / "data/fixtures/asset_manifest.sample.json"
    depot = Depot(tmp_path)
    for asset_id, value, source in [
        ("instrumental", "true", "Underground boom bap hip hop, instrumental no vocals"),
        ("vocal", "false", "Underground boom bap hip hop with rap verses"),
        ("unknown", None, "Underground boom bap hip hop"),
        ("legacy-instrumental", None, "Underground boom bap hip hop, no vocals"),
    ]:
        manifest = AssetManifest.model_validate_json(fixture.read_text())
        manifest.asset_id = manifest.family_id = asset_id
        manifest.provenance.source_prompt = source
        manifest.tags = {"instrumental": [Estimate(value=value, confidence=0.8)]} if value else {}
        depot.add(manifest, tmp_path / f"{asset_id}.wav")
    selected = {asset.manifest.asset_id for _, asset in depot.search(
        "underground theme music with no lyrics")}
    assert selected == {"instrumental", "legacy-instrumental"}
    selected = {asset.manifest.asset_id for _, asset in depot.search("underground rap")}
    assert "vocal" in selected and "instrumental" not in selected


def test_api_applies_vocal_setting_at_start_and_theme_change():
    seen = []
    live = SimpleNamespace(
        state=None, depot=SimpleNamespace(assets={}), stop=lambda: None,
        start=lambda theme, **kw: seen.append(theme) or {"generating": True},
        steer=lambda theme: seen.append(theme) or {"theme": theme},
        status=lambda: {"session_id": "test"},
    )
    app = create_app(live=live)

    def route(path):
        return next(item.endpoint for item in app.routes if getattr(item, "path", "") == path
                    and "POST" in item.methods)

    created = route("/sessions")(CreateSessionBody(
        theme="underground rap", session_id="test", instrumental=True))
    assert created.generating and vocal_intent(seen[0]) is False
    assert app.state.store.sessions["test"].state.theme == seen[0]
    changed = route("/sessions/{session_id}/theme")("test", ThemeBody(
        theme="underground rap with no lyrics", instrumental=False))
    assert vocal_intent(seen[-1]) is True and changed["theme"] == seen[-1]
    route("/sessions/{session_id}/theme")("test", ThemeBody(
        theme="underground theme music with no lyrics"))
    assert vocal_intent(seen[-1]) is False


def test_instrumental_writer_and_music_query_do_not_request_lyric_stories(monkeypatch):
    from vexa_orchestrator.generation import _plan_remote_theme
    from vexa_orchestrator.music_plan import track_query

    theme = "underground theme music with no lyrics"
    seen = []
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "ollama")

    def prompts(_base, _model, messages, *args):
        seen.extend(messages)
        return json.dumps({"tracks": [{
            "style": "Underground instrumental beat, deep bass and dusty drums, no vocals.",
            "lyrics": "",
        }]})

    monkeypatch.setattr("vexa_orchestrator.generation._ollama_prompts", prompts)
    brief = _plan_remote_theme(theme, count=1, track_index=0)[0]
    assert brief.lyrics == ""
    assert "Describe instrumental sections only" in seen[0]["content"]
    assert "Count actual lyric lines" not in seen[0]["content"]
    genre, palette, vocal = _theme_profile(theme)
    query = track_query(theme, genre=genre, palette=palette, vocal=vocal,
                        duration_s=165, energy=0.35, index=0, role="opener")
    assert "Write musical metadata only" in query and "Vocal language: unknown" in query
    assert "complete lyrics" not in query and "Each verse develops" not in query


def test_yue2_boundary_rejects_vocals_before_rendering_an_instrumental_request(
    monkeypatch, tmp_path
):
    from vexa_orchestrator.generation import generate_theme

    component = tmp_path / "fake-component"
    component.touch()
    for key in ("VEXA_YUE_BINARY", "VEXA_YUE_MODEL", "VEXA_YUE_VAE"):
        monkeypatch.setenv(key, str(component))
    with pytest.raises(ValueError, match="empty lyrics"):
        generate_theme("underground music with no lyrics", tmp_path,
                       style="Dusty breakbeats and a deep dark bass.", lyrics="Unwanted words.")
    with pytest.raises(ValueError, match="vocal delivery"):
        generate_theme("underground music with no lyrics", tmp_path,
                       style="Dusty breakbeats and a deep bass with spoken rap verses.")
    assert not list(tmp_path.glob("*.request.json"))


@pytest.mark.parametrize("style", [
    "JDM phonk, no vocals, singing or humming. Cowbell and 808 riffs.",
    "Instrumental phonk without any vocal samples or sung choruses.",
    "Phonk without vocals and without singing, just distorted bass.",
    "Phonk, exclude spoken words, rap samples and vocal chops.",
])
def test_negative_vocal_lists_are_not_positive_vocal_directions(style):
    assert not has_vocal_delivery(style)


@pytest.mark.parametrize("style", [
    "Phonk, no singing but chopped vocal samples.",
    "No lyrics, but Memphis rap samples and humming form a hook.",
    "Phonk without vocals in the intro. Add a singer at the drop.",
])
def test_negation_does_not_hide_positive_vocals_elsewhere(style):
    assert has_vocal_delivery(style)


def test_repaired_instrumental_phonk_preserves_music_and_clears_vocal_samples():
    from vexa_orchestrator.generation import _repair_instrumental_plan

    style = (
        "JDM Phonk, Late 2010s. Starts with a sparse, distorted cowbell loop and a heavy 808. "
        "Timbre includes gritty, chopped vocal samples used as texture. "
        "The bass is deep and distorted. Builds with hi-hat triplets and a rhythmic outro."
    )
    raw = json.dumps({"tracks": [{"style": style, "lyrics": "Unwanted rap words."}]})
    theme = "jdm phonk. Instrumental, no vocals, no lyrics"
    repaired, changes = _repair_instrumental_plan(raw, theme, 1)
    row = json.loads(repaired)["tracks"][0]
    assert row["lyrics"] == ""
    assert "cowbell loop" in row["style"] and "hi-hat triplets" in row["style"]
    assert "samples" not in row["style"] and not has_vocal_delivery(row["style"])
    assert changes[0]["cleared_lyrics"] and len(changes[0]["removed_sentences"]) == 1
    assert _parse_track_plan(repaired, theme, 1, "test")[0].lyrics == ""


def test_entirely_vocal_description_recovers_as_ten_distinct_phonk_briefs():
    from vexa_orchestrator.generation import _repair_instrumental_plan
    from vexa_orchestrator.music_plan import tempo_hint

    theme = "jdm phonk no lyrics"
    raw = json.dumps({"tracks": [{
        "style": "Female vocals and male rap verses with chopped vocal hooks.", "lyrics": ""
    } for _ in range(10)]})
    repaired, changes = _repair_instrumental_plan(raw, theme, 10)
    briefs = _parse_track_plan(repaired, theme, 10, "test")
    assert len(changes) == len(briefs) == len({row.style for row in briefs}) == 10
    assert all("cowbell" in row.style and row.lyrics == "" for row in briefs)
    genre, palette, vocal = _theme_profile(theme)
    assert genre == "JDM drift phonk" and not vocal and "808" in palette
    assert tempo_hint(theme, genre) == 150
    assert tempo_hint("jdm phonk 160 BPM", genre) == 160


def test_writer_recovers_instrumental_voice_drift_without_aborting_or_retrying(
    monkeypatch, tmp_path
):
    from vexa_orchestrator.generation import _plan_remote_theme

    monkeypatch.setenv("VEXA_LLM_PROVIDER", "ollama")
    monkeypatch.setattr("vexa_orchestrator.generation.ROOT", tmp_path)
    calls = []

    def prompts(*args):
        calls.append(args)
        return json.dumps({"tracks": [{
            "style": "JDM phonk, cowbell riffs and bass. Chopped vocal samples add a hook.",
            "lyrics": "",
        }]})

    monkeypatch.setattr("vexa_orchestrator.generation._ollama_prompts", prompts)
    brief = _plan_remote_theme("jdm phonk instrumental", count=1)[0]
    assert len(calls) == 1 and "cowbell riffs" in brief.style
    assert "vocal samples" not in brief.style and brief.lyrics == ""
    record = json.loads(next((tmp_path / "var/plans/raw").glob("writer-*.json")).read_text())
    assert "Chopped vocal samples" in record["response"]
    assert record["repairs"][0]["removed_sentences"]


def test_retry_clears_old_planning_error(monkeypatch, tmp_path):
    from vexa_orchestrator.live import LiveSet

    live = LiveSet(SimpleNamespace(library=tmp_path), preview_dir=tmp_path / "previews")
    live.error = "track prompt planning failed: previous description"
    monkeypatch.setattr("vexa_orchestrator.live.threading.Thread", lambda **kw: SimpleNamespace(
        start=lambda: None))
    assert live._queue_generation("jdm phonk instrumental", "test")
    assert live.error is None and live.status()["generating"]
