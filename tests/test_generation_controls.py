"""Settings survive restart and reach planning/rendering without rewriting active batches."""

import json
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_contracts import make_asset
from vexa_contracts import BeatGrid, GenerationSettings, MusicalClock, SessionState
from vexa_orchestrator.api import create_app
from vexa_orchestrator.generation import TrackBrief, generate_theme
from vexa_orchestrator.generation_runner import GenerationRunner
from vexa_orchestrator.generation_settings import conditioning_theme
from vexa_orchestrator.generation_v2 import build_plan, write_plan
from vexa_orchestrator.generators import components, render_alternative
from vexa_orchestrator.live import LiveSet


def make_live(tmp_path):
    return LiveSet(SimpleNamespace(library=tmp_path, assets={}), preview_dir=tmp_path / "previews")


@pytest.mark.parametrize(
    "values",
    [
        {"track_count": 11},
        {"duration_s": 90},
        {"duration_s": float("nan")},
        {"bpm": 230},
        {"plan_attempts": 99},
        {"vocal_mode": "samples"},
        {"start_policy": "random"},
        {"include_instruments": ["piano"], "exclude_instruments": ["Piano"]},
        {"vocal_mode": "instrumental", "custom_lyrics": "verse"},
        {"unsupported": True},
    ],
)
def test_invalid_controls_are_rejected(values):
    with pytest.raises(ValidationError):
        GenerationSettings.model_validate(values)


def test_preferences_override_theme_in_backend_forms_and_native_caption(tmp_path):
    controls = GenerationSettings(
        track_count=2,
        duration_s=180,
        genre="warm house",
        vocal_mode="instrumental",
        bpm=118,
        energy_arc="steady",
        energy=0.4,
        texture="dusty cassette",
        groove="shuffled hats",
    )
    theme = conditioning_theme("dark techno with vocals, 150 BPM", controls)
    assert conditioning_theme(theme, controls) == theme
    plan = build_plan(theme, 1, "minimax", controls=controls.model_dump())
    assert plan.intent.genre == "warm house" and plan.intent.bpm == 118
    assert plan.intent.vocal_mode == "instrumental"
    assert plan.duration_s == 180 and plan.energy == 0.4
    assert plan.sections[-1].end_s == 180
    brief = write_plan(plan, {"mode": "template", "model": "none"}, tmp_path / "draft.json")
    assert "dusty cassette" in brief.music_plan["minimax_caption"]
    assert "118 BPM" in brief.music_plan["minimax_caption"] and brief.lyrics == ""
    with pytest.raises(ValueError, match="track count"):
        build_plan(theme, 2, "minimax", controls=controls.model_dump())


@pytest.mark.parametrize(
    "genre, mode", [("90s underground rap", "rap"), ("memphis phonk", "instrumental")]
)
def test_genre_override_resolves_canonical_subgenre_and_auto_voice(genre, mode):
    plan = build_plan("a dark night", 0, "minimax", controls={"genre": genre})
    assert plan.intent.vocal_mode == mode
    assert "hip hop" in plan.intent.genre if mode == "rap" else plan.intent.genre == "Memphis phonk"


def original_rap():
    return "\n".join(
        [
            "[Verse 1]",
            *[f"Night bus takes the factory workers home after shift {i}" for i in range(16)],
            "[Refrain]",
            *[f"Rent keeps rising while we work the night shift {i}" for i in range(4)],
            "[Verse 2]",
            *[f"Morning light reveals the unpaid wages on my street {i}" for i in range(16)],
        ]
    )


def test_custom_lyrics_skip_lyric_writer_and_keep_instrumental_sections(tmp_path):
    controls = GenerationSettings(
        vocal_mode="rap", custom_lyrics=original_rap(), lyric_topic="night buses"
    )
    plan = build_plan("90s underground rap", 0, "yue2", controls=controls.model_dump())
    brief = write_plan(plan, {"mode": "template", "model": "none"}, tmp_path / "draft.json")
    assert "Night bus" in brief.lyrics and brief.lyrics.count("[Refrain]") == 2
    assert "[Intro]\n\n[Verse 1]" in brief.lyrics
    with pytest.raises(ValueError, match="heading"):
        build_plan("rap", 0, "yue2", controls={"custom_lyrics": "unstructured original words"})


def test_settings_persist_and_do_not_change_queued_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    live = make_live(tmp_path)
    live._queued_snapshot = live._new_snapshot()
    saved = json.dumps(live._queued_snapshot, sort_keys=True)
    marker = object()
    live.prepared = marker
    live.configure_generation_settings(GenerationSettings(track_count=3, duration_s=150, seed=234))
    assert json.dumps(live._queued_snapshot, sort_keys=True) == saved
    assert live.prepared is marker and not live._render_cancel.is_set()
    restarted = make_live(tmp_path)
    assert restarted.generation_settings()["selected"]["track_count"] == 3
    assert restarted.generation_settings()["selected"]["seed"] == 234


@pytest.mark.parametrize("policy, required", [("depot", 1), ("buffer", 2), ("complete", 3)])
def test_wait_policy_starts_only_from_required_prepared_tracks(tmp_path, policy, required):
    live = make_live(tmp_path)
    controls = GenerationSettings(track_count=3, start_policy=policy, buffer_tracks=2)
    snapshot = {"controls": controls.model_dump()}
    live._pending_start = ("techno", "test")
    starts = []
    live.start = lambda theme, **kw: starts.append(kw)
    rows = [{"state": "ready", "asset_id": f"track-{i}"} for i in range(required - 1)]
    verified = {row["asset_id"] for row in rows}
    live._start_prepared_batch("techno", snapshot, rows, verified)
    assert not starts
    rows.append({"state": "ready", "asset_id": "last"})
    verified.add("last")
    live._start_prepared_batch("techno", snapshot, rows, verified)
    assert starts[0]["prepared_ids"] == [row["asset_id"] for row in rows]
    assert live._pending_start is None


@pytest.mark.parametrize("policy, required", [("buffer", 2), ("complete", 3)])
def test_resumed_startup_waits_for_verified_assets(tmp_path, policy, required):
    live = make_live(tmp_path)
    controls = GenerationSettings(track_count=3, start_policy=policy, buffer_tracks=2)
    snapshot = {"controls": controls.model_dump()}
    live._queued_snapshot = snapshot
    live._generation_completed = 3
    live._pending_start = ("techno", "resumed")
    starts = []
    live.start = lambda theme, **kw: starts.append(kw)
    rows = [{"state": "ready", "asset_id": f"track-{i}"} for i in range(3)]
    live._start_prepared_batch("techno", snapshot, rows, {"track-0"})
    assert not starts
    assert live.status()["startup"]["ready"] == 1
    verified = {f"track-{i}" for i in range(required)}
    live._start_prepared_batch("techno", snapshot, rows, verified)
    assert starts[0]["prepared_ids"] == [f"track-{i}" for i in range(required)]


def test_live_recovery_does_not_start_complete_batch_with_changed_track(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    live = make_live(tmp_path)
    live._session_controls = GenerationSettings(track_count=2, start_policy="complete")
    snapshot = live._new_snapshot()
    batch = live._jobs.open_batch("techno", snapshot, 2)
    for slot in range(2):
        live._jobs.update(
            batch,
            slot,
            state="ready",
            asset_id=f"track-{slot}",
            asset_path=str(tmp_path / f"track-{slot}.wav"),
        )
    verified = []
    live.depot.add = lambda manifest, path: verified.append(manifest.asset_id)
    recovered = SimpleNamespace(
        path=tmp_path / "track-0.wav", manifest=SimpleNamespace(asset_id="track-0")
    )
    monkeypatch.setattr(
        "vexa_orchestrator.generation_runner.recovered_asset",
        lambda path: recovered if path == recovered.path else None,
    )
    starts = []
    live.start = lambda theme, **kw: starts.append(kw)
    live._queued_snapshot = snapshot
    live._resume_batch_id = batch
    live._generation_theme = "techno"
    live._pending_start = ("techno", "resumed")
    live._generating = True
    live._generate()
    assert verified == ["track-0"] and not starts
    assert [row["state"] for row in live._jobs.tracks(batch)] == ["ready", "rejected"]
    assert live.status()["startup"] == {
        "policy": "complete",
        "required": 2,
        "ready": 1,
        "buffer_s": 0.0,
    }


def test_cancel_generation_does_not_stop_playback(tmp_path):
    live = make_live(tmp_path)
    live.state = SessionState(session_id="playing", theme="techno", clock=MusicalClock(bpm=128))
    live._generating = True
    live.cancel_generation()
    assert live._render_cancel.is_set() and not live._stop.is_set()
    assert live.state.session_id == "playing"


def test_cancel_before_worker_enters_queue_is_not_lost(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    live = make_live(tmp_path)
    live._generation_theme = "techno"
    live._queued_snapshot = live._new_snapshot()
    live._generating = True
    live.cancel_generation()
    live._generate()
    assert not live._generating and live._render_cancel.is_set()
    assert not live._jobs.batches()


def test_runner_uses_frozen_controls_and_replacement_has_new_render_id(tmp_path):
    live = make_live(tmp_path)
    controls = GenerationSettings(track_count=1, duration_s=150, seed=123, render_attempts=2)
    snapshot = {
        "library": str(tmp_path.resolve()),
        "generator": "yue2",
        "preset": "q8",
        "writer": {"mode": "template", "model": "none"},
        "controls": controls.model_dump(),
    }
    batch = live._jobs.open_batch("techno instrumental", snapshot, 1)
    rendered, planned = [], []

    def planner(theme, slot, generator, **kw):
        planned.append(kw)
        return TrackBrief("techno", kw["controls"]["duration_s"], 0.5, "fixture")

    def renderer(*args, **kw):
        rendered.append(kw)
        if len(rendered) == 1:
            raise ValueError("fixture render failure")
        return SimpleNamespace(
            path=tmp_path / f"{kw['render_id']}.wav",
            manifest=SimpleNamespace(asset_id=kw["render_id"]),
        )

    def run():
        GenerationRunner(
            live._jobs,
            batch,
            tmp_path,
            planner=planner,
            renderer=renderer,
            admitted=lambda *a: None,
            event=lambda *a: None,
            cancel=threading.Event(),
        ).run()

    run()
    assert [kw["seed"] for kw in rendered] == [123, 1132]
    assert rendered[0]["duration_s"] == 150 and planned[0]["settings"]["seed"] == 123
    published_id = live._jobs.tracks(batch)[0]["asset_id"]
    original = tmp_path / f"{published_id}.wav"
    original.write_bytes(b"original playing audio")
    live._jobs.reset_slot(batch, 0, replace=True)
    run()
    assert live._jobs.tracks(batch)[0]["asset_id"] != published_id
    assert planned[1]["settings"]["seed"] != planned[0]["settings"]["seed"]
    assert original.read_bytes() == b"original playing audio"


@pytest.mark.parametrize(
    "speed,steps", [(None, 30), ("fast", 12), ("balanced", 20), ("quality", 30)]
)
def test_native_adapter_uses_requested_seed(monkeypatch, tmp_path, speed, steps):
    monkeypatch.setenv("VEXA_MINIMAX_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("VEXA_MINIMAX_BINARY", str(tmp_path / "mm-synth"))
    for name, path in components("minimax", "balanced").items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        if name == "binary":
            path.chmod(0o755)
    monkeypatch.setattr("vexa_orchestrator.generators.run_renderer", lambda *a: None)
    wav, request = tmp_path / "track.wav", tmp_path / "request.json"
    receipt = render_alternative(
        "minimax",
        "balanced",
        wav,
        request,
        style="Techno instrumental",
        lyrics="",
        duration_s=165,
        music_plan={"generation_settings": {"render_speed": speed}} if speed else None,
        seed=777,
    )
    assert json.loads(request.read_text())["seed"] == 777
    assert receipt["seed"] == 777
    assert json.loads(request.read_text())["steps"] == steps
    assert receipt["denoising_steps"] == steps


def test_render_speed_is_frozen_in_compiled_plan(tmp_path):
    controls = GenerationSettings(render_speed="fast")
    plan = build_plan("dark trap instrumental", 0, "minimax", controls=controls.model_dump())
    brief = write_plan(plan, {"mode": "template", "model": "none"}, tmp_path / "draft.json")
    assert brief.music_plan["generation_settings"]["render_speed"] == "fast"
    with pytest.raises(ValidationError):
        GenerationSettings(render_speed="instant")


@pytest.mark.parametrize("bpm, accepted", [(128, True), (64, True), (256, True), (139, False)])
def test_strict_tempo_gate_uses_measured_pulse_before_publication(
    monkeypatch, tmp_path, bpm, accepted
):
    monkeypatch.setenv("VEXA_CLAP_MODEL", "")
    manifest = make_asset(beat_grid=BeatGrid(bpm=bpm, grid_version=1, confidence=0.9))
    manifest.audio.duration_s = 165

    def render(generator, preset, wav, *args, **kw):
        import numpy as np
        import soundfile as sf

        samples = 0.1 * np.sin(2 * np.pi * 100 * np.arange(48000 * 165) / 48000)
        sf.write(wav, np.stack([samples, samples], axis=1), 48000)
        return {"generator": generator, "seed": kw["seed"], "preset": preset}

    monkeypatch.setattr(
        "vexa_orchestrator.generators.render_alternative",
        render,
    )
    monkeypatch.setattr("vexa_orchestrator.generation.master", lambda *a: None)

    def analyse(*args, **kwargs):
        manifest.provenance = kwargs["provenance"]
        return SimpleNamespace(admitted=True, manifest=manifest)

    monkeypatch.setattr("vexa_orchestrator.generation.analyse", analyse)
    monkeypatch.setattr(
        "vexa_orchestrator.generation.analyse_cues",
        lambda *a, **kw: SimpleNamespace(
            entry_s=8,
            exit_s=150,
            save=lambda *a: None,
        ),
    )
    args = {
        "generator": "minimax",
        "style": "Techno instrumental",
        "seed": 567,
        "music_plan": {"bpm": 128, "generation_settings": {"tempo_policy": "strict"}},
    }
    if accepted:
        asset = generate_theme("techno instrumental", tmp_path, **args)
        assert asset.path.with_suffix(".json").exists()
        assert asset.manifest.provenance.seed == 567
    else:
        with pytest.raises(ValueError, match="strict target"):
            generate_theme("techno instrumental", tmp_path, **args)
        assert not list(tmp_path.glob("live-????????????.json"))


def test_api_settings_preview_reports_and_idle_action_guards(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    live = make_live(tmp_path)
    with TestClient(create_app(live=live)) as client:
        assert client.post("/generation/settings", json={"track_count": 90}).status_code == 422
        body = client.post(
            "/generation/settings",
            json={"track_count": 2, "duration_s": 150, "vocal_mode": "instrumental"},
        ).json()
        assert body["selected"]["track_count"] == 2
        preview = client.post("/generation/preview", json={"theme": "techno"}).json()
        assert preview["plan"]["duration_s"] == 150
        snapshot = live._new_snapshot()
        batch = live._jobs.open_batch("techno", snapshot, 2)
        plan = build_plan("techno", 0, "yue2", controls=snapshot["controls"])
        brief = write_plan(plan, {"mode": "template", "model": "none"}, tmp_path / "cache.json")
        path = tmp_path / "ready.wav"
        path.with_suffix(".json").write_text(make_asset().model_dump_json())
        path.with_suffix(".cues.json").write_text("{}")
        live._jobs.update(batch, 0, state="ready", brief=asdict(brief), asset_path=str(path))
        report = client.get(f"/generation/batches/{batch}/tracks/1").json()
        assert report["requested"]["duration_s"] == 150
        assert report["measured"]["duration_s"] is not None
        assert report["measured"]["vocal_mode"] is None and report["measured"]["energy"] is None
        assert client.get(f"/generation/batches/{batch}/tracks/0").status_code == 404
        live._generating = True
        assert client.post(f"/generation/batches/{batch}/tracks/1/replace").status_code == 409
        live._generating = False
