"""Generator validation, request routing and same-theme cancellation preserve playback."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from vexa_contracts import MusicalClock, SessionState
from vexa_orchestrator.api import create_app
from vexa_orchestrator.generators import (
    catalog,
    components,
    minimax_caption,
    render_alternative,
    selection,
)
from vexa_orchestrator.live import LiveSet
from vexa_yue2.backends import BackendUnavailable


def installed_minimax(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_MINIMAX_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("VEXA_MINIMAX_BINARY", str(tmp_path / "mm-synth"))
    for name, path in components("minimax", "balanced").items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        if name == "binary":
            path.chmod(0o755)


def test_catalog_validates_each_preset_and_rejects_unknown_models(monkeypatch, tmp_path):
    installed_minimax(monkeypatch, tmp_path)
    row = next(row for row in catalog() if row["id"] == "minimax")
    assert [p["available"] for p in row["presets"]] == [True, False, False]
    assert selection("minimax", "balanced", require_available=True) == ("minimax", "balanced")
    with pytest.raises(ValueError):
        selection("../../binary")
    with pytest.raises(ValueError):
        selection("acestep", "balanced")
    with pytest.raises(BackendUnavailable):
        selection("minimax", "quality", require_available=True)


def test_native_timing_report_does_not_confuse_replay_with_fresh_generation(tmp_path):
    from vexa_orchestrator.generators import minimax_timings

    log = tmp_path / "native.log"
    log.write_text(
        "[AR] Replay: 4101 frames from audio_codes (164.0s of music), 20.5 s\n"
        "[DiT] CFG=1.70, 41 windows, 100.5 s\n"
        "[VAE] Decode: 41 windows -> 164.2s of audio, 10129 ms\n"
        "[Done] 132.4 s total\n"
    )
    assert minimax_timings(log) == {
        "replay_s": 20.5,
        "dit_s": 100.5,
        "vae_s": 10.129,
        "total_s": 132.4,
    }


def test_minimax_routes_exact_components_and_preserves_replay(monkeypatch, tmp_path):
    installed_minimax(monkeypatch, tmp_path)
    wav = tmp_path / "track.wav"
    request = tmp_path / "track.request.json"

    def render(command, binary, log, cancel, duration_s):
        payload = json.loads(request.read_text())
        assert payload["lm_model"].endswith("Q6_K.gguf")
        assert payload["dit_model"].endswith("Q8_0.gguf")
        assert payload["lyrics"] == "[Instrumental]"
        assert payload["output_format"] == "wav16"
        assert payload["duration"] == duration_s == 165
        assert payload["min_duration"] == 164
        assert command[command.index("--out") + 1] == str(wav)
        wav.write_bytes(b"fixture")
        wav.with_suffix(".json").write_text('{"audio_codes":"123"}')

    monkeypatch.setattr("vexa_orchestrator.generators.run_renderer", render)
    result = render_alternative(
        "minimax",
        "balanced",
        wav,
        request,
        style="Instrumental drift phonk, cowbell, distorted 808 bass",
        lyrics="",
        duration_s=165,
        music_plan={"bpm": 140},
    )
    assert result["generator"] == "minimax" and result["seed"] >= 0
    assert wav.with_suffix(".replay.json").is_file()
    assert not wav.with_suffix(".json").exists()


def test_memphis_structured_caption_reaches_minimax_without_a_second_generic_arrangement():
    from tools.check_acestep_audio import CASES

    bpm, key, caption = CASES["memphis"]
    result = minimax_caption(caption, "", {"bpm": bpm, "key": key})
    assert result == caption and result.count("Arrangement\n") == 1
    assert "88 BPM" in result and "cassette" in result and "cowbell" not in result


def make_live(tmp_path):
    return LiveSet(
        SimpleNamespace(library=tmp_path, assets={}, search=lambda *a, **k: []),
        preview_dir=tmp_path / "previews",
    )


def test_api_rejects_unavailable_selection_before_creating_session(monkeypatch, tmp_path):
    live = make_live(tmp_path)
    monkeypatch.setenv("VEXA_MINIMAX_MODELS", str(tmp_path / "absent"))
    app = create_app(live=live)
    with TestClient(app) as client:
        assert client.get("/generation/models").status_code == 200
        response = client.post("/sessions", json={"theme": "techno", "generator": "minimax"})
        assert response.status_code == 503
        assert app.state.store.sessions == {}
        assert live.generator == "yue2"
        response = client.post("/sessions", json={"theme": "techno", "generator": "unknown"})
        assert response.status_code == 422


@pytest.mark.parametrize("phase", ["planning", "rendering"])
def test_same_theme_generator_switch_discards_stale_work(monkeypatch, tmp_path, phase):
    installed_minimax(monkeypatch, tmp_path)
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    live = make_live(tmp_path)
    live.state = SessionState(session_id="test", theme="techno", clock=MusicalClock(bpm=128))
    marker = SimpleNamespace(
        asset=SimpleNamespace(
            manifest=SimpleNamespace(asset_id="prepared", title="Prepared"),
            path=tmp_path / "prepared.wav",
        ),
        track=SimpleNamespace(duration_s=165),
    )
    live.prepared = marker
    begun, release = threading.Event(), threading.Event()
    accepted, calls = [], []
    live.depot.add = lambda manifest, path: accepted.append(manifest.asset_id)
    from vexa_orchestrator.generation_v2 import plan_track

    first = True

    def plan(theme, slot, generator, cancel, **kwargs):
        nonlocal first
        if phase == "planning" and first:
            first = False
            begun.set()
            assert release.wait(3)
            raise RuntimeError("cancelled old planner")
        return plan_track(theme, slot, generator, cancel=cancel, **kwargs)

    def generate(theme, library, **kw):
        calls.append(kw["generator"])
        identity = "old" if kw["generator"] == "yue2" else f"new-{len(calls)}"
        path = tmp_path / f"{identity}.wav"
        path.with_suffix(".json").write_text("{}")
        if phase == "rendering" and kw["generator"] == "yue2":
            begun.set()
            assert release.wait(3)
            assert kw["cancel"].is_set()
        return SimpleNamespace(path=path, manifest=SimpleNamespace(asset_id=identity))

    monkeypatch.setattr("vexa_orchestrator.live.plan_track", plan)
    monkeypatch.setattr("vexa_orchestrator.live.generate_theme", generate)
    try:
        live._queue_generation("techno", "test")
        assert begun.wait(3)
        live.configure_generator("minimax", "balanced")
        live.steer("techno")
        assert live.prepared is marker  # generator-only changes preserve prepared deck audio
        release.set()
        live._generation_thread.join(5)
        assert not live._generation_thread.is_alive()
        assert len(accepted) == 10 and "old" not in accepted
        assert calls[-10:] == ["minimax"] * 10
        assert live.error is None
        if phase == "rendering":
            assert (tmp_path / "old.stale.json").exists()
            assert not (tmp_path / "old.json").exists()
        assert LiveSet(live.depot, preview_dir=tmp_path / "previews").generator == "minimax"
    finally:
        release.set()
        live.prepared = None
        live.stop()


def test_api_accepts_selected_model_and_reports_it_without_rendering(monkeypatch, tmp_path):
    installed_minimax(monkeypatch, tmp_path)
    live = make_live(tmp_path)

    def queue(theme, session_id, **kw):
        live._pending_start = (theme, session_id)
        live._generation_theme = theme
        live._generating = True
        return True

    monkeypatch.setattr(live, "_queue_generation", queue)
    with TestClient(create_app(live=live)) as client:
        response = client.post(
            "/sessions",
            json={
                "theme": "techno",
                "session_id": "model-test",
                "generator": "minimax",
                "generator_preset": "balanced",
            },
        )
        assert response.status_code == 200 and response.json()["generating"]
        status = client.get("/live").json()
        assert status["generation"]["generator"] == "minimax"
        assert status["generation"]["preset"] == "balanced"
        rejected = client.post(
            "/sessions/model-test/theme",
            json={"theme": "changed", "generator": "minimax", "generator_preset": "quality"},
        )
        assert rejected.status_code == 503
        assert live._pending_start == ("techno", "model-test")


def test_managed_install_and_frozen_components_override_later_configuration(monkeypatch, tmp_path):
    import json

    from vexa_orchestrator import generators

    monkeypatch.setattr(generators, "ROOT", tmp_path)
    names = generators._legacy_components("minimax", "balanced")
    frozen = {}
    for key in names:
        path = tmp_path / "verified" / key
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture")
        if key == "binary":
            path.chmod(0o755)
        frozen[key] = str(path)
    registry = tmp_path / "var/config/model-registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"minimax:balanced": {"preferred": True, "components": frozen}}))
    monkeypatch.setenv("VEXA_MINIMAX_MODELS", str(tmp_path / "older-install"))
    assert {k: str(v) for k, v in generators.components("minimax", "balanced").items()} == frozen
    monkeypatch.setattr(generators, "components", lambda *a: {})
    assert {
        k: str(v) for k, v in generators.resolve_components("minimax", "balanced", frozen).items()
    } == frozen


def test_yue_render_uses_managed_resolver(monkeypatch, tmp_path):
    import json

    from vexa_orchestrator import generation, generators

    monkeypatch.setattr(generators, "ROOT", tmp_path)
    paths = {}
    for key in ("binary", "model", "vae"):
        path = tmp_path / "managed" / key
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture")
        path.chmod(0o755)
        paths[key] = str(path)
    registry = tmp_path / "var/config/model-registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"yue2:q8": {"preferred": True, "components": paths}}))
    monkeypatch.setattr(generation, "_gpu_index", lambda: "fixture-gpu")
    commands = []

    def stop_before_render(command, **kwargs):
        commands.append(command)
        raise AssertionError("captured render command")

    monkeypatch.setattr(generation.subprocess, "Popen", stop_before_render)
    with pytest.raises(AssertionError, match="captured render"):
        generation.generate_theme(
            "instrumental techno",
            tmp_path / "library",
            style="instrumental techno",
            generator="yue2",
            preset="q8",
        )
    assert commands and all(path in commands[0] for path in paths.values())
