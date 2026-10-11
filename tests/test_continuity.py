"""Continuity must reflect actual supply and keep stale/late decisions off the decks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from test_contracts import make_asset
from vexa_audio.cues import CueMap
from vexa_audio.engine import AudioEngine, EngineConfig
from vexa_audio.equalizer import EqualizerSettings
from vexa_audio.loader import PreloadedTrack, file_sha256
from vexa_audio.mixer import Mixer
from vexa_audio.preview import PreviewRenderer, PreviewSpec
from vexa_audio.queue import Command, CommandKind
from vexa_contracts import MusicalClock, SessionState
from vexa_orchestrator.continuity import LatencyHistory, forecast, latency_key
from vexa_orchestrator.depot import DepotAsset
from vexa_orchestrator.live import LiveSet, Prepared
from vexa_orchestrator.transitions import TransitionReport


def test_forecast_subtracts_overlaps_deduplicates_and_has_no_cold_eta():
    result = forecast(
        current_remaining_s=100,
        prepared_span_s=160,
        candidate_spans=[
            ("hash1", "family1", 160),
            ("hash1", "family2", 160),
            ("hash2", "family1", 160),
            ("hash3", "family3", 160),
        ],
        supply={},
        preparation={},
        required=2,
    )
    assert result["secured_s"] == 256
    assert result["conditional_inventory_s"] == 568
    assert result["candidate_count"] == 2
    assert result["startup_eta_s"] is None and result["shortfall"] is None


def test_latency_persists_and_profiles_do_not_share_eta(tmp_path):
    path = tmp_path / "latency.sqlite"
    history = LatencyHistory(path)
    key = latency_key({"generator": "minimax", "preset": "balanced"})
    for seconds in (100, 120, 150):
        history.record("supply", key, seconds, success=True)
    history.record("supply", key, 300, success=False)
    measured = LatencyHistory(path).estimate("supply", key)
    assert measured["median_s"] == 220 and measured["failed_samples"] == 1
    assert measured["failure_overhead_s"] == 100
    assert measured["confidence"] == "measured"
    assert history.estimate("supply", latency_key({"generator": "yue2"}))["median_s"] is None
    result = forecast(
        current_remaining_s=20,
        prepared_span_s=0,
        candidate_spans=[],
        supply=measured,
        preparation={"p90_s": 10},
        required=3,
        ready=1,
        elapsed_s=30,
    )
    assert result["startup_eta_s"] == 410 and result["shortfall"]


@pytest.mark.parametrize("incoming_deck", [0, 1])
def test_preview_matches_actual_callback_with_eq_limiter_and_both_directions(
    tmp_path, incoming_deck
):
    sr, blocksize = 8000, 256
    t = np.arange(sr * 5) / sr
    a = (
        np.column_stack([np.sin(2 * np.pi * 100 * t), -np.sin(2 * np.pi * 100 * t)]).astype(
            "float32"
        )
        * 0.8
    )
    b = (
        np.column_stack([np.cos(2 * np.pi * 250 * t), np.cos(2 * np.pi * 250 * t)]).astype(
            "float32"
        )
        * 0.7
    )
    first, second = PreloadedTrack("a", a, sr, 2), PreloadedTrack("b", b, sr, 2)
    eq = EqualizerSettings(True, (8, 3, -2, 0, 0), -1)
    gains = (0.9, 0.8)
    spec = PreviewSpec("test", 1, 1, 0.5)
    renderer = PreviewRenderer(
        sample_rate=sr, blocksize=blocksize, equalizer=eq, gains=gains, out_dir=tmp_path
    )
    preview = renderer.render_transition(
        spec, first, second, out_name="test", outgoing_start_s=0.5, incoming_entry_s=0.25
    )
    actual = Mixer(sample_rate=sr, max_block=blocksize)
    actual.equalizer.configure(eq)
    outgoing = 1 - incoming_deck
    actual.queue.put(
        Command(CommandKind.LOAD_TRACK, deck=outgoing, value=1, value2=sr * 0.5, track=first)
    )
    for deck, gain in [(outgoing, gains[0]), (incoming_deck, gains[1])]:
        actual.queue.put(Command(CommandKind.SET_GAIN, deck=deck, value=gain))
    audio = np.zeros((int(2.5 * sr), 2), dtype="float32")
    for begin, end in [(0, sr), (sr, len(audio))]:
        if begin == sr:
            actual.queue.put(
                Command(
                    CommandKind.START_CROSSFADE,
                    deck=incoming_deck,
                    value=sr * 0.25,
                    value2=sr,
                    track=second,
                )
            )
        for offset in range(begin, end, blocksize):
            count = min(blocksize, end - offset)
            actual.render(audio[offset : offset + count], count)
    expected, _ = sf.read(preview.path, dtype="float32", always_2d=True)
    np.testing.assert_allclose(audio, expected, atol=2e-7)
    assert not actual.decks[outgoing].playing
    assert np.abs(audio).max() <= 0.891001


def make_live(tmp_path):
    sr = 8000
    engine = AudioEngine(EngineConfig(sample_rate=sr, blocksize=256))
    manifest = make_asset(asset_id="current", family_id="current-family")
    current = DepotAsset(manifest, tmp_path / "current.wav", frozenset(), {})
    depot = SimpleNamespace(
        assets={"current": current}, library=tmp_path, search=lambda *a, **k: []
    )
    live = LiveSet(depot, engine, preview_dir=tmp_path / "previews")
    live.state = SessionState(session_id="test", theme="dark", clock=MusicalClock(bpm=122))
    live.current = current
    live.current_track = PreloadedTrack(
        "a" * 64, np.ones((sr * 12, 2), dtype="float32") * 0.1, sr, 2, duration_s=12
    )
    live.cue = CueMap("a" * 64, 2, 122, 0.9, (), 0.9, (), 1, 6, "measured")
    engine.mixer.set_track(0, live.current_track)
    engine.mixer.decks[0].playing = True
    live._queue_generation = lambda *a, **k: False
    live._played_hashes.add(manifest.content_sha256)
    live._played_families.add(manifest.family_id)
    return live


def prepared(live):
    manifest = make_asset(asset_id="next", family_id="next-family", content_sha256="b" * 64)
    asset = DepotAsset(manifest, Path("next.wav"), frozenset(), {})
    track = replace(live.current_track, content_sha256="b" * 64)
    cue = replace(live.cue, source_sha256="b" * 64, entry_s=2)
    report = TransitionReport(
        0.8, 0, 0, -1, Path("preview.wav"), 2, 4, live._dsp_id(), "a" * 64, "b" * 64
    )
    return Prepared(asset, track, cue, report)


def test_commit_uses_measured_entry_and_never_enables_loop(tmp_path):
    live = make_live(tmp_path)
    next_track = prepared(live)
    live._commit(next_track)
    commands = live.engine.queue.drain()
    fade = next(c for c in commands if c.kind is CommandKind.START_CROSSFADE)
    assert fade.value == 2 * live.engine.sample_rate
    assert not any(deck.loop for deck in live.engine.mixer.decks)
    assert live.history[-1]["play_start_s"] == 2


def test_eq_change_invalidates_prepared_audio_before_commit(tmp_path):
    live = make_live(tmp_path)
    next_track = prepared(live)
    live.prepared = next_track
    live.configure_equalizer(EqualizerSettings(True, (1, 0, 0, 0, 0)))
    live._commit(next_track)
    assert not live.engine.queue.drain()
    assert live.prepared is None and live._prepare_cancel.is_set()


def test_empty_depot_finishes_once_then_waits_no_reuse(tmp_path):
    live = make_live(tmp_path)
    deck = live.engine.mixer.decks[0]
    deck.position = live.engine.sample_rate * 7
    live.tick()
    live.tick()
    assert live._preparation_misses == 1
    deck.position = live.current_track.frames
    deck.playing = False
    live.tick()
    if live._prepare_thread:
        live._prepare_thread.join(2)
    assert live._waiting and not deck.loop
    assert live.status()["continuity"]["secured_s"] == 0
    assert len([e for e in live.history if e["event"] == "waiting_for_music"]) == 1


def ready_for_restart(tmp_path):
    live = make_live(tmp_path)
    candidate = prepared(live)
    path = tmp_path / "generated.wav"
    sf.write(path, candidate.track.samples, live.engine.sample_rate, subtype="FLOAT")
    digest = file_sha256(path)
    manifest = candidate.asset.manifest.model_copy(
        update={
            "content_sha256": digest,
            "beat_grid": candidate.asset.manifest.beat_grid.model_copy(update={"bpm": 95.0}),
        }
    )
    asset = replace(candidate.asset, manifest=manifest, path=path)
    replace(candidate.cue, source_sha256=digest, bpm=95).save(path.with_suffix(".cues.json"))
    live.depot.assets[manifest.asset_id] = asset
    batch = live._jobs.open_batch("dark", {"generator": "minimax", "preset": "balanced"}, 2)
    live._jobs.update(batch, 0, state="ready", asset_id=manifest.asset_id, asset_path=str(path))
    live._jobs.update(batch, 1, state="rendering")
    live._generation_batch_id = batch
    live._generating = True
    live._generation_completed = 1
    live._generation_stage = "started"
    deck = live.engine.mixer.decks[0]
    deck.position = live.current_track.frames
    deck.playing = False
    live._waiting = True
    return live, asset


def test_manual_start_plays_incompatible_generated_track_without_cancelling_batch(tmp_path):
    from vexa_contracts import DeckId

    live, asset = ready_for_restart(tmp_path)
    assert not live.filter.build(
        state=live.state, library=[asset.manifest], current_asset=live.current.manifest
    ).transitions
    live.prepared = prepared(live)
    old_prepare_cancel = live._prepare_cancel
    result = live.start_playback()
    assert result["current_asset_id"] == asset.manifest.asset_id
    assert not result["continuity"]["waiting"]
    assert live.engine.clock.bpm == live.state.clock.bpm == 95
    assert live.state.deck(DeckId.B).asset_id == asset.manifest.asset_id
    assert live._generating and not live._render_cancel.is_set()
    assert live._jobs.tracks(live._generation_batch_id)[1]["state"] == "rendering"
    assert live._generation_completed == 1
    assert old_prepare_cancel.is_set() and live.prepared is None
    block = np.zeros((256, 2), dtype="float32")
    live.engine.mixer.render(block, len(block))
    assert np.any(block) and live.engine.mixer.decks[1].position == len(block)
    assert not any(deck.loop for deck in live.engine.mixer.decks)
    assert asset.manifest.content_sha256 in live._played_hashes
    assert asset.manifest.family_id in live._played_families
    with pytest.raises(RuntimeError):
        live.start_playback()


@pytest.mark.parametrize(
    "unavailable", ["played_hash", "played_family", "changed_audio", "rejected"]
)
def test_manual_start_rejects_repeats_and_unverified_audio(tmp_path, unavailable):
    live, asset = ready_for_restart(tmp_path)
    current = live.current
    if unavailable == "played_hash":
        live._played_hashes.add(asset.manifest.content_sha256)
    elif unavailable == "played_family":
        live._played_families.add(asset.manifest.family_id)
    elif unavailable == "changed_audio":
        with asset.path.open("ab") as output:
            output.write(b"changed")
    else:
        live._jobs.update(live._generation_batch_id, 0, state="rejected")
    with pytest.raises(ValueError, match="No unplayed generated track"):
        live.start_playback()
    assert live.current is current and live._waiting
    assert live.state.clock.bpm == 122
    assert live._generating and not live._render_cancel.is_set()
    assert not len(live.engine.queue)


def test_manual_start_refuses_to_interrupt_playing_music(tmp_path):
    live, asset = ready_for_restart(tmp_path)
    live.engine.mixer.decks[0].playing = True
    with pytest.raises(RuntimeError, match="already playing"):
        live.start_playback()
    assert live.current is not asset and not len(live.engine.queue)


def test_manual_start_api_reports_waiting_and_missing_music(tmp_path):
    from fastapi import HTTPException
    from vexa_orchestrator.api import create_app

    live = make_live(tmp_path)
    app = create_app(live=live)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/live/start")
    with pytest.raises(HTTPException) as conflict:
        endpoint()
    assert conflict.value.status_code == 409
    live._waiting = True
    live.engine.mixer.decks[0].playing = False
    with pytest.raises(HTTPException) as missing:
        endpoint()
    assert missing.value.status_code == 422


def test_stop_during_mix_silences_both_decks_but_keeps_generation(tmp_path, monkeypatch):
    live, asset = ready_for_restart(tmp_path)
    monkeypatch.setattr("vexa_orchestrator.live.publish_audio_priority", lambda reason: None)
    live._waiting = False
    live.prepared = prepared(live)
    for deck in live.engine.mixer.decks:
        deck.track = live.current_track
        deck.playing = True
        deck.position = 2000
        deck.fade_remaining = 500
        deck.fade_gain = 0.5
        deck.loop = True
    live.engine.load_and_play(live.prepared.track, 1, fade_frames=500)
    batch_id, state = live._generation_batch_id, live.state
    result = live.stop_playback()
    assert result["continuity"]["stopped"] and result["continuity"]["waiting"]
    assert live._generating and live._generation_batch_id == batch_id and live.state is state
    assert not live._stop.is_set() and not live._render_cancel.is_set()
    assert live._jobs.tracks(batch_id)[1]["state"] == "rendering"
    assert live._prepare_cancel.is_set() and live.prepared is None
    assert not len(live.engine.queue)
    assert all(
        not d.playing and d.position == d.fade_remaining == 0 and not d.loop
        for d in live.engine.mixer.decks
    )
    block = np.zeros((256, 2), dtype="float32")
    live.engine.mixer.render(block, len(block))
    assert not np.any(block)
    # A ready track arriving after STOP cannot trigger preparation or automatic playback.
    live.tick()
    assert live._prepare_thread is None and not len(live.engine.queue)
    assert asset.manifest.asset_id in live.depot.assets


def test_start_after_stop_restarts_current_track_without_cancelling_batch(tmp_path, monkeypatch):
    live, _ = ready_for_restart(tmp_path)
    monkeypatch.setattr("vexa_orchestrator.live.publish_audio_priority", lambda reason: None)
    live.stop_playback()
    monkeypatch.setattr(live.engine, "start", live.engine.mixer.start)
    result = live.start_playback()
    assert not result["continuity"]["stopped"] and not result["continuity"]["waiting"]
    assert live._generating and not live._render_cancel.is_set()
    block = np.zeros((256, 2), dtype="float32")
    live.engine.mixer.render(block, len(block))
    assert np.any(block) and live.engine.mixer.decks[live.active_deck].position == len(block)
    assert not live.engine.mixer.decks[1 - live.active_deck].playing
    assert live.history[-1]["event"] == "playback_restart"


def test_stop_before_first_track_prevents_autostart_until_start(tmp_path, monkeypatch):
    live, asset = ready_for_restart(tmp_path)
    monkeypatch.setattr("vexa_orchestrator.live.publish_audio_priority", lambda reason: None)
    live.state = live.current = live.current_track = live.cue = None
    live._pending_start = ("dark", "test")
    live.stop_playback()
    snapshot = {"controls": live._session_controls.model_dump()}
    rows = live._jobs.tracks(live._generation_batch_id)
    live._start_prepared_batch("dark", snapshot, rows, {asset.manifest.asset_id})
    assert live.state is None and live._pending_start == ("dark", "test")
    assert live._generating and not live._render_cancel.is_set()
    assert live._startup_verified_count == 1
    monkeypatch.setattr(
        live.engine,
        "start",
        lambda: setattr(live.engine, "_stream", SimpleNamespace(latency=0.0, cpu_load=0.0)),
    )
    monkeypatch.setattr(live, "_run", lambda: None)
    result = live.start_playback()
    assert result["running"] and not result["continuity"]["stopped"]
    assert live.current is asset and live._pending_start is None and live._generating


def test_start_after_early_stop_without_ready_tracks_stays_stopped(tmp_path, monkeypatch):
    live = make_live(tmp_path)
    monkeypatch.setattr("vexa_orchestrator.live.publish_audio_priority", lambda reason: None)
    live.state = live.current = live.current_track = live.cue = None
    live._pending_start = ("dark", "test")
    live._generating = True
    live.stop_playback()
    with pytest.raises(ValueError, match="No verified generated track"):
        live.start_playback()
    assert live._playback_stopped and live._pending_start == ("dark", "test")
    assert live._generating and not live._render_cancel.is_set()


def test_stop_api_only_stops_playback_full_shutdown_still_cancels(tmp_path, monkeypatch):
    from vexa_orchestrator.api import create_app

    live, _ = ready_for_restart(tmp_path)
    monkeypatch.setattr("vexa_orchestrator.live.publish_audio_priority", lambda reason: None)
    app = create_app(live=live)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/live/stop")
    assert endpoint()["continuity"]["stopped"]
    assert live._generating and live.state is not None and not live._render_cancel.is_set()
    live.stop()
    assert live._render_cancel.is_set() and live._stop.is_set()
    assert live.state is None and not live._generating and not live._playback_stopped


def test_stale_preparation_is_not_published(monkeypatch, tmp_path):
    live = make_live(tmp_path)
    next_track = prepared(live)
    live.depot.assets["next"] = next_track.asset
    live.depot.search = lambda *a, **k: [(1, next_track.asset)]
    live._cue = lambda asset: next_track.cue

    def prepare(*args, **kwargs):
        live.state.generation += 1
        return next_track.track, next_track.cue, next_track.report

    monkeypatch.setattr("vexa_orchestrator.live.prepare_candidate", prepare)
    live._prepare(0)
    assert live.prepared is None


def test_expired_preparation_does_not_launch_worker(monkeypatch, tmp_path):
    import time

    live = make_live(tmp_path)
    next_track = prepared(live)
    live.depot.search = lambda *a, **k: [(1, next_track.asset)]
    live._cue = lambda asset: next_track.cue
    live._prepare_deadline = time.monotonic() - 1

    def forbidden(*a, **k):
        pytest.fail("late worker launched")

    monkeypatch.setattr("vexa_orchestrator.live.prepare_candidate", forbidden)
    live._prepare(0)
    assert live.prepared is None


def test_buffer_policy_uses_measured_seconds_not_only_track_count(tmp_path):
    from vexa_contracts import GenerationSettings

    live = make_live(tmp_path)
    controls = GenerationSettings(track_count=4, start_policy="buffer", buffer_tracks=2)
    snapshot = {"generator": "minimax", "preset": "balanced", "controls": controls.model_dump()}
    key = latency_key(snapshot)
    for seconds in (350, 360, 370):
        live._latencies.record("supply", key, seconds, success=True)
    rows = []
    for index in range(3):
        identity = f"buffer-{index}"
        asset = DepotAsset(
            make_asset(asset_id=identity, family_id=identity),
            tmp_path / f"{identity}.wav",
            frozenset(),
            {},
        )
        live.depot.assets[identity] = asset
        replace(live.cue, entry_s=5, exit_s=145).save(asset.path.with_suffix(".cues.json"))
        rows.append({"asset_id": identity, "state": "ready"})
    starts = []
    live.start = lambda theme, **kwargs: starts.append(kwargs)
    live._pending_start = ("dark", "test")
    live._start_prepared_batch("dark", snapshot, rows[:2], {r["asset_id"] for r in rows[:2]})
    assert not starts and live._startup_required == 3
    live._start_prepared_batch("dark", snapshot, rows, {r["asset_id"] for r in rows})
    assert starts and live._startup_buffer_s == 429


def test_pcm_cache_repairs_changed_file_instead_of_reusing_it(tmp_path):
    from vexa_orchestrator.prepared_audio import persist_pcm

    live = make_live(tmp_path)
    track = live.current_track
    path = persist_pcm(track, tmp_path)
    original = path.read_bytes()
    path.write_bytes(b"changed")
    assert persist_pcm(track, tmp_path).read_bytes() == original


def test_callback_deadline_misses_are_distinct_from_mixer_errors(monkeypatch):
    mixer = Mixer(sample_rate=8000, max_block=256)
    times = iter([1.0, 1.1])
    monkeypatch.setattr("vexa_audio.mixer.time.perf_counter", lambda: next(times))
    mixer.render(np.zeros((256, 2), dtype="float32"), 256)
    assert mixer.stats.deadline_misses == 1 and mixer.stats.underruns == 0


def test_optional_observers_defer_only_while_audio_lease_is_active(monkeypatch, tmp_path):
    from vexa_orchestrator import audio_priority
    from vexa_orchestrator.semantic_audio import observe

    monkeypatch.setattr(audio_priority, "DIRECTORY", tmp_path)
    monkeypatch.setenv("VEXA_CLAP_MODEL", str(tmp_path))
    clock = [100.0]
    monkeypatch.setattr(audio_priority.time, "time", lambda: clock[0])
    audio_priority.publish("imminent deck preparation")
    result = observe(tmp_path / "absent.wav", {})
    assert result["status"] == "deferred" and result["genre"] is None
    clock[0] = 111.0
    assert audio_priority.deferred_reason() is None
    audio_priority.publish(None)
    assert not list(tmp_path.glob("*.json"))


def test_theme_steer_retains_only_prepared_track_in_new_scope(tmp_path):
    live = make_live(tmp_path)
    pending = prepared(live)
    live.prepared = pending
    live._prepared_loaded = True
    live.depot.search = lambda *a, **k: [(1, pending.asset)]
    live.steer("warm")
    assert live.prepared is pending and live._prepared_loaded
    assert any(e["event"] == "prepared_retained" for e in live.history)
    live.depot.search = lambda *a, **k: []
    live.steer("phonk")
    assert live.prepared is None and not live._prepared_loaded


def test_commit_rechecks_quality_even_if_preview_has_high_score(tmp_path):
    from vexa_contracts import ReadinessState

    live = make_live(tmp_path)
    pending = prepared(live)
    pending.asset.manifest.readiness = ReadinessState.UNANALYZED
    live._commit(pending)
    assert not live.engine.queue.drain()


def test_overdue_eta_becomes_unknown_instead_of_counting_down_to_zero():
    result = forecast(
        current_remaining_s=0,
        prepared_span_s=0,
        candidate_spans=[],
        supply={"median_s": 100, "p90_s": 120},
        preparation={},
        required=2,
        elapsed_s=150,
    )
    assert result["eta_overdue"] and result["startup_eta_s"] is None


def test_session_deck_activity_follows_callback_state(tmp_path):
    from vexa_contracts import DeckId

    live = make_live(tmp_path)
    live._prepared_for = ("current", live.state.generation)
    live.engine.mixer.decks[0].playing = True
    live.engine.mixer.decks[1].playing = False
    live.tick()
    assert live.state.deck(DeckId.A).playing and not live.state.deck(DeckId.B).playing
    live.engine.mixer.decks[0].playing = False
    live.tick()
    assert not live.state.deck(DeckId.A).playing


def test_backend_preference_boilerplate_does_not_poison_depot_scope(tmp_path):
    from vexa_contracts import Estimate, GenerationSettings
    from vexa_orchestrator.depot import Depot
    from vexa_orchestrator.generation_settings import conditioning_theme

    depot = Depot(tmp_path)
    depot.add(
        make_asset(tags={"mood": [Estimate(value="dark", confidence=0.9)]}), tmp_path / "dark.wav"
    )
    plain = depot.search("dark")
    conditioned = depot.search(
        conditioning_theme("dark", GenerationSettings(bpm=128, language="en"))
    )
    assert len(plain) == len(conditioned) == 1
    assert plain[0][0] == conditioned[0][0]
