"""Bridge tests: scheduler decisions become audio.

This is the seam that had no coverage because nothing connected the two halves. These tests run
**offline** — no device, no sound — by driving the mixer directly, which is exactly how a
scheduled transition would be verified without disturbing anyone.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from vexa_audio.bridge import AudioBridge, BridgeConfig
from vexa_audio.engine import AudioEngine, EngineConfig
from vexa_contracts import (
    ApprovalState,
    AssetManifest,
    AudioProperties,
    BeatGrid,
    QualityReport,
    ReadinessState,
    ScheduledAction,
    SectionMarker,
    SessionState,
    SourceType,
)

SR = 44100
PASSING = QualityReport(
    decode_ok=True, duration_ok=True, loudness_ok=True,
    beat_grid_ok=True, loop_boundary_ok=True, audio_quality_ok=True,
)


def make_asset(tmp_path, asset_id: str, *, bpm: float = 120.0, seconds: float = 16.0):
    """A real file plus the manifest that describes it, with matching hashes."""
    import soundfile as sf

    path = tmp_path / f"{asset_id}.wav"
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    tone = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    sf.write(str(path), tone, SR, subtype="PCM_16")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = AssetManifest(
        asset_id=asset_id,
        family_id=f"family_{asset_id}",
        source_type=SourceType.YUE2_RENDER,
        content_sha256=digest,
        audio=AudioProperties(codec="wav", sample_rate_hz=SR, channels=2, duration_s=seconds,
                              integrated_lufs=-18.0, true_peak_dbtp=-3.0),
        beat_grid=BeatGrid(bpm=bpm, grid_version=1, confidence=0.9),
        sections=[SectionMarker(section_id="s0", kind="intro", start_bar=0, end_bar=8)],
        quality=PASSING,
        approval=ApprovalState.APPROVED,
        readiness=ReadinessState.READY,
    )
    return manifest, path


def offline_engine() -> AudioEngine:
    """An engine with no device open. Commands still flow when the mixer is driven."""
    return AudioEngine(EngineConfig(sample_rate=SR, blocksize=512))


def pump(engine: AudioEngine, blocks: int = 60, block: int = 512) -> None:
    scratch = np.zeros((block, 2), dtype=np.float32)
    for _ in range(blocks):
        engine.mixer.render(scratch, block)


def session(**kwargs) -> SessionState:
    defaults = dict(session_id="s1", theme="rain-soaked jazz", generation=1)
    defaults.update(kwargs)
    return SessionState(**defaults)


# --- the happy path ---------------------------------------------------------


def test_a_committed_transition_becomes_audio(tmp_path):
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine)
    bridge.register(manifest, path)

    state = session()
    action = ScheduledAction(
        action_id="t1", session_id="s1", commit_bar=8, asset_id=manifest.asset_id,
        asset_sha256=manifest.content_sha256, generation=1,
    )
    assert bridge.apply(action=action, state=state) is True

    pump(engine, blocks=80)
    snap = engine.snapshot()
    assert snap["deck_b"]["loaded"] is True
    assert snap["deck_b"]["hash"] == manifest.content_sha256[:12]
    assert snap["bar"] >= 0


def test_the_crossfade_length_comes_from_the_musical_plan(tmp_path):
    """The fade should be as long as the plan said, not a hard-coded constant."""
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine, config=BridgeConfig(min_fade_bars=4, max_fade_bars=32))
    bridge.register(manifest, path)

    state = session()
    state.clock.bar = 0
    action = ScheduledAction(
        action_id="t1", session_id="s1", commit_bar=8, asset_sha256=manifest.content_sha256,
        generation=1,
    )
    bridge.apply(action=action, state=state)

    # 120 BPM -> 2 s per bar, so an 8-bar lead is 16 s.
    expected = int(state.clock.seconds_per_bar * 8 * engine.sample_rate)
    assert bridge._fade_frames(state, action) == expected

    # The crossfade is *enqueued*, not applied: the deck only starts fading once the callback
    # drains the queue. Checking before that would assert against a command nobody has run yet.
    pump(engine, blocks=1)
    assert engine.mixer.decks[1].fade_remaining > 0


def test_the_fade_is_clamped_at_both_ends(tmp_path):
    manifest, path = make_asset(tmp_path, "a")
    bridge = AudioBridge(offline_engine(), config=BridgeConfig(min_fade_bars=4, max_fade_bars=16))
    bridge.register(manifest, path)
    state = session()

    far = ScheduledAction(action_id="t", session_id="s1", commit_bar=100,
                          asset_sha256=manifest.content_sha256, generation=1)
    sr = bridge.engine.sample_rate
    assert bridge._fade_frames(state, far) == int(state.clock.seconds_per_bar * 16 * sr)

    state.clock.bar = 99
    near = ScheduledAction(action_id="t", session_id="s1", commit_bar=100,
                           asset_sha256=manifest.content_sha256, generation=1)
    assert bridge._fade_frames(state, near) == int(state.clock.seconds_per_bar * 4 * sr)


def test_the_outgoing_deck_fades_under_the_incoming_one(tmp_path):
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine)
    bridge.register(manifest, path)

    state = session()
    action = ScheduledAction(action_id="t1", session_id="s1", commit_bar=8,
                             asset_sha256=manifest.content_sha256, generation=1)
    bridge.apply(action=action, state=state)
    # 8 bars at 120 BPM is 16 s = 705600 frames, so 1400 blocks of 512 covers the fade.
    pump(engine, blocks=1400)

    assert engine.mixer.decks[1].fade_gain == pytest.approx(1.0, abs=0.01)
    assert engine.mixer.decks[0].fade_gain == pytest.approx(0.0, abs=0.01)


# --- the failure paths that matter -----------------------------------------


def test_an_action_naming_an_unknown_asset_is_refused(tmp_path):
    """A scheduler decision the library cannot satisfy must not disturb playback."""
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine)
    bridge.register(manifest, path)

    stray = ScheduledAction(action_id="t", session_id="s1", commit_bar=8,
                            asset_sha256="f" * 64, generation=1)
    assert bridge.apply(action=stray, state=session()) is False
    assert bridge.stats.skipped_no_asset == 1


def test_an_action_with_no_asset_hash_is_refused():
    engine = offline_engine()
    bridge = AudioBridge(engine)
    assert bridge.apply(action=ScheduledAction(action_id="t", session_id="s", commit_bar=8,
                                              generation=1),
                        state=session()) is False


def test_a_missing_file_is_refused_rather_than_raising(tmp_path):
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine)
    bridge.register(manifest, path)
    path.unlink()  # the library lost the file after the decision was made

    action = ScheduledAction(action_id="t", session_id="s", commit_bar=8,
                             asset_sha256=manifest.content_sha256, generation=1)
    assert bridge.apply(action=action, state=session()) is False
    assert bridge.stats.skipped_no_asset == 1


def test_a_hash_mismatch_is_refused(tmp_path):
    """A file that changed after the decision was made must not be played."""
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine)
    bridge.register(manifest, path)

    # Replace the file with different audio; its hash no longer matches the plan.
    import soundfile as sf

    other = np.zeros((SR * 2, 2), dtype=np.float32)
    sf.write(str(path), other, SR, subtype="PCM_16")

    action = ScheduledAction(action_id="t", session_id="s", commit_bar=8,
                             asset_sha256=manifest.content_sha256, generation=1)
    assert bridge.apply(action=action, state=session()) is False
    assert bridge.stats.preload_failures == 1


def test_the_engine_keeps_playing_when_a_transition_fails(tmp_path):
    """The whole point: a failed transition is not a failed set."""
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine)
    bridge.register(manifest, path)

    engine.load_and_play(engine.preload(path, bpm=120.0), deck=0)
    pump(engine, blocks=20)

    # A second, unresolvable transition.
    bridge.apply(
        action=ScheduledAction(action_id="bad", session_id="s", commit_bar=8,
                               asset_sha256="a" * 64, generation=1),
        state=session(),
    )
    pump(engine, blocks=20)

    assert engine.mixer.decks[0].playing is True, "the current track must survive"
    assert engine.mixer.decks[0].position > 0


def test_start_plays_an_asset_immediately(tmp_path):
    manifest, path = make_asset(tmp_path, "a")
    engine = offline_engine()
    bridge = AudioBridge(engine)
    bridge.register(manifest, path)

    assert bridge.start(manifest) is True
    pump(engine, blocks=20)
    assert engine.mixer.decks[0].playing is True


def test_start_refuses_an_unregistered_asset(tmp_path):
    manifest, _ = make_asset(tmp_path, "a")
    assert AudioBridge(offline_engine()).start(manifest) is False
