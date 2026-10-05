"""Audio engine tests.

These run the mixer **offline** — no device, no PortAudio thread — which is what makes them fast
and deterministic while still exercising the exact code the callback runs. The device path is
covered separately by ``tools/audio_smoke.py``, because claiming a callback is realtime-safe
without opening a device proves nothing.

What is worth testing here is the properties that protect the audio:
"""

from __future__ import annotations

import numpy as np
import pytest
from vexa_audio.engine import AudioEngine
from vexa_audio.loader import MasterLimiter, TrackCache, file_sha256, load_track
from vexa_audio.mixer import Mixer
from vexa_audio.queue import Clock, Command, CommandKind, CommandQueue

SR = 44100


def make_track(path, seconds: float = 4.0, freq: float = 220.0, amp: float = 0.25):
    """A stereo tone written to disk, so load paths are exercised too."""
    import soundfile as sf

    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    tone = amp * np.sin(2 * np.pi * freq * t)
    stereo = np.stack([tone, tone], axis=1).astype(np.float32)
    sf.write(str(path), stereo, SR, subtype="PCM_16")
    return path


def render(mixer: Mixer, blocks: int = 20, block: int = 512) -> np.ndarray:
    out = []
    for _ in range(blocks):
        buf = np.zeros((block, 2), dtype=np.float32)
        mixer.render(buf, block)
        out.append(buf.copy())
    return np.concatenate(out, axis=0)


def test_prepared_track_is_loaded_on_silent_deck_before_crossfade(tmp_path):
    engine = AudioEngine()
    track = engine.preload(make_track(tmp_path / "next.wav", seconds=0.5))
    engine.prepare_deck(track, 1)
    render(engine.mixer, blocks=1)
    assert engine.mixer.decks[1].track is track
    assert engine.mixer.decks[1].playing is False
    engine.crossfade_to(track, 1, fade_frames=512)
    render(engine.mixer, blocks=1)
    assert engine.mixer.decks[1].playing is True


# --- the queue --------------------------------------------------------------


def test_snapshot_exposes_measured_fade_timing_for_visuals(tmp_path):
    engine = AudioEngine()
    track = engine.preload(make_track(tmp_path / "fade.wav", seconds=0.5))
    engine.crossfade_to(track, 1, fade_frames=4096)
    render(engine.mixer, blocks=1, block=512)
    data = engine.mixer.snapshot()["deck_b"]
    assert data["fading"] is True
    assert data["fade_progress"] == pytest.approx(0.125)
    assert data["fade_duration_s"] == pytest.approx(4096 / SR, abs=1e-5)
    render(engine.mixer, blocks=7, block=512)
    data = engine.mixer.snapshot()["deck_b"]
    assert data["fading"] is False
    assert data["fade_progress"] == 1


def test_a_command_survives_the_round_trip():
    queue = CommandQueue(capacity=8)
    assert queue.put(Command(CommandKind.PLAY, deck=1))
    assert len(queue) == 1
    drained = queue.drain()
    assert [c.kind for c in drained] == [CommandKind.PLAY]
    assert drained[0].deck == 1
    assert len(queue) == 0


def test_draining_an_empty_queue_is_harmless():
    assert CommandQueue().drain() == []


def test_commands_arrive_in_order():
    queue = CommandQueue(capacity=16)
    for deck in (0, 1, 0, 1):
        queue.put(Command(CommandKind.SET_GAIN, deck=deck, value=float(deck)))
    assert [c.deck for c in queue.drain()] == [0, 1, 0, 1]


def test_a_full_queue_drops_rather_than_blocking_the_producer():
    """The producer is the control plane. It must never wait for audio."""
    queue = CommandQueue(capacity=4)
    accepted = [queue.put(Command(CommandKind.PLAY, deck=0)) for _ in range(6)]
    assert accepted.count(True) == 3, "capacity-1 slots, as a full ring always leaves one empty"
    assert queue.stats.dropped == 3
    assert len(queue) == 3


def test_the_queue_is_never_falsy_just_because_it_is_empty():
    """The bug that stopped all audio once: `queue or CommandQueue()` discards a valid empty
    queue, because `__len__` makes it falsy."""
    queue = CommandQueue()
    assert len(queue) == 0
    assert not queue  # falsy...
    mixer = Mixer(sample_rate=SR, queue=queue)
    assert mixer.queue is queue, "...but must not be replaced by a default"


def test_a_non_power_of_two_capacity_is_refused():
    with pytest.raises(ValueError, match="power of two"):
        CommandQueue(capacity=100)


# --- the musical clock ------------------------------------------------------


def test_the_clock_advances_in_bars_and_beats():
    clock = Clock(sample_rate=SR, bpm=120.0)
    # 120 BPM at 44.1 kHz: 2 s per bar.
    assert clock.samples_per_bar == pytest.approx(SR * 2)
    clock.advance(int(SR * 2))
    assert clock.bar == 1
    assert clock.beat == 0


def test_a_tempo_change_moves_the_bar_boundary():
    clock = Clock(sample_rate=SR, bpm=120.0)
    slow = Clock(sample_rate=SR, bpm=60.0)
    clock.advance(int(SR))
    slow.advance(int(SR))
    assert clock.bar == 0
    assert slow.bar == 0
    clock.advance(int(SR))
    assert clock.bar == 1


# --- loading ----------------------------------------------------------------


def test_a_loaded_track_is_addressable_by_content_hash(tmp_path):
    path = make_track(tmp_path / "a.wav")
    track = load_track(path, sample_rate=SR, bpm=120.0)
    assert track.content_sha256 == file_sha256(path)
    assert len(track.content_sha256) == 64
    assert track.channels == 2
    assert track.samples.shape[1] == 2
    assert track.can_seek_by_bar


def test_a_hash_mismatch_is_refused(tmp_path):
    """A file that changed after a decision was made against it must not be played."""
    path = make_track(tmp_path / "a.wav")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_track(path, sample_rate=SR, expected_hash="b" * 64)


def test_the_cache_returns_the_same_object_for_a_repeat_load(tmp_path):
    path = make_track(tmp_path / "a.wav")
    cache = TrackCache(sample_rate=SR)
    first = cache.preload(path, expected_hash=file_sha256(path))
    second = cache.preload(path, expected_hash=file_sha256(path))
    assert first is second, "a cached track must not be decoded twice on the realtime path"


def test_an_unreadable_file_raises_rather_than_returning_silence(tmp_path):
    from vexa_audio.loader import load_track as load

    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises((RuntimeError, ValueError, OSError)):
        load(bad, sample_rate=SR)


# --- the limiter ------------------------------------------------------------


def test_the_limiter_holds_the_signal_under_the_ceiling():
    limiter = MasterLimiter(frames=1024, channels=2, ceiling=0.891)
    loud = np.full((256, 2), 1.0, dtype=np.float32)
    for _ in range(8):
        out = limiter.process(loud.copy())
    assert float(np.max(np.abs(out))) <= 0.892


def test_a_quiet_signal_passes_through_untouched():
    limiter = MasterLimiter(frames=1024, channels=2)
    quiet = np.full((256, 2), 0.1, dtype=np.float32)
    out = limiter.process(quiet.copy())
    assert float(np.max(np.abs(out))) == pytest.approx(0.1, rel=0.02)


def test_the_limiter_catches_a_transient_after_a_quiet_passage():
    """Sample peak protection must hold when a quiet signal suddenly becomes loud."""
    limiter = MasterLimiter(frames=1024, channels=2)
    quiet = np.full((256, 2), 0.01, dtype=np.float32)
    limiter.process(quiet.copy())
    spike = np.full((256, 2), 1.0, dtype=np.float32)
    captured = [limiter.process(spike.copy()) for _ in range(8)]
    assert max(float(np.max(np.abs(c))) for c in captured) <= 0.892


# --- the mixer --------------------------------------------------------------


def test_an_unloaded_mixer_outputs_silence_and_does_not_raise():
    """The most important property: nothing downstream may stop the music."""
    audio = render(Mixer(sample_rate=SR), blocks=10)
    assert not np.any(audio)


def test_a_playing_deck_produces_audible_output(tmp_path):
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(make_track(tmp_path / "a.wav"), sample_rate=SR, bpm=120.0))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    audio = render(mixer, blocks=20)
    assert float(np.sqrt(np.mean(np.square(audio)))) > 0.01


def test_a_paused_deck_is_silent(tmp_path):
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(make_track(tmp_path / "a.wav"), sample_rate=SR))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    render(mixer, blocks=5)
    mixer.queue.put(Command(CommandKind.PAUSE, deck=0))
    assert not np.any(render(mixer, blocks=5))


def test_the_musical_clock_only_advances_while_rendering(tmp_path):
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(make_track(tmp_path / "a.wav"), sample_rate=SR, bpm=120.0))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    # 250 blocks of 512 is 2.9 s, comfortably past one bar at 120 BPM. The point is that the
    # clock tracks rendered audio, not how long the test happens to run.
    render(mixer, blocks=250, block=512)
    assert mixer.bar >= 1
    assert mixer.clock.samples == pytest.approx(250 * 512)
def test_a_deck_stops_when_its_track_ends(tmp_path):
    short = make_track(tmp_path / "s.wav", seconds=0.5)
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(short, sample_rate=SR))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    render(mixer, blocks=200)  # ~2.3 s, well past the 0.5 s track
    assert mixer.decks[0].playing is False


def test_gain_scales_the_output(tmp_path):
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(make_track(tmp_path / "a.wav"), sample_rate=SR))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    mixer.queue.put(Command(CommandKind.SET_GAIN, deck=0, value=0.0))
    assert float(np.max(np.abs(render(mixer, blocks=10)))) == 0.0


def test_a_partial_gain_reduces_the_output(tmp_path):
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(make_track(tmp_path / "a.wav"), sample_rate=SR))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    loud = float(np.max(np.abs(render(mixer, blocks=10))))

    quiet_mixer = Mixer(sample_rate=SR)
    quiet_mixer.set_track(0, load_track(make_track(tmp_path / "a.wav"), sample_rate=SR))
    quiet_mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    quiet_mixer.queue.put(Command(CommandKind.SET_GAIN, deck=0, value=0.5))
    quiet = float(np.max(np.abs(render(quiet_mixer, blocks=10))))

    assert 0.0 < quiet < loud


def test_seek_by_bar_moves_the_position(tmp_path):
    mixer = Mixer(sample_rate=SR)
    track = load_track(
        make_track(tmp_path / "a.wav", seconds=10), sample_rate=SR, bpm=120.0
    )
    mixer.set_track(0, track)
    mixer.queue.put(Command(CommandKind.SEEK_BAR, deck=0, value=4))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    render(mixer, blocks=2)
    # 120 BPM -> 2 s per bar, so bar 4 is sample 352800.
    assert mixer.decks[0].position >= 352800


def test_emergency_stop_halts_both_decks(tmp_path):
    mixer = Mixer(sample_rate=SR)
    for deck in (0, 1):
        mixer.set_track(deck, load_track(make_track(tmp_path / f"{deck}.wav"), sample_rate=SR))
        mixer.queue.put(Command(CommandKind.PLAY, deck=deck))
    render(mixer, blocks=5)
    mixer.queue.put(Command(CommandKind.EMERGENCY_STOP))
    render(mixer, blocks=2)
    assert mixer.decks[0].playing is False
    assert mixer.decks[1].playing is False


def test_a_malformed_command_cannot_break_the_callback():
    """An out-of-range deck index must be ignored, not raise into PortAudio."""
    mixer = Mixer(sample_rate=SR)
    mixer.queue.put(Command(CommandKind.PLAY, deck=99))
    mixer.queue.put(Command(CommandKind.SET_GAIN, deck=-5, value=99.0))
    audio = render(mixer, blocks=5)
    assert audio.shape[0] == 5 * 512
    assert mixer.stats.underruns == 0


def test_a_render_exception_outputs_silence_rather_than_raising():
    """The callback must survive anything. Silence beats a dead stream."""
    mixer = Mixer(sample_rate=SR)

    def boom(_out, _frames):
        raise RuntimeError("something impossible")

    mixer._render = boom  # type: ignore[method-assign]
    buf = np.zeros((512, 2), dtype=np.float32)
    mixer.render(buf, 512)
    assert not np.any(buf)
    assert mixer.stats.underruns == 1


# --- the crossfade ----------------------------------------------------------


def test_a_crossfade_ramps_rather_than_cutting(tmp_path):
    """A fade that only sets a flag is a hard cut wearing a crossfade's name."""
    mixer = Mixer(sample_rate=SR)
    for deck in (0, 1):
        mixer.set_track(deck, load_track(make_track(tmp_path / f"{deck}.wav"), sample_rate=SR))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    render(mixer, blocks=10)

    mixer.queue.put(Command(CommandKind.START_CROSSFADE, deck=1, value2=float(SR * 2)))
    mixer.render(np.zeros((512, 2), dtype=np.float32), 512)

    incoming = mixer.decks[1]
    outgoing = mixer.decks[0]
    assert incoming.fading and 0.0 <= incoming.fade_gain < 1.0
    assert outgoing.fading and 0.0 < outgoing.fade_gain <= 1.0

    # Halfway through a 2 s fade both decks should be audible: that is the overlap.
    half = int(SR / 512 / 2)
    render(mixer, blocks=half)
    assert mixer.decks[0].fade_gain > 0.05
    assert mixer.decks[1].fade_gain > 0.05

    # Finish it. A 2 s fade at 512 frames per block needs 173 blocks from the fade start.
    render(mixer, blocks=200)
    assert mixer.decks[1].fade_gain == pytest.approx(1.0, abs=0.01)
    assert mixer.decks[0].fade_gain == pytest.approx(0.0, abs=0.01)
    assert mixer.decks[1].fade_remaining == 0


def test_a_crossfade_does_not_drop_the_output_level(tmp_path):
    """An equal-power overlap should hold roughly constant loudness through the fade."""
    mixer = Mixer(sample_rate=SR)
    for deck in (0, 1):
        mixer.set_track(deck, load_track(make_track(tmp_path / f"{deck}.wav"), sample_rate=SR))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    render(mixer, blocks=20)

    before = float(np.sqrt(np.mean(np.square(render(mixer, blocks=10)))))
    mixer.queue.put(Command(CommandKind.START_CROSSFADE, deck=1, value2=float(SR * 4)))
    mids = []
    for _ in range(int(SR * 2 / 512)):
        buf = np.zeros((512, 2), dtype=np.float32)
        mixer.render(buf, 512)
        mids.append(float(np.sqrt(np.mean(np.square(buf)))))
    after = float(np.sqrt(np.mean(np.square(render(mixer, blocks=10)))))

    assert before > 0.01 and after > 0.01
    # Linear fades dip in the middle; the point is that the music does not vanish.
    assert min(mids) > before * 0.25, "output collapsed during the crossfade"

def test_a_looping_deck_keeps_playing_past_the_end(tmp_path):
    """A DJ bed runs indefinitely. This is also what lets a soak test stay small in memory."""
    short = make_track(tmp_path / "loop.wav", seconds=0.5)
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(short, sample_rate=SR))
    mixer.decks[0].loop = True
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))

    audio = render(mixer, blocks=400)  # ~4.6 s, well past the 0.5 s track
    assert mixer.decks[0].playing is True
    assert float(np.sqrt(np.mean(np.square(audio)))) > 0.01, "a looping deck must not fall silent"


def test_a_loop_returns_to_the_measured_entry_not_the_silent_intro(tmp_path):
    import soundfile as sf

    samples = np.zeros((SR, 2), dtype=np.float32)
    samples[int(0.3 * SR):] = 0.2
    path = tmp_path / "intro.wav"
    sf.write(path, samples, SR)
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(path, sample_rate=SR))
    mixer.decks[0].loop = True
    mixer.queue.put(Command(CommandKind.LOAD_TRACK, deck=0, value=1.0,
                            value2=float(int(0.35 * SR))))
    audio = render(mixer, blocks=180)
    assert mixer.decks[0].cue_point == int(0.35 * SR)
    assert float(np.sqrt(np.mean(np.square(audio[-SR:])))) > 0.05


def test_a_non_looping_deck_still_stops_at_the_end(tmp_path):
    """Looping is opt-in; the default behaviour is unchanged."""
    short = make_track(tmp_path / "once.wav", seconds=0.5)
    mixer = Mixer(sample_rate=SR)
    mixer.set_track(0, load_track(short, sample_rate=SR))
    mixer.queue.put(Command(CommandKind.PLAY, deck=0))
    render(mixer, blocks=400)
    assert mixer.decks[0].playing is False


def test_rerendering_a_decision_clears_its_stale_clips(tmp_path):
    """Overlapping indices with different audio under one name is a labelling trap."""
    from vexa_audio.preview import PreviewRenderer, PreviewSpec

    renderer = PreviewRenderer(sample_rate=SR, out_dir=tmp_path)
    track = load_track(make_track(tmp_path / "a.wav", seconds=10), sample_rate=SR)

    renderer.render_transition(PreviewSpec("x"), track, track, out_name="dec0_0_old")
    renderer.render_transition(PreviewSpec("x"), track, track, out_name="dec0_1_old")
    assert len(list(tmp_path.glob("dec0_*.wav"))) == 2

    # Re-render the same decision with a different option set.
    out = renderer.render_comparison(track, [("alpha", track), ("beta", track)], name="dec0")
    assert set(out) == {"alpha", "beta"}
    remaining = sorted(p.name for p in tmp_path.glob("dec0_*.wav"))
    assert not any("old" in n for n in remaining), remaining
    assert len(remaining) == 2


def test_clear_only_touches_the_named_decision(tmp_path):
    from vexa_audio.preview import PreviewRenderer, PreviewSpec

    renderer = PreviewRenderer(sample_rate=SR, out_dir=tmp_path)
    track = load_track(make_track(tmp_path / "a.wav", seconds=10), sample_rate=SR)
    renderer.render_transition(PreviewSpec("x"), track, track, out_name="keepme_0")
    renderer.render_transition(PreviewSpec("x"), track, track, out_name="dropme_0")

    assert renderer.clear("dropme") == 1
    assert list(tmp_path.glob("keepme_*.wav"))
    assert not list(tmp_path.glob("dropme_*.wav"))


def test_limiter_release_is_continuous_and_independent_of_callback_partition():
    # A loud passage followed by a quieter one must not reset gain at the next callback.
    signal = np.concatenate((np.ones((1200, 2)), np.full((1800, 2), 0.2))).astype(np.float32)
    whole = MasterLimiter(frames=4096, channels=2).process(signal).copy()
    limiter = MasterLimiter(frames=512, channels=2)
    chunks = [limiter.process(signal[i:i + 256]).copy() for i in range(0, len(signal), 256)]
    partitioned = np.concatenate(chunks)
    np.testing.assert_allclose(partitioned, whole, atol=1e-7)
    assert partitioned[1200, 0] < 0.19  # release retains reduction from the loud passage
    assert np.max(np.diff(partitioned[1200:, 0])) < 0.00001


def test_audio_start_warms_before_device_callback_and_counts_device_underflows(monkeypatch):
    import sys
    from types import SimpleNamespace

    from vexa_audio.engine import AudioEngine, EngineConfig

    engine = AudioEngine(EngineConfig(blocksize=2048, warmup_callbacks=3))

    class Stream:
        latency = 0.1
        cpu_load = 0.01

        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
            assert kwargs['latency'] == 'high'

        def start(self):
            assert engine.mixer.stats.callbacks == 3
            output = np.empty((2048, 2), dtype=np.float32)
            self.callback(output, 2048, None, SimpleNamespace(output_underflow=True))
            assert not np.any(output)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(OutputStream=Stream))
    engine.start()
    assert engine.mixer.stats.callbacks == 4  # no rendering on the control thread after start
    assert engine.snapshot()['stats']['output_underflows'] == 1
    assert engine.snapshot()['stats']['underruns'] == 1
    engine.stop()


def test_engine_track_and_playhead_swap_only_at_callback_boundary(tmp_path):
    engine = AudioEngine()
    a = engine.preload(make_track(tmp_path / "old.wav", seconds=2))
    b = engine.preload(make_track(tmp_path / "new.wav", seconds=1))
    engine.load_and_play(a)
    render(engine.mixer, blocks=2)
    before = engine.mixer.decks[0].position
    engine.load_and_play(b)
    assert engine.mixer.decks[0].track is a
    assert engine.mixer.decks[0].position == before
    render(engine.mixer, blocks=1)
    assert engine.mixer.decks[0].track is b
    assert engine.mixer.decks[0].position == 512
    assert engine.mixer.stats.underruns == 0
