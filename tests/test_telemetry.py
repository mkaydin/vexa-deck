import time

import numpy as np
import pytest
from vexa_audio.equalizer import EqualizerSettings
from vexa_audio.loader import PreloadedTrack
from vexa_audio.mixer import Mixer
from vexa_audio.telemetry import AudioTelemetry, spectrum_levels


@pytest.mark.parametrize("sample_rate", [44100, 48000])
@pytest.mark.parametrize("frequency", [50, 1000, 8000])
def test_final_output_tones_and_opposite_phase_stereo(sample_rate, frequency):
    telemetry = AudioTelemetry(sample_rate, 2)
    tone = 0.2 * np.sin(2 * np.pi * frequency * np.arange(8192) / sample_rate)
    telemetry.capture(np.column_stack((tone, -tone)), 8192)
    result = telemetry.measure()
    maximum = np.argmax(result["levels"])
    lower, upper = result["frequency_edges"][maximum : maximum + 2]
    assert lower - sample_rate / 8192 <= frequency <= upper + sample_rate / 8192
    assert max(result["levels"]) > 0.6
    assert result["rms_dbfs"] == pytest.approx(-17, abs=0.1)
    assert result["source"] == "post-master" and not result["stale"]


def test_capture_is_actual_master_output_including_eq_and_limiter():
    mixer = Mixer(sample_rate=44100, max_block=512)
    tone = np.sin(2 * np.pi * 60 * np.arange(16384) / 44100).astype("float32")
    mixer.set_track(0, PreloadedTrack("test", np.column_stack((tone, tone)), 44100, 2))
    mixer.decks[0].playing = True
    mixer.equalizer.configure(EqualizerSettings(True, (8, 0, 0, 0, 0), -2))
    output = []
    for _ in range(32):
        block = np.zeros((512, 2), dtype="float32")
        mixer.render(block, 512)
        output.append(block.copy())
    expected = np.concatenate(output)[-8192:]
    t = mixer.telemetry
    actual = np.concatenate((t.buffer[t.position :], t.buffer[: t.position]))
    assert np.array_equal(expected, actual)
    snapshot = t.measure()
    assert snapshot["peak_dbfs"] == pytest.approx(20 * np.log10(np.abs(expected).max()))
    assert mixer.stats.underruns == 0


def test_stale_stop_and_worker_restart():
    telemetry = AudioTelemetry(44100, 2)
    telemetry.capture(np.ones((256, 2), dtype="float32") * 0.1, 256)
    assert not telemetry.measure()["stale"]
    telemetry._last_audio = time.monotonic() - 1
    assert telemetry.measure()["stale"] and not any(telemetry.cached["levels"])
    telemetry.start()
    thread = telemetry._thread
    telemetry.start()
    assert telemetry._thread is thread
    telemetry.stop()
    assert not thread.is_alive() and not any(telemetry.cached["levels"])
    telemetry.start()
    assert telemetry._thread is not thread
    telemetry.stop()


def test_partial_ring_wrap_preserves_samples():
    telemetry = AudioTelemetry(44100, 2)
    samples = np.arange(10000, dtype="float32")[:, None].repeat(2, axis=1)
    for start in range(0, len(samples), 700):
        block = samples[start : start + 700]
        telemetry.capture(block, len(block))
    assert np.array_equal(
        np.concatenate(
            (telemetry.buffer[telemetry.position :], telemetry.buffer[: telemetry.position])
        ),
        samples[-8192:],
    )


def test_spectrum_reflects_playing_audio_and_returns_to_zero():
    sr = 44100
    mixer = Mixer(sample_rate=sr)
    assert spectrum_levels(mixer) == [0.0] * 32
    t = np.arange(sr) / sr
    wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    deck = mixer.decks[0]
    deck.track = PreloadedTrack("a" * 64, np.stack([wave, wave], axis=1), sr, 2)
    deck.playing = True
    block = np.zeros((1024, 2), dtype=np.float32)
    for _ in range(8):
        mixer.render(block, len(block))
    mixer.telemetry.measure()
    levels = spectrum_levels(mixer)
    assert max(levels) > 0.7 and levels.index(max(levels)) < 20
    deck.playing = False
    for _ in range(8):
        mixer.render(block, len(block))
    mixer.telemetry.measure()
    assert spectrum_levels(mixer) == [0.0] * 32


def test_nonfinite_samples_do_not_break_json():
    import json

    telemetry = AudioTelemetry(44100, 2)
    samples = np.zeros((8192, 2), dtype="float32")
    samples[0, 0] = np.nan
    samples[1, 1] = np.inf
    telemetry.capture(samples, len(samples))
    result = telemetry.measure()
    json.dumps(result, allow_nan=False)
    assert not any(result["levels"])
