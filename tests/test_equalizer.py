"""Measured DSP behavior, callback integration and persisted control-plane validation."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from vexa_audio.engine import AudioEngine
from vexa_audio.equalizer import FREQUENCIES, EqualizerSettings, MasterEqualizer
from vexa_orchestrator.api import create_app
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.live import LiveSet

SR = 44100


def tone(frequency, seconds=1, amplitude=0.05):
    wave = amplitude * np.sin(2 * np.pi * frequency * np.arange(int(seconds * SR)) / SR)
    return np.column_stack((wave, wave)).astype(np.float32)


def process(eq, signal):
    for start in range(0, len(signal), 1024):
        eq.process(signal[start : start + 1024])
    return signal


@pytest.mark.parametrize("band", range(5))
@pytest.mark.parametrize("gain", [-9, 9])
def test_measured_gain_at_each_band_center(band, gain):
    eq = MasterEqualizer(SR, 2)
    gains = [0.0] * 5
    gains[band] = gain
    eq.configure(EqualizerSettings(True, tuple(gains)))
    original = tone(FREQUENCIES[band])
    filtered = process(eq, original.copy())
    # Ignore coefficient ramp and filter settling; measure the steady-state RMS response.
    measured = 20 * np.log10(np.std(filtered[SR // 2 :, 0]) / np.std(original[SR // 2 :, 0]))
    assert measured == pytest.approx(gain, abs=0.05)
    assert not eq.snapshot()["pending"]


def test_flat_bypass_is_bit_exact_and_channels_remain_independent():
    eq = MasterEqualizer(SR, 2)
    original = tone(60)
    assert np.array_equal(process(eq, original.copy()), original)
    eq.configure(EqualizerSettings(True, (9, 0, 0, 0, 0), -9))
    original[:, 1] = 0
    filtered = process(eq, original.copy())
    assert np.isfinite(filtered).all()
    assert np.count_nonzero(filtered[:, 1]) == 0
    eq.configure(EqualizerSettings(False, (9, 0, 0, 0, 0), -9))
    process(eq, tone(60))
    assert np.array_equal(process(eq, original.copy()), original)


def test_rapid_changes_are_smoothed_and_latest_value_wins():
    eq = MasterEqualizer(SR, 2)
    original = tone(1000, seconds=2)
    output = original.copy()
    for start in range(0, SR, 512):
        gains = (12, -12, 12, -12, 12) if start % 1024 else (-12, 12, -12, 12, -12)
        eq.configure(EqualizerSettings(True, gains, -12))
        eq.process(output[start : start + 512])
    eq.configure(EqualizerSettings(True, (12,) * 5, -12))
    eq.configure(EqualizerSettings(False))
    process(eq, output[SR:])
    assert np.isfinite(output).all()
    assert np.max(np.abs(np.diff(output[:, 0]))) < 0.03
    assert np.array_equal(output[-1024:], original[-1024:])
    assert eq.snapshot()["enabled"] is False


def test_mixer_applies_eq_before_limiter_and_reports_state(tmp_path):
    import soundfile as sf

    engine = AudioEngine()
    source = tone(60, amplitude=0.8)
    sf.write(tmp_path / "tone.wav", source, SR)
    track = engine.preload(tmp_path / "tone.wav")
    engine.load_and_play(track)
    engine.mixer.equalizer.configure(EqualizerSettings(True, (12,) * 5))
    out = np.zeros((2048, 2), dtype=np.float32)
    for _ in range(20):
        engine.mixer.render(out, len(out))
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0
    assert engine.mixer.stats.underruns == 0
    assert engine.snapshot()["equalizer"]["enabled"] is True


def test_api_validation_persistence_and_stopped_engine_settings(tmp_path):
    previews = tmp_path / "previews"
    live = LiveSet(Depot(tmp_path / "library"), preview_dir=previews)
    client = TestClient(create_app(live=live))
    assert client.get("/audio/equalizer").json()["pending"] is False
    assert client.get("/live").status_code == 200
    config = {"enabled": True, "gains_db": [4, 1, -2, 2, 4], "preamp_db": -4}
    reply = client.post("/audio/equalizer", json=config)
    assert reply.status_code == 200
    assert reply.json()["gains_db"] == config["gains_db"]
    assert reply.json()["pending"] is True  # no callback until playback starts
    assert client.get("/live").json()["audio"]["equalizer"]["enabled"] is True
    saved = tmp_path / "config/equalizer.json"
    assert json.loads(saved.read_text()) == config
    restarted = LiveSet(Depot(tmp_path / "library"), preview_dir=previews)
    assert restarted.engine.snapshot()["equalizer"]["gains_db"] == tuple(config["gains_db"])
    for invalid in ([13, 0, 0, 0, 0], [0] * 4, ["NaN", 0, 0, 0, 0]):
        invalid_reply = client.post("/audio/equalizer", json={**config, "gains_db": invalid})
        assert invalid_reply.status_code == 422
    assert client.post("/audio/equalizer", json={**config, "preamp_db": 1}).status_code == 422
    assert json.loads(saved.read_text()) == config
    assert TestClient(create_app()).get("/audio/equalizer").status_code == 409
