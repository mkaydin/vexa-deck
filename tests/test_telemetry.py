"""The GUI's bars must come from deck audio rather than an animation timer."""

import numpy as np
from vexa_audio.loader import PreloadedTrack
from vexa_audio.mixer import Mixer
from vexa_audio.telemetry import spectrum_levels


def test_spectrum_reflects_playing_audio_and_returns_to_zero():
    sr = 44100
    mixer = Mixer(sample_rate=sr)
    assert spectrum_levels(mixer) == [0.0] * 32
    t = np.arange(sr) / sr
    wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    samples = np.stack([wave, wave], axis=1)
    deck = mixer.decks[0]
    deck.track = PreloadedTrack("a" * 64, samples, sr, 2)
    deck.playing = True
    deck.position = sr // 2
    levels = spectrum_levels(mixer)
    assert max(levels) > 0.7
    assert levels.index(max(levels)) < 20
    deck.playing = False
    assert spectrum_levels(mixer) == [0.0] * 32
