"""Read-only spectrum measurement for the GUI, outside the realtime callback."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .mixer import Mixer


def spectrum_levels(mixer: Mixer, *, bands: int = 32, frames: int = 2048) -> list[float]:
    """Measure the material under the playheads before the master limiter.

    This is called by the HTTP control thread. It never adds FFT work to PortAudio's callback.
    """
    if bands < 1 or frames < 128:
        raise ValueError("spectrum needs at least one band and 128 frames")
    mixed = np.zeros(frames, dtype=np.float32)
    for deck in mixer.decks:
        track = deck.track
        if track is None or not deck.playing:
            continue
        end = min(max(deck.position, 0), track.frames)
        start = max(0, end - frames)
        if end <= start:
            continue
        block = track.samples[start:end].mean(axis=1)
        mixed[frames - len(block):] += block * deck.gain * deck.fade_gain
    if not np.any(mixed):
        return [0.0] * bands
    window = np.hanning(frames).astype(np.float32)
    amplitudes = np.abs(np.fft.rfft(mixed * window)) / max(float(window.sum()) / 2, 1)
    frequencies = np.fft.rfftfreq(frames, 1.0 / mixer.sample_rate)
    edges = np.geomspace(35.0, min(16000.0, mixer.sample_rate / 2), bands + 1)
    levels: list[float] = []
    for lower, upper in pairwise(edges):
        region = amplitudes[(frequencies >= lower) & (frequencies < upper)]
        peak = float(region.max()) if region.size else 0.0
        db = 20.0 * np.log10(max(peak, 1e-8))
        levels.append(round(float(np.clip((db + 70.0) / 60.0, 0.0, 1.0)), 3))
    return levels
