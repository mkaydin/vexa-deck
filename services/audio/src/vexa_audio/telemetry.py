"""Bounded post-master audio capture; FFT and delivery never run in the callback."""

from __future__ import annotations

import threading
import time
from itertools import pairwise

import numpy as np


def regroup(levels, bands=32):
    """Preserve peaks when reducing the display resolution."""
    return [max(group, default=0.0) for group in np.array_split(levels, bands)]


class AudioTelemetry:
    def __init__(self, sample_rate, channels, frames=8192, bands=64):
        self.sample_rate, self.frames, self.bands = sample_rate, frames, bands
        self.buffer = np.zeros((frames, channels), dtype=np.float32)
        self.position = self.sequence = 0
        self.window = np.hanning(frames).astype(np.float32)
        frequencies = np.fft.rfftfreq(frames, 1 / sample_rate)
        desired = np.geomspace(min(35, sample_rate / 8), min(16000, sample_rate / 2), bands + 1)
        indices = np.searchsorted(frequencies, desired)
        for index in range(1, len(indices)):
            indices[index] = max(indices[index], indices[index - 1] + 1)
        self.regions = list(pairwise(indices.clip(0, len(frequencies))))
        self.edges = frequencies[indices.clip(0, len(frequencies) - 1)].tolist()
        self._stop = threading.Event()
        self._thread = None
        self._observed = 0
        self._last_audio = 0.0
        self.cached = self.silent()

    def silent(self):
        return dict(
            levels=[0.0] * self.bands,
            rms_dbfs=-120.0,
            peak_dbfs=-120.0,
            sequence=self.sequence,
            captured_at=0.0,
            sample_rate=self.sample_rate,
            frequency_edges=self.edges,
            source="post-master",
            stale=True,
        )

    def capture(self, output, frames):
        """Single writer; fixed storage. A reader rejects snapshots taken across a write."""
        count = min(frames, self.frames)
        self.sequence += 1
        first = min(count, self.frames - self.position)
        self.buffer[self.position : self.position + first] = output[
            frames - count : frames - count + first
        ]
        remaining = count - first
        if remaining:
            self.buffer[:remaining] = output[frames - remaining : frames]
        self.position = (self.position + count) % self.frames
        self.sequence += 1

    def measure(self):
        now = time.monotonic()
        if self.sequence == self._observed:
            if now - self._last_audio > 0.35:
                self.cached = self.silent()
            return self.cached
        for _ in range(3):
            sequence, position = self.sequence, self.position
            if sequence % 2:
                continue
            audio = np.concatenate((self.buffer[position:], self.buffer[:position]))
            if sequence != self.sequence:
                continue
            np.nan_to_num(audio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self._observed = sequence
            self._last_audio = now
            amplitude = np.abs(np.fft.rfft(audio * self.window[:, None], axis=0))
            amplitude /= max(float(self.window.sum()) / 2, 1)
            stereo = np.sqrt(np.mean(amplitude**2, axis=1))
            levels = []
            for lower, upper in self.regions:
                peak = float(stereo[lower:upper].max()) if upper > lower else 0
                db = 20 * np.log10(max(peak, 1e-6))
                levels.append(float(np.clip((db + 70) / 70, 0, 1)))
            self.cached = dict(
                levels=levels,
                rms_dbfs=float(20 * np.log10(max(float(np.sqrt(np.mean(audio**2))), 1e-6))),
                peak_dbfs=float(20 * np.log10(max(float(np.abs(audio).max()), 1e-6))),
                sequence=sequence,
                captured_at=now,
                sample_rate=self.sample_rate,
                frequency_edges=self.edges,
                source="post-master",
                stale=False,
            )
            break
        return self.cached

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def work():
            while not self._stop.wait(0.05):
                self.measure()

        self._thread = threading.Thread(target=work, name="audio-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.buffer.fill(0)
        self.cached = self.silent()


def spectrum_levels(mixer, *, bands=32, frames=8192):
    if bands < 1 or bands > 64 or frames < 128:
        raise ValueError("spectrum needs 1-64 bands and at least 128 frames")
    return regroup(mixer.telemetry.cached["levels"], bands)
