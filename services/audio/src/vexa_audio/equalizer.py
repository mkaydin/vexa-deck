"""Five-band master EQ. Prepare settings off-thread; smooth DSP at block boundaries.

The compiled kernel is warmed at construction, before PortAudio opens. Its buffers are fixed,
its sample loop releases the GIL, and it does no filesystem work or allocation. A latest-value
mailbox keeps rapid slider changes from filling the transport command queue.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numba import njit

FREQUENCIES = (60.0, 250.0, 1000.0, 4000.0, 12000.0)


@dataclass(frozen=True)
class EqualizerSettings:
    enabled: bool = False
    gains_db: tuple[float, ...] = (0.0,) * 5
    preamp_db: float = 0.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        object.__setattr__(self, "gains_db", tuple(self.gains_db))
        if len(self.gains_db) != 5:
            raise ValueError("equalizer requires five band gains")
        if any(not math.isfinite(g) or not -12 <= g <= 12 for g in self.gains_db):
            raise ValueError("band gains must be finite and between -12 and +12 dB")
        if not math.isfinite(self.preamp_db) or not -12 <= self.preamp_db <= 0:
            raise ValueError("preamp must be finite and between -12 and 0 dB")

    def to_dict(self):
        return asdict(self)


def coefficients(settings: EqualizerSettings, sample_rate: int) -> np.ndarray:
    result = np.zeros((5, 5), dtype=np.float64)
    result[:, 0] = 1.0
    if not settings.enabled:
        return result
    for index, (frequency, gain) in enumerate(zip(FREQUENCIES, settings.gains_db, strict=True)):
        if gain == 0:
            continue
        # RBJ peaking biquad, Q=0.8. Clamp centers below Nyquist for lower-rate output devices.
        omega = 2 * math.pi * min(frequency, sample_rate * 0.45) / sample_rate
        alpha = math.sin(omega) / (2 * 0.8)
        amplitude = 10 ** (gain / 40)
        a0 = 1 + alpha / amplitude
        cosine = math.cos(omega)
        result[index] = (
            (1 + alpha * amplitude) / a0,
            -2 * cosine / a0,
            (1 - alpha * amplitude) / a0,
            -2 * cosine / a0,
            (1 - alpha / amplitude) / a0,
        )
    return result


@njit(cache=True, nogil=True)
def _process(audio, current, target, delta, states, ramp):
    for frame in range(audio.shape[0]):
        if ramp[0] > 0:
            ramp[0] -= 1
            for band in range(5):
                for term in range(5):
                    current[band, term] = (
                        target[band, term]
                        if ramp[0] == 0
                        else current[band, term] + delta[band, term]
                    )
            ramp[1] = ramp[3] if ramp[0] == 0 else ramp[1] + ramp[2]
        for channel in range(audio.shape[1]):
            value = float(audio[frame, channel])
            for band in range(5):
                b0, b1, b2, a1, a2 = current[band]
                output = b0 * value + states[band, channel, 0]
                z1 = b1 * value - a1 * output + states[band, channel, 1]
                z2 = b2 * value - a2 * output
                states[band, channel, 0] = 0.0 if abs(z1) < 1e-20 else z1
                states[band, channel, 1] = 0.0 if abs(z2) < 1e-20 else z2
                value = output
            audio[frame, channel] = value * ramp[1]


class MasterEqualizer:
    def __init__(self, sample_rate: int, channels: int):
        self.sample_rate = sample_rate
        self.settings = EqualizerSettings()
        self._applied = self.settings
        self._mailbox = (self.settings, coefficients(self.settings, sample_rate), 1.0)
        self._consumed = self._mailbox
        self._current = self._mailbox[1].copy()
        self._target = self._current.copy()
        self._delta = np.zeros_like(self._current)
        self._states = np.zeros((5, channels, 2), dtype=np.float64)
        self._ramp = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        self._flat = True
        # Compile and warm the exact dtype/layout used by the mixer before streaming starts.
        _process(
            np.zeros((1, channels), dtype=np.float32),
            self._current,
            self._target,
            self._delta,
            self._states,
            self._ramp,
        )

    def configure(self, settings: EqualizerSettings):
        prepared = coefficients(settings, self.sample_rate)
        gain = 10 ** (settings.preamp_db / 20) if settings.enabled else 1.0
        self.settings = settings
        self._mailbox = (settings, prepared, gain)

    def snapshot(self):
        return {
            **self.settings.to_dict(),
            "frequencies_hz": FREQUENCIES,
            "pending": bool(self._consumed is not self._mailbox or self._ramp[0] > 0),
        }

    def process(self, audio: np.ndarray):
        message = self._mailbox
        if message is not self._consumed:
            self._applied, prepared, gain = message
            np.copyto(self._target, prepared)
            steps = max(1, int(self.sample_rate * 0.05))
            np.subtract(self._target, self._current, out=self._delta)
            np.divide(self._delta, steps, out=self._delta)
            self._ramp[0] = steps
            self._ramp[2] = (gain - self._ramp[1]) / steps
            self._ramp[3] = gain
            self._flat = not self._applied.enabled or (
                not any(self._applied.gains_db) and self._applied.preamp_db == 0
            )
            self._consumed = message
        if self._flat and self._ramp[0] == 0:
            return
        _process(audio, self._current, self._target, self._delta, self._states, self._ramp)
        if self._flat and self._ramp[0] == 0:
            self._states.fill(0)
