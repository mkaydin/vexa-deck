"""Mastering at ingestion.

A raw YuE2 render is not library material. Measured across five renders: integrated loudness
-15 to -19 LUFS, true peaks **at or above 0 dBTP**, and within-track loudness range of 30-65 LU.

That is unusable for a DJ library for three specific reasons, each of which the architecture
already names:

* **Peaks at 0 dBTP leave no headroom.** A crossfade sums two signals; the master limiter needs
  somewhere to work (``ARCHITECTURE.md:86``).
* **Un-normalised loudness makes every transition a jump.** "Large perceived loudness jumps" is a
  named success measure to avoid (``PRODUCT.md:69``).
* **Wide internal range** means a quiet intro and a loud chorus, which is musically normal but
  means the *integrated* number is the only stable thing to normalise against.

So the render is mastered before it is analysed: normalise the gated integrated loudness to a
target, then limit the true peak. This is a level operation, not a creative one — it does not
change the arrangement, the tempo, or the beat grid, so nothing downstream has to be re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import measure_loudness, measure_true_peak, probe_audio

#: DJ library target. Loud enough to be felt, quiet enough to leave crossfade headroom.
TARGET_LUFS = -14.0

#: Ceiling for the mastered render. The master limiter at playback sits below this.
CEILING_DBTP = -1.5


@dataclass(frozen=True, slots=True)
class MasterReport:
    """What mastering changed, so it is visible rather than silent."""

    input_lufs: float
    output_lufs: float
    input_peak_dbtp: float
    output_peak_dbtp: float
    gain_db: float
    #: True when limiting was needed to reach the ceiling.
    limited: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "input_lufs": round(self.input_lufs, 2),
            "output_lufs": round(self.output_lufs, 2),
            "input_peak_dbtp": round(self.input_peak_dbtp, 2),
            "output_peak_dbtp": round(self.output_peak_dbtp, 2),
            "gain_db": round(self.gain_db, 2),
            "limited": self.limited,
        }


def _soft_limit(samples: np.ndarray, ceiling: float) -> tuple[np.ndarray, bool]:
    """Bring peaks under ``ceiling`` with a soft knee rather than a hard clip.

    Hard clipping at 0 dBTP is what the render already does to itself. Clipping again would just
    add distortion on top of it.
    """
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        return samples, False

    ceiling_linear = 10 ** (ceiling / 20.0)
    if peak <= ceiling_linear:
        return samples, False

    # The knee must sit BELOW the ceiling. Putting it above means nothing ever exceeds it, so the
    # curve is never entered, and a peak well above the ceiling passes through untouched.
    knee = ceiling_linear * 0.7
    span = ceiling_linear - knee

    out = np.empty_like(samples)
    magnitude = np.abs(samples)
    over = magnitude > knee
    out[~over] = samples[~over]
    shaped = knee + span * np.tanh((magnitude[over] - knee) / span)
    out[over] = np.sign(samples[over]) * shaped

    # tanh asymptotes at the ceiling but never reaches it, so the tail is scaled to guarantee it.
    if float(np.max(np.abs(out))) > ceiling_linear:
        out *= ceiling_linear / float(np.max(np.abs(out)))
    return out, True


def master(
    path: str | Path,
    *,
    target_lufs: float = TARGET_LUFS,
    ceiling_dbtp: float = CEILING_DBTP,
    overwrite: bool = True,
) -> MasterReport:
    """Normalise and limit an audio file in place. Returns what changed.

    The true-peak ceiling is applied *before* the loudness gain, because gaining a signal up can
    push a previously-compliant peak over the limit.
    """
    import soundfile as sf

    probe, samples = probe_audio(path)
    before = measure_loudness(samples, probe.sample_rate_hz)
    before_peak = measure_true_peak(samples, probe.sample_rate_hz)

    # Normalise FIRST, then limit. Doing it the other way round means the gain pushes the peaks
    # back over the ceiling and the limiter's work is undone — which is exactly what happened:
    # four of five tracks came out at exactly 0.0 dBTP while reporting `limited=True`.
    if not np.isfinite(before.integrated_lufs) or before.integrated_lufs <= -70:
        gain_db = 0.0
    else:
        gain_db = target_lufs - before.integrated_lufs

    gained = np.clip(samples * (10 ** (gain_db / 20.0)), -1.0, 1.0)
    mastered, was_limited = _soft_limit(gained, ceiling_dbtp)
    if overwrite:
        sf.write(str(path), mastered, probe.sample_rate_hz, subtype="PCM_24")

    after = measure_loudness(mastered, probe.sample_rate_hz)
    return MasterReport(
        input_lufs=before.integrated_lufs,
        output_lufs=after.integrated_lufs,
        input_peak_dbtp=before_peak,
        output_peak_dbtp=measure_true_peak(mastered, probe.sample_rate_hz),
        gain_db=gain_db,
        limited=was_limited,
    )


def library_loudness(paths: list[Path]) -> dict[str, object]:
    """Integrated loudness across a whole library.

    This is the check that matters for a set, and it is a *library* property rather than a
    per-track one: any spread here becomes an audible jump at every transition.
    """
    readings = []
    for path in paths:
        probe, samples = probe_audio(path)
        level = measure_loudness(samples, probe.sample_rate_hz).integrated_lufs
        readings.append((path.stem, level))
    if not readings:
        return {"tracks": 0, "spread_lu": 0.0}

    values = [v for _, v in readings]
    return {
        "tracks": len(readings),
        "spread_lu": round(max(values) - min(values), 2),
        "min_lufs": round(min(values), 2),
        "max_lufs": round(max(values), 2),
        "readings": {name: round(value, 2) for name, value in sorted(readings)},
    }
