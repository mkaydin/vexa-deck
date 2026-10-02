"""Audio analysis.

Everything here runs at **ingestion**, never in the audio callback (``ARCHITECTURE.md:89``). The
loudness and peak measurements are implemented directly in numpy so they are deterministic,
dependency-light, and testable without loading a model.

Two rules shape this module:

* **Measured, not estimated.** Integrated loudness and true peak are arithmetic on samples. Beat
  grid and key are estimates and carry confidence, per ``ARCHITECTURE.md:44``.
* **Silence and clipping are caught here, not discovered during a set.** An asset that fails a
  gate must never reach the scheduler (``ARCHITECTURE.md:46``).

Channel handling is explicit throughout: every measurement reduces to mono *before* resampling.
That ordering matters — ``shape[-1]`` on a ``(n, channels)`` array is the channel count, so a
resample helper that assumes 1-D will quietly resample the wrong axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: BS.1770-4 is defined at 48 kHz. Anything else is resampled before measurement, which is what
#: every production loudness meter does.
REFERENCE_RATE = 48000

#: Gating thresholds from BS.1770-4.
_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0
_BLOCK_S = 0.4
_OVERLAP = 0.75

# K-weighting coefficients at 48 kHz, from ITU-R BS.1770-4.
_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_HIGHPASS_B = np.array([1.0, -2.0, 1.0])
_HIGHPASS_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

#: Gain the K-weighting curve applies at 1 kHz. The standard is calibrated so a sine whose *RMS*
#: is -23 dBFS reads exactly -23 LUFS; the +0.7 dB shelf/highpass contribution is what makes that
#: true, so it is stated here rather than discovered as a mysterious offset.
K_WEIGHTING_GAIN_DB_1KHZ = 0.698


class AnalysisError(RuntimeError):
    """Raised when audio cannot be read at all. Distinct from a gate *failure*."""


@dataclass(frozen=True, slots=True)
class AudioProbe:
    """What a decode pass establishes before any measurement runs."""

    path: str
    codec: str
    sample_rate_hz: int
    channels: int
    duration_s: float
    frames: int
    #: Fraction of samples at or beyond full scale. Non-zero means the source is already clipped.
    clipped_sample_ratio: float
    #: RMS of the whole signal. Near-zero is silence.
    rms: float


@dataclass(frozen=True, slots=True)
class LoudnessMeasurement:
    integrated_lufs: float
    #: Loudest short-term window in LUFS. More useful than the integrated value for spotting a
    #: track that suddenly jumps.
    max_short_term_lufs: float
    #: Range across short-term windows, measured over the *body* of the track.
    #:
    #: The outer 10% is excluded because nobody transitions out of a fade-in or into a fade-out.
    #: Including them inflated the figure badly on generated material: one 45 s render measured
    #: 64.9 LU over the whole file but 8.9 LU once the fades were dropped.
    loudness_range_lu: float

@dataclass(frozen=True, slots=True)
class TemporalAnalysis:
    """Estimated, not measured. Every field carries how much to trust it."""

    bpm: float
    beat_confidence: float
    key: str | None
    key_confidence: float
    #: Section boundaries in bars, when a tempo is known.
    section_bars: list[int] = field(default_factory=list)
    #: Whether a beat-aligned loop could be cut at all.
    has_stable_grid: bool = False



# --- channel and rate handling ----------------------------------------------


def _mono(samples: np.ndarray) -> np.ndarray:
    """Mix to mono by averaging channels.

    Sound arrays are ``(frames, channels)``, so channels are the **last** axis. Averaging axis 0
    instead collapses the sample axis and returns one value per channel, which then reads as a
    two-sample signal and measures as silence.
    """
    return samples if samples.ndim == 1 else samples.mean(axis=-1)



def _resample_1d(samples: np.ndarray, sr: int, target: int = REFERENCE_RATE) -> np.ndarray:
    """Resample a **1-D** signal. Reduce channels before calling."""
    if sr == target:
        return samples
    if samples.ndim != 1:
        raise ValueError(
            f"expected a 1-D signal, got shape {samples.shape}; reduce channels first"
        )
    n_out = round(samples.shape[0] / sr * target)
    if n_out < 1:
        return samples
    source_idx = np.linspace(0, samples.shape[0] - 1, n_out)
    return np.interp(source_idx, np.arange(samples.shape[0]), samples)


def _mono_at_reference(samples: np.ndarray, sr: int) -> np.ndarray:
    """Mono, at 48 kHz. The single entry point every measurement starts from."""
    return _resample_1d(_mono(samples), sr)


def _biquad(samples: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Direct-form-I biquad.

    A Python loop, deliberately: these buffers are a few hundred thousand samples at ingestion
    time and never on a realtime path, so clarity beats a compiled dependency here.
    """
    out = np.empty_like(samples)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for n in range(samples.shape[0]):
        x0 = float(samples[n])
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[n] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return out


def k_weight(samples: np.ndarray, sr: int = REFERENCE_RATE) -> np.ndarray:
    """Apply the BS.1770-4 K-weighting curve. Accepts 1-D or ``(n, channels)``."""
    mono = _mono_at_reference(samples, sr)
    mono = _biquad(mono, _SHELF_B, _SHELF_A)
    return _biquad(mono, _HIGHPASS_B, _HIGHPASS_A)


# --- loudness ---------------------------------------------------------------


def measure_loudness(
    samples: np.ndarray, sr: int, *, block_s: float = _BLOCK_S, overlap: float = _OVERLAP
) -> LoudnessMeasurement:
    """Gated integrated loudness per ITU-R BS.1770-4.

    Implements the two-stage gating the standard actually specifies: 400 ms mean-square blocks at
    75% overlap, an absolute gate at -70 LUFS, then a relative gate 10 LU below the mean of
    whatever survived the absolute gate. Without the relative gate a long quiet tail drags the
    integrated figure down and a quiet track is misreported as loud.
    """
    weighted = k_weight(samples, sr)
    if weighted.shape[0] == 0:
        return LoudnessMeasurement(-np.inf, -np.inf, 0.0)

    block = int(block_s * REFERENCE_RATE)
    step = max(1, int(block * (1.0 - overlap)))
    if weighted.shape[0] <= block:
        block = int(weighted.shape[0])
        step = max(1, block)

    starts = range(0, weighted.shape[0] - block + 1, step)
    powers = np.array(
        [float(np.mean(np.square(weighted[s : s + block]))) for s in starts]
    )
    if powers.size == 0:
        return LoudnessMeasurement(-np.inf, -np.inf, 0.0)

    block_lufs = -0.691 + 10.0 * np.log10(np.maximum(powers, 1e-20))

    above_absolute = block_lufs > _ABSOLUTE_GATE_LUFS
    if not np.any(above_absolute):
        return LoudnessMeasurement(-np.inf, -np.inf, 0.0)

    mean_above = -0.691 + 10.0 * np.log10(max(float(np.mean(powers[above_absolute])), 1e-20))
    relative_gate = mean_above + _RELATIVE_GATE_LU

    keep = above_absolute & (block_lufs > relative_gate)
    selected = powers[keep] if np.any(keep) else powers[above_absolute]

    integrated = -0.691 + 10.0 * np.log10(max(float(np.mean(selected)), 1e-20))

    # Range over the usable body, not the whole file. A fade-in is not a loudness jump a
    # listener would ever hear at a transition, and counting it made ordinary material look
    # unusable.
    body = block_lufs[keep]
    if body.size >= 10:
        margin = body.size // 10
        core = body[margin : body.size - margin]
    else:
        core = body
    spread = float(core.max() - core.min()) if core.size else 0.0

    return LoudnessMeasurement(
        integrated_lufs=float(integrated),
        max_short_term_lufs=float(block_lufs.max()),
        loudness_range_lu=spread,
    )


def measure_true_peak(samples: np.ndarray, sr: int, *, oversample: int = 4) -> float:
    """Oversampled true peak in dBTP.

    Sample peaks under-report intersample overshoot, which is exactly where a crossfade clips.
    """
    mono = _mono_at_reference(samples, sr)
    if mono.size == 0:
        return -np.inf
    if oversample > 1:
        n_up = mono.shape[0] * oversample
        target = np.linspace(0, mono.shape[0] - 1, n_up)
        upsampled = np.interp(target, np.arange(mono.shape[0]), mono)
        peak = float(np.max(np.abs(upsampled)))
    else:
        peak = float(np.max(np.abs(mono)))
    return 20.0 * np.log10(max(peak, 1e-12))


# --- decode -----------------------------------------------------------------


def probe_audio(path: str | Path) -> tuple[AudioProbe, np.ndarray]:
    """Decode a file and establish the facts every other check depends on."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
        samples, sr = sf.read(str(path), always_2d=True)
    except Exception as exc:
        raise AnalysisError(f"cannot decode {path}: {exc}") from exc

    channels = int(samples.shape[1]) if samples.ndim == 2 and samples.shape[1] else 1
    mono = _mono(samples)
    clipped = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0

    probe = AudioProbe(
        path=str(path),
        codec=info.format or "unknown",
        sample_rate_hz=int(sr),
        channels=channels,
        duration_s=float(info.frames) / float(sr) if sr else 0.0,
        frames=int(info.frames),
        clipped_sample_ratio=clipped,
        rms=rms,
    )
    return probe, samples


# --- tempo, key, structure --------------------------------------------------


def detect_temporal(samples: np.ndarray, sr: int) -> TemporalAnalysis:
    """Estimate tempo, key and section boundaries.

    Wraps librosa because a hand-rolled beat tracker would be worse in every way that matters
    here. Confidence is deliberately conservative: ``ROADMAP.md:52`` says to refuse ambiguous
    automatic transitions, so a tracker that cannot decide must not look confident.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - analysis extra is optional
        raise AnalysisError("librosa is required for tempo and key analysis") from exc

    mono = _mono_at_reference(samples, sr)
    if mono.shape[0] < REFERENCE_RATE:
        return TemporalAnalysis(bpm=0.0, beat_confidence=0.0, key=None, key_confidence=0.0)

    onset_env = librosa.onset.onset_strength(y=mono, sr=REFERENCE_RATE)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=REFERENCE_RATE, units="frames"
    )
    bpm = float(np.atleast_1d(tempo)[0])

    # Confidence from how regular the detected beats are. A clean 4/4 grid is tight; a free
    # rhythm or a wrong half/double reading is not.
    beats = np.asarray(beat_frames)
    if beats.size > 3:
        intervals = np.diff(beats).astype(float)
        mean_interval = max(float(np.mean(intervals)), 1e-9)
        confidence = 1.0 - float(np.std(intervals)) / mean_interval
    else:
        confidence = 0.0
    confidence = float(min(1.0, max(0.0, confidence) * 1.2))

    key, key_confidence = detect_key(mono)

    section_bars: list[int] = []
    if bpm > 0:
        seconds_per_bar = 240.0 / bpm
        total_bars = int(mono.shape[0] / REFERENCE_RATE / seconds_per_bar)
        try:
            boundaries = librosa.segment.agglomerative(mono, frame_length=2048, hop_length=512)
            for boundary in boundaries:
                seconds = float(librosa.frames_to_time(boundary, sr=REFERENCE_RATE, hop_length=512))
                bar = int(seconds / seconds_per_bar)
                if 0 < bar < total_bars:
                    section_bars.append(bar)
            section_bars = sorted(set(section_bars))[:16]
        except Exception:
            section_bars = []

    return TemporalAnalysis(
        bpm=bpm,
        beat_confidence=round(confidence, 4),
        key=key,
        key_confidence=round(key_confidence, 4),
        section_bars=section_bars,
        has_stable_grid=confidence >= 0.6 and bpm > 0,
    )


#: Krumhansl-style major/minor profiles. A small, transparent heuristic rather than a model —
#: the key is an estimate with confidence, never a fact.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(mono: np.ndarray) -> tuple[str | None, float]:
    """Estimate the key from a chroma profile. Returns ``(key, confidence)``."""
    try:
        import librosa
    except ImportError:  # pragma: no cover
        return None, 0.0

    if mono.shape[0] < REFERENCE_RATE:
        return None, 0.0

    chroma = librosa.feature.chroma_cqt(y=mono, sr=REFERENCE_RATE)
    profile = np.mean(chroma, axis=1)
    if not np.any(profile):
        return None, 0.0

    best: tuple[float, int, str] | None = None
    for index in range(12):
        for name, template in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(template, index)
            score = float(np.corrcoef(profile, rotated)[0, 1])
            if best is None or score > best[0]:
                best = (score, index, name)

    if best is None:
        return None, 0.0

    score, index, name = best
    # Pearson correlation mapped onto 0..1 and capped, so a weak match never looks authoritative.
    confidence = float(max(0.0, min(0.85, (score + 1.0) / 2.0 * 0.85)))
    return f"{_NOTE_NAMES[index]} {name}", round(confidence, 4)