"""Offline beat and cue analysis for an already admitted stereo asset.

This does not certify phrase structure. A detected beat can support an entry cue; a downbeat is
offered only when one of the four beat phases has a clear accent. Both retain confidence and the
source hash so changed audio cannot silently reuse old cue positions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .loader import file_sha256

CUE_VERSION = 3


@dataclass(frozen=True, slots=True)
class CueMap:
    source_sha256: str
    version: int
    bpm: float
    beat_confidence: float
    beat_times_s: tuple[float, ...]
    downbeat_confidence: float
    downbeat_times_s: tuple[float, ...]
    entry_s: float | None
    exit_s: float | None
    cue_kind: str

    def valid_for(self, path: Path) -> bool:
        # v3 adds bounded early exits. Existing admitted v2 cues remain readable; a
        # policy update must not discard an unrelated ready slot or erase its buffer.
        return self.version in (2, CUE_VERSION) and self.source_sha256 == file_sha256(path)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CueMap:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["beat_times_s"] = tuple(data["beat_times_s"])
        data["downbeat_times_s"] = tuple(data["downbeat_times_s"])
        return cls(**data)


def analyse_cues(path: Path, *, expected_bpm: float | None = None) -> CueMap:
    """Measure a beat grid and conservative entry/exit cues outside the audio callback."""
    import librosa
    import soundfile as sf

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    duration = len(mono) / sr
    digest = file_sha256(path)
    if duration < 5 or float(np.sqrt(np.mean(np.square(mono)))) < 1e-4:
        return CueMap(digest, CUE_VERSION, 0.0, 0.0, (), 0.0, (), None, None, "unknown")

    hop = 512
    onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])
    times = np.asarray(librosa.frames_to_time(frames, sr=sr, hop_length=hop), dtype=float)
    if len(times) < 8 or bpm <= 0:
        return CueMap(digest, CUE_VERSION, bpm, 0.0, (), 0.0, (), None, None, "unknown")

    intervals = np.diff(times)
    regularity = max(0.0, 1.0 - float(np.std(intervals) / np.mean(intervals)))
    tempo_fit = 1.0
    if expected_bpm and expected_bpm > 0:
        ratio = bpm / expected_bpm
        ratio = min(abs(ratio - 1), abs(ratio / 2 - 1), abs(ratio * 2 - 1))
        tempo_fit = max(0.0, 1.0 - ratio / 0.1)
    confidence = round(min(regularity, tempo_fit), 3)
    if confidence < 0.6:
        return CueMap(digest, CUE_VERSION, bpm, confidence, (), 0.0, (), None, None, "unknown")

    beat_times = tuple(round(float(t), 4) for t in times)
    accent = onset[np.clip(frames, 0, len(onset) - 1)]
    phase_means = np.array([float(np.mean(accent[i::4])) for i in range(4)])
    best_phase = int(np.argmax(phase_means))
    mean = float(np.mean(phase_means))
    phase_strength = (float(phase_means[best_phase]) - mean) / max(mean, 1e-9)
    downbeat_confidence = round(min(1.0, max(0.0, phase_strength)), 3)
    downbeats = (
        tuple(beat_times[i] for i in range(best_phase, len(beat_times), 4))
        if downbeat_confidence >= 0.25
        else ()
    )
    # A long beatless outro is not a reason to render the whole track again. Prefer the
    # usual late exit, then search a bounded earlier window. Never extrapolate the grid:
    # the four-second fade and two-second margin must stay inside measured, regular beats.
    period = 60.0 / bpm

    def supported_exit(t: float) -> bool:
        nearby = times[(times >= t - max(2.0, 2 * period)) & (times <= t + 6.0 + period)]
        if len(nearby) < 4 or nearby[-1] < t + 6.0:
            return False
        gaps = np.diff(nearby)
        if np.max(gaps) > 1.6 * period or abs(float(np.median(gaps)) / period - 1) > 0.2:
            return False
        # The tracker can bridge quiet passages. Require actual onset energy around
        # the overlap rather than accepting equally spaced timestamps alone.
        local_frames = frames[(times >= t) & (times <= t + 4.0)]
        floor = max(float(np.median(onset)), float(np.max(onset)) * 0.01, 1e-8)
        strengths = onset[np.clip(local_frames, 0, len(onset) - 1)]
        return bool(len(strengths) >= 2 and np.mean(strengths > floor) >= 0.5)

    exit_s = None
    candidates = beat_times
    # At most a quarter of the file (capped at 45 seconds) may be left after an exit.
    earliest = duration - max(18.0, min(45.0, duration * 0.25))
    for start in (duration - 18.0, earliest):
        for grid in (downbeats, beat_times):
            exits = [t for t in grid if start <= t <= duration - 8.0 and supported_exit(t)]
            if exits:
                exit_s, candidates = exits[-1], grid
                break
        if exit_s is not None:
            break
    # YuE2 sometimes renders a long non-rhythmic intro. The first detected beat is still a
    # useful entry if it leaves enough room to play before the exit cue.
    entry = next((t for t in candidates if exit_s is not None and t <= exit_s - 8.0), None)
    return CueMap(
        digest,
        CUE_VERSION,
        round(bpm, 3),
        confidence,
        beat_times,
        downbeat_confidence,
        downbeats,
        entry,
        exit_s,
        "downbeat" if downbeats and candidates is downbeats else "beat",
    )
