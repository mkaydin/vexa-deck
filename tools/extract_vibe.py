"""Extract the features that make "vibe" something a selector can condition on.

``DecisionContext`` carries bpm, key, section, bars_to_boundary, energy, vocals and recent families.
That is enough to schedule a transition and not enough to choose one that *fits*. Two tracks at 122
BPM in A minor can be identical to every feature the selector sees and still be obviously
incompatible to a listener, because one is four-to-the-floor house and the other a drifting lo-fi
beat.

These are the measurements that close that gap:

* **Chord labels** -- rough harmony estimates from the full mix, not a transcription.
* **Tuning** -- estimated concert-pitch offset. Unknown is recorded as unknown.
* **Onset-grid alignment** -- the fraction of detected transients close to detected beats. This
  alone does not identify a drum pattern or classify a track as suitable for mixing.
* **Tonal balance** -- low against high energy. Warm versus bright is a real perceptual axis and
  currently unrepresented.

An earlier 14-track Essentia tempo/key comparison found no clear improvement. This extractor uses
librosa CQT chroma without a new dependency. Its descriptors remain estimates until checked against
listening or annotated audio.

Spot-checked against known-good musical results: one D-major track reads
``D major -> B minor -> G major -> F# minor`` (I-vi-IV-ii), and an F-minor one alternates
``F minor / C# major`` (i-VI).

Results go to a sidecar JSON per track. Nothing is overwritten: the gates still run on the original
measurements, and these are extra evidence rather than a replacement for it.

Usage::

    uv run --no-sync python tools/extract_vibe.py                  # all tracks
    uv run --no-sync python tools/extract_vibe.py --limit 5        # spot check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers/yue2/src"))

LIBRARY = ROOT / "assets" / "library"

#: Chord templates as semitone offsets from the root. Hearing "Am" means the root *and* its
#: quality, so both triads are matched.
MAJOR = (0, 4, 7)
MINOR = (0, 3, 7)
ROOT_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: How many chord labels to keep per track. Enough to characterise a progression without turning the
#: sidecar into a transcript.
MAX_CHORD_LABELS = 16
EXTRACTOR_VERSION = 3


def _mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    return samples.mean(axis=1), sr


def _match_chord(profile: np.ndarray) -> str:
    """Best (root, quality) for one chroma frame, by rotated template correlation."""
    values = profile - profile.mean()
    norm = float(np.linalg.norm(values))
    if norm < 1e-9:
        return "unknown"
    values = values / norm
    best_score, best = -2.0, ("C", "major")
    for root in range(12):
        for quality, template in (("major", MAJOR), ("minor", MINOR)):
            candidate = np.zeros(12)
            for interval in template:
                candidate[(root + interval) % 12] = 1.0
            candidate -= candidate.mean()
            score = float(np.dot(values, candidate))
            if score > best_score:
                best_score, best = score, (ROOT_NAMES[root], quality)
    return f"{best[0]} {best[1]}"


def chords_and_tuning(path: Path) -> tuple[list[str], float | None]:
    """Chord labels across the track, and the tuning it was mastered at.

    Chroma comes from a constant-Q transform, which resolves pitch better than an STFT at the low
    register where bass roots live. Labels are collapsed to their distinct sequence so the sidecar
    records a progression rather than every window.
    """
    import librosa

    audio, sr = _mono(path)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=2048)
    if chroma.shape[1] == 0:
        return [], None

    step = max(1, chroma.shape[1] // MAX_CHORD_LABELS)
    labels: list[str] = []
    for start in range(0, chroma.shape[1], step):
        window = chroma[:, start:start + step].mean(axis=1)
        labels.append(_match_chord(window))
        if len(labels) >= MAX_CHORD_LABELS:
            break

    try:
        # librosa returns fractions of a semitone, not hertz. Its audio argument is keyword-only.
        offset = float(librosa.estimate_tuning(y=audio, sr=sr))
        tuning_hz = 440.0 * 2.0 ** (offset / 12.0) if np.isfinite(offset) else None
    except (ValueError, RuntimeError):
        tuning_hz = None
    return labels, tuning_hz


def rhythm_and_tone(path: Path) -> dict[str, float | str]:
    """Rhythm type and tonal balance, measured from the audio rather than taken from the prompt."""
    import librosa
    import soundfile as sf

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)

    # This is only an onset/beat proximity measurement. It does not classify four-on-the-floor,
    # syncopation, or whether two tracks sound good together.
    tempo = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
    )[0])
    beats = np.asarray(
        librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="time")[1], dtype=float
    )
    on_grid = 0
    total = 0
    if beats.size > 4 and tempo > 0:
        # Half a beat classifies almost every onset as on-grid. A tenth of a beat measures
        # whether the transient is actually near a detected beat rather than merely nearby.
        tolerance = 0.1 * 60.0 / tempo
        peaks = librosa.onset.onset_detect(y=mono, sr=sr, units="time", backtrack=False)
        for peak in peaks:
            nearest = float(np.min(np.abs(beats - peak)))
            if nearest <= tolerance:
                on_grid += 1
            total += 1
    fraction = on_grid / total if total else None

    contrast = librosa.feature.spectral_contrast(y=mono, sr=sr).mean(axis=1)
    low, high = float(contrast[1]), float(contrast[-2])

    window = mono[: min(mono.size, sr * 20)]
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(window.size, d=1.0 / sr)
    power = np.square(spectrum)
    total_energy = float(power.sum()) or 1.0
    # Preserve the numerical measurement without turning an arbitrary threshold into a genre tag.
    return {
        "rhythm": "unclassified" if fraction is not None else "unknown",
        "onset_grid_alignment": round(fraction, 3) if fraction is not None else None,
        "on_grid_onset_fraction": round(fraction, 3) if fraction is not None else None,
        "detected_onsets": total,
        "low_high_ratio": round(low / (high + 1e-9), 3),
        "sub_bass_fraction": round(float(power[freqs < 120].sum()) / total_energy, 4),
        "presence_fraction": round(
            float(power[(freqs >= 1000) & (freqs < 6000)].sum()) / total_energy, 4
        ),
    }


def _collapse(labels: list[str]) -> list[str]:
    """Distinct sequence, not distinct set: ``A A B B`` is a progression, ``A B`` is not."""
    collapsed: list[str] = []
    for label in labels:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)
    return collapsed


def extract(path: Path) -> dict:
    labels, tuning_hz = chords_and_tuning(path)
    features = rhythm_and_tone(path)
    progression = _collapse(labels)
    features.update(
        extractor_version=EXTRACTOR_VERSION,
        source_sha256=_sha256(path),
        chords=progression,
        chord_count=len(progression),
        distinct_chords=sorted(set(progression) - {"unknown"}),
        tuning_hz=round(tuning_hz, 2) if tuning_hz is not None else None,
    )
    return features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as audio:
        for block in iter(lambda: audio.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="stop after N tracks (0 = all)")
    parser.add_argument("--out", type=Path, default=LIBRARY)
    args = parser.parse_args()

    tracks = sorted(args.out.glob("*.wav"))
    if args.limit:
        tracks = tracks[: args.limit]
    if not tracks:
        print("no tracks found", file=sys.stderr)
        return 2

    print(f"extracting vibe features from {len(tracks)} tracks ...", flush=True)
    written = 0
    for index, wav in enumerate(tracks, 1):
        try:
            features = extract(wav)
        except Exception as exc:  # one bad track must not stop the run
            print(f"  [{index}/{len(tracks)}] {wav.name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        wav.with_suffix(".vibe.json").write_text(
            json.dumps(features, indent=2, sort_keys=True), encoding="utf-8"
        )
        written += 1
        if index % 25 == 0 or index == len(tracks):
            print(f"  {index}/{len(tracks)}", flush=True)

    print(f"wrote {written} vibe sidecars")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
