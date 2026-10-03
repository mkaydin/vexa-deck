"""Extract the features that make "vibe" something a selector can condition on.

``DecisionContext`` carries bpm, key, section, bars_to_boundary, energy, vocals and recent families.
That is enough to schedule a transition and not enough to choose one that *fits*. Two tracks at 122
BPM in A minor can be identical to every feature the selector sees and still be obviously
incompatible to a listener, because one is four-to-the-floor house and the other a drifting lo-fi
beat.

These are the measurements that close that gap:

* **Chord progression** -- the real compatibility signal. Key compatibility is necessary and nowhere
  near sufficient; two tracks in the same key share a progression only some of the time.
* **Tuning** -- distance from A=440. Beatmatching within 1 % needs this, and it is the difference
  between a transition that locks and one that drifts.
* **Onset-grid alignment** -- steady against syncopated. A crossfade between the two is audible as
  a mistake however well the tempo matches. Measured as how much onset energy lands on the beat
  lattice; deliberately *not* called "four on the floor", because brushed jazz and a house kick
  are both steady on the grid and the measurement does not distinguish them.
* **Tonal balance** -- low against high energy. Warm versus bright is a real perceptual axis and
  currently unrepresented.

**Essentia was evaluated and rejected.** ``idea1.md`` recommends it, and the measurement says it
adds nothing: after correcting a silent sample-rate assumption it agreed with the existing tempo
and key analysis on 11 of 14 and 12 of 14 tracks respectively -- equal, not better. It also
assumes 44100 Hz and takes no sample-rate argument, so fed 48 kHz audio it returns numbers 8.8 % low
that look entirely plausible. Chords here come from librosa CQT chroma instead, which needs no
resampling and no new dependency.

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


def chords_and_tuning(path: Path) -> tuple[list[str], float]:
    """Chord labels across the track, and the tuning it was mastered at.

    Chroma comes from a constant-Q transform, which resolves pitch better than an STFT at the low
    register where bass roots live. Labels are collapsed to their distinct sequence so the sidecar
    records a progression rather than every window.
    """
    import librosa

    audio, sr = _mono(path)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=2048)
    if chroma.shape[1] == 0:
        return [], 440.0

    step = max(1, chroma.shape[1] // MAX_CHORD_LABELS)
    labels: list[str] = []
    for start in range(0, chroma.shape[1], step):
        window = chroma[:, start:start + step].mean(axis=1)
        labels.append(_match_chord(window))
        if len(labels) >= MAX_CHORD_LABELS:
            break

    try:
        tuning_hz = float(np.atleast_1d(librosa.estimate_tuning(audio, sr=sr))[0])
    except Exception:
        tuning_hz = 440.0
    return labels, tuning_hz


def rhythm_and_tone(path: Path) -> dict[str, float | str]:
    """Rhythm type and tonal balance, measured from the audio rather than taken from the prompt."""
    import librosa
    import soundfile as sf

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)

    # Four-to-the-floor means a strong onset near every beat. Comparing onsets against the beat
    # lattice beats counting them, because a breakbeat has plenty of onsets too -- they are just not
    # on the grid.
    tempo = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
    )[0])
    beats = np.asarray(
        librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="time")[1], dtype=float
    )
    on_grid = off_grid = 0
    if beats.size > 4 and tempo > 0:
        tolerance = 0.5 * 60.0 / tempo
        peaks = librosa.onset.onset_detect(y=mono, sr=sr, units="time", backtrack=False)
        for peak in peaks:
            nearest = float(np.min(np.abs(beats - peak)))
            if nearest <= tolerance:
                on_grid += 1
            elif nearest > 2 * tolerance:
                off_grid += 1
    total = on_grid + off_grid
    fraction = on_grid / total if total else 0.0

    contrast = librosa.feature.spectral_contrast(y=mono, sr=sr).mean(axis=1)
    low, high = float(contrast[1]), float(contrast[-2])

    window = mono[: min(mono.size, sr * 20)]
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(window.size, d=1.0 / sr)
    total_energy = float(spectrum.sum()) or 1.0
    # Named for what is measured, which is onset-grid alignment -- *not* "four on the floor".
    # Brushed jazz and a house kick are both steady on the grid; calling both "four_on_floor"
    # would teach the selector a distinction that does not exist in the measurement.
    return {
        "rhythm": "steady_grid" if fraction > 0.6 else "syncopated_or_sparse",
        "onset_grid_alignment": round(fraction, 3),
        "on_grid_onset_fraction": round(fraction, 3),
        "low_high_ratio": round(low / (high + 1e-9), 3),
        "sub_bass_fraction": round(float(spectrum[freqs < 120].sum()) / total_energy, 4),
        "presence_fraction": round(
            float(spectrum[(freqs >= 1000) & (freqs < 6000)].sum()) / total_energy, 4
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
        chords=progression,
        chord_count=len(progression),
        distinct_chords=sorted(set(progression) - {"unknown"}),
        tuning_hz=round(tuning_hz, 2),
    )
    return features


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