"""Measure whether the depot is actually varied, without listening to it.

The depot is built from 12 tempo palettes, so at 200 tracks each palette contributes ~17 renders
that share a style prompt. Nothing forces them to be *distinguishable*. If they are near-duplicates
then a listener's A/B judgements carry almost no signal — every option in a menu sounds the same, so
"which do you prefer" has no answer worth learning.

That risk is checkable without ears. Renders produced from the same prompt land close together in
timbre, so an MFCC distance separates them much the way a listener would describe them: "these four
are basically the same loop".

This reports, per palette, how many distinct timbres exist and how many near-duplicate pairs.
**It is triage, not a quality judgement.** A low timbre distance does not prove a track is bad, only
that labelling it separately adds little.

One file is loaded at a time. Holding 200 x 45 s of stereo audio at 48 kHz at once is ~3.5 GB,
which is the kind of thing that caused the memory trouble earlier in this project.

Usage::

    uv run --no-sync python tools/depot_diversity.py
    uv run --no-sync python tools/depot_diversity.py --palette 124
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers/yue2/src"))

LIBRARY = ROOT / "assets" / "library"

#: Standardised fingerprints closer than this count as the same timbre. Set from the shape of the
#: observed distribution rather than from theory -- the histogram is printed on every run so the
#: threshold can be sanity-checked against real numbers.
SAME_TIMBRE_DISTANCE = 3.0

N_MFCC = 20


def descriptor(path: Path) -> np.ndarray | None:
    """A compact fingerprint: MFCC means plus broad spectral shape.

    Deliberately cheap and small. It exists to rank tracks by how similar they sound to each other,
    not to characterise them.
    """
    import librosa
    import soundfile as sf

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if mono.shape[0] < sr:
        return None
    mono = mono[: sr * 30]  # fixed slice keeps tracks comparable regardless of length
    if float(np.max(np.abs(mono))) < 1e-4:
        return None

    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([
        mfcc.mean(axis=1) / 100.0,
        librosa.feature.spectral_centroid(y=mono, sr=sr).mean(axis=1) / (sr / 2),
        librosa.feature.spectral_contrast(y=mono, sr=sr).mean(axis=1),
        librosa.feature.spectral_bandwidth(y=mono, sr=sr).mean(axis=1) / (sr / 2),
    ])


def standardise(matrix: np.ndarray) -> np.ndarray:
    """Z-score every dimension across the corpus.

    This is the difference between a working measure and a broken one. Raw mean-MFCC vectors share
    a dominant low-frequency profile, so cosine similarity between *any* two of them lands near 1
    and the distance near 0 -- measured on the first run, all 780 pairs of a corpus containing ten
    different briefs came back below 0.06 and every palette collapsed into one cluster. That was
    the descriptor, not the depot. Standardising removes the shared offset and lets the dimensions
    that actually vary decide the distance.
    """
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    return (matrix - mean) / std


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between standardised fingerprints, ignoring the level coefficient MFCCs carry."""
    return float(np.linalg.norm(a[1:] - b[1:]))


def cluster(names: list[str], vectors: list[np.ndarray], threshold: float) -> tuple[int, int]:
    """Single-linkage clustering. Returns ``(cluster_count, near_duplicate_pairs)``.

    Single linkage is a lower bound on distinctness, which is the safe direction: it cannot
    overstate how varied the depot is.
    """
    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    near_duplicates = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if euclidean(vectors[i], vectors[j]) >= threshold:
                continue
            near_duplicates += 1
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    return len({find(i) for i in range(len(names))}), near_duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palette", default=None, help="only this tempo palette")
    parser.add_argument("--distance", type=float, default=SAME_TIMBRE_DISTANCE)
    args = parser.parse_args()

    tracks: dict[str, Path] = {}
    for wav in sorted(LIBRARY.glob("*.wav")):
        if wav.stem.startswith("depot-"):
            parts = wav.stem.split("-")
            if len(parts) >= 3 and (args.palette is None or parts[1] == args.palette):
                tracks[wav.stem] = wav
        else:
            tracks[wav.stem] = wav

    if not tracks:
        print("no tracks found", file=sys.stderr)
        return 2

    print(f"describing {len(tracks)} tracks ...", flush=True)
    fingerprints: dict[str, np.ndarray] = {}
    for name, path in tracks.items():
        vector = descriptor(path)
        if vector is not None:
            fingerprints[name] = vector
    print(f"  {len(fingerprints)} described\n")

    by_palette: dict[str, list[str]] = collections.defaultdict(list)
    for name in fingerprints:
        parts = name.split("-")
        palette = parts[1] if name.startswith("depot-") and len(parts) >= 3 else "original"
        by_palette[palette].append(name)

    all_names = sorted(fingerprints)
    matrix = standardise(np.stack([fingerprints[n] for n in all_names]))
    all_vectors = [matrix[i] for i in range(len(all_names))]
    distances = [
        euclidean(all_vectors[i], all_vectors[j])
        for i in range(len(all_names))
        for j in range(i + 1, len(all_names))
    ]
    if distances:
        histogram = np.histogram(distances, bins=10)
        print("--- pairwise timbre distance, standardised (sanity-check the threshold against these) ---")
        for lo, count in zip(histogram[1][:-1], histogram[0], strict=True):
            bar = "#" * int(count / max(1, histogram[0].max()) * 40)
            span = float(np.subtract(*np.histogram(distances, bins=11)[1][1:3]))
            print(f"  {lo:6.2f}  {count:5d}  {bar}")
        print(f"\n  threshold {args.distance} sits in the dense region; "
              f"{sum(1 for d in distances if d < args.distance)} pairs fall below it\n")

    print("--- distinct timbres per palette ---")
    print(f"{'palette':>9} {'tracks':>7} {'clusters':>9} {'near-dup pairs':>15}")
    total_clusters = total_pairs = 0
    for palette in sorted(by_palette, key=lambda k: (len(k), k)):
        names = sorted(by_palette[palette])
        by_name = dict(zip(all_names, all_vectors, strict=True))
        clusters, near_dup = cluster(names, [by_name[n] for n in names], args.distance)
        total_clusters += clusters
        total_pairs += near_dup
        print(f"{palette:>9} {len(names):>7} {clusters:>9} {near_dup:>15}")

    print(
        f"\ntotals: {len(fingerprints)} tracks, ~{total_clusters} distinct timbres, "
        f"{total_pairs} near-duplicate pairs"
    )
    print(
        "\nReading this: a palette where tracks ~= clusters means every render of that style\n"
        "sounds the same, so a menu drawn from it offers no real choice and labels collected\n"
        "there carry little signal. This measures similarity, not quality -- a low score means\n"
        "'not worth listening to separately', not 'bad'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())