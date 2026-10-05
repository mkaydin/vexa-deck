"""Score how well one track follows another, using measured analysis rather than a prompt.

This is a measured compatibility heuristic for retrieval, not a replacement for listening labels.
It helps choose which transitions to preview and compare.

``ARCHITECTURE.md:64`` already puts tempo ratio and key clash in the deterministic feasibility
filter. This scorer emphasizes additional features the filter does not use:

* **Chord overlap** - whether the two progressions share harmony, and whether the shared chords are
  reached in a comparable order. Key compatibility is necessary and far from sufficient; two tracks
  in the same key can share almost nothing.
* **Tonal balance** - sub-bass and presence fractions. Four-to-the-floor house and an acoustic jazz
  cut are compatible at any tempo, and this is what shows they are not.
* **Onset-grid alignment** - compares measured onset fractions without assigning a rhythm genre.

Tempo and key are still scored, but at lower weight, because the filter already handles them.

**What this cannot be.** The result is a preference signal nobody endorsed. ``IDEAS.md:57`` is right
that a model trained on rules reproduces the rules. The difference here is that the rules encode
harmony and timbre rather than bare tempo, so the learned function is strictly richer than the
filter. Its ceiling is this module's ceiling, and it should be measured against the filter rather
than assumed to beat it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LIBRARY = Path(__file__).resolve().parent.parent / "assets" / "library"

#: How much each term contributes. Tempo and key are present but deliberately subdominant, since the
#: feasibility filter already enforces them and a model relearning them adds nothing.
WEIGHTS: dict[str, float] = {
    "chord_overlap": 0.30,
    "tonal_distance": 0.22,
    "grid_match": 0.16,
    "energy_flow": 0.14,
    "tempo": 0.10,
    "key": 0.08,
}

#: A floor on key compatibility, mirroring the filter's refusal. Anything below this is a mode clash
#: regardless of how well the harmony lines up.
KEY_COMPATIBLE = 0.25

PITCH_CLASSES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "F": 5, "E#": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


@dataclass(frozen=True, slots=True)
class TrackProfile:
    """Everything the scorer needs about one track."""

    asset_id: str
    bpm: float
    key: str | None
    scale: str | None
    energy: float
    chords: frozenset[str]
    chord_order: tuple[str, ...]
    sub_bass: float
    presence: float
    low_high: float
    onset_alignment: float | None


def _pitch_class(token: str) -> int | None:
    name = token.split()[0] if token.split() else token
    return PITCH_CLASSES.get(name)


def _chord_root(chord: str) -> int | None:
    return _pitch_class(chord)


def load_profiles(library: Path = LIBRARY) -> dict[str, TrackProfile]:
    """One profile per admitted track, merging the manifest with its vibe sidecar."""
    profiles: dict[str, TrackProfile] = {}
    for manifest_path in sorted(library.glob("*.json")):
        if any(part in manifest_path.name for part in (".request.", ".vibe.", ".cues.")):
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "asset_id" not in data:
            continue
        vibe_path = manifest_path.with_suffix(".vibe.json")
        vibe = (
            json.loads(vibe_path.read_text(encoding="utf-8"))
            if vibe_path.exists()
            else {}
        )
        tags = {
            name: entries[0]["value"]
            for name, entries in data.get("tags", {}).items()
            if entries
        }
        beat_grid = data.get("beat_grid") or {}
        key_info = data.get("key") or {}
        key_value = key_info.get("value") or ""

        chords = tuple(vibe.get("chords", []))
        profiles[data["asset_id"]] = TrackProfile(
            asset_id=data["asset_id"],
            bpm=float(beat_grid.get("bpm") or 0.0),
            key=key_value or None,
            scale=key_value.split()[1] if " " in key_value else None,
            energy=float(tags.get("energy", 0.5)),
            chords=frozenset(c for c in chords if c != "unknown"),
            chord_order=chords,
            sub_bass=float(vibe.get("sub_bass_fraction", 0.1)),
            presence=float(vibe.get("presence_fraction", 0.25)),
            low_high=float(vibe.get("low_high_ratio", 0.6)),
            onset_alignment=vibe.get("onset_grid_alignment"),
        )
    return profiles


# --- individual terms, each in [0, 1] where 1 is most compatible ---


def chord_overlap(current: TrackProfile, candidate: TrackProfile) -> float:
    """Shared harmony, weighted by how similar the two progressions are in *order*.

    Set overlap alone is too generous: two tracks in a major key share I, IV and V whether they move
    between them as a loop or as a slow progression. Comparing the sequences catches that.
    """
    if not current.chords or not candidate.chords:
        return 0.5  # unknown is neutral, not penalised
    shared = current.chords & candidate.chords
    if not shared:
        return 0.0

    union = len(current.chords | candidate.chords)
    set_score = len(shared) / union if union else 0.0

    roots_a = [r for r in (_chord_root(c) for c in current.chord_order) if r is not None]
    roots_b = [r for r in (_chord_root(c) for c in candidate.chord_order) if r is not None]
    order_score = 0.5
    if len(roots_a) >= 2 and len(roots_b) >= 2:
        # Mean shortest interval between aligned roots: staying put and moving by step score high,
        # leaping to a distant chord every bar scores low.
        steps = [
            min(abs(a - b), 12 - abs(a - b))
            for a, b in zip(roots_a, roots_b, strict=False)
        ]
        order_score = 1.0 - min(1.0, (sum(steps) / len(steps)) / 6.0)

    return max(0.0, min(1.0, 0.6 * set_score + 0.4 * order_score))


def tonal_distance(current: TrackProfile, candidate: TrackProfile) -> float:
    """How alike the two mixes are in weight and brightness."""
    sub = abs(current.sub_bass - candidate.sub_bass)
    presence = abs(current.presence - candidate.presence)
    low_high = abs(current.low_high - candidate.low_high)
    penalty = 2.0 * sub + 1.5 * presence + 0.5 * low_high
    return max(0.0, min(1.0, 1.0 - penalty))


def grid_match(current: TrackProfile, candidate: TrackProfile) -> float:
    """Similarity of measured transient proximity to the beat, without a genre label."""
    if current.onset_alignment is None or candidate.onset_alignment is None:
        return 0.5
    return max(0.0, 1.0 - abs(current.onset_alignment - candidate.onset_alignment))


def energy_flow(current: TrackProfile, candidate: TrackProfile) -> float:
    """Close energy is safe; a small lift is what a set does; a jump is a mistake.

    Scored asymmetrically: a modest increase reads as building, the same jump downward reads as a
    drop nobody asked for.
    """
    delta = candidate.energy - current.energy
    if delta < -0.12:
        return max(0.0, 1.0 + delta * 2.5)
    if delta <= 0.12:
        return 1.0
    return max(0.0, 1.0 - (delta - 0.12) * 2.0)


def tempo_fit(current: TrackProfile, candidate: TrackProfile) -> float:
    """Present but subdominant: the feasibility filter already gates this hard."""
    if current.bpm <= 0 or candidate.bpm <= 0:
        return 0.5
    ratio = candidate.bpm / current.bpm
    return max(0.0, min(1.0, 1.0 - (ratio - 1.0) * 4.0))


def key_fit(current: TrackProfile, candidate: TrackProfile) -> float:
    """Relative-key compatibility: the same key, a fifth, or a relative minor all work."""
    a, b = _pitch_class(current.key or ""), _pitch_class(candidate.key or "")
    if a is None or b is None:
        return 0.5
    semitones = abs(a - b)
    compatible = {0, 2, 3, 5, 7, 9, 10}  # unison, relative, parallel, dominant, subdominant
    if semitones not in compatible:
        return 0.0
    return 1.0 - semitones / 12.0


def compatibility(current: TrackProfile, candidate: TrackProfile) -> float:
    """Weighted score in [0, 1]. A hard mode clash zeroes the result, matching the filter."""
    terms = {
        "chord_overlap": chord_overlap(current, candidate),
        "tonal_distance": tonal_distance(current, candidate),
        "grid_match": grid_match(current, candidate),
        "energy_flow": energy_flow(current, candidate),
        "tempo": tempo_fit(current, candidate),
        "key": key_fit(current, candidate),
    }
    score = sum(WEIGHTS[name] * value for name, value in terms.items())
    if terms["key"] <= 0.0:
        return 0.0
    return max(0.0, min(1.0, score))


def explain(current: TrackProfile, candidate: TrackProfile) -> dict[str, float]:
    """Per-term breakdown, so a label can be argued with rather than merely trusted."""
    return {
        "chord_overlap": chord_overlap(current, candidate),
        "tonal_distance": tonal_distance(current, candidate),
        "grid_match": grid_match(current, candidate),
        "energy_flow": energy_flow(current, candidate),
        "tempo": tempo_fit(current, candidate),
        "key": key_fit(current, candidate),
    }
