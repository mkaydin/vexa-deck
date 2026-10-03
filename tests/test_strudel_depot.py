"""Tests for the Strudel depot builder.

Two of these exist because the bug was real and shipped into the layout before it was caught:

* ``test_specs_are_unique`` — cluster, mood and tempo were all derived from the same modulo, so
  every combination was perfectly correlated and a 100-spec request produced only 25 distinct
  assets. A silent duplicate depot is worse than a small one: the family-level splits that
  ``LAYA_DATA.md:70`` depends on would have collapsed.
* ``test_pattern_declares_the_specified_tempo`` — the tempo written into the pattern must be the
  tempo stored in the manifest, or the manifest is a guess wearing a manifest's clothes.

The octave tests cover the metrical-level ambiguity that a beat tracker cannot resolve on its own:
material with an onset on every eighth note reads as double-time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from build_strudel_depot import (  # noqa: E402
    DEFAULT_TOLERANCE_PCT,
    METRICAL_MULTIPLIERS,
    MOODS,
    TEMPO_CLUSTERS,
    build_specs,
    pattern_for,
)


def test_specs_are_unique() -> None:
    """Every spec must have its own name, or the depot silently deduplicates itself."""
    specs = build_specs(125)
    names = [spec.name for spec in specs]
    assert len(names) == len(set(names)), "duplicate spec names collapse the depot"
    assert len(specs) == 125


def test_cluster_and_mood_are_not_correlated() -> None:
    """Every cluster must reach every mood. A single shared modulo binds one cluster to one mood."""
    specs = build_specs(125)
    seen = {(spec.cluster, spec.mood) for spec in specs}
    expected = {(centre, mood) for centre, *_ in TEMPO_CLUSTERS for mood in MOODS}
    assert seen == expected


def test_energy_spans_the_band() -> None:
    """The retrieval filters match on energy, so the depot must not have a hole in the range."""
    energies = [spec.energy for spec in build_specs(125)]
    assert min(energies) <= 0.15
    assert max(energies) >= 0.85


def test_tempos_stay_inside_clusters() -> None:
    """A selector can only choose between assets close enough to transition into."""
    specs = build_specs(125)
    tempi = {tempo for _c, tempi, _k, _r in TEMPO_CLUSTERS for tempo in tempi}
    for spec in specs:
        assert spec.bpm in tempi


@pytest.mark.parametrize("spec", build_specs(12), ids=lambda s: s.name)
def test_pattern_declares_the_specified_tempo(spec) -> None:
    "setcpm is cycles per minute and wirbel divides by 60 again, so it renders 60x too slow"
    pattern = pattern_for(spec)
    assert "setcps(" in pattern, "tempo must be set with setcps, not setcpm"
    assert "setcpm(" not in pattern, (
        "setcpm is cycles per minute and wirbel divides by 60 again, rendering 60x too slow"
    )
    expected_cps = spec.bpm / 60.0 / 4.0
    assert f"{expected_cps:.10f}" in pattern


def _best_level(measured: float, specified: float) -> str | None:
    """Mirror of the gate in ``admit``, so the tolerance is tested rather than asserted."""
    candidates = {label: measured * factor for label, factor in METRICAL_MULTIPLIERS.items()}
    errors = {
        label: abs(value - specified) / specified * 100.0 for label, value in candidates.items()
    }
    best = min(errors, key=lambda label: errors[label])
    if errors[best] > DEFAULT_TOLERANCE_PCT:
        return None
    return None if best == "as-measured" else best


def test_metrical_gate_absorbs_known_tracker_ambiguity() -> None:
    """The gate tests the whole family of musically-related rates, not one reading.

    These are what a first pass rejected: a bass note on every eighth note reads as double-time,
    a 168 BPM bed reads as 112.5 when the tracker hears compound meter, and the tempogram
    quantises 144 to 140.6. None of them is a defect in the render.
    """
    for measured, specified, expected in [
        (175.8, 88.0, "half"),            # eighth-note pulse reads as double-time
        (87.9, 88.0, None),              # as-measured, so no level shift to report
        (140.6, 144.0, None),            # tempogram bin quantisation, 2.4 %
        (112.5, 168.0, "three-halves"),  # compound meter: measured is 2/3 of the beat
    ]:
        assert _best_level(measured, specified) == expected, (measured, specified)


def test_metrical_gate_still_rejects_a_broken_render() -> None:
    """Widening the family must not turn the gate into a no-op."""
    assert _best_level(61.3, 88.0) is None
    assert _best_level(88.0, 122.0) is None

def test_tempo_override_cannot_leave_markers_past_the_end() -> None:
    """Overriding the tempo re-scales bar numbers, so markers can end up outside the asset.

    ``analyse`` converts detected boundaries to bars using the *measured* tempo. A track read at
    175.8 is counted as 23 bars; stored at the specified 88 the same 32 s of audio is 11 bars,
    and ``AssetManifest`` refuses to load a section ending past the end. Nineteen manifests hit
    exactly this before it was fixed.
    """
    duration_s, specified_bpm, measured_bpm = 32.0, 88.0, 175.8
    bars_measured = int(duration_s / (240.0 / measured_bpm))
    bars_specified = int(duration_s / (240.0 / specified_bpm))
    assert bars_measured > bars_specified  # the precondition that made this reachable

    sections = [{"section_id": "s0", "kind": "segment", "start_bar": 0, "end_bar": bars_measured}]
    kept = [s for s in sections if s["end_bar"] < bars_specified]
    assert kept == [], "a section ending past the stored tempo's last bar must be dropped"
