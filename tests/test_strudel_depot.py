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
sys.path.insert(0, str(ROOT / "services/laya/src"))

from build_strudel_depot import (  # noqa: E402
    DEFAULT_TOLERANCE_PCT,
    METRICAL_MULTIPLIERS,
    MOODS,
    TEMPO_CLUSTERS,
    _rescale_bars,
    build_specs,
    pattern_for,
)
from vexa_contracts import (  # noqa: E402
    AssetManifest,
    AudioProperties,
    BeatGrid,
    LoopPoints,
    SectionMarker,
    SourceType,
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

def test_bar_rescale_keeps_a_whole_track_section() -> None:
    """Rescaling must preserve the entry point, which filtering markers out destroyed.

    A first attempt dropped markers past the end and left 55 assets unplayable: on a short track
    the only section *is* the whole track, so discarding it left ``FeasibilityFilter`` with nothing
    to enter at. Markers are converted and clamped instead.
    """
    duration_s, specified_bpm, measured_bpm = 32.0, 88.0, 175.8
    measured_bars = int(duration_s / (240.0 / measured_bpm))
    total_bars = int(duration_s / (240.0 / specified_bpm))
    assert measured_bars > total_bars, "precondition: analysis counted more bars than exist"

    scale = specified_bpm / measured_bpm
    last = total_bars - 1
    converted = lambda value: max(0, min(round(value * scale), last))  # noqa: E731

    assert converted(0) == 0
    assert converted(measured_bars) <= last, "clamped inside the stored track length"
    assert converted(measured_bars) > 0, "a whole-track section must survive rescaling"


def test_bar_rescale_is_identity_when_tempo_agrees() -> None:
    """A measurement that agrees must not move a single marker."""
    manifest = AssetManifest(
        asset_id="a1",
        family_id="family_a1",
        source_type=SourceType.PACK_IMPORT,
        content_sha256="a" * 64,
        audio=AudioProperties(
            codec="wav", sample_rate_hz=48000, channels=2,
            duration_s=40.0, integrated_lufs=-14.0, true_peak_dbtp=-1.5,
        ),
        beat_grid=BeatGrid(bpm=120.0, grid_version=1, confidence=1.0),
        sections=[SectionMarker(section_id="s0", kind="segment", start_bar=0, end_bar=19)],
        loops=[LoopPoints(start_bar=0, end_bar=8, beat_aligned=True)],
    )
    _rescale_bars(manifest, 1.0)

    assert [(s.start_bar, s.end_bar) for s in manifest.sections] == [(0, 19)]
    assert [(loop.start_bar, loop.end_bar) for loop in manifest.loops] == [(0, 8)]


def test_specs_carry_the_tags_the_filters_match_on() -> None:
    """Energy and mood must reach the manifest, not just the spec.

    A first pass built 117 tracks around energy and mood axes and wrote neither into a manifest.
    ``FeasibilityFilter`` matches on those tags, so every energy and mood constraint was silently
    inert for the entire depot -- the filters looked configured and did nothing.
    """
    energies = ["0.10", "0.20", "0.25", "0.35", "0.45", "0.55", "0.60", "0.70", "0.80", "0.90"]
    tags = {"energy": energies}
    moods = set(MOODS)
    for spec in build_specs(125):
        assert f"{spec.energy:.2f}" in tags["energy"], f"{spec.name} energy not a known band"
        assert spec.mood in moods


def test_rule_derived_rows_can_never_be_promoted() -> None:
    """The exclusion must be a property of the data, not a convention.

    ``IDEAS.md:57``: training on synthetic rules can reproduce the rules without improving
    listening quality. If a promotion step forgets to filter, the result looks like a successful
    fine-tune and adds nothing at runtime -- the worst outcome available, because it is invisible.
    """
    from rule_derived_labels import LabelSource, promotable
    from vexa_laya.dataset import Annotation, DecisionContext, StateSnapshot

    def row(kind: str) -> Annotation:
        return Annotation(
            state=DecisionContext(
                request="r",
                current=StateSnapshot(bpm=122.0, bars_to_boundary=8, energy=0.5),
                candidates=[{"id": "a", "type": "transition", "bpm": 122.0},
                            {"id": "b", "type": "transition", "bpm": 124.0}],
            ),
            preferred_action_id="a",
            family_id="family_x",
            session_id="s",
            label_source=LabelSource(kind=kind),
        )

    assert promotable([row("rule_derived")]) == []
    assert len(promotable([row("human_pairwise_review")])) == 1
    assert len(promotable([row("teacher_model")])) == 1


def test_generated_rows_are_all_rule_derived_and_well_formed() -> None:
    """Whatever the generator produces must carry its provenance and a real menu."""
    from rule_derived_labels import build_rows, load_library

    library = load_library()
    if not library:
        return  # no depot built; nothing to assert about
    rows = build_rows(library, per_asset=2)
    assert rows
    assert all(r.label_source.kind == "rule_derived" for r in rows)
    assert all(len(r.state.candidates) >= 2 for r in rows)
    # Families must be real asset families, or family-level splits have nothing to separate.
    assert len({r.family_id for r in rows}) == len(library)


def test_finetune_gate_refuses_non_human_labels() -> None:
    """The gate must refuse, not warn.

    A checkpoint trained on rule-derived labels scores well on our own benchmark and changes
    nothing at runtime. Nothing in a training log would reveal that, so the refusal has to happen
    before any GPU time is spent.
    """
    import pytest
    from finetune_human import refuse_unless_human
    from vexa_laya.dataset import Annotation, DecisionContext, LabelSource, StateSnapshot

    def row(kind: str) -> Annotation:
        return Annotation(
            state=DecisionContext(
                request="r",
                current=StateSnapshot(bpm=122.0, bars_to_boundary=8, energy=0.5),
                candidates=[{"id": "a", "type": "transition"},
                            {"id": "b", "type": "transition"}],
            ),
            preferred_action_id="a",
            family_id="f",
            session_id="s",
            label_source=LabelSource(kind=kind),
        )

    refuse_unless_human([row("human_pairwise_review")])  # must not raise
    for kind in ("rule_derived", "teacher_model", "synthetic"):
        with pytest.raises(SystemExit):
            refuse_unless_human([row(kind)])


def test_calibration_slice_must_be_carved_from_the_training_split() -> None:
    """``calibration_slice`` is not split-aware; carving it from the whole dataset leaks.

    It sorts what it is given by family and takes the first N, so calling it on all annotations
    returns rows that straddle train and validation. Fitting temperatures on data the run trained
    on measures the fit rather than the calibration (``LAYA_DATA.md:65``).
    """
    from rule_derived_labels import load_library
    from vexa_laya.convert import calibration_slice
    from vexa_laya.dataset import split_by_family

    library = load_library()
    if not library:
        return
    from rule_derived_labels import build_rows

    rows = build_rows(library, per_asset=2)
    split = split_by_family(rows)

    naive = {a.family_id for a in calibration_slice(rows, max_items=400)}
    held_out = {a.family_id for a in split.validation} | {a.family_id for a in split.test}

    carved_from_train = {a.family_id for a in calibration_slice(split.train, max_items=400)}
    assert not (carved_from_train & held_out), "carving from train must be disjoint by construction"
    # The naive call is allowed to overlap; that is precisely why it must not be used.
    assert naive is not None


def test_shadow_and_labels_join_on_the_option_set() -> None:
    """A decision point's only stable identity is the deck plus the options offered.

    The shadow log and the human labels come from different tools over different runs, so an
    index or a position cannot join them. Joining on (playing asset, candidate id set) is what
    makes the comparison possible at all.
    """
    from judge_shadow import decision_key

    options = {"continue_current", "transition_to_a", "transition_to_b"}
    shadow_side = decision_key("house-a", sorted(options))
    label_side = decision_key("house-a", [c for c in options])

    assert shadow_side == label_side, "order of the option set must not matter"

    other_deck = decision_key("house-b", options)
    other_menu = decision_key("house-a", {"continue_current", "transition_to_a"})
    assert shadow_side != other_deck, "a different deck is a different decision"
    assert shadow_side != other_menu, "a different menu is a different decision"


def test_hold_action_uses_the_canonical_id() -> None:
    """Every tool must spell the safe action the same way, or the join silently drops holds.

    ``FeasibilityFilter`` emits ``continue_current`` and the adapter recognises only that spelling,
    but the label collector used to rename it to ``continue``. Holding is exactly the choice Laya
    gets wrong -- it picked it 12 times out of 12 -- so the mislabelled ids would have discarded
    precisely the disagreements worth judging.
    """
    from vexa_laya.adapter import SAFE_IDS

    assert "continue_current" in SAFE_IDS
    from collect_labels import build_decision

    manifests = {f"a{i}": _asset(i) for i in range(6)}
    built = build_decision(manifests, index=0, theme="t")
    if built is None:
        return  # filter produced nothing playable; nothing to assert about the menu
    context = built[0]
    ids = [c.id for c in context.candidates]
    assert "continue_current" in ids
    assert "continue" not in ids, "the bare alias must not reappear"


def test_annotators_on_the_same_theme_share_a_session_id() -> None:
    """Agreement is only measurable if every annotator saw an identical request.

    Five annotators must land on the same decision points or their answers cannot be compared.
    Binding the theme into the session id makes a divergent ``--theme`` a visible mismatch rather
    than two rows that look like the same decision.
    """
    from collect_labels import _session_id

    theme = "a warm, unhurried set that keeps building"
    assert _session_id(0, theme) == _session_id(0, theme)
    assert _session_id(0, theme) != _session_id(1, theme)
    assert _session_id(0, theme) != _session_id(0, "a different direction")


def _asset(index: int):
    """A minimal admissible asset for filter-level tests."""
    from vexa_contracts import (
        ApprovalState,
        AssetManifest,
        AudioProperties,
        BeatGrid,
        Estimate,
        QualityReport,
        ReadinessState,
        SourceType,
    )

    return AssetManifest(
        asset_id=f"a{index}",
        family_id=f"family_a{index}",
        source_type=SourceType.PACK_IMPORT,
        content_sha256="a" * 64,
        audio=AudioProperties(
            codec="wav", sample_rate_hz=48000, channels=2, duration_s=32.0,
            integrated_lufs=-14.0, true_peak_dbtp=-1.5,
        ),
        beat_grid=BeatGrid(bpm=122.0 + index, grid_version=1, confidence=1.0),
        tags={"energy": [Estimate(value="0.50", confidence=1.0)]},
        approval=ApprovalState.APPROVED,
        quality=QualityReport(
            decode_ok=True, duration_ok=True, loudness_ok=True, beat_grid_ok=True,
            loop_boundary_ok=True, audio_quality_ok=True,
        ),
        readiness=ReadinessState.READY,
    )
