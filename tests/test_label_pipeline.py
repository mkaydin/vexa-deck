"""Tests for the label collection and shadow-mode pipeline.

These cover the properties that silently corrupt a fine-tune when they break:

* **The safe action's spelling.** ``FeasibilityFilter`` emits ``continue_current`` and the adapter
  recognises only that spelling. Renaming it in the collector made every "hold" label unjoinable
  to a shadow observation -- and holding is exactly what Laya gets wrong, so the disagreements
  worth judging would have been the ones dropped.
* **Annotators must see the same request.** Agreement across five people is only meaningful if
  they were asked the same question, so the theme is bound into the session id and a divergent
  ``--theme`` is visible rather than silent.
* **Rule-derived labels can never train a shipped model.** A checkpoint trained on them scores
  well on our own benchmark and changes nothing at runtime, which is invisible in a training log.
* **The calibration slice must not overlap held-out data.** ``calibration_slice`` is not
  split-aware, so carving it from the whole dataset leaks.

The depot builder itself moved to ``tools/build_yue2_depot.py``; the metrical-level and bar-rescale
findings that came out of the Strudel experiment are recorded in ``FINETUNE-DECISION.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "services/laya/src"))

from vexa_contracts import (  # noqa: E402
    ApprovalState,
    AssetManifest,
    AudioProperties,
    BeatGrid,
    Estimate,
    QualityReport,
    ReadinessState,
    SourceType,
)


def _asset(index: int) -> AssetManifest:
    """A minimal admissible asset for filter-level tests."""
    return AssetManifest(
        asset_id=f"a{index}",
        family_id=f"family_a{index}",
        source_type=SourceType.YUE2_RENDER,
        content_sha256="a" * 64,
        audio=AudioProperties(
            codec="wav", sample_rate_hz=48000, channels=2, duration_s=45.0,
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


def test_hold_action_uses_the_canonical_id() -> None:
    """Every tool must spell the safe action the same way, or the join silently drops holds."""
    from collect_labels import build_decision
    from vexa_laya.adapter import SAFE_IDS

    assert "continue_current" in SAFE_IDS

    manifests = {f"a{i}": _asset(i) for i in range(6)}
    built = build_decision(manifests, index=0, theme="t")
    if built is None:
        return  # filter produced nothing playable
    ids = [c.id for c in built[0].candidates]
    assert "continue_current" in ids
    assert "continue" not in ids, "the bare alias must not reappear"


def test_annotators_on_the_same_theme_share_a_session_id() -> None:
    """Agreement is only measurable if every annotator saw an identical request."""
    from collect_labels import _session_id

    theme = "a warm, unhurried set that keeps building"
    assert _session_id(0, theme) == _session_id(0, theme)
    assert _session_id(0, theme) != _session_id(1, theme)
    assert _session_id(0, theme) != _session_id(0, "a different direction")


def test_rule_derived_rows_can_never_be_promoted() -> None:
    """The exclusion must be a property of the data, not a convention."""
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
            family_id="family_x",
            session_id="s",
            label_source=LabelSource(kind=kind),
        )

    from rule_derived_labels import promotable

    assert promotable([row("rule_derived")]) == []
    assert len(promotable([row("human_pairwise_review")])) == 1


def test_finetune_gate_refuses_non_human_labels() -> None:
    """The gate must refuse, not warn: nothing in a training log would reveal the mistake."""
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


def test_shadow_and_labels_join_on_the_option_set() -> None:
    """A decision point's only stable identity is the deck plus the options offered."""
    from judge_shadow import decision_key

    options = {"continue_current", "transition_to_a", "transition_to_b"}
    shadow_side = decision_key("house-a", sorted(options))
    label_side = decision_key("house-a", list(options))

    assert shadow_side == label_side, "order of the option set must not matter"
    assert shadow_side != decision_key("house-b", options)
    assert shadow_side != decision_key("house-a", {"continue_current", "transition_to_a"})


def test_yue2_depot_briefs_are_unique_and_balanced() -> None:
    """A depot that silently deduplicates itself is worse than a small one.

    Family-level splits depend on every entry being distinct, so duplicate names would collapse
    the split key before training starts.
    """
    from build_yue2_depot import ENERGY_BANDS, MOODS, PALETTES, build_briefs

    briefs = build_briefs(200)
    assert len(briefs) == 200
    assert len({b.name for b in briefs}) == 200

    palettes = {b.name.split("-")[1] for b in briefs}
    assert len(palettes) == len(PALETTES), "every tempo palette must be represented"
    assert {b.mood for b in briefs} == set(MOODS)
    assert {b.energy for b in briefs} == {e for e, _a in ENERGY_BANDS}


def test_yue2_depot_prompts_stay_instrumental() -> None:
    """Every brief must forbid vocals.

    ``PRODUCT.md`` treats "no vocals" as a filterable property, and the depot has to contain
    material that exercises that filter rather than material that trivially satisfies it.
    """
    from build_yue2_depot import build_briefs

    for brief in build_briefs(200):
        assert "no vocals" in brief.style.lower()