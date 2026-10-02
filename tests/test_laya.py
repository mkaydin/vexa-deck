"""Laya dataset, conversion and adapter behaviour.

The dataset rules are the ones that quietly ruin a fine-tune if broken — a family that leaks
across splits, a label that points at an option nobody was offered, a soft target that does not
sum to one. Those get tested directly.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from vexa_contracts import ActionType, DecisionRequest, FeasibleAction, SessionState
from vexa_laya.adapter import MAX_OPTIONS, CalibrationGate, LayaAdapter, LayaUnavailable
from vexa_laya.convert import (
    calibration_slice,
    criteria,
    to_eval_record,
    to_training_items,
)
from vexa_laya.dataset import (
    Annotation,
    CandidateType,
    DecisionContext,
    LabelSource,
    StateSnapshot,
    read_jsonl,
    split_by_family,
)


def annotation(
    family: str = "family_1",
    session: str = "session_1",
    preferred: str = "b",
    also: list[str] | None = None,
    rejected: list[str] | None = None,
) -> Annotation:
    return Annotation(
        state=DecisionContext(
            request="darker, more energetic, no vocals",
            current=StateSnapshot(
                bpm=122.0, key="A minor", section="outro", bars_to_boundary=8,
                energy=0.46, vocals=False, last_action="continue",
                recent_families=["family_8", "family_3"],
            ),
            candidates=[
                CandidateType(id="a", type="continue", energy=0.46),
                CandidateType(id="b", type="transition", asset="asset_42", bpm=124.0,
                              energy=0.68, vocals=False),
                CandidateType(id="c", type="transition", asset="asset_53", bpm=121.0,
                              energy=0.72, vocals=True),
            ],
        ),
        preferred_action_id=preferred,
        also_acceptable=also or [],
        rejected_action_ids=rejected or [],
        family_id=family,
        session_id=session,
        label_source=LabelSource(kind="human_pairwise_review", agreement=0.8),
    )


# --- The annotation record ---------------------------------------------------


def test_a_label_must_point_at_an_offered_option():
    with pytest.raises(ValidationError, match="not among the offered"):
        annotation(preferred="not_offered")


def test_acceptable_options_must_also_have_been_offered():
    with pytest.raises(ValidationError, match="not offered"):
        annotation(also=["ghost"])


def test_an_option_cannot_be_both_preferred_and_rejected():
    with pytest.raises(ValidationError, match="both preferred and rejected"):
        annotation(preferred="b", rejected=["b"])


def test_a_continue_label_is_recognised_as_holding_position():
    """LAYA_DATA.md:57 — the dataset must contain hold-position choices or the model
    over-transitions."""
    assert annotation(preferred="a").is_safe_choice


def test_provenance_is_required_not_optional():
    """LAYA_DATA.md:85 — a source-and-license manifest per training row."""
    row = annotation()
    assert row.label_source.kind == "human_pairwise_review"
    assert row.label_source.agreement == 0.8

def test_jsonl_reader_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "# a comment\n\n" + json.dumps(annotation().model_dump(mode="json")) + "\n\n",
        encoding="utf-8",
    )
    assert len(list(read_jsonl(path))) == 1


# --- Splits -----------------------------------------------------------------


def test_a_family_never_spans_two_splits():
    """LAYA_DATA.md:69 — variants of one render must stay together, or the model memorises."""
    rows = [
        annotation(family=f"family_{i}", session=f"session_{i % 3}") for i in range(30)
    ]
    # Add extra variants of one family, as revisions of the same render would produce.
    rows += [annotation(family="family_0", session="session_1") for _ in range(5)]

    split = split_by_family(rows, seed=7)
    for bucket in (split.train, split.validation, split.test):
        others = [g for g in (split.train, split.validation, split.test) if g is not bucket]
        mine = {a.family_id for a in bucket}
        theirs = {a.family_id for group in others for a in group}
        assert not (mine & theirs), "a family leaked into two splits"

    assert set(split.all_families()) == {f"family_{i}" for i in range(30)}
    assert len(split.train) + len(split.validation) + len(split.test) == len(rows)


def test_split_is_deterministic_for_a_given_seed():
    rows = [annotation(family=f"family_{i}") for i in range(20)]
    first = split_by_family(rows, seed=3)
    second = split_by_family(rows, seed=3)
    assert [a.family_id for a in first.test] == [a.family_id for a in second.test]


def test_split_rejects_nonsense_ratios():
    rows = [annotation()]
    with pytest.raises(ValueError, match="train_ratio"):
        split_by_family(rows, train_ratio=1.5)
    with pytest.raises(ValueError, match="validation_ratio"):
        split_by_family(rows, validation_ratio=-0.1)


# --- Conversion -------------------------------------------------------------


def test_eval_record_matches_the_documented_shape():
    """laya/docs/evals.md: state, questions, expected, optional tags and language."""
    row = to_eval_record(annotation())
    assert set(row) >= {"state", "questions", "expected"}
    assert isinstance(row["state"], str)
    question = next(iter(row["questions"].values()))
    assert question["type"] in ("choice", "score", "noul")
    # A choice question's criteria are a dict of label -> description, which is what
    # `render_options` consumes. A list here fails at render time.
    assert isinstance(question["criteria"], dict)
    assert all(value == "" for value in question["criteria"].values())
    assert row["language"] == "en"


def test_eval_options_describe_measured_features_not_invented_prose():
    labels = criteria(annotation())
    assert any("124 BPM" in label for label in labels)
    assert any("instrumental" in label for label in labels)


def test_a_spread_label_adds_a_graded_score_question():
    row = to_eval_record(annotation(also=["a"], rejected=["c"]))
    types = [q["type"] for q in row["questions"].values()]
    assert "score" in types


def test_a_flat_label_does_not_invent_a_grade():
    row = to_eval_record(annotation())
    types = [q["type"] for q in row["questions"].values()]
    assert types == ["choice"]


def test_training_targets_form_a_valid_distribution():
    """RLCD imitates a distribution; targets that do not sum to one are meaningless."""
    items = to_training_items([annotation(also=["a"], rejected=["c"])])
    distribution = next(iter(items[0]["gold"].values()))["distribution"]
    assert pytest.approx(1.0, abs=1e-6) == sum(distribution.values())
    labels = list(items[0]["questions"][next(iter(items[0]["questions"]))]["criteria"])
    assert set(distribution) == set(labels), "gold must be keyed by the criteria labels"
    assert distribution[labels[2]] == 0.0, "a rejected option must carry no mass"
    assert distribution[labels[1]] > distribution[labels[0]]


def test_state_text_is_stable():
    """The eval harness fingerprints question text; rewording silently invalidates baselines."""
    assert to_eval_record(annotation())["state"] == to_eval_record(annotation())["state"]


def test_calibration_slice_is_bounded_and_deterministic():
    rows = [annotation(family=f"family_{i}") for i in range(100)]
    first = calibration_slice(rows, max_items=20)
    second = calibration_slice(rows, max_items=20)
    assert len(first) <= 20
    assert [a.family_id for a in first] == [a.family_id for a in second]


# --- The calibration gate ---------------------------------------------------


def test_confidence_is_untrusted_until_calibration_is_fitted():
    """Laya ships over-confident; a threshold on a raw score means nothing."""
    gate = CalibrationGate()
    assert not gate.calibrated
    assert gate.adjust("choice", 0.95) == 0.95, "an uncalibrated gate must not rescale"


def test_opening_the_gate_requires_actual_temperatures():
    with pytest.raises(ValueError, match="without fitted temperatures"):
        CalibrationGate().open({})


def test_fitting_deletes_the_inherited_option_temperatures():
    """The documented trap: inherited per-option temperatures mask a new fit at inference."""
    gate = CalibrationGate()
    gate.open({"choice": 1.4})
    config = {"temperature_by_options": {"a": 9.0}, "model": "x"}
    applied = gate.apply(config)
    assert applied["temperature"] == {"choice": 1.4}
    assert "temperature_by_options" not in applied
    assert "temperature_by_options" in config, "the input config must not be mutated"


def test_temperature_scaling_softens_an_over_confident_model():
    gate = CalibrationGate()
    gate.open({"choice": 2.0})
    softened = gate.adjust("choice", 0.95)
    assert 0.5 < softened < 0.95, "a temperature above 1 must pull confidence toward 0.5"
    assert gate.adjust("choice", 0.5) == pytest.approx(0.5, abs=1e-6)


# --- The adapter ------------------------------------------------------------


def session() -> SessionState:
    return SessionState(session_id="s", theme="rain-soaked jazz", generation=1)


def decision_request(n_transitions: int = 3) -> DecisionRequest:
    candidates = [
        FeasibleAction(action_id="continue_current", action_type=ActionType.CONTINUE_CURRENT,
                       start_at_session_bar=8),
    ]
    for i in range(n_transitions):
        candidates.append(
            FeasibleAction(action_id=f"t{i}", action_type=ActionType.TRANSITION,
                           asset_id=f"a{i}", start_at_session_bar=8, energy_fit=0.5)
        )
    return DecisionRequest(decision_id="d", session_id="s", state=session(), candidates=candidates)


def test_the_option_list_is_capped_for_the_model_token_budget():
    """LAYA_DATA.md:11 — high option counts degrade confidence selection."""
    request = decision_request(n_transitions=40)
    adapter = LayaAdapter()
    shortlist = adapter._shortlist(request.candidates)
    assert len(shortlist) <= MAX_OPTIONS


def test_every_safe_action_survives_shortlisting():
    """Without a hold-position option the model cannot express 'do nothing'."""
    request = decision_request(n_transitions=40)
    adapter = LayaAdapter()
    shortlist = {a.action_id for a in adapter._shortlist(request.candidates)}
    assert "continue_current" in shortlist


def test_a_single_option_menu_is_refused_by_the_contract():
    """A decision needs at least two options to be a decision."""
    with pytest.raises(ValidationError):
        DecisionRequest(
            decision_id="d", session_id="s", state=session(),
            candidates=[FeasibleAction(action_id="keep", action_type=ActionType.CONTINUE_CURRENT,
                                       start_at_session_bar=8)],
        )


def test_the_question_carries_exactly_the_shortlisted_options():
    request = decision_request(n_transitions=40)
    question = LayaAdapter().build_question(request)["choice"]
    assert question["type"] == "choice"
    assert len(question["criteria"]) <= MAX_OPTIONS
    assert any("keep the current track" in c for c in question["criteria"])


def test_the_model_is_never_sent_the_waveform():
    """README.md:39 — the planner and selector get structured metadata, not audio."""
    text = LayaAdapter.state_text(decision_request())
    assert "rain-soaked jazz" in text
    assert "BPM" in text
    assert "waveform" not in text.lower()


def test_inference_failure_falls_back_instead_of_raising(monkeypatch):
    """A dead model must never stall the set (README.md:18).

    Driven by a stub rather than by whether ``laya`` happens to be installed, so the guarantee is
    tested the same way in every environment.
    """
    class Exploding:
        def predict(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

    adapter = LayaAdapter()
    adapter._agent = Exploding()
    request = decision_request()
    response = adapter.choose(request)

    assert response.fell_back
    assert response.confidence == 0.0
    assert response.action_id in request.ids


def test_unrecognised_option_falls_back(monkeypatch):
    """A model that names an option the filter never offered must not be trusted."""

    class Nonsense:
        def predict(self, *args, **kwargs):
            return {"answers": {"choice": {"choice": "an option nobody offered",
                                          "answer_confidence": 0.99}}}

    adapter = LayaAdapter()
    adapter._agent = Nonsense()
    response = adapter.choose(decision_request())
    assert response.fell_back
    assert response.action_id == "continue_current"


def test_missing_checkpoint_reports_unavailable(monkeypatch):
    """A checkpoint that cannot be loaded is a configuration state, surfaced explicitly."""
    adapter = LayaAdapter()
    monkeypatch.setattr(adapter, "load", lambda: (_ for _ in ()).throw(
        LayaUnavailable("no such checkpoint")))
    response = adapter.choose(decision_request())
    assert response.fell_back
    assert "unavailable" in response.reasoning