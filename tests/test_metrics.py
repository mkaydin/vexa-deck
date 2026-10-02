"""Metrics and promotion gates.

The gate tests exist because of a specific failure mode: a report full of ``0.0`` reads like a
clean run. An unmeasured metric must never satisfy a gate that decides whether a model is allowed
to control audio.
"""

from __future__ import annotations

import pytest
from vexa_laya.metrics import (
    Prediction,
    brier_score,
    choice_accuracy,
    expected_calibration_error,
    fit_temperature,
    invalid_action_rate,
    latency_percentiles,
    pairwise_win_rate,
    score_mae,
    soft_accuracy,
    temperature_scale,
)
from vexa_laya.train_config import Metrics

#: The listener chose "a"; "b" was acceptable; "c" was explicitly rejected.
GOLD = {"a": 0.7, "b": 0.2, "c": 0.0}


def pred(correct="a", predicted="a", confidence=0.8, probabilities=None, latency=10.0):
    """A prediction whose defaults match ``GOLD``, so a metric reading zero means zero.

    The gold answer is "a". A "wrong" prediction must therefore name something else — picking "a"
    again scores as correct, which is the easiest way to write a test that proves nothing.
    """
    probs = probabilities or {"a": 0.7, "b": 0.2, "c": 0.1}
    return Prediction(
        correct_label=correct,
        predicted_label=predicted,
        confidence=confidence,
        gold=dict(GOLD),
        probabilities=probs,
        latency_ms=latency,
    )


def wrong(confidence: float) -> Prediction:
    """A prediction that picks the explicitly-rejected option."""
    return pred(
        predicted="c",
        confidence=confidence,
        probabilities={"a": 0.1, "b": 0.1, "c": 0.8},
    )


# --- basic metrics ----------------------------------------------------------


def test_choice_accuracy_counts_exact_matches():
    assert choice_accuracy([pred(), wrong(0.8)]) == 0.5


def test_soft_accuracy_measures_overlap_with_the_gold_distribution():
    """A model that spreads mass over two acceptable options is not wrong here."""
    spread = pred(probabilities={"a": 0.4, "b": 0.6, "c": 0.0})
    committed_wrong = pred(predicted="c", probabilities={"a": 0.1, "b": 0.1, "c": 0.8})
    assert soft_accuracy([spread]) > soft_accuracy([committed_wrong])


def test_brier_is_zero_for_an_exact_distribution():
    assert brier_score([pred(probabilities=dict(GOLD))]) == pytest.approx(0.0)


def test_brier_penalises_confidence_in_the_wrong_place():
    assert brier_score([wrong(0.8)]) > brier_score([pred(probabilities=dict(GOLD))])


def test_ece_is_zero_when_confidence_matches_accuracy():
    """80% correct while claiming 80% is exactly calibrated."""
    preds = [pred(confidence=0.8) for _ in range(4)] + [wrong(0.8)]
    assert expected_calibration_error(preds) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_an_over_confident_model():
    """The same 80% accuracy claimed at 99%. This gap is what calibration exists to close."""
    preds = [pred(confidence=0.99) for _ in range(4)] + [wrong(0.99)]
    assert expected_calibration_error(preds) == pytest.approx(0.19, abs=0.01)


def test_empty_input_is_not_a_crash():
    assert choice_accuracy([]) == 0.0
    assert brier_score([]) == 0.0
    assert expected_calibration_error([]) == 0.0
    assert latency_percentiles([]) == (0.0, 0.0)


def test_latency_percentiles_are_ordered():
    preds = [pred(latency=float(n)) for n in range(1, 101)]
    p50, p95 = latency_percentiles(preds)
    assert p50 <= p95
    assert 45 <= p50 <= 55


# --- the invariant that matters --------------------------------------------


def test_an_option_nobody_offered_counts_as_invalid():
    """Laya may only pick from the menu the filter produced. Anything else is a bug."""
    invented = pred(predicted="an-option-nobody-offered")
    offered = [["a", "b", "c"], ["a", "b", "c"]]
    assert invalid_action_rate([pred(), invented], offered) == 0.5


def test_a_fully_valid_run_scores_zero_invalid():
    offered = [["a", "b", "c"], ["a", "b", "c"]]
    assert invalid_action_rate([pred(), pred()], offered) == 0.0


# --- the rule baseline ------------------------------------------------------


def test_a_model_that_beats_the_baseline_scores_above_a_half():
    assert pairwise_win_rate(["b", "b", "a"], ["a", "a", "b"], ["b", "b", "a"]) == 1.0


def test_an_agreed_choice_is_shared_credit_not_a_win():
    assert pairwise_win_rate(["b", "b"], ["b", "b"], ["b", "b"]) == 0.5


def test_a_model_that_agrees_with_a_wrong_baseline_scores_below_a_half():
    assert pairwise_win_rate(["a", "a"], ["a", "a"], ["b", "b"]) == 0.0


def test_score_mae_is_the_mean_absolute_error():
    pairs = [(0.9, 1.0), (0.2, 0.0), (0.55, 0.5)]
    assert score_mae(pairs) == pytest.approx((0.1 + 0.2 + 0.05) / 3)


# --- temperature fitting ----------------------------------------------------


def test_scaling_softens_without_destroying_the_signal():
    """Temperature scaling moves confidence, and must leave a real signal readable.

    Exact values are ``sigmoid(logit(0.95) / T)``, i.e. ``sigmoid(ln(19) / T)``.
    """
    assert temperature_scale(0.95, 10.0) == pytest.approx(0.5731, abs=1e-3)
    assert temperature_scale(0.95, 2.0) == pytest.approx(0.8134, abs=1e-3)
    # Even at the maximum a calibrated model stays meaningfully above chance.
    assert temperature_scale(0.95, 10.0) > 0.55


def test_a_temperature_of_one_is_the_identity():
    assert temperature_scale(0.95, 1.0) == pytest.approx(0.95, abs=1e-6)


def test_fitting_on_a_thin_slice_refuses_and_returns_the_default():
    """Under ten points cannot fit a scalar; pretending otherwise returns a confident lie."""
    assert fit_temperature([0.9] * 5, [True] * 5, default=1.2) == 1.2


def test_fitting_pulls_an_over_confident_model_toward_its_accuracy():
    confidences = [0.99] * 40
    correct = [True] * 30 + [False] * 10  # 75% accurate, claims 99%
    assert fit_temperature(confidences, correct) > 1.0


def test_a_well_calibrated_set_stays_near_one():
    confidences = [0.75] * 40
    correct = [True] * 30 + [False] * 10
    assert fit_temperature(confidences, correct) == pytest.approx(1.0, abs=0.35)


def test_fitted_temperatures_respect_the_bounds():
    out = fit_temperature([0.99] * 40, [False] * 40, bounds=(0.1, 10.0))
    assert 0.1 <= out <= 10.0


# --- promotion gates --------------------------------------------------------


def test_an_unmeasured_run_is_not_promotable():
    """The gate that protects the audio path from an unevaluated checkpoint."""
    gates = dict(Metrics().promotion_gates(min_accuracy=0.5, max_ece=0.1))
    assert gates["metrics were measured"] is False
    assert not Metrics().promotable(min_accuracy=0.5, max_ece=0.1)


def test_a_fully_measured_good_run_promotes():
    metrics = Metrics(
        choice_accuracy=0.82, ece=0.04, invalid_action_rate=0.0, win_rate_vs_rules=0.61
    )
    assert metrics.measured
    assert metrics.promotable(min_accuracy=0.5, max_ece=0.1)


def test_a_nonzero_invalid_rate_blocks_promotion_outright():
    """Absolute, not a score: a model reaching outside its menu is broken, not merely worse."""
    metrics = Metrics(
        choice_accuracy=0.95, ece=0.01, invalid_action_rate=0.02, win_rate_vs_rules=0.9
    )
    gates = dict(metrics.promotion_gates(min_accuracy=0.5, max_ece=0.1))
    assert gates["invalid_action_rate == 0"] is False
    assert not metrics.promotable(min_accuracy=0.5, max_ece=0.1)


def test_a_losing_run_does_not_promote_despite_good_accuracy():
    metrics = Metrics(
        choice_accuracy=0.9, ece=0.02, invalid_action_rate=0.0, win_rate_vs_rules=0.3
    )
    assert not metrics.promotable(min_accuracy=0.5, max_ece=0.1)


def test_a_half_calibrated_run_does_not_promote():
    """Missing ECE is not "zero ECE"."""
    metrics = Metrics(
        choice_accuracy=0.9, invalid_action_rate=0.0, win_rate_vs_rules=0.8
    )
    assert not metrics.measured
    assert not metrics.promotable(min_accuracy=0.5, max_ece=0.1)


def test_unmeasured_metrics_serialise_as_null_not_zero():
    """A JSON report full of 0.0 would read as a clean run to anything downstream."""
    payload = Metrics().as_dict()
    assert payload["measured"] is False
    assert payload["choice_accuracy"] is None
    assert payload["invalid_action_rate"] is None