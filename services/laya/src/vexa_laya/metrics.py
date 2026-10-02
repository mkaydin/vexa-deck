"""Evaluation metrics.

Pure Python and numpy. No torch, no checkpoint, no GPU — so a dataset can be sanity-checked and
a report can be recomputed without loading a model.

The two that matter most for this project are **ECE** and **Brier**, because the whole point of
calibrating Laya is to make its confidence meaningful enough to gate on. An accuracy number alone
would let an over-confident model pass promotion while still being unusable in a live set.

Definitions follow the standard scoring rules:

* Brier — mean squared error of the probability assigned to the option that was actually correct.
* ECE — expected calibration error over 15 equal-width bins, on the reported confidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Bin count used by Laya's own harness.
ECE_BINS = 15


@dataclass(frozen=True, slots=True)
class Prediction:
    """One scored decision."""

    #: The label the listener actually chose.
    correct_label: str
    #: What the model chose.
    predicted_label: str
    #: Reported confidence in the chosen option, 0..1.
    confidence: float
    #: Gold target distribution over labels.
    gold: dict[str, float]
    #: Model probabilities over labels.
    probabilities: dict[str, float]
    #: Wall time for the call.
    latency_ms: float


def _softmax_confidence(probabilities: dict[str, float]) -> float:
    return max(probabilities.values()) if probabilities else 0.0


def choice_accuracy(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        return 0.0
    hits = sum(1 for p in predictions if p.predicted_label == p.correct_label)
    return hits / len(predictions)


def soft_accuracy(predictions: Sequence[Prediction]) -> float:
    """Agreement with the gold *distribution*, not the gold label.

    A model that assigns 0.7 to the right option and 0.3 to an option the listener also accepted
    is scoring well here even though the argmax is only one acceptable answer. This is the metric
    that matches the training signal.
    """
    if not predictions:
        return 0.0
    total = 0.0
    for p in predictions:
        # Overlap between predicted mass and gold mass, over the labels both mention.
        labels = set(p.gold) | set(p.probabilities)
        total += sum(
            min(p.gold.get(label, 0.0), p.probabilities.get(label, 0.0))
            for label in labels
        )
    return total / len(predictions)


def brier_score(predictions: Sequence[Prediction]) -> float:
    """Mean squared error over the option the listener actually chose."""
    if not predictions:
        return 0.0
    total = 0.0
    for p in predictions:
        labels = set(p.gold) | set(p.probabilities)
        total += sum(
            (p.probabilities.get(label, 0.0) - p.gold.get(label, 0.0)) ** 2
            for label in labels
        )
    return total / len(predictions)


def expected_calibration_error(
    predictions: Sequence[Prediction], bins: int = ECE_BINS
) -> float:
    """Weighted gap between reported confidence and observed accuracy."""
    if not predictions:
        return 0.0
    buckets: list[list[Prediction]] = [[] for _ in range(bins)]
    for p in predictions:
        confidence = min(max(p.confidence, 0.0), 1.0)
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append(p)

    error = 0.0
    total = len(predictions)
    for bucket in buckets:
        if not bucket:
            continue
        accuracy = sum(1 for p in bucket if p.predicted_label == p.correct_label) / len(bucket)
        mean_confidence = sum(p.confidence for p in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(accuracy - mean_confidence)
    return error


def mean_confidence(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        return 0.0
    return sum(p.confidence for p in predictions) / len(predictions)


def latency_percentiles(predictions: Sequence[Prediction]) -> tuple[float, float]:
    """p50 and p95 wall time.

    Informational only. Laya's harness deliberately excludes timing metrics from baseline
    comparisons, because timing noise is not a quality regression.
    """
    if not predictions:
        return 0.0, 0.0
    ordered = sorted(p.latency_ms for p in predictions)
    return _percentile(ordered, 0.50), _percentile(ordered, 0.95)


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return float(ordered[index])


def invalid_action_rate(
    predictions: Sequence[Prediction], offered: Sequence[Sequence[str]]
) -> float:
    """Fraction of choices that named an option the filter never offered.

    Must be 0. A model that can reach outside its menu is not bounded, and the whole
    architecture rests on it being unable to.
    """
    if not predictions:
        return 0.0
    bad = sum(
        1
        for p, options in zip(predictions, offered, strict=False)
        if p.predicted_label not in options
    )
    return bad / len(predictions)


def pairwise_win_rate(
    model_choices: Sequence[str], baseline_choices: Sequence[str], gold: Sequence[str]
) -> float:
    """How often the model beats the baseline, counting an agreed choice as a shared win."""
    if not model_choices:
        return 0.0
    wins = 0.0
    for model, baseline, target in zip(model_choices, baseline_choices, gold, strict=False):
        model_right = model == target
        baseline_right = baseline == target
        if model_right and not baseline_right:
            wins += 1.0
        elif model_right and baseline_right:
            wins += 0.5  # agreed: shared credit, not a win for the model
    return wins / len(model_choices)


def score_within(
    predictions: Sequence[tuple[float, float]], tolerance: float
) -> float:
    """Fraction of graded answers within ``tolerance`` absolute."""
    if not predictions:
        return 0.0
    hits = sum(1 for predicted, expected in predictions if abs(predicted - expected) <= tolerance)
    return hits / len(predictions)


def score_mae(predictions: Sequence[tuple[float, float]]) -> float:
    if not predictions:
        return 0.0
    return sum(abs(p - e) for p, e in predictions) / len(predictions)


def brier_from_probability(confidence: float, correct: bool) -> float:
    """Single-decision Brier term, for smoke checks against the harness."""
    return (confidence - (1.0 if correct else 0.0)) ** 2


def temperature_scale(probability: float, temperature: float) -> float:
    """Temperature-scale one probability. ``temperature > 1`` softens."""
    if temperature <= 0:
        return probability
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    logit = math.log(probability / (1 - probability))
    scaled = logit / temperature
    if scaled >= 0:
        return 1.0 / (1.0 + math.exp(-scaled))
    e = math.exp(scaled)
    return e / (1.0 + e)


def fit_temperature(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bounds: tuple[float, float] = (0.1, 10.0),
    default: float = 1.2,
    steps: int = 200,
) -> float:
    """Fit one temperature by grid search on log-temperature.

    A grid rather than LBFGS: the objective is one-dimensional and smooth, the search is cheap,
    and a coarse grid cannot diverge to a degenerate scale — which is precisely the failure the
    published fit warns about when it is handed a slice it already trained on.
    """
    if len(confidences) < 10:
        return default

    def loss(log_t: float) -> float:
        temperature = math.exp(log_t)
        if not (bounds[0] <= temperature <= bounds[1]):
            return float("inf")
        total = 0.0
        for confidence, is_correct in zip(confidences, correct, strict=False):
            scaled = temperature_scale(confidence, temperature)
            total += (scaled - (1.0 if is_correct else 0.0)) ** 2
        return total / len(confidences)

    low, high = math.log(bounds[0]), math.log(bounds[1])
    best_t, best_loss = math.log(default), float("inf")
    for index in range(steps + 1):
        log_t = low + (high - low) * index / steps
        value = loss(log_t)
        if value < best_loss:
            best_t, best_loss = log_t, value
    return round(math.exp(best_t), 4)