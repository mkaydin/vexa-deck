"""Deterministic decision policies.

Two implementations of one interface:

* :class:`RulePolicy` — the baseline. Deterministic, always available, and the permanent fallback.
  ``LAYA_DATA.md:81`` requires it to exist so that a fine-tuned model is an *improvement* rather
  than a dependency.
* :class:`ShadowPolicy` — wraps any policy, records what the inner policy would have chosen, and
  still lets the rules control audio. This is how Laya is introduced without letting it break a
  set: disagreements accumulate until there is enough evidence to switch control over.

Nothing here touches audio. A policy returns an action; the scheduler decides whether to commit it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from vexa_contracts import (
    ActionType,
    DecisionRequest,
    DecisionResponse,
    FeasibleAction,
)


class DecisionPolicy(Protocol):
    """Chooses one action from a validated menu."""

    name: str

    def choose(self, request: DecisionRequest) -> DecisionResponse: ...


@dataclass(frozen=True, slots=True)
class RuleWeights:
    """How much each consideration is worth.

    These are the knobs the docs call out when they say the rule policy is the baseline every
    learned policy must beat.
    """

    #: Distance from the requested energy worth taking a transition at all.
    energy_weight: float = 1.0
    #: Reward for a transition whose asset is already loaded.
    loaded_bonus: float = 0.15
    #: Penalty for changing direction relative to the current energy.
    inertia: float = 0.25
    #: Below this score the policy prefers holding position over transitioning.
    transition_threshold: float = 0.1



@dataclass(slots=True)
class RulePolicy:
    """Deterministic selection. No model, no randomness, no I/O."""

    weights: RuleWeights = field(default_factory=RuleWeights)
    name: str = "rules"
    #: Assets currently resident, so transitions that will actually land can be preferred.
    _loaded_asset_ids: set[str] = field(default_factory=set, init=False)

    def mark_loaded(self, asset_ids: set[str]) -> None:
        """Tell the policy what is resident, so it can prefer transitions that will land."""
        self._loaded_asset_ids = set(asset_ids)

    def choose(self, request: DecisionRequest) -> DecisionResponse:
        transitions = [a for a in request.candidates if not a.is_safe]
        safe = [a for a in request.candidates if a.is_safe]
        if not safe:  # pragma: no cover - DecisionRequest enforces this
            raise ValueError("no safe action offered")

        current = request.state.energy
        best: FeasibleAction | None = None
        best_score = float("-inf")

        for action in transitions:
            score = self.weights.energy_weight * (1.0 - abs(action.energy_fit - current))
            score += self.weights.inertia * (1.0 - abs(action.energy_fit - current))
            if action.asset_id in self._loaded_asset_ids:
                score += self.weights.loaded_bonus
            if score > best_score:
                best, best_score = action, score

        if best is not None and best_score >= self.weights.transition_threshold:
            return DecisionResponse(
                decision_id=request.decision_id,
                action_id=best.action_id,
                confidence=min(1.0, max(0.0, best_score)),
                selector=self.name,
                reasoning=f"rule policy: energy fit {best.energy_fit:.2f} vs current {current:.2f}",
            )

        chosen = next(
            (a for a in safe if a.action_type is ActionType.CONTINUE_CURRENT), safe[0]
        )
        return DecisionResponse(
            decision_id=request.decision_id,
            action_id=chosen.action_id,
            confidence=0.5,
            selector=self.name,
            reasoning="rule policy: no transition cleared the threshold",
        )


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One shadow-mode observation: what the model wanted versus what the rules did."""

    decision_id: str
    selector: str
    chosen_action_id: str
    rule_action_id: str
    confidence: float
    fell_back: bool

    @property
    def agrees(self) -> bool:
        return self.chosen_action_id == self.rule_action_id


@dataclass(slots=True)
class ShadowPolicy:
    """Runs an inner policy for observation only; the rules keep control.

    ``LAYA_DATA.md:81``: *"Run a shadow mode first: Laya chooses but the rule policy controls
    audio. Compare choices and listen to disagreements."*
    """

    inner: DecisionPolicy
    rules: DecisionPolicy
    #: Abstain below this confidence. Until Laya is calibrated this must stay disabled, because
    #: an uncalibrated confidence is meaningless (``LAYA_DATA.md:65``).
    min_confidence: float | None = None
    disagreements: list[Disagreement] = field(default_factory=list)
    name: str = "shadow"

    def choose(self, request: DecisionRequest) -> DecisionResponse:
        rule_response = self.rules.choose(request)
        model_response = self.inner.choose(request)

        self.disagreements.append(
            Disagreement(
                decision_id=request.decision_id,
                selector=self.inner.name,
                chosen_action_id=model_response.action_id,
                rule_action_id=rule_response.action_id,
                confidence=model_response.confidence,
                fell_back=(
                    self.min_confidence is not None
                    and model_response.confidence < self.min_confidence
                ),
            )
        )

        if (
            self.min_confidence is not None
            and model_response.confidence < self.min_confidence
        ):
            return DecisionResponse(
                decision_id=request.decision_id,
                action_id=rule_response.action_id,
                confidence=rule_response.confidence,
                selector=self.rules.name,
                fell_back=True,
                reasoning=f"{self.inner.name} below confidence floor; rules decided",
            )

        return DecisionResponse(
            decision_id=request.decision_id,
            action_id=rule_response.action_id,
            confidence=rule_response.confidence,
            selector=self.rules.name,
            reasoning=f"shadow mode: {self.inner.name} chose {model_response.action_id}",
        )

    def agreement_rate(self) -> float:
        """Fraction of observations where the model matched the rules."""
        if not self.disagreements:
            return 0.0
        agreeing = sum(1 for d in self.disagreements if d.agrees)
        return agreeing / len(self.disagreements)


@dataclass(slots=True)
class ChainPolicy:
    """Tries policies in order, falling through on low confidence or an invalid choice.

    This is what lets Laya be switched on without a redeploy: keep the chain, change the order.
    """

    policies: Sequence[DecisionPolicy]
    validator: Callable[[DecisionRequest, FeasibleAction], bool] | None = None
    min_confidence: float = 0.0
    name: str = "chain"

    def choose(self, request: DecisionRequest) -> DecisionResponse:
        for policy in self.policies:
            try:
                response = policy.choose(request)
            except Exception:
                continue
            if response.confidence < self.min_confidence:
                continue
            offered = request.by_id(response.action_id)
            if offered is None:
                continue
            if self.validator is not None and not self.validator(request, offered):
                continue
            return response

        # Every policy declined or failed. Holding position is always safe.
        safe = next((a for a in request.candidates if a.is_safe), None)
        if safe is None:  # pragma: no cover - enforced by DecisionRequest
            raise ValueError("no safe action offered")
        return DecisionResponse(
            decision_id=request.decision_id,
            action_id=safe.action_id,
            confidence=0.0,
            selector="fallback",
            fell_back=True,
            reasoning="every policy declined; holding position",
        )