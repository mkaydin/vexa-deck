"""The decision contract.

This is the boundary that keeps models bounded (``README.md:19``). The orchestrator builds a small
list of *feasible* actions and a selector — Laya or the deterministic rule policy — returns an
``action_id``. A selector can never emit a DSP command, a gain value, or an arbitrary parameter;
it picks from a menu that has already been validated.

Two independent gates protect the audio thread:

1. The **feasibility filter** builds the menu (``ARCHITECTURE.md:64``).
2. A **second deterministic validator** re-checks the chosen action before it is scheduled.

Both are required. The second one exists because a selector can be wrong, and because a
fine-tuned model is a probabilistic component that will occasionally return something the filter
would never have produced.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .asset import Contract
from .enums import ActionType
from .session import SessionState


def _utcnow() -> datetime:
    return datetime.now(UTC)


class FeasibleAction(BaseModel):
    """One concrete, already-validated option.

    Mirrors the shape in ``ARCHITECTURE.md:52-62``. The fields are the *only* things a selector
    may influence; there is deliberately no free-form command here.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    action_type: ActionType
    asset_id: str | None = None

    #: Bar within the target asset where playback begins.
    entry_bar: int = Field(default=0, ge=0)
    #: Bar in the session at which the transition commits.
    start_at_session_bar: int = Field(ge=0)
    #: Crossfade length in bars.
    fade_bars: int = Field(default=0, ge=0, le=64)
    #: Playback-rate ratio applied to reach the target tempo.
    tempo_ratio: float = Field(default=1.0, gt=0.5, le=2.0)
    #: e.g. ``bass_swap``, ``cut``, ``fade``.
    transition_preset: str = "fade"

    #: 0..1, how far this action moves the set toward the active request's target energy.
    energy_fit: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Human-readable justification, derived from measured features — never invented after the
    #: fact (``IDEAS.md:19``).
    rationale: str = ""

    @model_validator(mode="after")
    def _fallback_needs_no_asset(self) -> FeasibleAction:
        if self.action_type in (ActionType.CONTINUE_CURRENT, ActionType.PLAY_FALLBACK_LOOP):
            return self
        if self.asset_id is None:
            raise ValueError(
                f"action {self.action_id!r} of type {self.action_type.value!r} requires an asset_id"
            )
        return self

    @property
    def is_safe(self) -> bool:
        """Safe actions are always present in the menu (``ARCHITECTURE.md:64``)."""
        return self.action_type in (ActionType.CONTINUE_CURRENT, ActionType.PLAY_FALLBACK_LOOP)


class DecisionRequest(Contract):
    """What a selector is asked to answer (``ARCHITECTURE.md:103``)."""

    decision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


    #: The snapshot the decision is made against. Its ``generation`` is the supersession key, so
    #: it is read from here rather than duplicated in a second field that could disagree.
    state: SessionState
    #: The complete feasible menu. 3-8 entries is the working range — Laya has an option-token
    #: budget and high option counts degrade its confidence (``LAYA_DATA.md:11``).
    candidates: list[FeasibleAction] = Field(min_length=2)
    #: Wall-clock instant by which an answer is still useful. Past it the scheduler drops the
    #: answer rather than committing late (``ARCHITECTURE.md:64``).
    deadline: datetime = Field(default_factory=_utcnow)
    policy_version: str = "rules-v1"

    @model_validator(mode="after")
    def _always_offers_a_safe_option(self) -> DecisionRequest:
        if not any(a.is_safe for a in self.candidates):
            raise ValueError(
                "a DecisionRequest must always offer continue_current or play_fallback_loop; "
                "without a safe option the set has nowhere to go if nothing is chosen"
            )
        return self

    @property
    def ids(self) -> set[str]:
        return {a.action_id for a in self.candidates}

    @property
    def generation(self) -> int:
        """The supersession key, read from the state snapshot."""
        return self.state.generation

    def by_id(self, action_id: str) -> FeasibleAction | None:
        return next((a for a in self.candidates if a.action_id == action_id), None)


class DecisionResponse(BaseModel):
    """A selector's answer. Deliberately thin — an ID, a confidence, and a reason."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    action_id: str
    #: 0..1. Only meaningful after calibration (``LAYA_DATA.md:65``); until then it is a raw
    #: score and must not gate anything.
    confidence: float = Field(ge=0.0, le=1.0)
    #: Which selector produced this, for the shadow-mode disagreement log.
    selector: str = "rules"
    #: Set when the selector abstained and the rules decided instead.
    fell_back: bool = False
    reasoning: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)


class ScheduledAction(Contract):
    """A committed transition (``ARCHITECTURE.md:104``).

    Carries the asset's version and content hash so a scheduler decision cannot act on a file
    that changed after the decision was made.
    """

    action_id: str
    session_id: str
    #: Session bar at which the transition commits.
    commit_bar: int = Field(ge=0)
    asset_id: str | None = None
    asset_version: int | None = None
    asset_sha256: str | None = None
    #: Generation this was planned under; a mismatch invalidates the action.
    generation: int = Field(ge=0)
    #: Where to go if the chosen asset fails to load in time.
    fallback_action_id: str | None = None
    committed_at: datetime = Field(default_factory=_utcnow)
    #: Deadline by which the asset must be loaded. After it, the action is dropped.
    load_deadline: datetime | None = None

    def deadline_expired(self, *, at: datetime | None = None) -> bool:
        if self.load_deadline is None:
            return False
        return (at or _utcnow()) > self.load_deadline

    def is_stale(self, current_generation: int) -> bool:
        """Whether a newer user request has superseded this action."""
        return self.generation < current_generation

    @classmethod
    def from_action(
        cls,
        action: FeasibleAction,
        session: SessionState,
        *,
        asset_version: int | None = None,
        asset_sha256: str | None = None,
        fallback_action_id: str | None = None,
        load_lead_time_s: float = 5.0,
    ) -> ScheduledAction:
        lead = timedelta(seconds=load_lead_time_s)
        return cls(
            action_id=action.action_id,
            session_id=session.session_id,
            commit_bar=action.start_at_session_bar,
            asset_id=action.asset_id,
            asset_version=asset_version,
            asset_sha256=asset_sha256,
            generation=session.generation,
            fallback_action_id=fallback_action_id,
            load_deadline=_utcnow() + lead,
        )