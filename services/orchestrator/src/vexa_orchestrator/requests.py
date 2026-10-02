"""The live-request state machine.

``ARCHITECTURE.md:75-81``. This is where the project's central promise is kept: a request has an
honest status, and a slow model can never overturn the listener's latest direction.

The mechanism is a **generation number**. Every accepted request bumps the session's generation.
Anything planned under an older generation is stale the moment a newer request lands — whether it
is a scheduled transition or a generation job that finished late.

``PRODUCT.md:37`` names three request classes. Note that the class describes the *nature* of the
request, not what must happen: a generation-dependent request that happens to be satisfiable from
the ready library is satisfied from there, because playing a fitting track beats queueing a new
one. The class only decides what happens when the library cannot help.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from vexa_contracts import (
    ActionType,
    AssetManifest,
    DecisionRequest,
    FeasibleAction,
    RequestClass,
    RequestState,
    RequestStatus,
    SessionState,
    UserRequest,
)

from .feasibility import FeasibilityFilter
from .policy import DecisionPolicy, RulePolicy

#: Request states that are history rather than plans, and so are never superseded.
_SETTLED = frozenset(
    {
        RequestState.APPLIED,
        RequestState.FAILED,
        RequestState.REVIEW_NEEDED,
        RequestState.SUPERSEDED,
    }
)


@dataclass(slots=True)
class RequestOutcome:
    """What the orchestrator did, and what the listener should be told."""

    status: RequestStatus
    scheduled_action: FeasibleAction | None = None
    matched_asset_id: str | None = None
    #: True when nothing suitable existed and generation was queued instead.
    queued_generation: bool = False


@dataclass(slots=True)
class RequestEngine:
    """Drives a request from text to an honest status.

    Holds no audio and opens no device. It answers one question: given the session and the
    library, what should happen, and what do we tell the listener?
    """

    policy: DecisionPolicy = field(default_factory=RulePolicy)
    filter: FeasibilityFilter = field(default_factory=FeasibilityFilter)
    statuses: dict[str, RequestStatus] = field(default_factory=dict)

    # -- lifecycle ----------------------------------------------------------

    def submit(self, *, state: SessionState, request: UserRequest) -> RequestStatus:
        """Accept a request, superseding older ones that have not been applied yet."""
        self._supersede_older(request.generation, keep=request.request_id)
        status = RequestStatus(request_id=request.request_id, generation=request.generation)
        self.statuses[request.request_id] = status
        return status

    def _supersede_older(self, generation: int, *, keep: str) -> None:
        """Move every in-flight request from an older generation to ``SUPERSEDED``.

        Applied requests are left alone: they already happened, and rewriting history would be a
        lie. ``PRODUCT.md:39`` is explicit that only *uncommitted* plans are superseded.
        """
        for request_id, status in self.statuses.items():
            if request_id == keep or status.generation >= generation:
                continue
            if status.state in _SETTLED:
                continue
            status.transition(RequestState.SUPERSEDED)

    # -- resolution ---------------------------------------------------------

    def handle(
        self,
        *,
        state: SessionState,
        request: UserRequest,
        library: Iterable[AssetManifest],
        current_asset: AssetManifest | None = None,
        recent_family_ids: tuple[str, ...] = (),
    ) -> RequestOutcome:
        """Resolve a request against the ready library."""
        status = self.statuses[request.request_id]
        if status.state is RequestState.RECEIVED:
            status.transition(RequestState.INTERPRETED)

        menu = self.filter.build(
            state=state,
            library=library,
            constraints=request.constraints,
            current_asset=current_asset,
            recent_family_ids=recent_family_ids,
        ).candidates

        if request.request_class is RequestClass.IMMEDIATE_CONTROL:
            return self._apply_immediate(status, menu)

        match = self._best_transition(menu)
        if match is None:
            return self._queue_generation(status, menu)

        return self._schedule(status, state, match, menu)

    @staticmethod
    def _best_transition(menu: list[FeasibleAction]) -> FeasibleAction | None:
        """The highest-scoring transition the filter admitted."""
        transitions = [a for a in menu if a.action_type is ActionType.TRANSITION]
        if not transitions:
            return None
        return max(transitions, key=lambda a: (a.energy_fit, a.action_id))

    def _apply_immediate(
        self, status: RequestStatus, menu: list[FeasibleAction]
    ) -> RequestOutcome:
        """A bounded parameter change takes effect now without moving to a new asset."""
        chosen = next(
            (a for a in menu if a.action_type is ActionType.CONTINUE_CURRENT), menu[0]
        )
        status.transition(RequestState.MATCHED_READY)
        status.asset_id = chosen.asset_id
        status.transition(RequestState.SCHEDULED)
        status.transition(RequestState.APPLIED)
        status.detail = "applied immediately at the next safe boundary"
        return RequestOutcome(
            status=status, scheduled_action=chosen, matched_asset_id=chosen.asset_id
        )

    def _schedule(
        self,
        status: RequestStatus,
        state: SessionState,
        fallback_choice: FeasibleAction,
        menu: list[FeasibleAction],
    ) -> RequestOutcome:
        """Ask the selector to choose, then execute only what was actually offered.

        A selector returns an action id and nothing else, so the executed action is always looked
        back up from the offered menu. That lookup *is* the second deterministic gate: the only
        command that can reach the scheduler is one the filter already produced.
        """
        status.transition(RequestState.MATCHED_READY)

        decision_request = DecisionRequest(
            decision_id=f"dec-{status.request_id}",
            session_id=state.session_id,
            state=state,
            candidates=menu,
        )
        response = self.policy.choose(decision_request)

        action = _offered(menu, response.action_id) or fallback_choice
        if action.action_id != response.action_id:
            status.detail = (
                f"selector returned {response.action_id!r}, which was not on the menu; "
                "using the best feasible option instead"
            )
        else:
            status.detail = response.reasoning

        status.asset_id = action.asset_id
        status.transition(RequestState.SCHEDULED)
        return RequestOutcome(
            status=status, scheduled_action=action, matched_asset_id=action.asset_id
        )

    def _queue_generation(
        self, status: RequestStatus, menu: list[FeasibleAction]
    ) -> RequestOutcome:
        """Nothing suitable exists. Say so honestly; playback is unaffected."""
        status.transition(RequestState.MISSING)
        status.transition(RequestState.QUEUED)
        safe = sum(1 for a in menu if a.is_safe)
        status.detail = (
            "no ready asset matches this request; queued for background generation. "
            f"{safe} safe option(s) remain available to keep playing."
        )
        return RequestOutcome(status=status, queued_generation=True)

    # -- background generation ---------------------------------------------

    def generation_ready(
        self, status: RequestStatus, *, asset: AssetManifest, current_generation: int
    ) -> bool:
        """Admit a finished generation into the session.

        A stale result is admitted to the *library* but never to the *set*
        (``ARCHITECTURE.md:81``).
        """
        if status.generation < current_generation:
            status.detail = (
                f"asset {asset.asset_id} stored, but request generation {status.generation} "
                f"was superseded by {current_generation}; not introduced to the set"
            )
            return False

        # Walk the machine rather than jumping. A job that reached `ready` genuinely passed
        # through generating and validating, and the UI is meant to show that history.
        if status.state is RequestState.QUEUED:
            status.transition(RequestState.GENERATING)
            status.transition(RequestState.VALIDATING)
        status.asset_id = asset.asset_id
        status.transition(RequestState.READY)
        return True

    def status(self, request_id: str) -> RequestStatus | None:
        return self.statuses.get(request_id)


def _offered(menu: list[FeasibleAction], action_id: str) -> FeasibleAction | None:
    """Resolve an action id against the menu that was actually offered."""
    return next((a for a in menu if a.action_id == action_id), None)