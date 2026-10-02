"""User requests and their live status.

The central invariant lives here. Every request carries a **generation number**, and a newer
request supersedes an older *uncommitted* one. A generation job that finishes late may keep its
asset, but the scheduler rechecks the current generation before using it
(``ARCHITECTURE.md:81``).

This is what stops a slow model from overruling the listener: the audio that is already playing is
never disturbed, and a result that arrives after the listener changed their mind cannot pull the
set backwards.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .asset import Contract
from .enums import (
    REQUEST_TRANSITIONS,
    USER_FACING_STATUS,
    RequestClass,
    RequestState,
)
from .version import CONTRACT_REVISION, CONTRACT_VERSION


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RequestConstraints(BaseModel):
    """Hard limits a request imposes on candidate selection."""

    model_config = ConfigDict(extra="forbid")

    #: ``None`` means unconstrained.
    min_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    max_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Requested tempo band in BPM.
    min_bpm: float | None = Field(default=None, gt=0.0)
    max_bpm: float | None = Field(default=None, gt=0.0)
    forbid_vocals: bool = False
    allowed_keys: list[str] = Field(default_factory=list)
    max_tempo_ratio: float = Field(default=1.06, gt=1.0, le=1.5)

    @model_validator(mode="after")
    def _ordered(self) -> RequestConstraints:
        if (
            self.min_energy is not None
            and self.max_energy is not None
            and self.min_energy > self.max_energy
        ):
            raise ValueError("min_energy exceeds max_energy")
        if self.min_bpm is not None and self.max_bpm is not None and self.min_bpm > self.max_bpm:
            raise ValueError("min_bpm exceeds max_bpm")
        return self


class UserRequest(Contract):
    """A parsed live request. Immutable once accepted."""

    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    #: Monotonic within a session. The supersession key.
    generation: int = Field(ge=0)

    text: str = Field(min_length=1)
    request_class: RequestClass
    constraints: RequestConstraints = Field(default_factory=RequestConstraints)
    target_energy: float = Field(default=0.5, ge=0.0, le=1.0)

    received_at: datetime = Field(default_factory=_utcnow)

    def supersedes(self, older: UserRequest) -> bool:
        return self.generation > older.generation


class RequestStatus(BaseModel):
    """Live status of a request, with transitions validated against the state machine."""

    model_config = ConfigDict(validate_assignment=True)

    request_id: str
    generation: int = Field(ge=0)
    state: RequestState = RequestState.RECEIVED
    #: Job enqueued to satisfy a generation-dependent request.
    job_id: str | None = None
    #: Asset promoted into the ready library for this request.
    asset_id: str | None = None
    updated_at: datetime = Field(default_factory=_utcnow)
    detail: str = ""

    def transition(self, new_state: RequestState) -> None:
        """Move to ``new_state``, refusing any transition the machine forbids.

        Raises:
            ValueError: if the transition is not in ``REQUEST_TRANSITIONS``.
        """
        if new_state is self.state:
            return
        allowed = REQUEST_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(
                f"request {self.request_id!r} cannot move "
                f"{self.state.value!r} -> {new_state.value!r}; "
                f"allowed: {sorted(s.value for s in allowed) or 'none (terminal)'}"
            )
        self.state = new_state
        self.updated_at = _utcnow()

    @property
    def user_facing(self) -> str:
        """One of the five statuses the UI must distinguish (``README.md:21``)."""
        return USER_FACING_STATUS[self.state]

    @property
    def is_terminal(self) -> bool:
        return not REQUEST_TRANSITIONS[self.state]

    @property
    def is_settled(self) -> bool:
        """Whether the listener can rely on this request having taken effect."""
        return self.state in (RequestState.APPLIED, RequestState.SCHEDULED)


class RequestLog(BaseModel):
    """Append-only session history (``PRODUCT.md:53``).

    Reproducibility is a stated design principle (``README.md:22``): request, selected assets,
    action, timing and outcome.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    entries: list[dict[str, object]] = Field(default_factory=list)

    def record(self, *, kind: str, **fields: object) -> None:
        self.entries.append({"kind": kind, "at": _utcnow().isoformat(), **fields})


def contract_fields() -> dict[str, type[int] | type[str]]:
    """The version fields every contract carries, for envelope validation on receipt."""
    return {"contract_version": int, "contract_revision": str}


__all__ = [
    "CONTRACT_REVISION",
    "CONTRACT_VERSION",
    "Contract",
    "RequestConstraints",
    "RequestLog",
    "RequestStatus",
    "UserRequest",
]