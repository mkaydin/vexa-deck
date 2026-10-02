"""Enumerations and the state machines that constrain them.

Two state machines are defined here rather than in the services, because both the orchestrator and
the GUI need to agree on them and neither should own the definition:

* :class:`RequestState` — the live-request machine from ``ARCHITECTURE.md:75``.
* :class:`JobState` — the background-generation machine from ``PRODUCT.md:51``.

Transitions are validated rather than merely documented. A service that tries to move a job from
``ready`` back to ``generating`` gets an error instead of a corrupt timeline.
"""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    """Where an asset came from. ``ARCHITECTURE.md:44``."""

    YUE2_RENDER = "yue2_render"
    REVISED_RENDER = "revised_render"
    USER_IMPORT = "user_import"
    STEM_SEPARATION = "stem_separation"
    LOOP_EXTRACT = "loop_extract"
    PACK_IMPORT = "pack_import"


class ApprovalState(StrEnum):
    """Human approval. Only approved assets may enter the ready library (``PRODUCT.md:27``)."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReadinessState(StrEnum):
    """Whether an asset may be scheduled.

    ``READY`` is reachable only after decode, duration, loudness, beat-grid, loop-boundary and
    audio-quality checks pass (``ARCHITECTURE.md:46``). Nothing in the runtime may schedule
    against any other value.
    """

    UNANALYZED = "unanalyzed"
    ANALYZING = "analyzing"
    QUARANTINED = "quarantined"
    AWAITING_REVIEW = "awaiting_review"
    READY = "ready"
    FAILED = "failed"


class ActionType(StrEnum):
    """The concrete things a decision can choose between (``LAYA_DATA.md:7-9``)."""

    CONTINUE_CURRENT = "continue_current"
    TRANSITION = "transition"
    PLAY_FALLBACK_LOOP = "play_fallback_loop"
    LAYER_ADD = "layer_add"
    LAYER_REMOVE = "layer_remove"
    BASS_SWAP = "bass_swap"


class RequestClass(StrEnum):
    """How a user request should be satisfied (``PRODUCT.md:33-37``)."""

    IMMEDIATE_CONTROL = "immediate_control"
    READY_LIBRARY = "ready_library"
    GENERATION_DEPENDENT = "generation_dependent"


class RequestState(StrEnum):
    """Live-request lifecycle. Transitions mirror ``ARCHITECTURE.md:75`` exactly."""

    RECEIVED = "received"
    INTERPRETED = "interpreted"
    MATCHED_READY = "matched_ready"
    SCHEDULED = "scheduled"
    APPLIED = "applied"
    MISSING = "missing"
    QUEUED = "queued"
    GENERATING = "generating"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"
    REVIEW_NEEDED = "review_needed"
    SUPERSEDED = "superseded"


#: ``ARCHITECTURE.md:75-79``, with one correction the diagrams do not make explicit.
#:
#: ``PRODUCT.md:39`` supersedes only older *uncommitted* plans. A request that has reached
#: ``SCHEDULED`` but whose transition has not fired yet is still uncommitted — the audio has not
#: changed — so it must be supersedable too. Anything already ``APPLIED``, ``FAILED`` or
#: ``REVIEW_NEEDED`` is history and is left alone.
REQUEST_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.RECEIVED: frozenset({RequestState.INTERPRETED, RequestState.SUPERSEDED}),
    RequestState.INTERPRETED: frozenset(
        {RequestState.MATCHED_READY, RequestState.MISSING, RequestState.SUPERSEDED}
    ),
    RequestState.MATCHED_READY: frozenset(
        {RequestState.SCHEDULED, RequestState.INTERPRETED, RequestState.SUPERSEDED}
    ),
    RequestState.MISSING: frozenset({RequestState.QUEUED, RequestState.SUPERSEDED}),
    RequestState.QUEUED: frozenset({RequestState.GENERATING, RequestState.SUPERSEDED}),
    RequestState.GENERATING: frozenset({RequestState.VALIDATING, RequestState.SUPERSEDED}),
    RequestState.VALIDATING: frozenset(
        {RequestState.READY, RequestState.FAILED, RequestState.SUPERSEDED}
    ),
    RequestState.READY: frozenset({RequestState.SCHEDULED, RequestState.SUPERSEDED}),
    RequestState.SCHEDULED: frozenset(
        {RequestState.APPLIED, RequestState.INTERPRETED, RequestState.SUPERSEDED}
    ),
    # Terminal.
    RequestState.APPLIED: frozenset(),
    RequestState.FAILED: frozenset(),
    RequestState.REVIEW_NEEDED: frozenset(),
    RequestState.SUPERSEDED: frozenset(),
}

#: The statuses the UI must distinguish (``README.md:21`` names five; see the note below).
#:
#: ``"interpreting"`` is a deliberate addition. Three states sit between accepting a request and
#: classifying it, and none of them honestly maps onto the five documented labels: reporting
#: ``generating`` would claim work that is not happening, and reporting ``ready`` would claim an
#: outcome nobody has decided. It is a sub-second transient the UI rarely paints, but when it
#: does paint a status it must be true.
REQUEST_STATUS_LABELS: frozenset[str] = frozenset(
    {"applied now", "scheduled", "generating", "ready", "unable to fulfil", "interpreting"}
)

USER_FACING_STATUS: dict[RequestState, str] = {
    RequestState.RECEIVED: "interpreting",
    RequestState.INTERPRETED: "interpreting",
    RequestState.MATCHED_READY: "ready",
    RequestState.SCHEDULED: "scheduled",
    RequestState.APPLIED: "applied now",
    RequestState.MISSING: "generating",
    RequestState.QUEUED: "generating",
    RequestState.GENERATING: "generating",
    RequestState.VALIDATING: "generating",
    RequestState.READY: "ready",
    RequestState.FAILED: "unable to fulfil",
    RequestState.REVIEW_NEEDED: "unable to fulfil",
    # A newer request replaced this one, so it will never be applied.
    RequestState.SUPERSEDED: "unable to fulfil",
}


class JobState(StrEnum):
    """Background generation lifecycle (``PRODUCT.md:51``)."""

    QUEUED = "queued"
    RUNNING = "running"
    ANALYZING = "analyzing"
    REVIEW_NEEDED = "review_needed"
    READY = "ready"
    FAILED = "failed"
    CANCELED = "canceled"


JOB_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELED}),
    JobState.RUNNING: frozenset(
        {JobState.ANALYZING, JobState.FAILED, JobState.CANCELED}
    ),
    JobState.ANALYZING: frozenset(
        {JobState.READY, JobState.REVIEW_NEEDED, JobState.FAILED, JobState.CANCELED}
    ),
    JobState.REVIEW_NEEDED: frozenset({JobState.READY, JobState.FAILED, JobState.CANCELED}),
    # Terminal.
    JobState.READY: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELED: frozenset(),
}


class BackendKind(StrEnum):
    """Generation backends. Selected per job by capacity — see PLAN.md §3.4.

    ``DRY_RUN`` is a test fixture that writes a tone instead of calling a model. It has its own
    identity precisely so it can never be mistaken for, or routed in place of, a real backend.
    """

    YUE2_CPP = "yue2.cpp"
    TORCH = "torch"
    AUDIO_CPP = "audio.cpp"
    DRY_RUN = "dry_run"


class CpuPolicy(StrEnum):
    """How hard a model is allowed to work on the audio machine.

    The docs treat generation competing with audio for CPU/GPU as a named risk
    (``ROADMAP.md:53``). A job may never be allowed to starve the audio callback.
    """

    #: Unrestricted. Only valid when no session is active.
    FULL = "full"
    #: Yield to the audio thread.
    REALTIME_SAFE = "realtime_safe"