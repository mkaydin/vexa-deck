"""VEXA//DECK versioned contracts.

Every message that crosses a process boundary is one of these types. Nothing in the system passes
loose dicts between components, and no component invents a field the others have not declared —
``extra="forbid"`` is enforced on the models so a typo is an error rather than a silently dropped
value.

Import from here rather than from submodules::

    from vexa_contracts import AssetManifest, UserRequest
"""

from __future__ import annotations

from .asset import (
    AssetManifest,
    AudioProperties,
    BeatGrid,
    Contract,
    Estimate,
    KeyEstimate,
    LoopPoints,
    Provenance,
    QualityReport,
    SectionMarker,
)
from .decision import (
    DecisionRequest,
    DecisionResponse,
    FeasibleAction,
    ScheduledAction,
)
from .enums import (
    JOB_TRANSITIONS,
    REQUEST_TRANSITIONS,
    USER_FACING_STATUS,
    ActionType,
    ApprovalState,
    BackendKind,
    CpuPolicy,
    JobState,
    ReadinessState,
    RequestClass,
    RequestState,
    SourceType,
)
from .jobs import GenerationBrief, GenerationJob, GenerationResult
from .requests import RequestConstraints, RequestLog, RequestStatus, UserRequest
from .schema import export_schemas, registry, schema_bundle
from .session import DeckId, DeckState, MusicalClock, SessionState
from .version import CONTRACT_REVISION, CONTRACT_VERSION, supports

__all__ = [
    # versioning
    "CONTRACT_VERSION",
    "CONTRACT_REVISION",
    "supports",
    # base
    "Contract",
    # asset
    "AssetManifest",
    "AudioProperties",
    "BeatGrid",
    "Estimate",
    "KeyEstimate",
    "LoopPoints",
    "Provenance",
    "QualityReport",
    "SectionMarker",
    # session
    "DeckId",
    "DeckState",
    "MusicalClock",
    "SessionState",
    # decision
    "DecisionRequest",
    "DecisionResponse",
    "FeasibleAction",
    "ScheduledAction",
    # requests
    "RequestConstraints",
    "RequestLog",
    "RequestStatus",
    "UserRequest",
    "REQUEST_TRANSITIONS",
    "USER_FACING_STATUS",
    "registry",
    # jobs
    "GenerationBrief",
    "GenerationJob",
    "GenerationResult",
    "JOB_TRANSITIONS",
    # enums
    "ActionType",
    "ApprovalState",
    "BackendKind",
    "CpuPolicy",
    "JobState",
    "ReadinessState",
    "RequestClass",
    "RequestState",
    "SourceType",
    # schema export
    "export_schemas",
    "schema_bundle",
]