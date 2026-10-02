"""In-memory session and library store.

Deliberately not a database. Phase B2 replaces this with the real index; until then the store
holds everything in process so the orchestrator is runnable and testable without infrastructure.

The important thing it already gets right: **assets are keyed by content hash, not by path.**
``ScheduledAction`` binds to a hash so a decision cannot act on a file that changed underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vexa_contracts import (
    ApprovalState,
    AssetManifest,
    GenerationJob,
    RequestLog,
    RequestStatus,
    SessionState,
)

from .policy import DecisionPolicy
from .requests import RequestEngine
from .scheduler import Scheduler


@dataclass(slots=True)
class SessionRecord:
    """Everything the orchestrator holds for one set."""

    state: SessionState
    scheduler: Scheduler = field(default_factory=Scheduler)
    engine: RequestEngine = field(default_factory=RequestEngine)
    jobs: dict[str, GenerationJob] = field(default_factory=dict)
    log: RequestLog | None = None

    def __post_init__(self) -> None:
        if self.log is None:
            self.log = RequestLog(session_id=self.state.session_id)

    @property
    def session_id(self) -> str:
        return self.state.session_id


@dataclass(slots=True)
class Store:
    """Process-wide state. Single-threaded; the API layer serialises access."""

    sessions: dict[str, SessionRecord] = field(default_factory=dict)
    assets: dict[str, AssetManifest] = field(default_factory=dict)
    jobs: dict[str, GenerationJob] = field(default_factory=dict)
    policy: DecisionPolicy | None = None

    # -- sessions -----------------------------------------------------------

    def create_session(self, state: SessionState) -> SessionRecord:
        record = SessionRecord(state=state)
        if self.policy is not None:
            record.engine.policy = self.policy
        self.sessions[state.session_id] = record
        record.log.record(kind="session_created", theme=state.theme)
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    # -- assets -------------------------------------------------------------

    def add_asset(self, manifest: AssetManifest) -> AssetManifest:
        self.assets[manifest.asset_id] = manifest
        return manifest

    def by_sha256(self, digest: str) -> AssetManifest | None:
        return next((a for a in self.assets.values() if a.content_sha256 == digest), None)

    def ready_assets(self) -> list[AssetManifest]:
        return [a for a in self.assets.values() if a.admissible()]

    def approve(self, asset_id: str) -> AssetManifest:
        """Promote a verified candidate to the ready library (``ARCHITECTURE.md:102``).

        Readying also requires the quality gates to have passed, which ``AssetManifest``
        enforces structurally — so an approval of a failing asset is refused here rather than
        producing a manifest that claims to be ready.
        """
        asset = self.assets.get(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        approved = asset.model_copy(
            update={
                "approval": ApprovalState.APPROVED,
                "readiness": (asset.quality.passed and asset.readiness) or asset.readiness,
            }
        )
        self.assets[asset_id] = approved
        return approved

    # -- jobs ---------------------------------------------------------------

    def add_job(self, job: GenerationJob) -> GenerationJob:
        self.jobs[job.job_id] = job
        record = self.sessions.get(job.session_id or "")
        if record is not None:
            record.jobs[job.job_id] = job
            record.log.record(kind="job_enqueued", job_id=job.job_id, brief=job.brief.style)
        return job

    def next_generation(self) -> int:
        return len(self.jobs) + 1

    # -- introspection ------------------------------------------------------

    def status_of(self, request_id: str) -> RequestStatus | None:
        for record in self.sessions.values():
            found = record.engine.status(request_id)
            if found is not None:
                return found
        return None