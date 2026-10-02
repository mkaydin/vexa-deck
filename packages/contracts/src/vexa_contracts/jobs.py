"""Generation jobs.

One contract serves every backend and every location. A job does not know or care whether it will
be executed by ``yue2.cpp`` on the 4060, ``torch`` on the 5060 Ti, or a user-operated remote
worker on another machine (``IDEAS.md:41``, ``ROADMAP.md:31``). That is the whole point of
versioning it up front: the handoff is a deployment change, not a protocol change.

Transitions are validated so a cancelled job cannot quietly become ``ready`` afterwards.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .asset import Contract
from .enums import JOB_TRANSITIONS, BackendKind, CpuPolicy, JobState


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GenerationBrief(BaseModel):
    """What to generate. The request the planner turns into one of these."""

    model_config = ConfigDict(extra="forbid")

    style: str = Field(min_length=1)
    lyrics: str = Field(default="")
    target_duration_s: float = Field(default=180.0, gt=0.0, le=900.0)
    #: ``full`` = melody + chord planning, ``melody`` = melody only, ``off`` = no symbolic plan.
    #: ``full`` and ``melody`` are what allow plan→edit→regenerate (``PLAN.md`` §3.4).
    cot: str = "full"
    cfg_scale: float = Field(default=1.2, ge=0.0, le=10.0)
    energy_target: float = Field(default=0.5, ge=0.0, le=1.0)
    forbid_vocals: bool = False
    language: str = "en"

    @model_validator(mode="after")
    def _known_cot(self) -> GenerationBrief:
        if self.cot not in ("full", "melody", "off"):
            raise ValueError(f"cot must be one of full/melody/off, got {self.cot!r}")
        return self


class GenerationJob(Contract):
    """One unit of background work."""

    job_id: str = Field(min_length=1)
    session_id: str | None = None
    #: The request that caused this job. Used to re-check supersession before scheduling.
    request_id: str | None = None
    request_generation: int = Field(default=0, ge=0)

    brief: GenerationBrief
    seed: int | None = None
    family_id: str | None = None

    #: Lower runs first. A live request outranks background library preparation.
    priority: int = Field(default=100, ge=0, le=1000)
    state: JobState = JobState.QUEUED
    #: Generation is never allowed to starve the audio callback (``ROADMAP.md:53``).
    cpu_policy: CpuPolicy = CpuPolicy.REALTIME_SAFE
    backend: BackendKind | None = None
    #: Index of the GPU this job may run on, if already assigned.
    gpu_index: int | None = None

    asset_id: str | None = None
    error: str | None = None
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @property
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts and self.state is JobState.FAILED

    @property
    def is_live_request(self) -> bool:
        return self.priority < 100

    def transition(self, new_state: JobState) -> None:
        """Move to ``new_state``, refusing transitions the machine forbids.

        Raises:
            ValueError: if the transition is not in ``JOB_TRANSITIONS``.
        """
        if new_state is self.state:
            return
        allowed = JOB_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(
                f"job {self.job_id!r} cannot move {self.state.value!r} -> {new_state.value!r}; "
                f"allowed: {sorted(s.value for s in allowed) or 'none (terminal)'}"
            )
        self.state = new_state
        self.updated_at = _utcnow()

    def claim(self, backend: BackendKind, gpu_index: int) -> None:
        """Assign a backend and GPU, and move to ``RUNNING``."""
        if self.state is not JobState.QUEUED:
            raise ValueError(f"job {self.job_id!r} is {self.state.value!r}, not queued")
        self.backend = backend
        self.gpu_index = gpu_index
        self.attempts += 1
        self.transition(JobState.RUNNING)


class GenerationResult(BaseModel):
    """What a backend hands back. Every field is provenance, not just audio."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    backend: BackendKind
    #: GGUF renders must stay distinguishable from bf16 ones downstream.
    quantization: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    seed: int | None = None

    audio_path: str
    content_sha256: str
    duration_s: float = Field(gt=0.0)
    #: The ABC score the model wrote, when the backend exposes one. Lets a user or agent edit the
    #: plan and regenerate rather than re-rolling from scratch.
    score_abc: str | None = None

    #: Seconds of wall time the render took. Recorded because the docs want measured generation
    #: times rather than estimates (``ROADMAP.md:33``).
    wall_time_s: float = Field(ge=0.0)
    peak_vram_mib: int | None = None
    artifacts_path: str | None = None
    finished_at: datetime = Field(default_factory=_utcnow)