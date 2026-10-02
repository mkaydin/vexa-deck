"""The orchestrator HTTP API.

Mirrors the sketches in ``ARCHITECTURE.md:98-104``. Transport is an implementation choice for the
prototype, but the *payloads* are the versioned contracts, so swapping HTTP for in-process messages
or gRPC later changes no caller.

Nothing here talks to a model directly, and nothing here opens an audio device. Killing this
process does not stop playback — that is Phase B1's exit criterion
(``ROADMAP.md:15``), and it holds because the audio engine is a separate process entirely.
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vexa_contracts import (
    ApprovalState,
    AssetManifest,
    DeckId,
    GenerationBrief,
    GenerationJob,
    MusicalClock,
    ReadinessState,
    RequestClass,
    RequestConstraints,
    SessionState,
    UserRequest,
)

from .feasibility import FeasibilityFilter
from .planner_client import PlannerClient
from .policy import DecisionPolicy, RulePolicy
from .store import Store

# --- request/response bodies -------------------------------------------------


class CreateSessionBody(BaseModel):
    theme: str = Field(min_length=1)
    session_id: str | None = None
    bpm: float = Field(default=120.0, gt=0.0)
    fallback_asset_id: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    theme: str
    #: Whether the session can start immediately. False means the library is empty, and the
    #: listener is told rather than left waiting (PRODUCT.md:19).
    can_start_now: bool
    ready_asset_count: int


class SubmitRequestBody(BaseModel):
    text: str = Field(min_length=1)
    #: Optional. When omitted the planner classifies the request; the local rules classify it
    #: if the planner is unavailable. A caller that knows the class may always state it.
    request_class: RequestClass | None = None
    constraints: RequestConstraints = Field(default_factory=RequestConstraints)
    target_energy: float = Field(default=0.5, ge=0.0, le=1.0)


class SubmitRequestResponse(BaseModel):
    request_id: str
    #: One of the honest statuses from README.md:21.
    status: str
    state: str
    generation: int
    scheduled_action_id: str | None = None
    matched_asset_id: str | None = None
    queued_generation: bool
    detail: str = ""


class EnqueueJobBody(BaseModel):
    brief: GenerationBrief
    session_id: str | None = None
    request_id: str | None = None
    request_generation: int = Field(default=0, ge=0)
    seed: int | None = None
    #: Lower runs first. A live request outranks background library preparation.
    priority: int = Field(default=100, ge=0, le=1000)


class EnqueueJobResponse(BaseModel):
    job_id: str
    state: str
    priority: int


class SessionStateResponse(BaseModel):
    session_id: str
    theme: str
    generation: int
    bar: int
    beat: int
    bpm: float
    energy: float
    deck_a: str | None
    deck_b: str | None
    fallback_asset_id: str | None
    next_transition: str | None
    next_transition_bar: int | None
    committed_action_id: str | None
    active_request_id: str | None
    job_count: int


# --- app ---------------------------------------------------------------------


def create_app(store: Store | None = None, policy: DecisionPolicy | None = None) -> FastAPI:
    """Build the ASGI app. A store can be injected so tests share one without globals."""
    app = FastAPI(
        title="VEXA//DECK orchestrator",
        version="0.1.0",
        description="Deterministic control plane. Models are bounded; audio is elsewhere.",
    )
    app.state.store = store or Store()
    if policy is not None:
        app.state.store.policy = policy
    # Empty by default, so the orchestrator runs standalone on the local rules. Setting
    # VEXA_PLANNER_URL routes intent through the isolated planner service instead.
    app.state.planner = PlannerClient(base_url=os.environ.get("VEXA_PLANNER_URL", ""))

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "sessions": len(app.state.store.sessions),
            "assets": len(app.state.store.assets),
            "ready_assets": len(app.state.store.ready_assets()),
            "planner": app.state.planner.health(),
        }

    @app.post("/sessions", response_model=CreateSessionResponse)
    def create_session(body: CreateSessionBody) -> CreateSessionResponse:
        session_id = body.session_id or f"sess-{uuid.uuid4().hex[:8]}"
        state = SessionState(
            session_id=session_id,
            theme=body.theme,
            clock=MusicalClock(bpm=body.bpm),
            fallback_asset_id=body.fallback_asset_id,
        )
        app.state.store.create_session(state)
        ready = len(app.state.store.ready_assets())
        return CreateSessionResponse(
            session_id=session_id,
            theme=body.theme,
            can_start_now=ready > 0 or body.fallback_asset_id is not None,
            ready_asset_count=ready,
        )

    @app.post("/sessions/{session_id}/requests", response_model=SubmitRequestResponse)
    def submit_request(session_id: str, body: SubmitRequestBody) -> SubmitRequestResponse:
        record = _record(app, session_id)
        # Resolve intent first. The planner — or the local rules, if it is unavailable — decides
        # what class of request this is and what constraints it carries. Classification is the
        # planner's job; that is the whole reason it exists.
        brief = app.state.planner.plan(
            body.text, context={"session_id": session_id, "theme": record.state.theme}
        )
        request_class = body.request_class or brief.request_class
        constraints = body.constraints if body.constraints else brief.constraints
        target_energy = body.target_energy if body.target_energy != 0.5 else brief.target_energy

        request = UserRequest(
            request_id=f"req-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            generation=record.state.generation + 1,
            text=body.text,
            request_class=request_class,
            constraints=constraints,
            target_energy=target_energy,
        )
        record.state.generation = request.generation
        record.engine.submit(state=record.state, request=request)

        outcome = record.engine.handle(
            state=record.state,
            request=request,
            library=app.state.store.ready_assets(),
        )
        record.log.record(
            kind="request_handled",
            request_id=request.request_id,
            text=request.text,
            state=outcome.status.state.value,
            status=outcome.status.user_facing,
            action_id=outcome.scheduled_action.action_id if outcome.scheduled_action else None,
        )
        return SubmitRequestResponse(
            request_id=request.request_id,
            status=outcome.status.user_facing,
            state=outcome.status.state.value,
            generation=request.generation,
            scheduled_action_id=(
                outcome.scheduled_action.action_id if outcome.scheduled_action else None
            ),
            matched_asset_id=outcome.matched_asset_id,
            queued_generation=outcome.queued_generation,
            detail=outcome.status.detail,
        )

    @app.get("/sessions/{session_id}/state", response_model=SessionStateResponse)
    def session_state(session_id: str) -> SessionStateResponse:
        record = _record(app, session_id)
        pending = record.scheduler.pending
        return SessionStateResponse(
            session_id=session_id,
            theme=record.state.theme,
            generation=record.state.generation,
            bar=record.state.clock.bar,
            beat=record.state.clock.beat,
            bpm=record.state.clock.bpm,
            energy=record.state.energy,
            deck_a=record.state.deck(DeckId.A).asset_id,
            deck_b=record.state.deck(DeckId.B).asset_id,
            fallback_asset_id=record.state.fallback_asset_id,
            next_transition=pending.action_id if pending else None,
            next_transition_bar=pending.commit_bar if pending else None,
            committed_action_id=record.state.committed_action_id,
            active_request_id=record.state.active_request_id,
            job_count=len(record.jobs),
        )

    @app.post("/jobs/generate", response_model=EnqueueJobResponse, status_code=202)
    def enqueue_job(body: EnqueueJobBody) -> EnqueueJobResponse:
        """Enqueue a YuE2 brief. Returns immediately; playback is untouched either way."""
        job = GenerationJob(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            session_id=body.session_id,
            request_id=body.request_id,
            request_generation=body.request_generation,
            brief=body.brief,
            seed=body.seed,
            priority=body.priority,
        )
        app.state.store.add_job(job)
        return EnqueueJobResponse(job_id=job.job_id, state=job.state.value, priority=job.priority)

    @app.post("/assets", response_model=AssetManifest, status_code=201)
    def register_asset(manifest: AssetManifest) -> AssetManifest:
        """Register an analyzed candidate. Ingestion normally calls this, not the GUI."""
        return app.state.store.add_asset(manifest)

    @app.post("/assets/{asset_id}/approve", response_model=AssetManifest)
    def approve_asset(asset_id: str) -> AssetManifest:
        """Promote a verified candidate to the ready library (``ARCHITECTURE.md:102``)."""
        try:
            return app.state.store.approve(asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such asset: {asset_id}") from exc

    @app.get("/assets", response_model=list[AssetManifest])
    def list_assets() -> list[AssetManifest]:
        return list(app.state.store.assets.values())

    @app.get("/jobs", response_model=list[GenerationJob])
    def list_jobs() -> list[GenerationJob]:
        return list(app.state.store.jobs.values())

    @app.get("/sessions/{session_id}/feasibility")
    def feasibility_for(session_id: str) -> dict[str, object]:
        """Explain what is playable right now, and why anything is not.

        The rejection list is the point: ``IDEAS.md:19`` requires reasons to be derived from
        recorded features rather than invented after the fact, and this is where those reasons
        are visible.
        """
        record = _record(app, session_id)
        result = FeasibilityFilter().build(
            state=record.state, library=list(app.state.store.assets.values())
        )
        return {
            "session_id": session_id,
            "candidates": [a.model_dump(mode="json") for a in result.candidates],
            "rejections": [
                {"asset_id": r.asset_id, "reason": r.reason} for r in result.rejections
            ],
        }

    return app


def _record(app: FastAPI, session_id: str):
    record = app.state.store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no such session: {session_id}")
    return record


__all__ = [
    "ApprovalState",
    "create_app",
    "ReadinessState",
    "RulePolicy",
]