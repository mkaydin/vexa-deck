"""The orchestrator HTTP API.

Mirrors the sketches in ``ARCHITECTURE.md:98-104``. Transport is an implementation choice for the
prototype, but the *payloads* are the versioned contracts, so swapping HTTP for in-process messages
or gRPC later changes no caller.

Nothing here talks to a model directly, and nothing here opens an audio device. Killing this
process does not stop playback — that is Phase B1's exit criterion
(``ROADMAP.md:15``), and it holds because the audio engine is a separate process entirely.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from vexa_audio.equalizer import EqualizerSettings
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
from .live import LiveSet
from .planner_client import PlannerClient
from .policy import DecisionPolicy, RulePolicy
from .store import Store
from .track_titles import title_for_asset
from .vocal_intent import apply_instrumental_mode

# --- request/response bodies -------------------------------------------------


class CreateSessionBody(BaseModel):
    theme: str = Field(min_length=1)
    instrumental: bool | None = None
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
    generating: bool = False


class SubmitRequestBody(BaseModel):
    text: str = Field(min_length=1)
    #: Optional. When omitted the planner classifies the request; the local rules classify it
    #: if the planner is unavailable. A caller that knows the class may always state it.
    request_class: RequestClass | None = None
    constraints: RequestConstraints = Field(default_factory=RequestConstraints)
    target_energy: float = Field(default=0.5, ge=0.0, le=1.0)


class ThemeBody(BaseModel):
    theme: str = Field(min_length=1)
    instrumental: bool | None = None


class EqualizerBody(BaseModel):
    enabled: bool = False
    gains_db: list[float] = Field(default_factory=lambda: [0.0] * 5, min_length=5, max_length=5)
    preamp_db: float = Field(default=0.0, ge=-12.0, le=0.0, allow_inf_nan=False)


class FeedbackBody(BaseModel):
    rating: str = Field(pattern="^(like|dislike)$")
    note: str = Field(default="", max_length=500)


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


def create_app(store: Store | None = None, policy: DecisionPolicy | None = None,
               live: LiveSet | None = None) -> FastAPI:
    """Build the ASGI app. A store can be injected so tests share one without globals."""
    app = FastAPI(
        title="VEXA//DECK orchestrator",
        version="0.1.0",
        description="Deterministic control plane. Models are bounded; audio is elsewhere.",
    )
    app.state.store = store or Store()
    app.state.live = live
    if live is not None:
        @app.on_event("shutdown")
        def close_live() -> None:
            live.stop()

        for asset in live.depot.assets.values():
            app.state.store.add_asset(asset.manifest)
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
        if live is not None and live.state is not None:
            raise HTTPException(status_code=409, detail="a live set is already running")
        theme = apply_instrumental_mode(body.theme, body.instrumental)
        session_id = body.session_id or f"sess-{uuid.uuid4().hex[:8]}"
        state = SessionState(
            session_id=session_id,
            theme=theme,
            clock=MusicalClock(bpm=body.bpm),
            fallback_asset_id=body.fallback_asset_id,
        )
        app.state.store.create_session(state)
        ready = len(app.state.store.ready_assets())
        live_status = live.start(theme, session_id=session_id) if live else None
        if live is not None and live.state is not None:
            app.state.store.sessions[session_id].state = live.state
        return CreateSessionResponse(
            session_id=session_id,
            theme=theme,
            can_start_now=bool(live_status.get("running")) if live_status else
                ready > 0 or body.fallback_asset_id is not None,
            ready_asset_count=ready,
            generating=bool(live_status.get("generating")) if live_status else False,
        )

    @app.get("/live")
    def live_status() -> dict[str, object]:
        return live.status() if live else {"running": False, "reason": "live audio disabled"}

    @app.post("/sessions/{session_id}/theme")
    def change_theme(session_id: str, body: ThemeBody) -> dict[str, object]:
        _record(app, session_id)
        if live is None or live.status()["session_id"] != session_id:
            raise HTTPException(status_code=409, detail="no active live set for session")
        return live.steer(apply_instrumental_mode(body.theme, body.instrumental))

    @app.get("/audio/equalizer")
    def equalizer_status() -> dict[str, object]:
        if live is None:
            raise HTTPException(status_code=409, detail="live audio disabled")
        return live.engine.mixer.equalizer.snapshot()

    @app.post("/audio/equalizer")
    def configure_equalizer(body: EqualizerBody) -> dict[str, object]:
        if live is None:
            raise HTTPException(status_code=409, detail="live audio disabled")
        try:
            settings = EqualizerSettings(body.enabled, tuple(body.gains_db), body.preamp_db)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return live.configure_equalizer(settings)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail="could not save equalizer settings"
            ) from exc

    @app.post("/live/stop")
    def stop_live() -> dict[str, bool]:
        if live is not None:
            live.stop()
        return {"running": False}

    @app.post("/sessions/{session_id}/feedback")
    def submit_feedback(session_id: str, body: FeedbackBody) -> dict[str, object]:
        _record(app, session_id)
        if live is None or live.state is None or live.state.session_id != session_id:
            raise HTTPException(status_code=409, detail="no active live set for session")
        return live.feedback(body.rating, body.note)

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

        if live is not None and live.status()["session_id"] == session_id:
            if request_class is RequestClass.IMMEDIATE_CONTROL:
                raise HTTPException(status_code=422,
                                    detail="use /live/stop for host playback control")
            snapshot = live.steer(body.text)
            if live.state is not None:
                record.state = live.state
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            record.log.record(kind="theme_queued", request_id=request_id, text=body.text)
            return SubmitRequestResponse(
                request_id=request_id, status="Preparing a theme-matched transition",
                state="queued", generation=record.state.generation,
                queued_generation=bool(snapshot["generating"]),
                detail="The current track keeps playing until a feasible successor is prepared",
            )

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

    @app.get("/depot/tracks")
    def list_depot_tracks() -> list[dict[str, object]]:
        if live is None:
            return []
        tracks = []
        for asset in live.depot.assets.values():
            if not asset.manifest.asset_id.startswith("live-"):
                continue
            if not asset.path.exists():
                continue
            brief_path = asset.path.with_suffix(".brief.json")
            try:
                brief = json.loads(brief_path.read_text()) if brief_path.exists() else {}
            except (OSError, ValueError):
                brief = {}
            if not isinstance(brief, dict):
                brief = {}
            tracks.append({"asset_id": asset.manifest.asset_id,
                           "title": title_for_asset(asset),
                           "duration_s": asset.manifest.audio.duration_s,
                           "bpm": asset.manifest.beat_grid.bpm,
                           "theme": brief.get("theme") or "legacy generation",
                           "style": brief.get("style") or asset.manifest.provenance.source_prompt,
                           "file_size": asset.path.stat().st_size,
                           "playing": bool(live.current and
                                           live.current.manifest.asset_id ==
                                           asset.manifest.asset_id),
                           "prepared": bool(live.prepared and
                                            live.prepared.asset.manifest.asset_id ==
                                            asset.manifest.asset_id)})
        return sorted(tracks, key=lambda row: str(row["asset_id"]), reverse=True)

    @app.get("/depot/tracks/{asset_id}/audio")
    def listen_depot_track(asset_id: str) -> FileResponse:
        asset = live.depot.assets.get(asset_id) if live is not None else None
        if asset is None or not asset_id.startswith("live-") or not asset.path.exists():
            raise HTTPException(status_code=404, detail="generated track not found")
        return FileResponse(asset.path, media_type="audio/wav", filename=asset.path.name)

    @app.delete("/depot/tracks/{asset_id}")
    def delete_depot_track(asset_id: str) -> dict[str, str]:
        if live is None:
            raise HTTPException(status_code=503, detail="host depot unavailable")
        with live._lock:
            asset = live.depot.assets.get(asset_id)
            if asset is None or not asset_id.startswith("live-"):
                raise HTTPException(status_code=404, detail="generated track not found")
            in_use = ({live.current.manifest.asset_id} if live.current else set())
            if live.prepared is not None:
                in_use.add(live.prepared.asset.manifest.asset_id)
            if live.state is not None:
                for deck_id, deck in ((DeckId.A, live.engine.mixer.decks[0]),
                                      (DeckId.B, live.engine.mixer.decks[1])):
                    if deck.playing:
                        loaded_id = live.state.deck(deck_id).asset_id
                        if loaded_id:
                            in_use.add(loaded_id)
            if asset_id in in_use:
                raise HTTPException(status_code=409, detail="track is playing or prepared")
            trash = Path(os.environ.get("VEXA_TRASH_DIR", "var/trash")) / uuid.uuid4().hex
            trash.mkdir(parents=True, exist_ok=True)
            for path in asset.path.parent.glob(f"{asset_id}.*"):
                shutil.move(str(path), str(trash / path.name))
            live.depot.reload()
            app.state.store.assets.pop(asset_id, None)
            return {"asset_id": asset_id, "trash": str(trash)}

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
    live = app.state.live
    if live is not None and live.state is not None and live.state.session_id == session_id:
        record.state = live.state
        for asset in live.depot.assets.values():
            if asset.manifest.asset_id not in app.state.store.assets:
                app.state.store.add_asset(asset.manifest)
    return record


__all__ = [
    "ApprovalState",
    "create_app",
    "ReadinessState",
    "RulePolicy",
]
