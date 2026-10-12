"""The orchestrator HTTP API.

Mirrors the sketches in ``ARCHITECTURE.md:98-104``. Transport is an implementation choice for the
prototype, but the *payloads* are the versioned contracts, so swapping HTTP for in-process messages
or gRPC later changes no caller.

Nothing here talks to a model directly, and nothing here opens an audio device. Killing this
process does not stop playback — that is Phase B1's exit criterion
(``ROADMAP.md:15``), and it holds because the audio engine is a separate process entirely.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from vexa_audio.equalizer import EqualizerSettings
from vexa_contracts import (
    ApprovalState,
    AssetManifest,
    DeckId,
    GenerationBrief,
    GenerationJob,
    GenerationSettings,
    MusicalClock,
    ReadinessState,
    RequestClass,
    RequestConstraints,
    SessionState,
    UserRequest,
)
from vexa_yue2.backends import BackendUnavailable

from .feasibility import FeasibilityFilter
from .generators import catalog
from .live import LiveSet
from .model_installs import ModelInstalls
from .planner_client import PlannerClient
from .policy import DecisionPolicy, RulePolicy
from .store import Store
from .track_titles import title_for_asset
from .vocal_intent import apply_instrumental_mode
from .writers import WriterProfile
from .writers import catalog as writer_catalog

# --- request/response bodies -------------------------------------------------


class CreateSessionBody(BaseModel):
    theme: str = Field(min_length=1)
    instrumental: bool | None = None
    generation_settings: GenerationSettings | None = None
    generator: str | None = None
    generator_preset: str | None = None
    generation_enabled: bool = True
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
    generation_settings: GenerationSettings | None = None
    generator: str | None = None
    generator_preset: str | None = None
    generation_enabled: bool = True


class EqualizerBody(BaseModel):
    enabled: bool = False
    gains_db: list[float] = Field(default_factory=lambda: [0.0] * 5, min_length=5, max_length=5)
    preamp_db: float = Field(default=0.0, ge=-12.0, le=0.0, allow_inf_nan=False)


class FeedbackBody(BaseModel):
    rating: str = Field(pattern="^(like|dislike)$")
    note: str = Field(default="", max_length=500)


class ResumeGenerationBody(BaseModel):
    retry_failed: bool = False


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


class ModelInstallBody(BaseModel):
    generator: str
    preset: str | None = None
    accept_terms: bool = False


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


def create_app(
    store: Store | None = None, policy: DecisionPolicy | None = None, live: LiveSet | None = None
) -> FastAPI:
    """Build the ASGI app. A store can be injected so tests share one without globals."""
    app = FastAPI(
        title="VEXA//DECK orchestrator",
        version="0.2.0",
        description="Deterministic control plane. Models are bounded; audio is elsewhere.",
    )
    app.state.store = store or Store()
    app.state.live = live
    installs = (
        ModelInstalls(
            live.renderer.out_dir.parent / "model-installs",
            busy=lambda: live._generating or live.engine.running,
        )
        if isinstance(live, LiveSet)
        else None
    )
    app.state.model_installs = installs
    if live is not None:

        @app.on_event("shutdown")
        def close_live() -> None:
            if installs is not None:
                installs.close()
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

    @app.get("/generation/models")
    def generation_models() -> dict[str, object]:
        return {
            "models": catalog(),
            "selected": {
                "generator": live.generator if live else "yue2",
                "preset": live.generator_preset if live else "q8",
            },
        }

    @app.get("/generation/batches")
    def generation_batches() -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        return {"batches": live.generation_batches()}

    @app.get("/generation/writers")
    def generation_writers() -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        return {"writers": writer_catalog(), "selected": live.writer_profile()}

    @app.post("/generation/writer")
    def configure_writer(body: WriterProfile) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        from vexa_yue2.backends import BackendUnavailable

        try:
            return {"selected": live.configure_writer(body)}
        except BackendUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/generation/settings")
    def generation_settings() -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        return live.generation_settings()

    @app.post("/generation/settings")
    def configure_generation_settings(body: GenerationSettings) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        return live.configure_generation_settings(body)

    @app.post("/generation/settings/validate")
    def validate_generation_settings(body: GenerationSettings) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        return {"selected": body.model_dump()}

    @app.post("/generation/preview")
    def preview_generation(body: ThemeBody) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        try:
            return live.preview_generation(
                apply_instrumental_mode(body.theme, body.instrumental),
                request_controls(body),
                generator=body.generator,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/generation/cancel")
    def cancel_generation() -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        return live.cancel_generation()

    @app.get("/generation/batches/{batch_id}/tracks/{slot}")
    def generation_track(batch_id: str, slot: int) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        try:
            return live.track_report(batch_id, slot)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/generation/batches/{batch_id}/tracks/{slot}/{action}")
    def generation_track_action(batch_id: str, slot: int, action: str) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        if action not in {"retry", "replace"}:
            raise HTTPException(status_code=404, detail="unknown track action")
        try:
            return attach_live_session(live.retry_slot(batch_id, slot, replace=action == "replace"))
        except BackendUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/generation/batches/{batch_id}")
    def generation_batch(batch_id: str) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        try:
            return live.generation_batch(batch_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/generation/batches/{batch_id}/resume")
    def resume_generation(batch_id: str, body: ResumeGenerationBody) -> dict:
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        from vexa_yue2.backends import BackendUnavailable

        try:
            live.generation_batch(batch_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            return attach_live_session(
                live.resume_generation(batch_id, retry_failed=body.retry_failed)
            )
        except BackendUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def attach_live_session(status):
        session_id = status.get("session_id")
        if session_id and session_id not in app.state.store.sessions:
            state = live.state or SessionState(
                session_id=session_id, theme=status.get("theme") or "", clock=MusicalClock(bpm=120)
            )
            app.state.store.create_session(state)
        return status

    def select_generator(body) -> None:
        if not body.generation_enabled or (
            body.generator is None and body.generator_preset is None
        ):
            return
        if live is None:
            raise HTTPException(status_code=409, detail="live generation disabled")
        from vexa_yue2.backends import BackendUnavailable

        try:
            live.configure_generator(body.generator, body.generator_preset)
        except BackendUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def request_controls(body) -> GenerationSettings | None:
        if body.instrumental is None:
            return body.generation_settings
        controls = body.generation_settings or GenerationSettings.model_validate(
            live.generation_settings()["selected"] if live else {}
        )
        try:
            return GenerationSettings.model_validate(
                {
                    **controls.model_dump(),
                    "vocal_mode": "instrumental" if body.instrumental else "auto",
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/sessions", response_model=CreateSessionResponse)
    def create_session(body: CreateSessionBody) -> CreateSessionResponse:
        if live is not None and live.state is not None:
            raise HTTPException(status_code=409, detail="a live set is already running")
        select_generator(body)
        theme = apply_instrumental_mode(body.theme, body.instrumental)
        controls = request_controls(body)
        if live:
            try:
                theme = live.preview_generation(theme, controls)["plan"]["intent"]["theme"]
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        session_id = body.session_id or f"sess-{uuid.uuid4().hex[:8]}"
        state = SessionState(
            session_id=session_id,
            theme=theme,
            clock=MusicalClock(bpm=body.bpm),
            fallback_asset_id=body.fallback_asset_id,
        )
        app.state.store.create_session(state)
        ready = len(app.state.store.ready_assets())
        live_status = (
            live.start(
                theme,
                session_id=session_id,
                controls=controls,
                generation_enabled=body.generation_enabled,
            )
            if live
            else None
        )
        if live is not None and live.state is not None:
            app.state.store.sessions[session_id].state = live.state
        return CreateSessionResponse(
            session_id=session_id,
            theme=theme,
            can_start_now=bool(live_status.get("running"))
            if live_status
            else ready > 0 or body.fallback_asset_id is not None,
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
        select_generator(body)
        theme = apply_instrumental_mode(body.theme, body.instrumental)
        try:
            controls = request_controls(body)
            live.preview_generation(theme, controls)
            return live.steer(theme, controls=controls, generation_enabled=body.generation_enabled)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    def installation_service():
        if installs is None:
            raise HTTPException(status_code=409, detail="model installation requires live backend")
        return installs

    def local_install_request(request):
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(
                status_code=403, detail="Model installation is available from the local app only"
            )

    @app.get("/models/catalog")
    def install_catalog():
        return {"models": installation_service().catalog()}

    @app.get("/models/catalog/{generator}/{preset}/preflight")
    def model_preflight(generator: str, preset: str):
        try:
            return installation_service().inspect(generator, preset)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="Cannot inspect download; check network and model access"
            ) from exc

    @app.get("/models/installations")
    def installation_jobs():
        return {"jobs": installation_service().list()}

    @app.post("/models/installations")
    def install_model(body: ModelInstallBody, request: Request):
        local_install_request(request)
        if not body.accept_terms:
            raise HTTPException(status_code=422, detail="Review the model terms before downloading")
        try:
            return installation_service().start(body.generator, body.preset)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/models/installations/{identity}")
    def install_job(identity: str):
        try:
            return installation_service().get(identity)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/models/installations/{identity}/cancel")
    def cancel_install(identity: str, request: Request):
        local_install_request(request)
        try:
            return installation_service().cancel(identity)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/models/installations/{identity}/resume")
    def resume_install(identity: str, request: Request):
        local_install_request(request)
        try:
            return installation_service().resume(identity)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/audio/telemetry")
    async def audio_telemetry(request: Request):
        if live is None:
            raise HTTPException(status_code=409, detail="live audio disabled")

        async def events():
            while not await request.is_disconnected():
                yield "data: " + json.dumps(live.engine.mixer.telemetry.cached) + "\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/live/stop")
    def stop_live() -> dict[str, object]:
        if live is not None:
            return live.stop_playback()
        return {"running": False}

    @app.post("/live/skip")
    def skip_playback():
        if live is None:
            raise HTTPException(status_code=409, detail="live audio disabled")
        try:
            return live.skip_track()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/live/start")
    def start_playback() -> dict[str, object]:
        if live is None:
            raise HTTPException(status_code=409, detail="live audio disabled")
        try:
            return live.start_playback()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
                raise HTTPException(
                    status_code=422, detail="use /live/stop for host playback control"
                )
            snapshot = live.steer(body.text)
            if live.state is not None:
                record.state = live.state
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            record.log.record(kind="theme_queued", request_id=request_id, text=body.text)
            return SubmitRequestResponse(
                request_id=request_id,
                status="Preparing a theme-matched transition",
                state="queued",
                generation=record.state.generation,
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
            tracks.append(
                {
                    "asset_id": asset.manifest.asset_id,
                    "title": title_for_asset(asset),
                    "duration_s": asset.manifest.audio.duration_s,
                    "bpm": asset.manifest.beat_grid.bpm,
                    "theme": brief.get("theme") or "legacy generation",
                    "style": brief.get("style") or asset.manifest.provenance.source_prompt,
                    "file_size": asset.path.stat().st_size,
                    "mixing_capabilities": asset.vibe.get("mixing_capabilities"),
                    "observation_status": (asset.vibe.get("audio_observations") or {}).get(
                        "status"
                    ),
                    "playing": bool(
                        live.current and live.current.manifest.asset_id == asset.manifest.asset_id
                    ),
                    "prepared": bool(
                        live.prepared
                        and live.prepared.asset.manifest.asset_id == asset.manifest.asset_id
                    ),
                }
            )
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
            in_use = {live.current.manifest.asset_id} if live.current else set()
            if live.prepared is not None:
                in_use.add(live.prepared.asset.manifest.asset_id)
            pending = getattr(live, "_skip_pending", None)
            if pending is not None and pending[2] is not None:
                in_use.add(pending[2][0].manifest.asset_id)
            if live.state is not None:
                for deck_id, deck in (
                    (DeckId.A, live.engine.mixer.decks[0]),
                    (DeckId.B, live.engine.mixer.decks[1]),
                ):
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
            "rejections": [{"asset_id": r.asset_id, "reason": r.reason} for r in result.rejections],
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
