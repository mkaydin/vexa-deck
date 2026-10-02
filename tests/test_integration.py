"""End-to-end integration across the backend.

These tests cross service boundaries the way the running system does: real audio through the real
analyzer into a real manifest, then that manifest through the real feasibility filter, decision
policy, and scheduler. The point is the seams, which unit tests deliberately avoid.

Every failure here would be one where each component passes its own tests and the system is still
broken.
"""

from __future__ import annotations

import numpy as np
import pytest
from vexa_contracts import (
    ActionType,
    ApprovalState,
    RequestClass,
    RequestState,
    UserRequest,
)
from vexa_laya.adapter import LayaAdapter
from vexa_orchestrator.api import create_app
from vexa_orchestrator.feasibility import FeasibilityFilter, revalidate
from vexa_orchestrator.policy import ChainPolicy, RulePolicy, ShadowPolicy
from vexa_orchestrator.requests import RequestEngine
from vexa_orchestrator.scheduler import Scheduler
from vexa_planner.planner import ChainedPlanner, RulePlanner
from vexa_yue2.gates import analyse

SR = 48000


def musical_track(seconds: float = 40.0, bpm: float = 120.0, bed_dbfs: float = -18.0) -> np.ndarray:
    """A stereo track with a real beat grid, so the analyzer can honestly find one."""
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    bed = 10 ** (bed_dbfs / 20.0) * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)
    click = np.zeros_like(bed)
    period = int(SR * 60.0 / bpm)
    for index in range(0, len(click), period):
        click[index : index + 400] += 0.5
    return np.stack([np.clip(bed + click, -1, 1)] * 2, axis=1)


@pytest.fixture
def analysed(tmp_path):
    """Run the real analyzer over real audio and return the manifest it produced."""
    import soundfile as sf

    path = tmp_path / "track.wav"
    sf.write(str(path), musical_track(), SR, subtype="PCM_16")
    return analyse(path, asset_id="track_1", approval=ApprovalState.APPROVED)


# --- analyzer to library ----------------------------------------------------


def test_audio_becomes_a_schedulable_asset(analysed):
    """The whole ingestion path: audio file to something the scheduler may play."""
    manifest = analysed.manifest
    assert manifest is not None
    assert manifest.quality.passed, analysed.quality.flags
    assert manifest.readiness.value == "ready"
    assert manifest.admissible()
    assert manifest.content_sha256 and len(manifest.content_sha256) == 64
    assert manifest.audio.integrated_lufs is not None


def test_a_broken_render_is_quarantined_and_says_why(tmp_path):
    """The pipeline's value is refusing bad material before a set ever sees it."""
    import soundfile as sf

    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros((SR * 20, 2)), SR, subtype="PCM_16")

    outcome = analyse(path, asset_id="silent")
    assert outcome.manifest is not None
    assert outcome.manifest.readiness.value == "quarantined"
    assert not outcome.manifest.admissible()
    assert any("silent" in flag for flag in outcome.quality.flags)


# --- library to menu --------------------------------------------------------


def test_an_analysed_asset_reaches_the_feasibility_menu(analysed):
    """The seam that matters: analysis output must be consumable by the filter unchanged."""
    session = _session()
    result = FeasibilityFilter().build(state=session, library=[analysed.manifest])

    assert result.transitions, [r.reason for r in result.rejections]
    action = result.transitions[0]
    assert action.asset_id == "track_1"
    assert action.rationale, "the reason must be derived, not blank"


def test_the_filter_refuses_to_schedule_a_quarantined_asset(tmp_path):
    """A gate failure must propagate all the way to the menu, not just sit in a report."""
    import soundfile as sf

    path = tmp_path / "clipped.wav"
    sf.write(str(path), np.clip(musical_track() * 2.0, -1, 1), SR, subtype="PCM_16")

    outcome = analyse(path, asset_id="hot")
    result = FeasibilityFilter().build(state=_session(), library=[outcome.manifest])

    assert not result.transitions
    assert any("not ready" in r.reason or "quality" in r.reason for r in result.rejections)


# --- full request lifecycle -------------------------------------------------


def test_a_live_request_runs_from_planner_text_to_a_committed_transition(analysed):
    """The whole path the docs promise for journey 3, with no model in the loop."""
    planner = ChainedPlanner([RulePlanner()])
    brief = planner.plan("make it darker and faster")
    assert brief.request_class is RequestClass.READY_LIBRARY

    session = _session()
    engine = RequestEngine()
    request = UserRequest(
        request_id="r1",
        session_id=session.session_id,
        generation=1,
        text=brief.theme,
        request_class=brief.request_class,
        constraints=brief.constraints,
        target_energy=brief.target_energy,
    )
    engine.submit(state=session, request=request)
    outcome = engine.handle(state=session, request=request, library=[analysed.manifest])

    assert outcome.status.state is RequestState.SCHEDULED
    chosen = outcome.scheduled_action
    assert chosen.action_type is ActionType.TRANSITION

    scheduler = Scheduler()
    committed = scheduler.commit(
        state=session, action=chosen, asset=analysed.manifest
    )
    assert committed.committed, committed.reason
    assert committed.action.asset_sha256 == analysed.manifest.content_sha256

    deck = scheduler.apply(state=session, action=chosen, asset=analysed.manifest)
    assert session.deck(deck).asset_id == "track_1"
    assert session.deck(deck).playing is True
    assert session.clock.bar == chosen.start_at_session_bar


def test_a_generation_arriving_late_never_enters_the_set(analysed):
    """The headline guarantee: a slow model cannot reverse the listener's latest direction."""
    session = _session(generation=1)
    engine = RequestEngine()

    first = UserRequest(
        request_id="r1", session_id=session.session_id, generation=1,
        text="add a Turkish vocal synthwave section",
        request_class=RequestClass.GENERATION_DEPENDENT,
    )
    engine.submit(state=session, request=first)
    engine.handle(state=session, request=first, library=[analysed.manifest])

    session.generation = 2
    second = UserRequest(
        request_id="r2", session_id=session.session_id, generation=2,
        text="make it darker", request_class=RequestClass.READY_LIBRARY,
    )
    engine.submit(state=session, request=second)

    admitted = engine.generation_ready(
        engine.status("r1"), asset=analysed.manifest, current_generation=2
    )
    assert admitted is False
    assert "superseded" in engine.status("r1").detail


def test_a_committed_transition_is_dropped_when_a_newer_request_lands(analysed):
    """The scheduler half of the same guarantee."""
    session = _session(generation=1)
    menu = FeasibilityFilter().build(state=session, library=[analysed.manifest]).candidates
    chosen = next(a for a in menu if a.action_type is ActionType.TRANSITION)

    scheduler = Scheduler()
    assert scheduler.commit(state=session, action=chosen, asset=analysed.manifest).committed

    session.generation = 2  # the listener changes their mind
    session.clock.bar = chosen.start_at_session_bar

    outcome = scheduler.due(state=session)
    assert outcome is not None
    assert not outcome.committed
    assert "superseded" in outcome.reason


# --- selectors --------------------------------------------------------------


def test_the_laya_adapter_only_ever_names_an_offered_action(analysed):
    """Whatever the model returns, the executed action came off the menu."""
    session = _session()
    menu = FeasibilityFilter().build(state=session, library=[analysed.manifest]).candidates

    from vexa_contracts import DecisionRequest

    request = DecisionRequest(
        decision_id="d", session_id=session.session_id, state=session, candidates=menu
    )
    response = LayaAdapter().choose(request)

    assert response.action_id in request.ids
    if not response.fell_back:
        # A non-fallback answer must survive the deterministic second gate.
        assert revalidate(request.by_id(response.action_id), menu) is not None


def test_shadow_mode_keeps_the_rules_in_control_while_laya_observes(analysed):
    """LAYA_DATA.md:81 — Laya chooses, the rules decide, disagreements accumulate."""
    session = _session()
    menu = FeasibilityFilter().build(state=session, library=[analysed.manifest]).candidates

    from vexa_contracts import DecisionRequest, DecisionResponse

    class AlwaysTransition:
        name = "laya-stub"

        def choose(self, request):
            transition = next(a for a in request.candidates if not a.is_safe)
            return DecisionResponse(decision_id=request.decision_id,
                                    action_id=transition.action_id, confidence=0.9)

    request = DecisionRequest(
        decision_id="d", session_id=session.session_id, state=session, candidates=menu
    )
    shadow = ShadowPolicy(inner=AlwaysTransition(), rules=RulePolicy())
    response = shadow.choose(request)

    assert response.selector == "rules"
    assert len(shadow.disagreements) == 1


def test_a_chain_that_every_selector_declines_holds_position(analysed):
    """The last line of defence before the audio thread."""
    session = _session()
    menu = FeasibilityFilter().build(state=session, library=[analysed.manifest]).candidates

    from vexa_contracts import DecisionRequest, DecisionResponse

    class Silent:
        name = "silent"

        def choose(self, request):
            return DecisionResponse(decision_id=request.decision_id,
                                    action_id="continue_current", confidence=0.0)

    chain = ChainPolicy(policies=[Silent()], min_confidence=0.9)
    response = chain.choose(
        DecisionRequest(decision_id="d", session_id=session.session_id,
                        state=session, candidates=menu)
    )
    assert response.action_id == "continue_current"
    assert response.fell_back


# --- the HTTP surface -------------------------------------------------------


def test_the_api_drives_the_same_path_end_to_end(analysed, tmp_path):
    """The contract a GUI actually calls."""
    from fastapi.testclient import TestClient

    manifest = analysed.manifest.model_dump(mode="json")
    with TestClient(create_app()) as client:
        created = client.post("/sessions", json={"theme": "rain-soaked jazz", "bpm": 120.0})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        # Nothing is schedulable until the library has something in it.
        assert client.get(f"/sessions/{session_id}/state").json()["job_count"] == 0

        assert client.post("/assets", json=manifest).status_code == 201
        state = client.get(f"/sessions/{session_id}/state").json()
        assert state["session_id"] == session_id

        steered = client.post(
            f"/sessions/{session_id}/requests",
            json={"text": "make it darker and faster", "request_class": "ready_library"},
        ).json()
        assert steered["status"] == "scheduled"
        assert steered["matched_asset_id"] == "track_1"

        queued = client.post(
            f"/sessions/{session_id}/requests",
            json={
                "text": "add a Turkish vocal synthwave section",
                "request_class": "generation_dependent",
                "constraints": {"min_energy": 0.99},
            },
        ).json()
        assert queued["queued_generation"] is True
        assert queued["generation"] > steered["generation"]


def test_the_api_refuses_a_session_it_does_not_know():
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as client:
        assert client.get("/sessions/nope/state").status_code == 404
        assert client.post("/sessions/nope/requests", json={"text": "x"}).status_code == 404


def test_the_api_refuses_an_asset_that_fails_its_gates():
    """Registration cannot smuggle an unplayable asset into the ready library."""
    from fastapi.testclient import TestClient
    from pydantic import ValidationError
    from vexa_contracts import AssetManifest

    broken = _valid_manifest_dict()
    broken["approval"] = "pending"

    # A manifest claiming READY without approval is not merely rejected by the API — it does not
    # validate at all. The readiness invariant is structural, not a check someone can forget.
    with pytest.raises(ValidationError):
        AssetManifest.model_validate(broken)

    with TestClient(create_app()) as client:
        assert client.post("/assets", json=broken).status_code == 422


def _valid_manifest_dict() -> dict:
    """A manifest that passes every gate, used as the base for a corrupted copy."""
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "t.wav"
    import soundfile as sf

    sf.write(str(path), musical_track(), SR, subtype="PCM_16")
    outcome = analyse(path, asset_id="t", approval=ApprovalState.APPROVED)
    return outcome.manifest.model_dump(mode="json")


def _session(**kwargs):
    from vexa_contracts import SessionState

    defaults = dict(session_id="s1", theme="rain-soaked jazz", generation=1)
    defaults.update(kwargs)
    return SessionState(**defaults)