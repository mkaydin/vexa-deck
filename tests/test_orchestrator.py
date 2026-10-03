"""Orchestrator behaviour.

Each test here corresponds to a rule the runtime depends on. They are written against behaviour,
not structure: what the filter admits, what the scheduler refuses, and — most importantly — what
happens when a slow model finishes after the listener has already changed their mind.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from vexa_contracts import (
    ActionType,
    ApprovalState,
    AssetManifest,
    AudioProperties,
    BeatGrid,
    Estimate,
    KeyEstimate,
    LoopPoints,
    QualityReport,
    ReadinessState,
    RequestClass,
    RequestConstraints,
    RequestState,
    SectionMarker,
    SessionState,
    SourceType,
    UserRequest,
)
from vexa_orchestrator.feasibility import FeasibilityFilter, FilterConfig, revalidate
from vexa_orchestrator.policy import ChainPolicy, RulePolicy, ShadowPolicy
from vexa_orchestrator.requests import RequestEngine
from vexa_orchestrator.scheduler import Scheduler, SchedulerConfig

PASSING = QualityReport(
    decode_ok=True, duration_ok=True, loudness_ok=True,
    beat_grid_ok=True, loop_boundary_ok=True, audio_quality_ok=True,
)


def asset(
    asset_id: str,
    *,
    bpm: float = 122.0,
    family: str | None = None,
    energy: float | None = None,
    key: str | None = "A minor",
    vocals: bool = False,
    ready: bool = True,
    beat_confidence: float = 0.9,
) -> AssetManifest:
    tags = {}
    if energy is not None:
        tags["energy"] = [Estimate(value=f"{energy:.2f}", confidence=0.9)]
    if vocals:
        tags["vocal"] = [Estimate(value="lead vocal", confidence=0.85)]
    return AssetManifest(
        asset_id=asset_id,
        family_id=family or f"family_{asset_id}",
        source_type=SourceType.YUE2_RENDER,
        content_sha256=hashlib.sha256(asset_id.encode()).hexdigest(),
        audio=AudioProperties(codec="flac", sample_rate_hz=48000, channels=2,
                              duration_s=200.0, integrated_lufs=-9.0, true_peak_dbtp=-1.5),
        beat_grid=BeatGrid(bpm=bpm, grid_version=1, confidence=beat_confidence),
        key=KeyEstimate(value=key, confidence=0.85) if key else None,
        sections=[SectionMarker(section_id="a", kind="intro", start_bar=0, end_bar=16)],
        loops=[LoopPoints(start_bar=0, end_bar=8, beat_aligned=True)],
        tags=tags,
        quality=PASSING,
        approval=ApprovalState.APPROVED if ready else ApprovalState.PENDING,
        readiness=ReadinessState.READY if ready else ReadinessState.UNANALYZED,
    )


def session(**kwargs) -> SessionState:
    defaults = dict(session_id="s1", theme="rain-soaked jazz", generation=1)
    defaults.update(kwargs)
    return SessionState(**defaults)


# --- Feasibility filter ------------------------------------------------------


def test_menu_is_never_empty_even_with_an_empty_library():
    """Without a safe option the set has nowhere to go at a phrase boundary."""
    result = FeasibilityFilter().build(state=session(), library=[])
    assert [a.action_type for a in result.candidates] == [ActionType.CONTINUE_CURRENT]


def test_unready_assets_never_reach_the_menu():
    result = FeasibilityFilter().build(state=session(), library=[asset("a", ready=False)])
    assert result.transitions == []
    assert any("not ready" in r.reason for r in result.rejections)


def test_an_ambiguous_beat_grid_blocks_scheduling():
    """ROADMAP.md:52 — reject ambiguous automatic transitions."""
    result = FeasibilityFilter().build(
        state=session(), library=[asset("a", beat_confidence=0.2)]
    )
    assert result.transitions == []
    assert any("beat grid unreliable" in r.reason for r in result.rejections)


def test_a_large_tempo_jump_is_refused():
    result = FeasibilityFilter().build(state=session(), library=[asset("a", bpm=200.0)])
    assert result.transitions == []
    assert any("exceeds" in r.reason and "tempo" in r.reason for r in result.rejections)


def test_tempo_bound_follows_the_request_not_just_the_engine():
    """A listener asking for gentle changes gets them even when the engine tolerates more."""
    library = [asset("a", bpm=126.0)]
    engine_view = FeasibilityFilter().build(state=session(), library=library)
    gentle = FeasibilityFilter().build(
        state=session(), library=library, constraints=RequestConstraints(max_tempo_ratio=1.01)
    )
    assert engine_view.transitions, "122->126 BPM should be fine for the engine"
    assert not gentle.transitions, "122->126 BPM exceeds a 1% request bound"


def test_recent_family_is_not_replayed():
    result = FeasibilityFilter().build(
        state=session(), library=[asset("a", family="family_x")], recent_family_ids=("family_x",)
    )
    assert result.transitions == []
    assert any("played recently" in r.reason for r in result.rejections)


def test_vocal_collision_is_refused():
    """ARCHITECTURE.md:87 — two prominent vocals must not overlap by default."""
    result = FeasibilityFilter().build(
        state=session(), library=[asset("a", vocals=True)], current_asset=asset("cur", vocals=True)
    )
    assert result.transitions == []
    assert any("two prominent vocals" in r.reason for r in result.rejections)


def test_forbid_vocals_filters_tagged_assets():
    result = FeasibilityFilter().build(
        state=session(), library=[asset("a", vocals=True), asset("b")],
        constraints=RequestConstraints(forbid_vocals=True),
    )
    assert [t.asset_id for t in result.transitions] == ["b"]


def test_mode_clash_is_refused_but_unknown_key_is_permissive():
    clash = FeasibilityFilter().build(
        state=session(), library=[asset("a", key="C major")],
        current_asset=asset("c", key="A minor"),
    )
    assert clash.transitions == []

    unknown = FeasibilityFilter().build(
        state=session(), library=[asset("a", key=None)],
        current_asset=asset("c", key="A minor"),
    )
    assert len(unknown.transitions) == 1, "an unknown key must not stall the set"


def test_an_unmeasured_asset_cannot_satisfy_a_stated_energy_bound():
    """An asset with no energy estimate must not be admitted against an explicit request.

    Otherwise an untagged track silently satisfies "at least 0.99 energy", and the listener gets
    something we cannot claim matches what they asked for.
    """
    result = FeasibilityFilter().build(
        state=session(), library=[asset("untagged")],
        constraints=RequestConstraints(min_energy=0.99),
    )
    assert result.transitions == []
    assert any("unmeasured" in r.reason for r in result.rejections)


def test_an_unmeasured_asset_is_fine_when_no_energy_bound_was_stated():
    """Missing data must not be punished when the request did not ask about it."""
    result = FeasibilityFilter().build(state=session(), library=[asset("untagged")])
    assert len(result.transitions) == 1


def test_energy_bounds_from_the_request_are_honoured():
    library = [asset("calm", energy=0.2), asset("loud", energy=0.9)]
    result = FeasibilityFilter().build(
        state=session(), library=library, constraints=RequestConstraints(max_energy=0.5)
    )
    assert [t.asset_id for t in result.transitions] == ["calm"]


def test_fallback_loop_appears_when_one_is_loaded():
    without = FeasibilityFilter().build(state=session(), library=[])
    assert "play_fallback_loop" not in {a.action_id for a in without.candidates}

    with_fallback = FeasibilityFilter().build(
        state=session(fallback_asset_id="bed"), library=[]
    )
    assert "play_fallback_loop" in {a.action_id for a in with_fallback.candidates}


def test_filter_is_deterministic():
    """Reproducibility is a stated design principle (README.md:22)."""
    library = [asset("a", energy=0.3), asset("b", energy=0.7), asset("c", energy=0.5)]
    first = FeasibilityFilter().build(state=session(), library=library).candidates
    second = FeasibilityFilter().build(state=session(), library=list(reversed(library))).candidates
    assert [a.action_id for a in first] == [a.action_id for a in second]


def test_menu_stays_short_for_the_model_token_budget():
    """LAYA_DATA.md:11 — high option counts degrade Laya's confidence."""
    library = [asset(f"a{i}", energy=0.5) for i in range(40)]
    result = FeasibilityFilter().build(state=session(), library=library)
    assert len(result.candidates) <= FilterConfig().max_candidates + 2


# --- The second gate ---------------------------------------------------------


def test_revalidate_accepts_an_unaltered_offered_action():
    offered = FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
    target = next(a for a in offered if a.action_type is ActionType.TRANSITION)
    assert revalidate(target, offered) is not None


def test_revalidate_refuses_a_mutated_action():
    """A selector cannot smuggle different DSP parameters past the filter."""
    offered = FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
    target = next(a for a in offered if a.action_type is ActionType.TRANSITION)
    tampered = target.model_copy(update={"tempo_ratio": 1.9, "fade_bars": 0})
    assert revalidate(tampered, offered) is None


def test_revalidate_refuses_an_action_that_was_never_offered():
    offered = FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
    from vexa_contracts import FeasibleAction

    invented = FeasibleAction(action_id="nope", action_type=ActionType.TRANSITION,
                              asset_id="zzz", start_at_session_bar=8)
    assert revalidate(invented, offered) is None


# --- Scheduler ---------------------------------------------------------------


def test_transition_commits_on_a_phrase_boundary():
    target = next(
        a for a in FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
        if a.action_type is ActionType.TRANSITION
    )
    outcome = Scheduler().commit(state=session(), action=target, asset=asset("a"))
    assert outcome.committed
    assert outcome.action.asset_sha256 == asset("a").content_sha256


def test_off_phrase_commit_is_refused():
    """Never execute mid-phrase by accident (ARCHITECTURE.md:64)."""
    target = next(
        a for a in FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
        if a.action_type is ActionType.TRANSITION
    )
    off = target.model_copy(update={"start_at_session_bar": target.start_at_session_bar + 1})
    outcome = Scheduler().commit(state=session(), action=off, asset=asset("a"))
    assert not outcome.committed
    assert "phrase boundary" in outcome.reason


def test_commit_in_the_past_is_refused():
    target = next(
        a for a in FeasibilityFilter().build(
            state=session(generation=1), library=[asset("a")]
        ).candidates
        if a.action_type is ActionType.TRANSITION
    )
    late = session(clock={"bar": target.start_at_session_bar})
    late.clock.bar = target.start_at_session_bar
    outcome = Scheduler().commit(state=late, action=target, asset=asset("a"))
    assert not outcome.committed


def test_stale_action_is_dropped_when_a_newer_request_arrives():
    """The rule that stops a slow model from reversing the listener's latest direction."""
    scheduler = Scheduler()
    target = next(
        a for a in FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
        if a.action_type is ActionType.TRANSITION
    )
    scheduler.commit(state=session(generation=1), action=target, asset=asset("a"))

    advanced = session(generation=2)
    advanced.clock.bar = target.start_at_session_bar
    outcome = scheduler.due(state=advanced)

    assert outcome is not None
    assert not outcome.committed
    assert "superseded" in outcome.reason


def test_unloaded_asset_is_dropped_rather_than_played_late():
    scheduler = Scheduler(SchedulerConfig(load_lead_time_s=1.0))
    target = next(
        a for a in FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
        if a.action_type is ActionType.TRANSITION
    )
    scheduler.commit(state=session(), action=target, asset=asset("a"))

    advanced = session()
    advanced.clock.bar = target.start_at_session_bar
    outcome = scheduler.due(state=advanced, now=datetime.now(UTC) + timedelta(seconds=30))

    assert outcome is not None and not outcome.committed
    assert "deadline" in outcome.reason


def test_due_returns_nothing_before_the_commit_bar():
    scheduler = Scheduler()
    target = next(
        a for a in FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
        if a.action_type is ActionType.TRANSITION
    )
    scheduler.commit(state=session(), action=target, asset=asset("a"))
    assert scheduler.due(state=session()) is None


def test_action_naming_one_asset_cannot_be_given_another():
    scheduler = Scheduler()
    target = next(
        a for a in FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates
        if a.action_type is ActionType.TRANSITION
    )
    outcome = scheduler.commit(state=session(), action=target, asset=asset("different"))
    assert not outcome.committed
    assert "was supplied" in outcome.reason


# --- Policies ----------------------------------------------------------------


def test_shadow_mode_records_disagreement_but_lets_rules_control():
    """LAYA_DATA.md:81 — Laya chooses, the rule policy controls audio."""
    library = [asset("a", energy=0.2)]
    menu = FeasibilityFilter().build(state=session(), library=library).candidates

    class AlwaysPicksFirst:
        name = "laya-stub"

        def choose(self, request):
            from vexa_contracts import DecisionResponse

            return DecisionResponse(decision_id=request.decision_id,
                                    action_id=menu[0].action_id, confidence=0.9)

    shadow = ShadowPolicy(inner=AlwaysPicksFirst(), rules=RulePolicy())
    response = shadow.choose(
        __import__("vexa_contracts").DecisionRequest(
            decision_id="d", session_id="s", state=session(), candidates=menu
        )
    )
    assert len(shadow.disagreements) == 1
    assert response.selector == "rules"


def test_chain_falls_through_to_a_safe_action_when_every_policy_declines():
    menu = FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates

    class Declines:
        name = "stub"

        def choose(self, request):
            from vexa_contracts import DecisionResponse

            return DecisionResponse(decision_id=request.decision_id, action_id="menu[0]",
                                    confidence=0.0)

    chain = ChainPolicy(policies=[Declines()], min_confidence=0.9)
    response = chain.choose(
        __import__("vexa_contracts").DecisionRequest(
            decision_id="d", session_id="s", state=session(), candidates=menu
        )
    )
    assert response.action_id == "continue_current"
    assert response.fell_back


def test_chain_survives_a_policy_that_raises():
    """A failing model must never stall the set."""
    menu = FeasibilityFilter().build(state=session(), library=[asset("a")]).candidates

    class Explodes:
        name = "boom"

        def choose(self, request):
            raise RuntimeError("model died")

    chain = ChainPolicy(policies=[Explodes(), RulePolicy()])
    response = chain.choose(
        __import__("vexa_contracts").DecisionRequest(
            decision_id="d", session_id="s", state=session(), candidates=menu
        )
    )
    assert response.selector == "rules"


# --- Request engine ----------------------------------------------------------


def test_generation_dependent_request_with_nothing_available_is_queued():
    """PRODUCT.md:37 — when the asset is absent, keep playing and say so."""
    engine = RequestEngine()
    state = session()
    request = UserRequest(request_id="r1", session_id="s1", generation=1,
                          text="add a Turkish vocal synthwave section",
                          request_class=RequestClass.GENERATION_DEPENDENT)
    engine.submit(state=state, request=request)

    outcome = engine.handle(state=state, request=request, library=[])
    assert outcome.queued_generation
    assert outcome.status.state is RequestState.QUEUED
    assert outcome.status.user_facing == "generating"


def test_generation_dependent_request_prefers_a_ready_asset_when_one_exists():
    """The class describes the request, not the mechanism. A fitting track beats queueing."""
    engine = RequestEngine()
    state = session()
    request = UserRequest(request_id="r1", session_id="s1", generation=1,
                          text="add a Turkish vocal synthwave section",
                          request_class=RequestClass.GENERATION_DEPENDENT)
    engine.submit(state=state, request=request)

    outcome = engine.handle(state=state, request=request, library=[asset("a")])
    assert not outcome.queued_generation
    assert outcome.status.state is RequestState.SCHEDULED


def test_a_newer_request_supersedes_an_older_uncommitted_one():
    """PRODUCT.md:39 — the latest active direction wins."""
    engine = RequestEngine()
    state = session(generation=1)
    first = UserRequest(request_id="r1", session_id="s1", generation=1, text="darker",
                        request_class=RequestClass.GENERATION_DEPENDENT)
    engine.submit(state=state, request=first)
    engine.handle(state=state, request=first, library=[])
    assert engine.status("r1").state is RequestState.QUEUED

    state.generation = 2
    second = UserRequest(request_id="r2", session_id="s1", generation=2, text="brighter",
                         request_class=RequestClass.READY_LIBRARY)
    engine.submit(state=state, request=second)

    assert engine.status("r1").state is RequestState.SUPERSEDED
    assert engine.status("r1").user_facing == "unable to fulfil"


def test_an_applied_request_is_not_rewritten_by_a_later_one():
    engine = RequestEngine()
    state = session(generation=1)
    first = UserRequest(request_id="r1", session_id="s1", generation=1, text="quieter",
                        request_class=RequestClass.IMMEDIATE_CONTROL)
    engine.submit(state=state, request=first)
    engine.handle(state=state, request=first, library=[asset("a")])
    assert engine.status("r1").state is RequestState.APPLIED

    state.generation = 2
    second = UserRequest(request_id="r2", session_id="s1", generation=2, text="louder",
                         request_class=RequestClass.READY_LIBRARY)
    engine.submit(state=state, request=second)
    assert engine.status("r1").state is RequestState.APPLIED, "history is not rewritten"


def test_a_late_generation_never_overturns_a_newer_direction():
    """ARCHITECTURE.md:81 — the headline scenario this whole mechanism exists for."""
    engine = RequestEngine()
    state = session(generation=1)
    request = UserRequest(request_id="r1", session_id="s1", generation=1,
                          text="add a Turkish vocal synthwave section",
                          request_class=RequestClass.GENERATION_DEPENDENT)
    engine.submit(state=state, request=request)
    engine.handle(state=state, request=request, library=[])

    state.generation = 2
    later = UserRequest(request_id="r2", session_id="s1", generation=2, text="make it darker",
                        request_class=RequestClass.READY_LIBRARY)
    engine.submit(state=state, request=later)

    admitted = engine.generation_ready(
        engine.status("r1"), asset=asset("generated"), current_generation=2
    )
    assert not admitted
    assert "superseded" in engine.status("r1").detail


def test_a_current_generation_is_admitted():
    engine = RequestEngine()
    state = session(generation=3)
    request = UserRequest(request_id="r1", session_id="s1", generation=3, text="add a bridge",
                          request_class=RequestClass.GENERATION_DEPENDENT)
    engine.submit(state=state, request=request)
    engine.handle(state=state, request=request, library=[])

    admitted = engine.generation_ready(
        engine.status("r1"), asset=asset("generated"), current_generation=3
    )
    assert admitted
    assert engine.status("r1").state is RequestState.READY
    assert engine.status("r1").user_facing == "ready"


def test_ready_library_request_schedules_a_transition():
    engine = RequestEngine()
    state = session()
    request = UserRequest(request_id="r1", session_id="s1", generation=1, text="make it darker",
                          request_class=RequestClass.READY_LIBRARY)
    engine.submit(state=state, request=request)
    outcome = engine.handle(state=state, request=request, library=[asset("a")])

    assert not outcome.queued_generation
    assert outcome.status.state is RequestState.SCHEDULED
    assert outcome.status.user_facing == "scheduled"


@pytest.mark.parametrize("state_value,expected", [
    (RequestState.APPLIED, "applied now"),
    (RequestState.SCHEDULED, "scheduled"),
    (RequestState.QUEUED, "generating"),
    (RequestState.READY, "ready"),
    (RequestState.FAILED, "unable to fulfil"),
])
def test_listener_always_sees_an_honest_status(state_value, expected):
    from vexa_contracts import RequestStatus

    assert RequestStatus(request_id="r", generation=1, state=state_value).user_facing == expected

def test_a_tempo_band_actually_filters() -> None:
    """``min_bpm``/``max_bpm`` were declared on the contract and never enforced.

    A listener asking for "keep it under 100" got the whole library back, because
    ``violates_constraints`` checked energy and vocals and stopped. Measured against the 127-track
    depot, a 118-126 band excluded nothing.
    """
    library = [asset("slow", bpm=92.0), asset("house", bpm=124.0), asset("peak", bpm=174.0)]

    kept = FeasibilityFilter().build(
        state=session(), library=library, constraints=RequestConstraints(min_bpm=118, max_bpm=126),
    )
    assert [a.asset_id for a in kept.transitions] == ["house"]


def test_tempo_bounds_compose_with_energy_bounds() -> None:
    """Both absolute bounds apply at once, not just the last one checked."""
    library = [
        asset("slow_house", bpm=96.0, energy=0.40),
        asset("right_energy", bpm=124.0, energy=0.50),
        asset("wrong_energy", bpm=124.0, energy=0.90),
    ]

    result = FeasibilityFilter().build(
        state=session(),
        library=library,
        constraints=RequestConstraints(min_bpm=118, max_bpm=126, max_energy=0.6),
    )
    assert [a.asset_id for a in result.transitions] == ["right_energy"]
