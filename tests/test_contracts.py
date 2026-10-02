"""Contract invariants.

These are not schema-shape tests. Each one pins a rule that the runtime relies on and that would
fail *silently* if broken — an asset that becomes schedulable without passing its gates, a late
generation result overrunning a newer request, a decision menu with no safe option in it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from vexa_contracts import (
    ActionType,
    ApprovalState,
    AssetManifest,
    AudioProperties,
    BackendKind,
    BeatGrid,
    CpuPolicy,
    DecisionRequest,
    FeasibleAction,
    GenerationBrief,
    GenerationJob,
    JobState,
    LoopPoints,
    MusicalClock,
    QualityReport,
    ReadinessState,
    RequestClass,
    RequestConstraints,
    RequestState,
    RequestStatus,
    ScheduledAction,
    SectionMarker,
    SessionState,
    SourceType,
    UserRequest,
    registry,
    supports,
)
from vexa_contracts.enums import REQUEST_STATUS_LABELS, USER_FACING_STATUS

PASSING_GATES = dict(
    decode_ok=True,
    duration_ok=True,
    loudness_ok=True,
    beat_grid_ok=True,
    loop_boundary_ok=True,
    audio_quality_ok=True,
)


def make_asset(**overrides) -> AssetManifest:
    base = {
        "asset_id": "asset_42",
        "family_id": "family_8",
        "source_type": SourceType.YUE2_RENDER,
        "content_sha256": "a" * 64,
        "audio": AudioProperties(
            codec="flac",
            sample_rate_hz=48000,
            channels=2,
            duration_s=214.0,
            integrated_lufs=-9.2,
            true_peak_dbtp=-1.4,
        ),
        "beat_grid": BeatGrid(bpm=122.0, grid_version=1, confidence=0.91),
        "sections": [SectionMarker(section_id="s1", kind="outro", start_bar=0, end_bar=32)],
        "approval": ApprovalState.APPROVED,
        "quality": QualityReport(**PASSING_GATES),
        "readiness": ReadinessState.READY,
    }
    base.update(overrides)
    return AssetManifest(**base)


def safe_menu(*, asset_id: str = "a1") -> list[FeasibleAction]:
    return [
        FeasibleAction(action_id="t1", action_type=ActionType.TRANSITION, asset_id=asset_id,
                       start_at_session_bar=8),
        FeasibleAction(action_id="keep", action_type=ActionType.CONTINUE_CURRENT,
                       start_at_session_bar=8),
    ]


# --- The readiness invariant -------------------------------------------------


def test_ready_requires_approval():
    """The core rule: only approved assets enter the ready library (PRODUCT.md:27)."""
    with pytest.raises(ValidationError, match="only approved assets"):
        make_asset(approval=ApprovalState.PENDING)


@pytest.mark.parametrize("gate", sorted(PASSING_GATES))
def test_ready_requires_every_gate(gate: str):
    """One failing gate must be enough to block readiness (ARCHITECTURE.md:46)."""
    failing = dict(PASSING_GATES, **{gate: False})
    with pytest.raises(ValidationError, match="gates failed"):
        make_asset(quality=QualityReport(**failing))


def test_quality_passed_is_derived_not_stored():
    """A caller cannot flip a readiness flag; the verdict comes from the gates."""
    assert QualityReport(**PASSING_GATES).passed
    partial = QualityReport(decode_ok=True)
    assert not partial.passed
    assert set(partial.failures) == {
        "duration", "loudness", "beat_grid", "loop_boundary", "audio_quality"
    }


def test_unready_asset_needs_no_approval():
    """Only READY is gated; an unanalyzed import is legitimately still pending."""
    asset = make_asset(approval=ApprovalState.PENDING, readiness=ReadinessState.UNANALYZED,
                       quality=QualityReport())
    assert asset.readiness is ReadinessState.UNANALYZED


def test_admissible_requires_a_reliable_beat_grid():
    """An unreliable grid must not be scheduled against (ROADMAP.md:52)."""
    assert make_asset().admissible()
    weak = make_asset(beat_grid=BeatGrid(bpm=122.0, grid_version=1, confidence=0.2))
    assert not weak.admissible()


def test_sections_cannot_outlive_the_asset():
    with pytest.raises(ValidationError, match="past the end"):
        make_asset(sections=[
            SectionMarker(section_id="s", kind="outro", start_bar=0, end_bar=9999)
        ])


def test_loops_must_be_ordered():
    with pytest.raises(ValidationError, match="ends at or before it starts"):
        LoopPoints(start_bar=16, end_bar=16)


def test_headroom_is_required_for_a_safe_mix():
    assert make_asset().audio.has_headroom()
    hot = make_asset()
    assert not hot.audio.model_copy(
        update={"audio": hot.audio.model_copy(update={"true_peak_dbtp": -0.1})}
    ).audio.has_headroom()


# --- The supersession rule ---------------------------------------------------


def test_newer_request_supersedes_older():
    older = UserRequest(request_id="r1", session_id="s", generation=1, text="darker",
                        request_class=RequestClass.READY_LIBRARY)
    newer = UserRequest(request_id="r2", session_id="s", generation=2, text="brighter",
                        request_class=RequestClass.READY_LIBRARY)
    assert newer.supersedes(older)
    assert not older.supersedes(newer)


def test_scheduled_action_is_stale_under_a_newer_generation():
    """The rule that stops a slow generation from reversing the listener's direction
    (ARCHITECTURE.md:81)."""
    action = ScheduledAction(action_id="a1", session_id="s", commit_bar=16, generation=3)
    assert not action.is_stale(current_generation=3)
    assert action.is_stale(current_generation=4)


def test_expired_load_deadline_marks_the_action_late():
    from datetime import UTC, datetime, timedelta
    past = datetime.now(UTC) - timedelta(seconds=1)
    action = ScheduledAction(action_id="a1", session_id="s", commit_bar=16, generation=0,
                             load_deadline=past)
    assert action.deadline_expired()
    assert not ScheduledAction(action_id="a2", session_id="s", commit_bar=16,
                               generation=0).deadline_expired()


# --- Musical time ------------------------------------------------------------


def test_beat_bound_tracks_the_time_signature():
    """3/4 time must not accept a beat index that only 4/4 forbids."""
    MusicalClock(bar=1, beat=2, bpm=120.0, beats_per_bar=3)
    with pytest.raises(ValidationError):
        MusicalClock(bar=1, beat=3, bpm=120.0, beats_per_bar=3)


def test_absolute_beat_and_bar_maths():
    clock = MusicalClock(bar=5, beat=2, bpm=120.0, beats_per_bar=4)
    assert clock.absolute_beat == 22
    assert clock.bars_until(9) == 4
    assert clock.bars_until(2) == 0


# --- Request state machine ---------------------------------------------------


def test_request_follows_the_documented_happy_path():
    status = RequestStatus(request_id="r1", generation=1)
    for state in (
        RequestState.INTERPRETED,
        RequestState.MISSING,
        RequestState.QUEUED,
        RequestState.GENERATING,
        RequestState.VALIDATING,
        RequestState.READY,
        RequestState.SCHEDULED,
        RequestState.APPLIED,
    ):
        status.transition(state)
    assert status.state is RequestState.APPLIED
    assert status.is_terminal


def test_illegal_request_transition_is_refused():
    status = RequestStatus(request_id="r1", generation=1)
    with pytest.raises(ValueError, match="cannot move"):
        status.transition(RequestState.APPLIED)


def test_terminal_request_state_is_final():
    status = RequestStatus(request_id="r1", generation=1)
    status.transition(RequestState.INTERPRETED)
    status.transition(RequestState.MISSING)
    status.transition(RequestState.QUEUED)
    status.transition(RequestState.GENERATING)
    status.transition(RequestState.SUPERSEDED)
    assert status.is_terminal
    with pytest.raises(ValueError, match=r"none \(terminal\)"):
        status.transition(RequestState.GENERATING)


def test_every_state_has_an_honest_user_facing_status():
    """README.md:21. A state with no label would leave the UI free to invent one."""
    assert set(USER_FACING_STATUS) == set(RequestState)
    assert set(USER_FACING_STATUS.values()) <= REQUEST_STATUS_LABELS


def test_superseded_request_reports_it_will_never_apply():
    status = RequestStatus(request_id="r1", generation=1, state=RequestState.SUPERSEDED)
    assert status.user_facing == "unable to fulfil"


# --- Job state machine -------------------------------------------------------


def test_job_cannot_become_ready_after_cancellation():
    """A cancelled job must stay cancelled, or it would re-enter the library."""
    job = GenerationJob(job_id="j1", brief=GenerationBrief(style="warm house"))
    job.transition(JobState.RUNNING)
    job.transition(JobState.CANCELED)
    assert job.state is JobState.CANCELED
    with pytest.raises(ValueError):
        job.transition(JobState.READY)


def test_claim_assigns_backend_and_counts_the_attempt():
    job = GenerationJob(job_id="j1", brief=GenerationBrief(style="warm house"))
    job.claim(BackendKind.YUE2_CPP, gpu_index=1)
    assert job.state is JobState.RUNNING
    assert job.backend is BackendKind.YUE2_CPP
    assert job.attempts == 1


def test_claim_rejects_a_job_that_is_not_queued():
    job = GenerationJob(job_id="j1", brief=GenerationBrief(style="warm house"))
    job.transition(JobState.RUNNING)
    with pytest.raises(ValueError, match="not queued"):
        job.claim(BackendKind.TORCH, gpu_index=0)


def test_generation_jobs_default_to_realtime_safe_cpu_policy():
    """Generation must never be allowed to starve the audio callback (ROADMAP.md:53)."""
    job = GenerationJob(job_id="j1", brief=GenerationBrief(style="warm house"))
    assert job.cpu_policy is CpuPolicy.REALTIME_SAFE


def test_unknown_cot_is_rejected():
    with pytest.raises(ValidationError, match="full/melody/off"):
        GenerationBrief(style="x", cot="medium")


# --- The bounded-decision rule ----------------------------------------------


def test_decision_request_must_always_offer_a_safe_action():
    """ARCHITECTURE.md:64 — the menu always contains a safe way to keep playing."""
    state = SessionState(session_id="s", theme="rain")
    unsafe = [
        FeasibleAction(action_id=f"t{i}", action_type=ActionType.TRANSITION, asset_id=f"a{i}",
                       start_at_session_bar=8)
        for i in range(2)
    ]
    with pytest.raises(ValidationError, match="always offer"):
        DecisionRequest(decision_id="d1", session_id="s", state=state, candidates=unsafe)


def test_continue_current_is_accepted_as_the_only_safe_option():
    request = DecisionRequest(
        decision_id="d1", session_id="s",
        state=SessionState(session_id="s", theme="rain"), candidates=safe_menu(),
    )
    assert request.by_id("keep").is_safe
    assert request.ids == {"t1", "keep"}


def test_decision_generation_is_read_from_the_state_snapshot():
    """One source of truth: staleness is judged against the session, not a parallel field."""
    state = SessionState(session_id="s", theme="rain", generation=7)
    request = DecisionRequest(decision_id="d1", session_id="s", state=state, candidates=safe_menu())
    assert request.generation == 7


def test_transition_action_requires_an_asset():
    with pytest.raises(ValidationError, match="requires an asset_id"):
        FeasibleAction(action_id="t1", action_type=ActionType.TRANSITION, start_at_session_bar=8)


# --- Versioning and strictness ----------------------------------------------


def test_every_registered_contract_is_inspectable():
    for name, model in registry().items():
        assert model.model_fields, f"{name} declares no fields"
        assert model.model_json_schema()


def test_contract_version_gate_rejects_a_foreign_major():
    assert supports(1)
    assert not supports(2)


def test_extra_fields_are_refused():
    """A typo across a process boundary is an error, not a silently dropped value."""
    with pytest.raises(ValidationError):
        SessionState(session_id="s", theme="t", not_a_field=1)


def test_user_request_rejects_an_inverted_energy_band():
    with pytest.raises(ValidationError, match="min_energy exceeds max_energy"):
        RequestConstraints(min_energy=0.9, max_energy=0.1)


def test_manifests_survive_a_json_round_trip():
    asset = make_asset()
    assert AssetManifest.model_validate_json(asset.model_dump_json()) == asset

# --- bars and seconds -------------------------------------------------------


@pytest.mark.parametrize("bpm,beats_per_bar,expected", [
    (120.0, 4, 2.00),   # 2 beats/second, 4 to a bar
    (120.0, 3, 1.50),
    (60.0, 4, 4.00),
    (90.0, 4, 2.6666667),
])
def test_seconds_per_bar_is_the_beat_arithmetic(bpm, beats_per_bar, expected):
    """A bar is `beats_per_bar` beats at `bpm`.

    This was 4x too long (240 instead of 60), which silently corrupted every bars-to-seconds
    conversion in the system — crossfade lengths, loop lengths, cue positions.
    """
    clock = MusicalClock(bpm=bpm, beats_per_bar=beats_per_bar)
    assert clock.seconds_per_bar == pytest.approx(expected, abs=1e-4)
    assert clock.seconds_per_bar == pytest.approx(
        beats_per_bar / (bpm / 60.0), abs=1e-6
    )


def test_beat_and_bar_conversions_agree():
    """The engine clock (which has a sample rate) must agree with the session clock."""
    from vexa_audio.queue import Clock

    session_clock = MusicalClock(bpm=128.0, beats_per_bar=7)
    engine_clock = Clock(sample_rate=48000, bpm=128.0, beats_per_bar=7)
    assert engine_clock.samples_per_bar / 48000 == pytest.approx(session_clock.seconds_per_bar)
    assert engine_clock.samples_per_beat == pytest.approx(48000 * 60.0 / 128.0)


def test_two_bars_advance_the_clock_by_two_bars():
    from vexa_audio.queue import Clock

    clock = Clock(sample_rate=44100, bpm=120.0)
    clock.advance(int(clock.samples_per_bar * 2))
    assert clock.bar == 2
