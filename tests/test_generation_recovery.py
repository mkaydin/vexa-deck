"""A failed slot and process restart cannot consume or erase the rest of a set."""

import threading
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from vexa_orchestrator.api import create_app
from vexa_orchestrator.generation import TrackBrief
from vexa_orchestrator.generation_jobs import GenerationJobs
from vexa_orchestrator.generation_runner import GenerationRunner, recovered_asset
from vexa_orchestrator.live import LiveSet
from vexa_yue2.backends import BackendUnavailable


def setup_jobs(tmp_path, count=3):
    jobs = GenerationJobs(tmp_path / "jobs.sqlite3")
    snapshot = {
        "generator": "yue2",
        "preset": "q8",
        "library": str(tmp_path.resolve()),
        "writer": {"mode": "template", "model": "none"},
    }
    batch = jobs.open_batch("dark techno", snapshot, count)
    return jobs, batch, snapshot


def asset(tmp_path, name):
    return SimpleNamespace(path=tmp_path / f"{name}.wav", manifest=SimpleNamespace(asset_id=name))


def run(jobs, batch, tmp_path, planner, renderer, cancel=None, admitted=None):
    runner = GenerationRunner(
        jobs,
        batch,
        tmp_path,
        planner=planner,
        renderer=renderer,
        admitted=admitted or (lambda *args: None),
        event=lambda event: None,
        cancel=cancel or threading.Event(),
    )
    runner.run()


def test_bad_planning_slot_does_not_abort_later_slots(tmp_path):
    jobs, batch, _ = setup_jobs(tmp_path)
    calls = Counter()

    def planner(theme, slot, generator, **kwargs):
        calls[slot] += 1
        if slot == 0:
            raise ValueError("bad creative response")
        return TrackBrief("Instrumental techno", 165, 0.5, "fixture", title="Same title")

    rendered = []
    run(
        jobs,
        batch,
        tmp_path,
        planner,
        lambda *a, **kw: rendered.append(kw) or asset(tmp_path, kw["render_id"]),
    )
    assert calls == {0: 3, 1: 1, 2: 1}
    rows = GenerationJobs(jobs.path).tracks(batch)
    assert [row["state"] for row in rows] == ["rejected", "ready", "ready"]
    assert len(rendered) == 2
    assert rows[1]["brief"]["title"] != rows[2]["brief"]["title"]
    assert jobs.batch(batch)["state"] == "partial"


def test_bad_render_slot_has_own_retry_budget(tmp_path):
    jobs, batch, _ = setup_jobs(tmp_path)
    calls = Counter()

    def planner(theme, slot, generator, **kwargs):
        return TrackBrief(str(slot), 165, 0.5, "fixture")

    def renderer(*args, **kw):
        calls[kw["style"]] += 1
        if kw["style"] == "0":
            raise ValueError("short render")
        return asset(tmp_path, kw["render_id"])

    run(jobs, batch, tmp_path, planner, renderer)
    assert calls == {"0": 3, "1": 1, "2": 1}
    assert jobs.failure_count(batch) == 3


def test_cue_failure_corrects_persisted_outro_before_next_render(tmp_path):
    import copy

    from vexa_orchestrator.generation_runner import cue_retry_brief

    jobs, batch, _ = setup_jobs(tmp_path, count=1)
    plan = {
        "minimax_caption": "Global Metadata\nTrap, 140 BPM\nVocal Details\nInstrumental\n"
        "Arrangement\n0:00-2:15 Groove: dark bass and swung drums.\n"
        "2:15-2:45 Outro: Remove all drums and fade the pads slowly.",
        "sections": [{"section_id": "s01"}, {"section_id": "s02"}],
        "creative_draft": {"section_descriptions": {"s01": "Groove", "s02": "Fade"}},
        "bpm": 140,
    }
    original = copy.deepcopy(plan)
    brief = TrackBrief(
        "Instrumental dark trap", 165, 0.35, "fixture", music_plan=plan, title="Night"
    )
    rendered = []

    def renderer(*args, **kwargs):
        rendered.append(kwargs)
        if len(rendered) == 1:
            raise ValueError("audio evidence rejected: transition_cues_unreliable")
        return asset(tmp_path, kwargs["render_id"])

    run(jobs, batch, tmp_path, lambda *a, **kw: brief, renderer)
    assert len(rendered) == 2
    assert "DJ outro correction" not in rendered[0]["style"]
    assert "DJ outro correction" in rendered[1]["style"]
    assert "Remove all drums" not in rendered[1]["music_plan"]["minimax_caption"]
    assert "2:15-2:45 Outro:" in rendered[1]["music_plan"]["minimax_caption"]
    assert rendered[0]["duration_s"] == rendered[1]["duration_s"] == 165
    assert rendered[0]["lyrics"] == rendered[1]["lyrics"] == ""
    assert plan == original
    saved = TrackBrief(**jobs.tracks(batch)[0]["brief"])
    assert saved.music_plan["cue_retry_correction"]["reason"] == "transition_cues_unreliable"
    assert cue_retry_brief(saved) is saved


def test_noncue_render_failure_does_not_change_the_music_brief(tmp_path):
    jobs, batch, _ = setup_jobs(tmp_path, count=1)
    styles = []

    def renderer(*args, **kwargs):
        styles.append(kwargs["style"])
        raise ValueError("audio evidence rejected: severe_clipping")

    run(
        jobs,
        batch,
        tmp_path,
        lambda *a, **kw: TrackBrief("Dark trap", 165, 0.5, "fixture"),
        renderer,
    )
    assert styles == ["Dark trap"] * 3


def test_retry_old_failed_slot_uses_saved_cue_feedback(tmp_path):
    jobs, batch, _ = setup_jobs(tmp_path, count=1)
    brief = TrackBrief("Instrumental dark trap", 165, 0.35, "fixture")
    jobs.update(batch, 0, brief=asdict(brief), state="rejected", render_attempts=3)
    jobs.attempt(batch, 0, "planning", "accepted")
    jobs.attempt(batch, 0, "rendering", "failed", "transition_cues_unreliable")
    jobs.reset_failed(batch)
    rendered = []
    run(
        jobs,
        batch,
        tmp_path,
        lambda *a, **kw: pytest.fail("saved plan should be reused"),
        lambda *a, **kw: rendered.append(kw) or asset(tmp_path, kw["render_id"]),
    )
    assert len(rendered) == 1
    assert "DJ outro correction" in rendered[0]["style"]


def test_provider_outage_pauses_without_consuming_later_slots(tmp_path):
    jobs, batch, _ = setup_jobs(tmp_path)

    def planner(theme, slot, generator, **kwargs):
        if slot == 1:
            raise BackendUnavailable("writer offline")
        return TrackBrief("techno", 165, 0.5, "fixture")

    run(jobs, batch, tmp_path, planner, lambda *a, **kw: asset(tmp_path, kw["render_id"]))
    assert jobs.batch(batch)["state"] == "unavailable"
    assert [r["state"] for r in jobs.tracks(batch)] == ["ready", "unavailable", "queued"]
    assert jobs.tracks(batch)[2]["plan_attempts"] == 0


def test_restart_reuses_plan_and_published_render(monkeypatch, tmp_path):
    jobs, batch, snapshot = setup_jobs(tmp_path, count=2)
    brief = TrackBrief("techno", 165, 0.5, "fixture", title="Night")
    recovered = asset(tmp_path, "published")
    jobs.update(
        batch,
        0,
        state="rendering",
        brief=asdict(brief),
        render_attempts=1,
        asset_path=str(recovered.path),
    )
    jobs.update(batch, 1, state="plan_ready", brief=asdict(brief), plan_attempts=1)
    # Represents a crash after publication, before the SQLite ready commit.
    monkeypatch.setattr(
        "vexa_orchestrator.generation_runner.recovered_asset", lambda path: recovered
    )
    restarted = GenerationJobs(jobs.path)
    assert restarted.open_batch("dark techno", snapshot, 2) == batch
    renders, admissions = [], []

    def planner(*args, **kwargs):
        pytest.fail("accepted plans must not be rewritten")

    run(
        restarted,
        batch,
        tmp_path,
        planner,
        lambda *a, **kw: renders.append(kw) or asset(tmp_path, kw["render_id"]),
        admitted=lambda a, slot: admissions.append(slot),
    )
    assert len(renders) == 1 and admissions == [0, 1]
    assert restarted.batch(batch)["state"] == "completed"
    changed = {**snapshot, "writer": {"mode": "remote", "model": "different"}}
    assert restarted.open_batch("dark techno", changed, 2) != batch


def test_cancelled_admission_quarantines_manifest(tmp_path):
    jobs, batch, _ = setup_jobs(tmp_path, count=1)
    cancel = threading.Event()

    def renderer(*args, **kw):
        result = asset(tmp_path, kw["render_id"])
        result.path.with_suffix(".json").write_text("{}")
        return result

    def admitted(*args):
        cancel.set()
        raise RuntimeError("new theme arrived")

    run(
        jobs,
        batch,
        tmp_path,
        lambda *a, **kw: TrackBrief("techno", 165, 0.5, "fixture"),
        renderer,
        cancel,
        admitted,
    )
    assert jobs.batch(batch)["state"] == "interrupted"
    assert not list(tmp_path.glob("live-????????????.json"))
    assert len(list(tmp_path.glob("live-*.stale.json"))) == 1


def test_deleted_ready_track_requires_explicit_retry(monkeypatch, tmp_path):
    jobs, batch, _ = setup_jobs(tmp_path, count=1)
    brief = TrackBrief("techno", 165, 0.5, "fixture")
    jobs.update(batch, 0, state="ready", brief=asdict(brief), asset_path=str(tmp_path / "gone.wav"))
    monkeypatch.setattr("vexa_orchestrator.generation_runner.recovered_asset", lambda p: None)
    run(
        jobs,
        batch,
        tmp_path,
        lambda *a, **kw: pytest.fail("must not plan"),
        lambda *a, **kw: pytest.fail("must not silently recreate a deleted track"),
    )
    assert jobs.tracks(batch)[0]["state"] == "rejected"
    jobs.reset_failed(batch)
    assert jobs.tracks(batch)[0]["state"] == "plan_ready"


def test_recovery_requires_matching_published_audio_and_cue_hash(tmp_path):
    import hashlib

    from test_contracts import make_asset
    from vexa_audio.cues import CUE_VERSION, CueMap

    path = tmp_path / "published.wav"
    path.write_bytes(b"admitted audio fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = make_asset(content_sha256=digest)
    path.with_suffix(".json").write_text(manifest.model_dump_json())
    CueMap(digest, CUE_VERSION, 120, 0.9, (), 0.8, (), 0, 150, "beat").save(
        path.with_suffix(".cues.json")
    )
    assert recovered_asset(path).manifest.asset_id == manifest.asset_id
    path.write_bytes(b"replaced audio")
    assert recovered_asset(path) is None


def test_api_exposes_saved_jobs_and_rejects_invalid_resume(tmp_path):
    live = LiveSet(SimpleNamespace(library=tmp_path, assets={}), preview_dir=tmp_path / "previews")
    batch = live._jobs.open_batch(
        "dark techno",
        {
            "generator": "yue2",
            "preset": "q8",
            "library": str(tmp_path.resolve()),
            "writer": {"model": "fixture", "base": "private"},
        },
        2,
    )
    with TestClient(create_app(live=live)) as client:
        body = client.get("/generation/batches").json()
        assert body["batches"][0]["target"] == 2
        assert "private" not in str(body)
        assert client.get(f"/generation/batches/{batch}").status_code == 200
        assert client.post("/generation/batches/missing/resume", json={}).status_code == 404
        live._generating = True
        assert client.post(f"/generation/batches/{batch}/resume", json={}).status_code == 409

        live._generating = False
        invoked = []
        live.resume_generation = lambda bid, **kw: invoked.append((bid, kw)) or {"generating": True}
        response = client.post(f"/generation/batches/{batch}/resume", json={"retry_failed": True})
        assert response.status_code == 200
        assert invoked == [(batch, {"retry_failed": True})]
        del live.resume_generation
        live._jobs.mark_batch(batch, "completed")
        assert client.post(f"/generation/batches/{batch}/resume", json={}).status_code == 409


def test_invalid_writer_configuration_does_not_leave_generation_stuck(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_WRITER_MODE", "invalid")
    live = LiveSet(SimpleNamespace(library=tmp_path, assets={}), preview_dir=tmp_path / "previews")
    assert not live._queue_generation("techno", "test")
    assert live._generation_thread is None
    assert live.status()["generating"] is False
    assert "VEXA_WRITER_MODE" in live.status()["error"]
    live.stop()


@pytest.mark.parametrize("failure", ["malformed_json", "oom", "worker_crash", "corrupt_render"])
def test_injected_failures_preserve_current_and_prepared_decks(tmp_path, failure):
    import json

    import numpy as np
    from test_audio import make_track, render
    from vexa_audio.engine import AudioEngine

    engine = AudioEngine()
    current = engine.preload(make_track(tmp_path / "playing.wav"))
    prepared = engine.preload(make_track(tmp_path / "prepared.wav", freq=330))
    engine.load_and_play(current, 0)
    engine.prepare_deck(prepared, 1)
    render(engine.mixer, blocks=1)
    position = engine.mixer.decks[0].position
    jobs, batch, _ = setup_jobs(tmp_path)

    def planner(theme, slot, generator, **kwargs):
        if slot == 0 and failure == "malformed_json":
            json.loads("{malformed")
        return TrackBrief(str(slot), 165, 0.5, "fixture")

    def renderer(*args, **kwargs):
        if kwargs["style"] == "0" and failure != "malformed_json":
            raise RuntimeError(
                {
                    "oom": "CUDA out of memory",
                    "worker_crash": "worker exited -9",
                    "corrupt_render": "decode_failed: corrupt render",
                }[failure]
            )
        return asset(tmp_path, kwargs["render_id"])

    blocks = []
    GenerationRunner(
        jobs,
        batch,
        tmp_path,
        planner=planner,
        renderer=renderer,
        admitted=lambda *_: None,
        event=lambda _: blocks.append(render(engine.mixer, blocks=1)),
        cancel=threading.Event(),
    ).run()
    rows = jobs.tracks(batch)
    assert [row["state"] for row in rows] == ["rejected", "ready", "ready"]
    assert jobs.failure_count(batch) == 3
    assert engine.mixer.decks[0].track is current
    assert engine.mixer.decks[0].playing
    assert engine.mixer.decks[0].position == position + 512 * len(blocks)
    assert engine.mixer.decks[1].track is prepared
    assert not engine.mixer.decks[1].playing and engine.mixer.decks[1].position == 0
    assert all(np.isfinite(block).all() and np.max(np.abs(block)) > 0 for block in blocks)


@pytest.mark.parametrize(
    "checkpoint", ["generation_planning", "generation_planned", "generation_started", "generated"]
)
def test_restart_at_each_persisted_work_stage_retains_accepted_slots(
    monkeypatch, tmp_path, checkpoint
):
    jobs, batch, _ = setup_jobs(tmp_path)
    planned, rendered, published, admitted = Counter(), Counter(), {}, set()

    def planner(theme, slot, generator, **kwargs):
        planned[slot] += 1
        return TrackBrief(str(slot), 165, 0.5, "fixture")

    def renderer(*args, **kwargs):
        rendered[kwargs["style"]] += 1
        result = asset(tmp_path, kwargs["render_id"])
        published[result.path] = result
        return result

    def crash(event):
        if event["track"] == 2 and event["event"] == checkpoint:
            # BaseException models an abrupt process death, bypassing retry handlers.
            raise SystemExit("injected process death")

    monkeypatch.setattr(
        "vexa_orchestrator.generation_runner.recovered_asset", lambda path: published.get(path)
    )
    with pytest.raises(SystemExit, match="process death"):
        GenerationRunner(
            jobs,
            batch,
            tmp_path,
            planner=planner,
            renderer=renderer,
            admitted=lambda a, slot: admitted.add(a.manifest.asset_id),
            event=crash,
            cancel=threading.Event(),
        ).run()
    assert jobs.tracks(batch)[0]["state"] == "ready"
    restarted = GenerationJobs(jobs.path)
    run(
        restarted,
        batch,
        tmp_path,
        planner,
        renderer,
        admitted=lambda a, slot: admitted.add(a.manifest.asset_id),
    )
    assert restarted.batch(batch)["state"] == "completed"
    assert len(admitted) == 3
    assert planned == {0: 1, 1: 1, 2: 1}
    assert rendered == {"0": 1, "1": 1, "2": 1}


def test_missing_planner_gpu_reports_unavailable_instead_of_consuming_slot_retries(
    monkeypatch, tmp_path
):
    import json

    from vexa_orchestrator import offline_worker

    job, result = tmp_path / "job.json", tmp_path / "result.json"
    job.write_text(json.dumps({"action": "plan_v2", "payload": {}}))
    monkeypatch.setattr("sys.argv", ["worker", str(job), str(result)])
    monkeypatch.setattr(offline_worker.os, "nice", lambda *_: None)
    monkeypatch.setattr(offline_worker.os, "sched_getaffinity", lambda *_: {0})
    monkeypatch.setattr(
        "vexa_orchestrator.generation._gpu_index",
        lambda: (_ for _ in ()).throw(RuntimeError("RTX 5060 Ti was not found")),
    )
    offline_worker.main()
    response = json.loads(result.read_text())
    assert not response["ok"] and response["unavailable"]
    assert "RTX 5060 Ti" in response["error"]
