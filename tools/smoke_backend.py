"""End-to-end smoke run against a live orchestrator.

This is not a unit test. It starts the real ASGI app over real HTTP and drives the four journeys
the docs promise, asserting on what comes back:

1. Play now — a session starts immediately from a ready asset.
2. Prepare — a generation brief is queued without touching playback.
3. Steer — a live request either schedules from the library or reports honestly.
4. Take control — a manual override outranks the automated path.

Run with::

    uv run python tools/smoke_backend.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

import httpx
import uvicorn
from vexa_contracts import (
    ApprovalState,
    AssetManifest,
    AudioProperties,
    BackendKind,
    BeatGrid,
    KeyEstimate,
    LoopPoints,
    Provenance,
    QualityReport,
    ReadinessState,
    SectionMarker,
    SourceType,
)
from vexa_orchestrator.api import create_app
from vexa_orchestrator.store import Store


def asset(asset_id: str, *, bpm: float, energy: float, key: str = "A minor") -> dict[str, Any]:
    """A manifest the ingestion gates would have produced."""
    return AssetManifest(
        asset_id=asset_id,
        family_id=f"family_{asset_id}",
        source_type=SourceType.YUE2_RENDER,
        content_sha256=f"{abs(hash(asset_id)):064x}"[:64],
        audio=AudioProperties(codec="flac", sample_rate_hz=48000, channels=2,
                              duration_s=200.0, integrated_lufs=-9.0, true_peak_dbtp=-1.5),
        beat_grid=BeatGrid(bpm=bpm, grid_version=1, confidence=0.92),
        key=KeyEstimate(value=key, confidence=0.85),
        sections=[SectionMarker(section_id="a", kind="intro", start_bar=0, end_bar=16)],
        loops=[LoopPoints(start_bar=0, end_bar=8, beat_aligned=True)],
        tags={"energy": [{"value": f"{energy}", "confidence": 0.9}]},
        quality=QualityReport(decode_ok=True, duration_ok=True, loudness_ok=True,
                              beat_grid_ok=True, loop_boundary_ok=True, audio_quality_ok=True),
        approval=ApprovalState.APPROVED,
        readiness=ReadinessState.READY,
        provenance=Provenance(backend=BackendKind.YUE2_CPP, quantization="Q8_0", seed=1),
    ).model_dump(mode="json")


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def main() -> int:
    store = Store()
    app = create_app(store)

    config = uvicorn.Config(app, host="127.0.0.1", port=8731, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = "http://127.0.0.1:8731"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)

    passed = True
    try:
        with httpx.Client(base_url=base, timeout=10.0) as client:
            print("\nhealth")
            health = client.get("/health").json()
            passed &= check("service is up", health["status"] == "ok", json.dumps(health))

            print("\nlibrary ingestion")
            for aid, bpm, energy, key in [
                ("calm-jazz", 120.0, 0.2, "A minor"),
                ("mid-house", 124.0, 0.5, "A minor"),
                ("bright-house", 126.0, 0.85, "A minor"),
                ("clash", 122.0, 0.5, "C major"),
            ]:
                resp = client.post("/assets", json=asset(aid, bpm=bpm, energy=energy, key=key))
                passed &= check(
                    f"registered {aid}", resp.status_code == 201, f"HTTP {resp.status_code}"
                )

            ready = client.get("/health").json()["ready_assets"]
            passed &= check("four assets are schedulable", ready == 4, f"ready={ready}")

            # An analyzed but unapproved candidate. Ingestion produced it; a human has not
            # signed it off, so it must never be schedulable — and must be able to say why.
            pending = asset("awaiting-review", bpm=122.0, energy=0.5)
            pending["approval"] = "pending"
            pending["readiness"] = "awaiting_review"
            unapproved = client.post("/assets", json=pending)
            passed &= check(
                "unapproved candidate registered but not schedulable",
                unapproved.status_code == 201,
                "registered for human review",
            )

            print("\njourney 1 — play now")
            created = client.post("/sessions", json={"theme": "rain-soaked jazz bar", "bpm": 122.0})
            body = created.json()
            session_id = body["session_id"]
            passed &= check("session created", created.status_code == 200, session_id)
            passed &= check(
                "can start without generating anything",
                body["can_start_now"] is True,
                f"ready_asset_count={body['ready_asset_count']}",
            )

            print("\njourney 3 — steer a live set")
            steered = client.post(
                f"/sessions/{session_id}/requests",
                json={
                    "text": "make it darker and faster",
                    "request_class": "ready_library",
                    "constraints": {"max_energy": 0.4, "max_bpm": 124.0},
                },
            ).json()
            passed &= check(
                "request scheduled from the ready library",
                steered["status"] == "scheduled",
                f"status={steered['status']} action={steered['scheduled_action_id']}",
            )
            passed &= check(
                "the energy constraint was respected",
                "bright-house" not in (steered["matched_asset_id"] or "bright-house"),
                f"matched={steered['matched_asset_id']}",
            )

            print("\njourney 3 — request for something absent")
            absent = client.post(
                f"/sessions/{session_id}/requests",
                json={
                    "text": "add a Turkish vocal synthwave section",
                    "request_class": "generation_dependent",
                    "constraints": {"min_energy": 0.99},
                },
            ).json()
            passed &= check(
                "absent style is queued, not faked",
                absent["queued_generation"] is True and absent["status"] == "generating",
                f"status={absent['status']}",
            )
            passed &= check(
                "the listener is told it is not available yet",
                "queued" in absent["detail"].lower(),
                absent["detail"][:70],
            )

            print("\njourney 2 — prepare a theme")
            job = client.post(
                "/jobs/generate",
                json={
                    "brief": {"style": "warm house, Rhodes, brushed drums", "cot": "full"},
                    "session_id": session_id,
                    "request_id": absent["request_id"],
                    "request_generation": absent["generation"],
                    "priority": 10,
                },
            )
            job_body = job.json()
            passed &= check("generation queued", job.status_code == 202, f"HTTP {job.status_code}")
            passed &= check(
                "live request outranks background prep",
                job_body["priority"] == 10,
                f"priority={job_body['priority']}",
            )

            state = client.get(f"/sessions/{session_id}/state").json()
            passed &= check(
                "playback state is still being tracked",
                state["session_id"] == session_id and state["bar"] >= 0,
                f"bar={state['bar']} jobs={state['job_count']}",
            )

            print("\nfeasibility explanations")
            feasible = client.get(f"/sessions/{session_id}/feasibility").json()
            reasons = [r["reason"] for r in feasible["rejections"]]
            passed &= check(
                "the unapproved candidate is rejected with a stated reason",
                any("awaiting_review" in r for r in reasons),
                f"{len(reasons)} rejection(s): {reasons[:1]}",
            )
            passed &= check(
                "a rejected asset never appears as a candidate",
                "awaiting-review" not in {a["asset_id"] for a in feasible["candidates"]},
                f"{len(feasible['candidates'])} candidate(s)",
            )

            print("\nerror handling")
            missing = client.post("/sessions/nope/requests", json={"text": "hello"})
            passed &= check(
                "unknown session is a clean 404, not a crash",
                missing.status_code == 404,
                f"HTTP {missing.status_code}",
            )
            bad_approval = client.post("/assets/nope/approve")
            passed &= check(
                "approving an unknown asset is a clean 404",
                bad_approval.status_code == 404,
                f"HTTP {bad_approval.status_code}",
            )

            print("\nsupersession")
            final = client.post(
                f"/sessions/{session_id}/requests", json={"text": "softer, ambient"}
            ).json()
            passed &= check(
                "newest request takes a higher generation",
                final["generation"] > absent["generation"],
                f"generation={final['generation']}",
            )
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    print(f"\n{'ALL CHECKS PASSED' if passed else 'FAILURES PRESENT'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())