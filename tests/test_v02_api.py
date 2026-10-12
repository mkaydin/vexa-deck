import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from test_continuity import make_live
from vexa_orchestrator.api import ModelInstallBody, create_app


def route(app, path):
    return next(r.endpoint for r in app.routes if r.path == path)


def test_installs_require_local_control_and_terms(tmp_path, monkeypatch):
    app = create_app(live=make_live(tmp_path))
    start = route(app, "/models/installations")
    # GET and POST share a path; select POST specifically.
    start = next(
        r.endpoint for r in app.routes if r.path == "/models/installations" and "POST" in r.methods
    )
    remote = SimpleNamespace(client=SimpleNamespace(host="192.0.2.1"))
    local = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    with pytest.raises(HTTPException) as error:
        start(ModelInstallBody(generator="minimax", accept_terms=True), remote)
    assert error.value.status_code == 403
    with pytest.raises(HTTPException) as error:
        start(ModelInstallBody(generator="minimax"), local)
    assert error.value.status_code == 422
    calls = []
    monkeypatch.setattr(
        app.state.model_installs, "start", lambda g, p: calls.append((g, p)) or {"state": "queued"}
    )
    assert (
        start(ModelInstallBody(generator="minimax", preset="balanced", accept_terms=True), local)[
            "state"
        ]
        == "queued"
    )
    assert calls == [("minimax", "balanced")]
    app.state.model_installs.close()


@pytest.mark.asyncio
async def test_sse_stream_reports_cached_output_and_disconnects(tmp_path):
    live = make_live(tmp_path)
    app = create_app(live=live)
    attempts = 0

    async def disconnected():
        nonlocal attempts
        attempts += 1
        return attempts > 2

    response = await route(app, "/audio/telemetry")(SimpleNamespace(is_disconnected=disconnected))
    frames = [item async for item in response.body_iterator]
    assert len(frames) == 2 and response.media_type == "text/event-stream"
    body = json.loads(frames[0][6:])
    assert body["source"] == "post-master" and len(body["levels"]) == 64
    assert body["stale"] and not any(body["levels"])
    app.state.model_installs.close()


def test_depot_only_never_starts_generation(tmp_path):
    live = make_live(tmp_path)
    live.generation_enabled = False
    # Use the production method: make_live replaces queueing for other continuity fixtures.
    from vexa_orchestrator.live import LiveSet

    assert LiveSet._queue_generation(live, "dark", "session") is False
    assert live._generation_stage == "depot_only" and live._generation_thread is None
