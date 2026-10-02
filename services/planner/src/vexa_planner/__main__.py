"""Planner HTTP facade.

Why this exists as its own process: the planner is the **only** component that talks to the
outside world. Isolating it means the rest of the backend has no route to a third-party API at
all, and an API key lives in exactly one container.

It also means the orchestrator can fall back to the local rules without the planner being up.
``GET /health`` reports whether a remote endpoint is configured, and the answer is deliberately
informative rather than a bare 200: an unconfigured planner is a valid, supported state, not a
failure.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .planner import ChainedPlanner, OpenAICompatiblePlanner, RulePlanner


class PlanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    #: Structured session metadata. Never raw audio (README.md:39).
    context: dict[str, Any] = Field(default_factory=dict)


def settings_from_env():
    from .planner import PlannerSettings

    return PlannerSettings(
        base_url=os.environ.get("PLANNER_BASE_URL", ""),
        model=os.environ.get("PLANNER_MODEL", ""),
        api_key=os.environ.get("PLANNER_API_KEY") or None,
        timeout_s=float(os.environ.get("PLANNER_TIMEOUT_S", "8")),
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="VEXA//DECK planner",
        version="0.1.0",
        description="Intent planning. OpenAI-compatible; degrades to local rules.",
    )

    @app.on_event("startup")
    def _build_chain() -> None:
        settings = settings_from_env()
        # The rules are always last in the chain, so this endpoint cannot fail to answer.
        app.state.planner = ChainedPlanner(
            [OpenAICompatiblePlanner(settings), RulePlanner()]
        )
        app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, Any]:
        settings = app.state.settings
        return {
            "status": "ok",
            "remote_configured": settings.configured,
            "remote_is_local": settings.is_local,
            "model": settings.model or None,
            "fallback": "rules",
        }

    @app.post("/plan")
    def plan(body: PlanRequest) -> dict[str, Any]:
        brief = app.state.planner.plan(body.text, context=body.context)
        return brief.as_dict()

    return app


def main() -> int:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("VEXA_HOST", "0.0.0.0"),
        port=int(os.environ.get("VEXA_PORT", "8001")),
        log_level=os.environ.get("VEXA_LOG_LEVEL", "info"),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())