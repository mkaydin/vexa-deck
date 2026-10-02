"""Calling the planner from the orchestrator.

The planner runs as its own process so the API key lives in exactly one container and the rest of
the backend has no route to a third party. That only earns anything if the orchestrator actually
talks to it — so this is the client, and its defining property is what happens when the call
fails.

**Every failure falls back to the local rules.** The planner is an enhancement to intent
understanding, never a dependency of playback. A planner that is down, slow, or misconfigured must
produce a slightly worse interpretation of a request, not a stopped set (``README.md:18``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from vexa_planner.planner import RulePlanner, SetBrief


@dataclass(slots=True)
class PlannerClient:
    """HTTP client for the planner service, with a rules fallback baked in.

    Args:
        base_url: the planner service. Unset or empty means "use the rules in-process", which is
            the default so the orchestrator runs standalone.
        timeout_s: deliberately short. The planner sits on the request path but not on the audio
            path, so a slow answer is worse than no answer.
        cache_size: repeated identical requests within a session are common ("quieter", "again"),
            and a repeated intent does not need a second round trip.
    """

    base_url: str = ""
    timeout_s: float = 3.0
    cache_size: int = 64
    _fallback: RulePlanner = field(default_factory=RulePlanner)
    _cache: dict[str, SetBrief] = field(default_factory=dict)

    @property
    def remote_configured(self) -> bool:
        return bool(self.base_url)

    def plan(self, text: str, *, context: dict | None = None) -> SetBrief:
        """Resolve a request's intent, falling back to the local rules on any failure."""
        if not self.remote_configured:
            return self._fallback.plan(text, context=context)

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        try:
            brief = self._remote(text, context)
        except Exception:
            return self._fallback.plan(text, context=context)

        if len(self._cache) >= self.cache_size:
            self._cache.clear()
        self._cache[text] = brief
        return brief

    def _remote(self, text: str, context: dict | None) -> SetBrief:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/plan",
                json={"text": text, "context": context or {}},
            )
            response.raise_for_status()
            payload = response.json()
        return SetBrief(
            theme=payload["theme"],
            request_class=_request_class(payload["request_class"]),
            constraints=_constraints(payload.get("constraints")),
            target_energy=float(payload.get("target_energy", 0.5)),
            explanation=payload.get("explanation", ""),
            source=payload.get("source", "planner"),
        )

    def health(self) -> dict:
        """Whether the planner is reachable, for ``/health``. Never raises."""
        if not self.remote_configured:
            return {"configured": False, "reachable": False, "fallback": "rules"}
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.get(f"{self.base_url.rstrip('/')}/health")
                response.raise_for_status()
                return {"configured": True, "reachable": True, **response.json()}
        except Exception:
            return {"configured": True, "reachable": False, "fallback": "rules"}


def _request_class(value: str):
    from vexa_contracts import RequestClass

    return RequestClass(value)


def _constraints(payload: dict | None):
    from vexa_contracts import RequestConstraints

    if not payload:
        return RequestConstraints()
    allowed = {
        "min_energy", "max_energy", "min_bpm", "max_bpm",
        "forbid_vocals", "allowed_keys", "max_tempo_ratio",
    }
    return RequestConstraints(**{k: v for k, v in payload.items() if k in allowed})