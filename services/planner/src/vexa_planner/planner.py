"""Intent planning: free-form request to a structured brief.

Two rules shape this module, both from the docs:

* The planner receives **structured music metadata**, never raw audio (``README.md:39``).
* A **local rule-based fallback must exist and be usable**. If the LLM is unreachable, slow, or
  simply not configured, the orchestrator keeps working. Basic operation never requires a remote
  service (``README.md:7``, ``ARCHITECTURE.md:24``).

The vendor is deliberately not in this code. The client speaks ``/v1/chat/completions``, so any
OpenAI-compatible endpoint works — OpenAI, a local llama.cpp or vLLM server, LM Studio, or a
gateway. Pointing at a different provider is a config change, which is the whole reason for the
seam.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from vexa_contracts import RequestClass, RequestConstraints


@dataclass(frozen=True, slots=True)
class PlannerSettings:
    """Configuration. No vendor appears here — only a base URL and a model id."""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    timeout_s: float = 8.0
    #: Above this many tokens the planner is too slow to help and the rules take over.
    max_tokens: int = 512
    temperature: float = 0.2

    @property
    def configured(self) -> bool:
        """Whether a remote planner is usable at all.

        A local llama.cpp server needs no key, so a missing key is only disqualifying for a hosted
        endpoint.
        """
        return bool(self.base_url and self.model)

    @property
    def is_local(self) -> bool:
        return any(host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))


@dataclass(slots=True)
class SetBrief:
    """What the planner produces: a structured intent, never a command."""

    theme: str
    request_class: RequestClass = RequestClass.READY_LIBRARY
    constraints: RequestConstraints = field(default_factory=RequestConstraints)
    target_energy: float = 0.5
    #: Shown in the UI so the listener can see what was understood.
    explanation: str = ""
    #: Which planner produced this, for the session log.
    source: str = "rules"

    def as_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "request_class": self.request_class.value,
            "constraints": self.constraints.model_dump(mode="json", exclude_none=True),
            "target_energy": self.target_energy,
            "explanation": self.explanation,
            "source": self.source,
        }


class Planner(Protocol):
    """Turns free text into a :class:`SetBrief`."""

    name: str

    def plan(self, text: str, *, context: dict[str, Any] | None = None) -> SetBrief: ...


# --- The rule-based fallback ------------------------------------------------

#: Deliberately small and legible. This is not trying to be clever; it is trying to be reliable
#: when nothing else is available.
_ENERGY_WORDS: dict[str, float] = {
    "bare": 0.05, "minimal": 0.1, "calm": 0.2, "quiet": 0.15, "chill": 0.25,
    "gentle": 0.25, "downtempo": 0.3, "jazzy": 0.3, "jazz": 0.3, "mellow": 0.25,
    "smooth": 0.35, "warm": 0.4, "steady": 0.45, "mid": 0.5,
    "driving": 0.65, "darker": 0.3, "dark": 0.3, "moody": 0.3,
    "driving house": 0.7, "house": 0.6, "energetic": 0.8, "peak": 0.95,
    "hard": 0.9, "intense": 0.9, "rave": 0.9, "loud": 0.85, "frantic": 0.95,
}

_VOCAL_OFF = ("no vocals", "instrumental", "without vocals", "no singing", "beat only")
_VOCAL_ON = ("vocals", "with vocals", "sung", "singing")
_CONTROL = ("volume", "louder", "quieter", "mute", "lower", "raise", "gain")


def rule_based_plan(text: str, *, context: dict[str, Any] | None = None) -> SetBrief:
    """The always-available planner.

    Classifies the request into one of the three documented classes and extracts what it can from
    the text. When it cannot tell, it says so rather than guessing an action.
    """
    lowered = text.lower().strip()

    if any(word in lowered for word in _CONTROL) and not _mentions_direction(lowered):
        return SetBrief(
            theme=lowered,
            request_class=RequestClass.IMMEDIATE_CONTROL,
            explanation="reads as a transport or gain control rather than a musical direction",
            source="rules",
        )

    energy = _estimate_energy(lowered)
    forbid_vocals = any(phrase in lowered for phrase in _VOCAL_OFF)
    wants_vocals = any(phrase in lowered for phrase in _VOCAL_ON) and not forbid_vocals

    if energy is None and forbid_vocals is False and not wants_vocals:
        return SetBrief(
            theme=lowered,
            request_class=RequestClass.GENERATION_DEPENDENT,
            explanation=(
                "no measurable attribute found; treating as a request for material that is "
                "probably not in the library yet"
            ),
            source="rules",
        )

    constraints = RequestConstraints(forbid_vocals=forbid_vocals)
    found: list[str] = []
    if energy is not None:
        found.append(f"energy {energy:.2f}")
    if forbid_vocals:
        found.append("no vocals")
    if wants_vocals:
        found.append("vocals requested")

    return SetBrief(
        theme=lowered,
        request_class=RequestClass.READY_LIBRARY,
        constraints=constraints,
        target_energy=energy if energy is not None else 0.5,
        explanation="read from: " + ", ".join(found) if found else "read from the request text",
        source="rules",
    )


def _mentions_direction(text: str) -> bool:
    return any(word in text for word in ("darker", "brighter", "faster", "slower", "warmer"))


def _estimate_energy(text: str) -> float | None:
    """Longest matching energy phrase wins, so 'driving house' beats 'house'."""
    best: tuple[int, float] | None = None
    for phrase, value in _ENERGY_WORDS.items():
        if phrase in text and (best is None or len(phrase) > best[0]):
            best = (len(phrase), value)
    return best[1] if best else None


class RulePlanner:
    """:class:`Planner` wrapper around :func:`rule_based_plan`."""

    name = "rules"

    def plan(self, text: str, *, context: dict[str, Any] | None = None) -> SetBrief:
        return rule_based_plan(text, context=context)


# --- The OpenAI-compatible client -------------------------------------------

_SYSTEM = """You turn a listener's request into a structured brief for an AI DJ.

Reply with JSON only, no prose and no code fences, using exactly these keys:
  theme            string  the request restated in a few words
  request_class    string  one of: immediate_control, ready_library, generation_dependent
  target_energy    number  0.0 to 1.0
  forbid_vocals    boolean
  explanation      string  one short sentence naming what you understood

You do not choose tracks or write audio parameters. You describe intent only."""


class OpenAICompatiblePlanner:
    """Talks to any ``/v1/chat/completions`` endpoint.

    Any failure degrades to the rules rather than propagating: an unreachable planner must not be
    able to stop a set (``README.md:18``).
    """

    def __init__(self, settings: PlannerSettings | None = None, fallback: Planner | None = None):
        self.settings = settings or PlannerSettings()
        self.fallback = fallback or RulePlanner()
        self.name = "openai-compatible"

    def plan(self, text: str, *, context: dict[str, Any] | None = None) -> SetBrief:
        if not self.settings.configured:
            return self.fallback.plan(text, context=context)
        try:
            payload = self._request(text, context)
        except Exception:
            brief = self.fallback.plan(text, context=context)
            brief.explanation = f"{brief.explanation} (planner unavailable, used rules)"
            return brief

        parsed = _parse_json_object(payload)
        if parsed is None:
            return self.fallback.plan(text, context=context)

        try:
            return SetBrief(
                theme=str(parsed.get("theme") or text)[:200],
                request_class=RequestClass(parsed.get("request_class", "ready_library")),
                constraints=RequestConstraints(forbid_vocals=bool(parsed.get("forbid_vocals"))),
                target_energy=_clamp(float(parsed.get("target_energy", 0.5))),
                explanation=str(parsed.get("explanation", ""))[:280],
                source=self.name,
            )
        except (ValueError, TypeError):
            return self.fallback.plan(text, context=context)

    def _request(self, text: str, context: dict[str, Any] | None) -> str:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        body = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps({"request": text, "session": context or {}}),
                },
            ],
        }

        with httpx.Client(timeout=self.settings.timeout_s) as client:
            response = client.post(
                f"{self.settings.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a fenced block.

    Returns ``None`` rather than raising: a malformed reply is a fallback case, not an error.
    """
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ChainedPlanner:
    """Tries planners in order, falling through on failure.

    This is what makes the OpenAI-compatible seam usable without a redeploy: keep the chain, change
    its order.
    """

    def __init__(self, planners: list[Planner]):
        self.planners = planners
        self.name = "chain"

    def plan(self, text: str, *, context: dict[str, Any] | None = None) -> SetBrief:
        for planner in self.planners:
            try:
                brief = planner.plan(text, context=context)
            except Exception:
                continue
            if brief is not None:
                return brief
        raise RuntimeError("every planner failed and no fallback was configured")