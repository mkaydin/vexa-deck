"""Planner behaviour.

The invariant worth protecting is the one the docs state twice: **an unreachable LLM must not be
able to stop a set.** Everything here tests that path, plus the classification the rule planner
performs when it is the only thing left.
"""

from __future__ import annotations

import json

import pytest
from vexa_contracts import RequestClass
from vexa_planner.planner import (
    ChainedPlanner,
    OpenAICompatiblePlanner,
    PlannerSettings,
    RulePlanner,
    _parse_json_object,
    rule_based_plan,
)

# --- Rule classification ----------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("lower the volume", RequestClass.IMMEDIATE_CONTROL),
    ("mute the vocals", RequestClass.IMMEDIATE_CONTROL),
    ("make it darker and faster", RequestClass.READY_LIBRARY),
    ("calm ambient piano", RequestClass.READY_LIBRARY),
    ("add a Turkish vocal synthwave section", RequestClass.GENERATION_DEPENDENT),
])
def test_rule_planner_classifies_requests(text, expected):
    assert rule_based_plan(text).request_class is expected


def test_longest_energy_phrase_wins():
    """'driving house' must not be read as plain 'house'."""
    assert rule_based_plan("driving house").target_energy > rule_based_plan("house").target_energy


def test_no_vocals_is_extracted():
    brief = rule_based_plan("warm house, no vocals")
    assert brief.constraints.forbid_vocals is True


def test_no_vocals_beats_a_bare_vocals_mention():
    brief = rule_based_plan("no vocals, unlike the previous track which had vocals")
    assert brief.constraints.forbid_vocals is True


def test_energy_stays_in_range():
    for text in ("bare", "frantic", "peak", "minimal"):
        assert 0.0 <= rule_based_plan(text).target_energy <= 1.0


def test_an_empty_conclusion_is_reported_not_guessed():
    """When nothing measurable is found the planner says so instead of inventing a direction."""
    brief = rule_based_plan("something completely unquantifiable")
    assert brief.request_class is RequestClass.GENERATION_DEPENDENT
    assert "no measurable attribute" in brief.explanation


def test_every_brief_is_serialisable():
    """Session logs persist briefs, so the shape must round-trip."""
    json.dumps(rule_based_plan("warm house, no vocals").as_dict())


# --- Fallback behaviour -----------------------------------------------------


def test_unconfigured_client_falls_back_to_rules():
    planner = OpenAICompatiblePlanner(PlannerSettings(base_url="", model=""))
    brief = planner.plan("warm house")
    assert brief.source == "rules"


def test_an_unreachable_endpoint_falls_back_instead_of_raising():
    """The core guarantee: a dead planner cannot stop a set."""
    planner = OpenAICompatiblePlanner(
        # Port 1 refuses immediately; no network egress and no long timeout.
        PlannerSettings(base_url="http://127.0.0.1:1/v1", model="anything", timeout_s=1.0)
    )
    brief = planner.plan("make it darker and faster")
    assert brief.source == "rules"
    assert "unavailable" in brief.explanation
    assert brief.request_class is RequestClass.READY_LIBRARY


def test_a_malformed_reply_falls_back(monkeypatch):
    planner = OpenAICompatiblePlanner(PlannerSettings(base_url="http://x/v1", model="m"))

    def boom(text, context):
        return "I think you should play something nicer"

    monkeypatch.setattr(planner, "_request", boom)
    assert planner.plan("warm house").source == "rules"


def test_a_reply_naming_an_unknown_class_falls_back(monkeypatch):
    planner = OpenAICompatiblePlanner(PlannerSettings(base_url="http://x/v1", model="m"))
    monkeypatch.setattr(
        planner, "_request", lambda text, context: json.dumps({"request_class": "vibes"})
    )
    assert planner.plan("warm house").source == "rules"


def test_a_well_formed_reply_is_used(monkeypatch):
    planner = OpenAICompatiblePlanner(PlannerSettings(base_url="http://x/v1", model="m"))
    monkeypatch.setattr(
        planner,
        "_request",
        lambda text, context: json.dumps(
            {
                "theme": "dark peak-time techno",
                "request_class": "ready_library",
                "target_energy": 0.88,
                "forbid_vocals": True,
                "explanation": "the listener asked for darker and faster",
            }
        ),
    )
    brief = planner.plan("make it darker and faster")
    assert brief.source == "openai-compatible"
    assert brief.target_energy == pytest.approx(0.88)
    assert brief.constraints.forbid_vocals is True


def test_an_out_of_range_energy_is_clamped_not_trusted():
    planner = OpenAICompatiblePlanner(PlannerSettings(base_url="http://x/v1", model="m"))
    planner._request = lambda text, context: json.dumps({"target_energy": 42.0})
    assert planner.plan("warm house").target_energy == 1.0


# --- JSON tolerance ---------------------------------------------------------


def test_a_fenced_reply_is_accepted():
    """Models fence JSON despite being asked not to; that is not a failure."""
    assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize("bad", [None, "", "not json", "[1, 2, 3]"])
def test_unusable_replies_return_none_rather_than_raising(bad):
    assert _parse_json_object(bad) is None


# --- Chaining ---------------------------------------------------------------


def test_chain_falls_through_to_the_next_planner():
    class Broken:
        name = "broken"

        def plan(self, text, *, context=None):
            raise RuntimeError("boom")

    chain = ChainedPlanner([Broken(), RulePlanner()])
    assert chain.plan("warm house").source == "rules"


def test_chain_without_a_fallback_fails_loudly():
    """Silently returning nothing would be worse than an error here."""
    class Broken:
        name = "broken"

        def plan(self, text, *, context=None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="no fallback"):
        ChainedPlanner([Broken()]).plan("warm house")


# --- Configuration ----------------------------------------------------------


def test_a_local_endpoint_needs_no_api_key():
    """Pointing at llama.cpp or vLLM must not require a credential."""
    assert PlannerSettings(base_url="http://localhost:8080/v1", model="qwen").is_local
    assert PlannerSettings(base_url="http://localhost:8080/v1", model="qwen").configured


def test_no_vendor_is_hardcoded_in_the_settings_shape():
    """Swapping providers must be a config change, so the settings expose no vendor field."""
    from dataclasses import fields

    names = {f.name for f in fields(PlannerSettings)}
    assert names == {
        "base_url", "model", "api_key", "timeout_s", "max_tokens", "temperature"
    }
    assert not any(name in {"provider", "vendor", "engine"} for name in names)


def test_the_client_derives_its_url_from_the_configured_base():
    """Any OpenAI-compatible endpoint, including a local server, works unchanged."""
    from dataclasses import replace

    settings = PlannerSettings(base_url="http://localhost:8080/v1/", model="qwen3-8b")
    assert f"{settings.base_url.rstrip('/')}/chat/completions" == (
        "http://localhost:8080/v1/chat/completions"
    )
    assert replace(settings, model="llama3").model == "llama3"