"""Backend-owned clocks, conditioning constraints and independent lyric recovery."""

import json
import re
from itertools import pairwise

import pytest
from pydantic import ValidationError
from vexa_contracts import CreativeDraft, LyricsDraft, TrackPlan
from vexa_orchestrator.generation_v2 import (
    build_plan,
    compile_brief,
    template_draft,
    validate_creative,
    write_plan,
    writer_settings,
)


@pytest.fixture(autouse=True)
def mocked_provider_identity(monkeypatch):
    monkeypatch.setattr(
        "vexa_orchestrator.generation_v2.preflight",
        lambda settings, base: {"model": settings["model"], "digest": "fixture"},
    )


@pytest.mark.parametrize(
    "theme",
    [
        "dark melodic techno",
        "Memphis phonk instrumental, 88 BPM",
        "90s underground rap in English",
        "warm house vocals in Turkish",
    ],
)
def test_backend_forms_cover_all_ten_tracks_without_timeline_gaps(theme):
    for slot in range(10):
        plan = build_plan(theme, slot, "minimax")
        assert 150 <= plan.duration_s <= 180
        assert plan.sections[0].start_s == 0
        assert plan.sections[-1].end_s == plan.duration_s
        assert all(a.end_s == b.start_s for a, b in pairwise(plan.sections))
        assert TrackPlan.model_validate_json(plan.model_dump_json()) == plan


def test_malformed_writer_fields_cannot_override_backend_clocks():
    plan = build_plan("dark techno instrumental", 0, "minimax")
    draft = template_draft(plan)
    with pytest.raises(ValidationError):
        CreativeDraft.model_validate({**draft.model_dump(), "duration_s": 30})
    draft.section_descriptions["invented"] = "Use drums and bass."
    with pytest.raises(ValueError, match="section IDs"):
        validate_creative(draft, plan)
    draft = template_draft(plan)
    draft.production_notes += " Begin at 1:47."
    with pytest.raises(ValueError, match="timecodes"):
        validate_creative(draft, plan)


def test_native_caption_is_compiled_and_instrumental_has_no_lyric_request():
    plan = build_plan("Memphis phonk instrumental, 88 BPM", 0, "minimax")
    brief = compile_brief(plan, template_draft(plan), None, "fixture")
    caption = brief.music_plan["minimax_caption"]
    ranges = re.findall(r"(\d+:\d+)\-(\d+:\d+)", caption)
    assert ranges[0][0] == "0:00" and ranges[-1][1] == "2:45"
    assert all(a[1] == b[0] for a, b in pairwise(ranges))
    assert brief.lyrics == "" and brief.music_plan["measured"] is False
    with pytest.raises(ValueError, match="cannot include lyrics"):
        compile_brief(plan, template_draft(plan), LyricsDraft(sections={}), "fixture")


def test_lyric_retry_reuses_accepted_creative_draft(monkeypatch, tmp_path):
    plan = build_plan("90s underground rap in English", 0, "minimax")
    assert plan.intent.vocal_mode == "rap"
    draft = template_draft(plan)
    calls = []
    instructions = []
    seeds = []
    lyrics = LyricsDraft(
        sections={
            s.section_id: []
            if s.name in {"Intro", "Outro", "Instrumental Break"}
            else [
                f"{s.name} city stories on the pavement, I keep moving through the rain {i}"
                for i in range(16 if "Verse" in s.name else 4)
            ]
            for s in plan.sections
        }
    )

    def request(_plan, _settings, schema, _instruction):
        instructions.append(_instruction)
        seeds.append(_settings["seed"])
        kind = "creative" if "title" in schema["properties"] else "lyrics"
        calls.append(kind)
        if kind == "creative":
            return draft.model_dump()
        return {"sections": {}} if calls.count("lyrics") == 1 else lyrics.model_dump()

    monkeypatch.setattr("vexa_orchestrator.generation_v2._write_json", request)
    settings = {"mode": "remote", "provider": "ollama", "base": "fixture", "model": "test"}
    cache = tmp_path / "draft.json"
    with pytest.raises(ValueError, match="section IDs"):
        write_plan(plan, settings, cache)
    assert "creative" in json.loads(cache.read_text())
    assert "lyrics" not in json.loads(cache.read_text())
    brief = write_plan(plan, settings, cache)
    assert calls == ["creative", "lyrics", "lyrics"]
    assert seeds == [42, 42, 1051]
    assert not cache.with_suffix(".feedback.json").exists()
    assert "Correct this previous validation failure" in instructions[-1]
    assert brief.lyrics.count("[Chorus]") == 2
    assert brief.music_plan["caption_origin"] == "compiler:v2+llm:test+v2"


def test_invalid_creative_is_not_cached(monkeypatch, tmp_path):
    plan = build_plan("dark techno instrumental", 0, "minimax")
    draft = template_draft(plan)
    draft.production_notes += " Female vocalist sings a melodic chorus."
    monkeypatch.setattr(
        "vexa_orchestrator.generation_v2._write_json", lambda *args: draft.model_dump()
    )
    cache = tmp_path / "draft.json"
    with pytest.raises(ValueError, match="vocal material"):
        write_plan(plan, {"mode": "remote", "base": "fixture", "model": "test"}, cache)
    assert not cache.exists()


def test_repeated_failure_varies_seed_and_settings_change_resets_feedback(monkeypatch, tmp_path):
    plan = build_plan("dark techno instrumental", 3, "minimax")
    draft = template_draft(plan)
    section = plan.sections[0].section_id
    draft.section_descriptions[section] += " Chopped vocal samples add a hook."
    seeds = []

    def request(_plan, settings, *_args):
        seeds.append(settings["seed"])
        return draft.model_dump()

    monkeypatch.setattr("vexa_orchestrator.generation_v2._write_json", request)
    settings = {"mode": "remote", "base": "fixture", "model": "test"}
    cache = tmp_path / "draft.json"
    for attempt in range(3):
        with pytest.raises(ValueError, match=f"section_descriptions.{section}"):
            write_plan(plan, settings, cache)
        assert (
            json.loads(cache.with_suffix(".feedback.json").read_text())["attempts"] == attempt + 1
        )
    assert seeds == [45, 1054, 2063]
    with pytest.raises(ValueError, match="vocal material"):
        write_plan(plan, {**settings, "seed": 100}, cache)
    assert seeds[-1] == 103
    assert json.loads(cache.with_suffix(".feedback.json").read_text())["attempts"] == 1


def test_writer_mode_override_and_cloud_requires_remote(monkeypatch):
    monkeypatch.setenv("VEXA_WRITER_MODE", "local")
    monkeypatch.setenv("VEXA_WRITER_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    assert writer_settings()["model"] == "qwen3.5:9b"
    monkeypatch.setenv("VEXA_WRITER_MODEL", "nemotron-3-super:cloud")
    with pytest.raises(ValueError, match="remote"):
        writer_settings()


def test_duplicate_verses_are_rejected_but_refrains_may_repeat():
    plan = build_plan("90s underground rap in English", 0, "minimax")
    lyrics = LyricsDraft(
        sections={
            s.section_id: []
            if s.name in {"Intro", "Outro", "Instrumental Break"}
            else [
                f"City street story line number {i}" for i in range(16 if "Verse" in s.name else 4)
            ]
            for s in plan.sections
        }
    )
    with pytest.raises(ValueError, match="repeat excessively"):
        compile_brief(plan, template_draft(plan), lyrics, "fixture")
    second = next(s for s in plan.sections if s.name == "Verse 2")
    lyrics.sections[second.section_id] = [
        f"Another turn in the story number {i}" for i in range(16)
    ]
    assert compile_brief(plan, template_draft(plan), lyrics, "fixture").lyrics


def test_translated_topics_do_not_require_english_words():
    from vexa_orchestrator.music_plan import validate_lyric_theme

    theme = "rap in Turkish about Istanbul ferries and unpaid rent"
    validate_lyric_theme(theme, "İstanbul vapurları geçiyor, kiram yine ödenmedi")  # noqa: RUF001
    with pytest.raises(ValueError, match="narrative topic"):
        validate_lyric_theme(theme.replace("Turkish", "English"), "Flowers in a sunny garden")
