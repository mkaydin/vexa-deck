"""The specialized planner must preserve intent, reject drift and stay off audio's process."""

import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from vexa_orchestrator.ace_planner import MODEL_NAME, parse_sample, plan_in_worker
from vexa_orchestrator.generation import _theme_profile, plan_theme
from vexa_orchestrator.music_plan import arrangement, track_query, validate_metadata


def sample(lyrics=None, caption=None):
    return {
        "caption": caption or "1990s underground boom bap rap, chopped jazz, swung drums, dry MC.",
        "lyrics": lyrics
        or "[Verse 1]\n"
        + "Factory workers face lost jobs beneath rain on concrete streets.\n" * 40,
        "bpm": "92",
        "keyscale": "D minor",
        "timesignature": "4",
        "language": "en",
        "duration": "165",
    }


def test_native_metadata_and_lyrics_are_parsed_without_audio_codes():
    result = parse_sample(
        "<think>\nbpm: 92\ncaption: |\n  Dusty boom bap\n  with dry drums.\n"
        "duration: 165\nkeyscale: D minor\nlanguage: en\ntimesignature: 4\n"
        "</think>\n# Lyric\n[Verse 1]\nMy original words.\n<|im_end|>"
    )
    assert result["caption"] == "Dusty boom bap with dry drums."
    assert result["lyrics"] == "[Verse 1]\nMy original words."
    with pytest.raises(ValueError, match="truncated"):
        parse_sample("<think>bpm: 92")
    with pytest.raises(ValueError, match="audio codes"):
        parse_sample("<think>caption: test</think><|audio_code_12|>")


def test_arrangement_fills_target_and_preserves_eight_bar_mix_regions():
    sections = arrangement(165, 92, vocal=True, rap=True)
    assert sections[0]["bars"] == sections[-1]["bars"] == 8
    assert [row["section"] for row in sections].count("Refrain") == 2
    assert abs(sections[-1]["end_s"] - 165) < 240 / 92
    assert all(left["end_s"] == right["start_s"] for left, right in pairwise(sections))
    assert all(row["bars"] > 0 for row in sections)


def test_theme_story_language_and_tempo_are_in_the_actual_model_query():
    theme = "Türkçe 90s underground rap, yağmurda işinden evine dönen bir işçi, 90 BPM"
    genre, palette, vocal = _theme_profile(theme)
    query = track_query(
        theme,
        genre=genre,
        palette=palette,
        vocal=vocal,
        duration_s=165,
        energy=0.35,
        index=0,
        role="opener",
    )
    assert theme in query and "Vocal language: tr" in query
    assert "90 BPM" in query and "40 original lyric lines" in query
    assert "coherent narrative" in query and "internal rhymes" in query
    assert "Intro 8 bars" in query and "Outro 8 bars" in query


def test_trap_is_not_forced_into_boom_bap_and_song_request_is_vocal():
    genre, palette, vocal = _theme_profile("dark trap rap about isolation")
    assert "trap" in genre and "808" in palette and vocal
    assert _theme_profile("a sad love song with guitar")[2]
    assert not _theme_profile("a sad love song instrumental")[2]


def test_metadata_keeps_explicit_bpm_and_is_not_a_measured_audio_feature():
    result = validate_metadata(sample(), "90s underground rap", "boom bap hip hop")
    assert result["measured"] is False and result["key"] == "D minor"
    with pytest.raises(ValueError, match="requested tempo"):
        validate_metadata(sample(), "90 BPM underground rap", "boom bap hip hop")
    with pytest.raises(ValueError, match="tempo range"):
        validate_metadata({**sample(), "bpm": 140}, "90s rap", "boom bap hip hop")


def test_local_planner_rejects_optional_pop_refinement_without_changing_lyrics(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "vexa_orchestrator.ace_planner.write_local_briefs",
        lambda *a: [
            SimpleNamespace(
                style="Dusty boom bap rap, swung drums.",
                lyrics=sample()["lyrics"],
                origin="local:gemma4:e4b",
            )
        ],
    )
    calls = []

    def generate(self, query, **kwargs):
        calls.append(query)
        return sample(
            caption="Glossy pop anthem with melodic singing." if len(calls) == 1 else None
        )

    monkeypatch.setattr(
        "vexa_orchestrator.ace_planner.TextPlanner", type("Fake", (), {"sample": generate})
    )
    monkeypatch.setattr("vexa_orchestrator.ace_planner.ROOT", tmp_path)
    result = plan_in_worker("90s underground rap about lost factory jobs", 1)
    row = result["tracks"][0]
    assert len(calls) == 1
    assert row["music_plan"]["refinement_accepted"] is False
    assert row["origin"] == f"local:gemma4:e4b+{MODEL_NAME}"
    assert row["music_plan"]["theme"].endswith("lost factory jobs")
    assert row["music_plan"]["measured"] is False
    assert json.loads(Path(result["report"]).read_text())["tracks"][0]["lyrics"] == row["lyrics"]


def test_invalid_local_lyrics_fail_visibly_after_bounded_retries(monkeypatch):
    monkeypatch.setattr(
        "vexa_orchestrator.ace_planner.write_local_briefs",
        lambda *a: [
            SimpleNamespace(
                style="Dusty boom bap rap, swung drums.",
                lyrics="[Verse]\nToo few original lines to fill a whole song.",
                origin="local:gemma4:e4b",
            )
        ],
    )
    monkeypatch.setattr(
        "vexa_orchestrator.ace_planner.TextPlanner",
        lambda: SimpleNamespace(
            sample=lambda *a, **k: sample(
                lyrics="[Verse]\nToo few original lines to fill a whole song."
            )
        ),
    )
    with pytest.raises(ValueError, match=r"after three attempts.*missing original vocal lyrics"):
        plan_in_worker("90s underground rap", 1)


def test_default_uses_isolated_local_plan_and_retains_all_details(monkeypatch):
    monkeypatch.delenv("VEXA_MUSIC_PLANNER", raising=False)
    monkeypatch.delenv("VEXA_LLM_PROVIDER", raising=False)
    calls = []

    def job(action, payload, **kwargs):
        calls.append((action, payload, kwargs))
        return {
            "tracks": [
                {
                    "style": "test",
                    "lyrics": "original",
                    "duration_s": 165,
                    "energy": 0.35,
                    "origin": f"local:{MODEL_NAME}",
                    "music_plan": {"bpm": 92, "measured": False},
                }
            ],
            "report": "plan.json",
        }

    monkeypatch.setattr("vexa_orchestrator.background.run_job", job)
    brief = plan_theme("90s underground rap", count=1)[0]
    assert calls[0][0] == "plan" and calls[0][1]["count"] == 1
    assert brief.music_plan["bpm"] == 92 and brief.lyrics == "original"


def test_no_silent_remote_fallback_when_local_planner_is_unavailable(monkeypatch):
    monkeypatch.setenv("VEXA_MUSIC_PLANNER", "acestep")
    monkeypatch.delenv("VEXA_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(
        "vexa_orchestrator.background.run_job",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model missing")),
    )
    with pytest.raises(RuntimeError, match="model missing"):
        plan_theme("90s underground rap", count=1)


def test_stage_directions_cannot_pad_full_song_line_count():
    from vexa_orchestrator.music_plan import lyric_lines, validate_lyric_theme

    assert lyric_lines("[Verse]\n(Rain sound)\nMy original words.\n[Outro]") == [
        "My original words."
    ]
    with pytest.raises(ValueError, match="narrative topic"):
        validate_lyric_theme(
            "rap about factory workers walking through rainy streets",
            "[Verse]\nWe party with love all night.",
        )
    validate_lyric_theme(
        "rap about factory workers walking through rainy streets",
        "[Verse]\nThe factory gate shuts; rain washes the streets.",
    )
