"""Music 3 format must preserve genre, lyric sections and instrumental intent end to end."""

import json
from contextlib import nullcontext
from dataclasses import replace

import pytest
from vexa_orchestrator import ace_planner
from vexa_orchestrator.generation import (
    PromptResponseError,
    _ollama_prompts,
    _parse_track_plan,
    _plan_remote_theme,
    plan_theme,
)
from vexa_orchestrator.generators import minimax_caption
from vexa_orchestrator.minimax_prompts import normalize_lyrics, retime_caption, validate_caption


def caption():
    return (
        "Global Metadata\nMemphis phonk, 88 BPM. Nocturnal cassette beat with a "
        "pitched-down piano loop, rounded 808 bass and sparse swung drum-machine groove. "
        "Keep the sample tone warm and narrow with low-pass filtering. Let the mood "
        "develop through slight variations in articulation and layer density.\n"
        "Vocal Details\nInstrumental. The piano carries the lead motif. No vocals.\n"
        "Arrangement\n0:00-0:20 Intro: Establish filtered piano and sparse kick. "
        "0:20-0:55 Groove: Add bass and dry snare with closed swung hats. "
        "0:55-1:30 Development: Change the sample slices "
        "and introduce a quiet instrumental response. 1:30-2:20 Peak and Release: Increase drum "
        "density briefly, then remove the supporting response. 2:20-2:45 Outro: Reduce the layers "
        "to drums, bass and filtered piano, with the rhythm active to the final bar."
    )


def raw(value=None):
    return json.dumps(
        {
            "tracks": [
                {
                    "style": "Memphis phonk, 88 BPM, dusty piano "
                    "chops and sparse swung drums with deep 808 bass.",
                    "lyrics": "",
                    "structured_caption": value or caption(),
                }
            ]
        }
    )


def test_structured_caption_survives_plan_and_native_adapter_exactly():
    brief = _parse_track_plan(
        raw(), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )[0]
    assert minimax_caption(brief.style, brief.lyrics, brief.music_plan) == caption()
    assert brief.music_plan["measured"] is False and brief.music_plan["bpm"] == 88
    assert brief.music_plan["caption_format"] == "minimax-structured-v1"


@pytest.mark.parametrize(
    "bad",
    [
        caption().replace("Vocal Details", "Voice"),
        caption().replace("88 BPM", "140 BPM"),
        caption().replace("No vocals.", "A singer hums the main hook."),
        caption().replace("Memphis phonk", "JDM drift phonk"),
    ],
)
def test_caption_cannot_reverse_explicit_constraints(bad):
    with pytest.raises(ValueError):
        _parse_track_plan(
            raw(bad), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
        )


def test_lyric_sections_are_kept_separate_and_appear_in_arrangement():
    value = (
        caption()
        .replace("Memphis phonk", "Warm soul")
        .replace(
            "Instrumental. The piano carries the lead motif. No vocals.",
            "Intimate female vocal, warm and restrained with a dry close microphone.",
        )
    )
    lyrics = "[Verse 1]\nOriginal words\n[Chorus]\nOriginal refrain"
    with pytest.raises(ValueError, match="lyric section"):
        validate_caption(value, style="Soul, 88 BPM", lyrics=lyrics, genre="soul", vocal=True)
    value += "\nVerse 1: Restrained piano and bass. Chorus: Add soft percussion and strings."
    assert (
        validate_caption(value, style="Soul, 88 BPM", lyrics=lyrics, genre="soul", vocal=True)
        == value
    )
    assert "Original words" not in value


def test_remote_minimax_writer_requests_and_retries_structured_json(monkeypatch):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "ollama")
    calls = []

    def write(*args, **kwargs):
        calls.append((args[2], kwargs))
        return raw(caption().replace("88 BPM", "140 BPM")) if len(calls) == 1 else raw()

    monkeypatch.setattr("vexa_orchestrator.generation._ollama_prompts", write)
    brief = _plan_remote_theme("Memphis phonk instrumental, 88 BPM", count=1, generator="minimax")[
        0
    ]
    assert len(calls) == 2 and calls[0][1] == {"structured": True}
    assert "MiniMax Music 3.0" in calls[0][0][0]["content"]
    assert "structured_caption" in calls[0][0][0]["content"]
    assert brief.music_plan["minimax_caption"] == caption()


def test_local_variant_uses_writer_captions_without_ace_audio_lm(monkeypatch, tmp_path):
    brief = _parse_track_plan(
        raw(), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )[0]
    selected = []
    monkeypatch.setattr(ace_planner, "ROOT", tmp_path)

    def writer(theme, count, **kwargs):
        selected.append(kwargs)
        return [replace(brief, duration_s=165)]

    monkeypatch.setattr(ace_planner, "write_local_briefs", writer)
    monkeypatch.setattr(ace_planner, "TextPlanner", lambda: pytest.fail("ACE LM is not needed"))
    result = ace_planner.plan_in_worker(
        "Memphis phonk instrumental, 88 BPM", 1, generator="minimax"
    )
    assert selected == [{"generator": "minimax"}]
    assert result["tracks"][0]["music_plan"]["minimax_caption"] == caption()


def test_model_selection_is_forwarded_to_offline_planning(monkeypatch):
    monkeypatch.setenv("VEXA_MUSIC_PLANNER", "acestep")
    monkeypatch.delenv("VEXA_LLM_PROVIDER", raising=False)
    calls = []
    brief = _parse_track_plan(
        raw(), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )[0]
    from dataclasses import asdict

    def job(action, payload, **kw):
        calls.append(payload)
        return {"tracks": [asdict(brief)], "report": "plan.json"}

    monkeypatch.setattr("vexa_orchestrator.background.run_job", job)
    assert (
        plan_theme("Memphis phonk instrumental, 88 BPM", count=1, generator="minimax")[
            0
        ].music_plan["minimax_caption"]
        == caption()
    )
    assert calls[0]["generator"] == "minimax"


def test_writer_heading_and_clock_formatting_is_normalized():
    value = caption().replace("Global Metadata\n", "## Global Metadata: ")
    value = value.replace("Vocal Details\n", "**Vocal Details**: ")
    value = value.replace("Arrangement\n", "Arrangement: ")
    value = value.replace("0:00-0:20", "M:00-M:0:20")
    value = value.replace("0:20-0:55", "M:0:20-M:0:55")
    assert validate_caption(
        value, style="88 BPM", lyrics="", genre="Memphis phonk", vocal=False, duration_s=165
    ) == caption()


def test_valid_short_timeline_is_retimed_and_recorded_without_changing_music():
    short, changed = retime_caption(caption(), 120)
    assert changed and "1:42-2:00 Outro" in short
    brief = _parse_track_plan(
        raw(short), "Memphis phonk instrumental, 88 BPM", 1, "gemma",
        generator="minimax", track_index=1,
    )[0]
    assert brief.duration_s == 166.5
    assert brief.music_plan["caption_retimed"] is True
    assert brief.music_plan["original_minimax_caption"] == short
    assert "2:46 Outro" in brief.music_plan["minimax_caption"]
    assert "piano" in brief.music_plan["minimax_caption"]


def test_memphis_style_can_use_drift_as_a_verb():
    payload = json.loads(
        raw(caption().replace("Let the mood", "Let the piano motifs drift. The mood"))
    )
    payload["tracks"][0]["style"] += " Piano motifs drift over the bass."
    assert _parse_track_plan(
        json.dumps(payload), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )


def test_lyric_tag_normalization_keeps_actual_words_intact():
    assert normalize_lyrics("(Verse 1) The room stays quiet\n(Refrain) Stay (with me)") == (
        "[Verse 1]\nThe room stays quiet\n[Refrain]\nStay (with me)"
    )


def test_duplicate_interval_notes_merge_without_losing_instrument_changes():
    value = caption().replace(
        "0:55-1:30 Development:",
        "0:20-0:55 Groove: Add evolving noise sweeps. 0:55-1:30 Development:",
    )
    brief = _parse_track_plan(
        raw(value), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )[0]
    result = minimax_caption(brief.style, brief.lyrics, brief.music_plan)
    assert result.count("0:20-0:55 Groove:") == 1
    assert "Add bass and dry snare" in result and "Add evolving noise sweeps." in result
    assert brief.music_plan["caption_retimed"]
    assert brief.music_plan["original_minimax_caption"] == value
    assert retime_caption(result, 165) == (result, False)


@pytest.mark.parametrize("range_", ["0:35-0:55", "0:10-0:55", "0:55-0:20", "0:20-0:20"])
def test_bad_writer_clocks_rebuild_into_contiguous_full_length_sections(range_):
    value = caption().replace("0:20-0:55", range_)
    brief = _parse_track_plan(
        raw(value), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )[0]
    result = minimax_caption(brief.style, brief.lyrics, brief.music_plan)
    assert brief.music_plan["caption_retimed"]
    assert result.index("Intro:") < result.index("Groove:") < result.index("Development:")
    assert "2:45 Outro:" in result
    assert "Add bass and dry snare with closed swung hats." in result


def test_same_time_different_sections_remain_distinct():
    value = caption().replace(
        "0:55-1:30 Development:",
        "0:20-0:55 Interlude: Keep the piano motif. 0:55-1:30 Development:",
    )
    fixed, changed = retime_caption(value, 165)
    assert changed
    assert "Groove:" in fixed and "Interlude:" in fixed
    validate_caption(fixed, style="88 BPM", lyrics="", genre="Memphis phonk",
                     vocal=False, duration_s=165)


def test_instrumental_voice_is_not_a_singer_but_positive_vocals_still_fail():
    value = caption().replace("The piano carries", "The lead instrumental voice carries")
    assert _parse_track_plan(
        raw(value), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )
    with pytest.raises(ValueError, match="vocal delivery"):
        _parse_track_plan(
            raw(value + "\nA female singer hums over the instrumental voice."),
            "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax",
        )


def test_timing_repair_cannot_hide_vocal_conflicts_or_missing_sections():
    value = caption().replace("0:20-0:55", "0:35-0:55")
    with pytest.raises(ValueError, match="vocal delivery"):
        _parse_track_plan(
            raw(value.replace("No vocals.", "A singer hums.")),
            "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax",
        )
    invalid = value.replace("Vocal Details", "Voice")
    with pytest.raises(ValueError, match="Vocal Details"):
        _parse_track_plan(
            raw(invalid), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
        )


@pytest.mark.parametrize("negative", [
    "There is no vocal delivery, backing, or effects applied to vocals.",
    "No vocals, singing, or humming.",
    "There are no vocal elements or vocal effects.",
])
def test_instrumental_negative_lists_do_not_create_false_vocal_requests(negative):
    value = caption().replace("No vocals.", negative)
    assert _parse_track_plan(
        raw(value), "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax"
    )
    with pytest.raises(ValueError, match="vocal delivery"):
        _parse_track_plan(
            raw(value + " But a singer performs the chorus."),
            "Memphis phonk instrumental, 88 BPM", 1, "gemma", generator="minimax",
        )


def test_synthesizer_melodic_voice_metaphor_does_not_hide_a_human_voice():
    from vexa_orchestrator.vocal_intent import has_vocal_delivery, instrumental_details

    text = (
        "A detuned synthesizer, which acts as the primary melodic voice, weaves mournful phrases."
    )
    assert not has_vocal_delivery(text)
    assert instrumental_details(text) == (text, [])
    assert has_vocal_delivery(text + " A female voice sings the hook.")
    assert has_vocal_delivery("A synthesizer supports a breathy female voice.")
    assert has_vocal_delivery("A synthesizer with vocal samples acts as the primary melodic voice.")
    assert not has_vocal_delivery(
        "The lead instrumental motif is a heavily modulated, detuned sawtooth wave, "
        "functioning as the primary melodic voice."
    )


def test_split_second_digits_and_stray_colons_are_repaired_before_validation():
    value = caption().replace("0:00-0:20", "M:0:00-0:2:0")
    value = value.replace("0:20-0:55", "M:0:2:0-0:5:5:")
    value = value.replace("0:55-1:30", "M:0:5:5-1:3:0")
    fixed, changed = retime_caption(value, 165)
    assert changed
    assert "0:20-0:55 Groove:" in fixed
    assert "0:55-1:30 Development:" in fixed
    validate_caption(fixed, style="88 BPM", lyrics="", genre="Memphis phonk",
                     vocal=False, duration_s=165)


@pytest.mark.parametrize("oversize", [False, True])
def test_failed_writer_stream_is_bounded_and_retains_actual_response(monkeypatch, oversize):
    text = "x" * 24001 if oversize else '{"tracks":['
    requests = []

    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"message": {"content": text}})
            yield json.dumps({"done": True, "done_reason": "length"})

    def stream(*args, **kwargs):
        requests.append(kwargs["json"])
        return nullcontext(Response())

    monkeypatch.setattr("vexa_orchestrator.generation.httpx.stream", stream)
    with pytest.raises(PromptResponseError) as failure:
        _ollama_prompts("http://localhost:1", "gemma4:e4b", [], 1, None, structured=True)
    assert failure.value.response_text == text
    assert requests[0]["options"]["num_predict"] == 5000
