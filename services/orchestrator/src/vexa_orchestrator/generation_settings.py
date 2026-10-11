"""Resolve explicit UI preferences before planning, searching or starting a batch."""

from __future__ import annotations

import re

from vexa_contracts import GenerationSettings, LyricsDraft

from .vocal_intent import apply_instrumental_mode

LANGUAGES = {
    "en": "English",
    "tr": "Turkish",
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}
MARKER = "\n[Vexa preferences]\n"


def conditioning_theme(theme: str, settings: GenerationSettings) -> str:
    # Idempotence matters when admitted tracks start a previously waiting session.
    theme = theme.split(MARKER, 1)[0].strip()
    if not theme:
        raise ValueError("enter a theme first")
    if settings.vocal_mode != "auto":
        theme = apply_instrumental_mode(theme, settings.vocal_mode == "instrumental")
    elif settings.custom_lyrics:
        theme = apply_instrumental_mode(theme, False)
    directives = []
    if settings.genre:
        directives.append(f"Genre override: {settings.genre}")
    if settings.vocal_mode in {"rap", "singing"}:
        directives.append(f"Vocal delivery override: {settings.vocal_mode}")
    if settings.language != "auto" and settings.vocal_mode != "instrumental":
        directives.append(f"Vocal language override: {LANGUAGES[settings.language]}")
    if settings.lyric_topic and settings.vocal_mode != "instrumental":
        directives.append(f"Lyrics about {settings.lyric_topic}")
    if settings.bpm is not None:
        directives.append(f"Tempo override: {settings.bpm:g} BPM")
    return theme + (MARKER + ". ".join(directives) if directives else "")


def custom_lyrics(plan, text: str) -> LyricsDraft:
    """Map named original lyric sections to backend IDs; repeated refrains may share text."""
    sections = {}
    name = None
    for line in text.splitlines():
        line = line.strip()
        heading = re.fullmatch(r"\[([^\]]+)\]", line)
        if heading:
            name = heading[1].casefold().replace("chorus", "refrain")
            if name in sections:
                raise ValueError(f"custom lyrics repeat heading [{heading[1]}]")
            sections[name] = []
        elif line:
            if name is None:
                raise ValueError("custom lyrics need [Verse 1], [Refrain] and [Verse 2] headings")
            sections[name].append(line)
    expected = {s.name.casefold().replace("chorus", "refrain") for s in plan.sections}
    if set(sections) - expected:
        raise ValueError("custom lyrics contain an unsupported section heading")
    return LyricsDraft(
        sections={
            s.section_id: sections.get(s.name.casefold().replace("chorus", "refrain"), [])
            for s in plan.sections
        }
    )


def capabilities(generator: str) -> dict:
    from .generators import MINIMAX_STEPS

    return {
        "track_count": {"min": 1, "max": 10},
        "duration_s": {"min": 150, "max": 180},
        "vocal_modes": ["auto", "instrumental", "singing", "rap"],
        "languages": LANGUAGES,
        "native_controls": ["duration", "seed"]
        + (["bpm"] if generator == "acestep" else [])
        + (["steps"] if generator == "minimax" else []),
        "render_speed_profiles": MINIMAX_STEPS if generator == "minimax" else {},
        "prompt_hints": ["genre", "bpm", "energy", "language", "texture", "groove", "instruments"],
        "unsupported": ["vocal_samples_only", "automatic_stem_analysis", "arbitrary_arrangement"],
        "quality_policy": "Raw defects are inspected before a bounded correction pass. "
        "Final loudness, duration, audio and cue gates remain mandatory.",
        "audio_observation_policy": "CPU-only local CLAP when installed. "
        "Uncalibrated scores never certify vocals/genre or reject a track.",
        "tempo_policy": "Strict rejects measured tempo more than 3 BPM from the target; "
        "half/double-time equivalents count as the same pulse.",
    }
