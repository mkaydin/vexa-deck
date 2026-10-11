"""Small creative writing requests; structure and native captions belong to the backend."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import httpx
from vexa_contracts import (
    CreativeDraft,
    GenerationSettings,
    LyricsDraft,
    PlannedSection,
    ThemeSpec,
    TrackPlan,
)
from vexa_yue2.backends import BackendUnavailable

from .generation import (
    ENERGY_ARC,
    ROLES,
    TRACK_DURATIONS,
    TrackBrief,
    _llm_connection,
    _record_prompt_attempt,
    _theme_profile,
)
from .music_plan import arrangement, lyric_lines, tempo_hint, validate_lyric_theme, vocal_language
from .track_titles import clean_title, fallback_title
from .vocal_intent import has_vocal_delivery, vocal_intent
from .writers import complete, pin_local, preflight

LOGGER = logging.getLogger(__name__)


def settings_identity(plan: TrackPlan, settings: dict) -> str:
    configuration = {key: value for key, value in settings.items() if not key.startswith("_")}
    return hashlib.sha256(
        json.dumps({"plan": plan.model_dump(), "writer": configuration}, sort_keys=True).encode()
    ).hexdigest()


def _environment_writer_settings() -> dict:
    mode = os.environ.get("VEXA_WRITER_MODE", "").strip()
    if mode and mode not in {"local", "remote", "template"}:
        raise ValueError("VEXA_WRITER_MODE must be local, remote or template")
    if mode == "template" or (not mode and os.environ.get("VEXA_LLM_PROVIDER") == "none"):
        return {"mode": "template", "model": "none", "provider": "none", "base": ""}
    legacy_local = not mode and os.environ.get("VEXA_MUSIC_PLANNER", "acestep") == "acestep"
    if mode == "local" or legacy_local:
        model = (
            os.environ.get("VEXA_WRITER_MODEL")
            or os.environ.get("VEXA_ACE_WRITER_MODEL")
            or "gemma4:e4b"
        )
        if model.endswith(":cloud"):
            raise ValueError("cloud writers require VEXA_WRITER_MODE=remote")
        return {
            "mode": "local",
            "model": model,
            "provider": "ollama",
            "base": os.environ.get("VEXA_WRITER_URL") or os.environ.get("VEXA_ACE_WRITER_URL", ""),
        }
    base, model, _key, provider = _llm_connection()
    return {
        "mode": "remote",
        "model": os.environ.get("VEXA_WRITER_MODEL") or model,
        "provider": provider,
        "base": os.environ.get("VEXA_WRITER_URL") or base,
    }


def writer_settings() -> dict:
    settings = _environment_writer_settings()
    for field, variable, parse in (
        ("context_length", "VEXA_WRITER_CONTEXT", int),
        ("max_tokens", "VEXA_WRITER_MAX_TOKENS", int),
        ("creativity", "VEXA_WRITER_CREATIVITY", float),
        ("lyric_creativity", "VEXA_WRITER_LYRIC_CREATIVITY", float),
    ):
        if os.environ.get(variable):
            settings[field] = parse(os.environ[variable])
    return pin_local(settings)


def build_plan(theme: str, slot: int, generator: str, *, controls: dict | None = None) -> TrackPlan:
    from .generation_settings import conditioning_theme

    preferences = GenerationSettings.model_validate(controls or {})
    if slot >= preferences.track_count:
        raise ValueError("track slot exceeds the configured track count")
    theme = conditioning_theme(theme, preferences)
    genre, palette, vocal = _theme_profile(theme)
    if preferences.genre:
        _genre, palette, _vocal = _theme_profile(preferences.genre)
        genre = _genre
        if preferences.vocal_mode == "auto" and vocal_intent(theme) is None:
            vocal = _vocal
    mode = preferences.vocal_mode
    if mode != "auto":
        vocal = mode != "instrumental"
    mode = (
        ("rap" if vocal and "hip hop" in genre else "singing" if vocal else "instrumental")
        if mode == "auto"
        else mode
    )
    bpm = preferences.bpm if preferences.bpm is not None else tempo_hint(theme, genre)
    intent = ThemeSpec(
        theme=theme,
        genre=genre,
        palette=palette,
        vocal_mode=mode,
        language=(preferences.language if preferences.language != "auto" else vocal_language(theme))
        if vocal
        else "unknown",
        bpm=bpm,
        bpm_explicit=preferences.bpm is not None
        or bool(re.search(r"\b\d{2,3}\s*bpm\b", theme, re.I)),
        preferences=preferences if controls is not None else None,
    )
    duration = preferences.duration_s or TRACK_DURATIONS[slot]
    form = arrangement(duration, bpm, vocal=vocal, rap=intent.vocal_mode == "rap")
    scale = duration / form[-1]["end_s"]
    sections = [
        PlannedSection(
            section_id=f"s{index + 1:02}",
            name="Chorus"
            if generator == "minimax" and row["section"] == "Refrain"
            else row["section"],
            bars=row["bars"],
            start_s=round(row["start_s"] * scale, 6),
            end_s=round(row["end_s"] * scale, 6),
        )
        for index, row in enumerate(form)
    ]
    sections[-1].end_s = duration
    position = slot / max(1, preferences.track_count - 1)
    energy = (
        preferences.energy
        if preferences.energy_arc == "steady"
        else 0.3 + 0.55 * position
        if preferences.energy_arc == "build"
        else 0.85 - 0.55 * position
        if preferences.energy_arc == "wind_down"
        else ENERGY_ARC[round(position * 9)]
        if preferences.track_count > 1
        else ENERGY_ARC[0]
    )
    plan = TrackPlan(
        slot=slot,
        generator=generator,
        intent=intent,
        duration_s=duration,
        energy=energy,
        role=ROLES[round(position * 9)] if preferences.track_count > 1 else ROLES[0],
        sections=sections,
    )
    if preferences.custom_lyrics:
        from .generation_settings import custom_lyrics

        compile_brief(
            plan,
            template_draft(plan),
            custom_lyrics(plan, preferences.custom_lyrics),
            "custom-lyrics-validation",
        )
    return plan


def validate_creative(draft: CreativeDraft, plan: TrackPlan) -> CreativeDraft:
    if set(draft.section_descriptions) != {section.section_id for section in plan.sections}:
        raise ValueError("creative descriptions must cover exactly the supplied section IDs")
    for text in (*draft.instruments, *draft.section_descriptions.values()):
        if not isinstance(text, str) or not 3 <= len(text.strip()) <= 1000:
            raise ValueError("instrument/section descriptions must contain bounded musical details")
    body = " ".join(
        [draft.production_notes, *draft.instruments, *draft.section_descriptions.values()]
    )
    if len(body) < 300:
        raise ValueError("creative draft needs developed musical descriptions")
    if len(body) + len(plan.intent.theme) > 6800:
        raise ValueError("creative draft exceeds the native caption budget")
    if re.search(
        r"\b\d{1,2}:\d{2}\b|(?m:^[ \t]*(?:Global Metadata|Vocal Details|Arrangement)$)", body
    ):
        raise ValueError("creative fields must not contain timecodes or native caption headings")
    if plan.intent.genre == "Memphis phonk" and re.search(
        r"drift[ -]+phonk|\bjdm\b|phonk\s+house|four.on.the.floor", body, re.I
    ):
        raise ValueError("creative draft changed the Memphis phonk subgenre")
    if plan.intent.vocal_mode == "instrumental" and has_vocal_delivery(body):
        fields = {
            "production_notes": draft.production_notes,
            **{f"instruments[{i}]": text for i, text in enumerate(draft.instruments)},
            **{
                f"section_descriptions.{key}": text
                for key, text in draft.section_descriptions.items()
            },
        }
        rejected = [
            f"{key}: {text[:250]}" for key, text in fields.items() if has_vocal_delivery(text)
        ]
        raise ValueError(
            "creative draft introduced vocal material into an instrumental request. "
            "Replace all voice-derived sounds, including wordless vocal chops, sighs and "
            "humming, with instrumental sources such as piano, horn or synth chops. "
            "Offending fields: " + "; ".join(rejected[:3])
        )
    return draft


def compile_brief(
    plan: TrackPlan, draft: CreativeDraft, lyrics: LyricsDraft | None, origin: str
) -> TrackBrief:
    validate_creative(draft, plan)
    vocal = plan.intent.vocal_mode != "instrumental"
    if vocal and lyrics is None:
        raise ValueError("vocal plan requires original lyrics")
    if not vocal and lyrics is not None:
        raise ValueError("instrumental plan cannot include lyrics")
    lyric_text = ""
    if lyrics is not None:
        ids = {section.section_id for section in plan.sections}
        if set(lyrics.sections) != ids:
            raise ValueError("lyrics must use exactly the supplied section IDs")
        blocks = []
        for section in plan.sections:
            lines = lyrics.sections[section.section_id]
            if section.name in {"Intro", "Outro", "Instrumental Break"} and lines:
                raise ValueError("DJ intro/outro and instrumental break must remain instrumental")
            if section.name not in {"Intro", "Outro", "Instrumental Break"} and len(lines) < 4:
                raise ValueError(f"lyric section {section.section_id} needs at least four lines")
            if any(not line.strip() or len(line) > 180 or "\n" in line for line in lines):
                raise ValueError("lyric lines must be nonempty, bounded single lines")
            blocks.append(f"[{section.name}]\n" + "\n".join(lines))
        lyric_text = "\n".join(blocks).strip()
        if len(lyric_text) > 5000:
            raise ValueError("lyrics exceed the native conditioning budget")
        if plan.intent.vocal_mode == "rap" and len(lyric_lines(lyric_text)) < 40:
            raise ValueError("full-length rap needs two developed verses and written-out refrains")
        verse_lines = [
            re.sub(r"\W+", " ", line.casefold()).strip()
            for section in plan.sections
            if section.name.startswith("Verse")
            for line in lyrics.sections[section.section_id]
        ]
        if verse_lines and len(set(verse_lines)) / len(verse_lines) < 0.8:
            raise ValueError("verse lines repeat excessively; Verse 2 must develop new lines")
        preferences = plan.intent.preferences
        topic_theme = (
            f"vocals in {plan.intent.language} about {preferences.lyric_topic}"
            if preferences and preferences.lyric_topic
            else plan.intent.theme
        )
        validate_lyric_theme(topic_theme, lyric_text, language=plan.intent.language)

    voice = (
        "Instrumental, no vocals."
        if not vocal
        else "Dry rhythmic spoken rap, with original lyrics."
        if plan.intent.vocal_mode == "rap"
        else "Sung vocal delivery follows the listener's theme, with original lyrics."
    )
    style = (
        f"{plan.intent.genre}. Listener theme: {plan.intent.theme}. {voice} "
        f"Target {plan.intent.bpm:g} BPM. {draft.production_notes}"
    )
    preferences = plan.intent.preferences
    hint_text = ""
    if preferences:
        hints = {
            "Production texture": preferences.texture,
            "Groove": preferences.groove,
            "Include instruments": ", ".join(preferences.include_instruments),
            "Exclude instruments": ", ".join(preferences.exclude_instruments),
        }
        hint_text = ". ".join(f"{key}: {value}" for key, value in hints.items() if value)
        style += " " + hint_text
    form = [s.model_dump() for s in plan.sections]
    metadata = {
        "pipeline_version": 2,
        "track_plan": plan.model_dump(),
        "creative_draft": draft.model_dump(),
        "genre": plan.intent.genre,
        "bpm": plan.intent.bpm,
        "measured": False,
        "requested_duration_s": plan.duration_s,
        "sections": form,
        "generation_settings": preferences.model_dump() if preferences else None,
        "lyric_topic_evidence": "unknown_cross_language"
        if vocal and plan.intent.language != "en"
        else "lexical_en"
        if vocal and re.search(r"\babout\b", plan.intent.theme, re.I)
        else "not_requested",
    }
    if plan.generator == "minimax":
        from .minimax_prompts import FORMAT, validate_caption

        def clock(value):
            value = round(value)
            return f"{value // 60}:{value % 60:02}"

        caption = (
            f"Global Metadata\n{plan.intent.genre}. Listener theme: {plan.intent.theme}. "
            f"Tempo target {plan.intent.bpm:g} BPM, 4/4. "
            f"Duration target {plan.duration_s:g} seconds. "
            f"Track role: {plan.role}. {draft.production_notes}. {hint_text}\n"
            f"Instrument roles: {'; '.join(draft.instruments)}.\n"
            f"Vocal Details\n{voice}\nArrangement\n"
            + "\n".join(
                f"{clock(s.start_s)}-{clock(s.end_s)} {s.name}: "
                f"{draft.section_descriptions[s.section_id]}"
                for s in plan.sections
            )
        )
        metadata.update(
            minimax_caption=validate_caption(
                caption,
                style=style,
                lyrics=lyric_text,
                genre=plan.intent.genre,
                vocal=vocal,
                duration_s=plan.duration_s,
            ),
            caption_format=FORMAT,
            caption_origin=f"compiler:v2+{origin}",
            caption_retimed=False,
        )
    else:
        style += " Arrangement: " + "; ".join(
            f"{s.name} ({s.bars} bars): {draft.section_descriptions[s.section_id]}"
            for s in plan.sections
        )
    return TrackBrief(
        style,
        plan.duration_s,
        plan.energy,
        origin,
        lyric_text,
        metadata,
        clean_title(draft.title) or fallback_title(plan.intent.theme, style, str(plan.slot)),
    )


def template_draft(plan: TrackPlan) -> CreativeDraft:
    return CreativeDraft(
        title=fallback_title(plan.intent.theme, "", str(plan.slot)),
        production_notes=f"{plan.intent.palette}. Preserve the requested groove and production "
        f"texture. Variation {plan.slot + 1}: develop the motif and bass phrasing "
        "within the requested genre, with rhythmic DJ intro and outro.",
        instruments=[plan.intent.palette],
        section_descriptions={
            s.section_id: "Keep sparse percussion and bass active for mixing; "
            "gradually withdraw supporting "
            "layers until the end."
            if s.name == "Outro"
            else "Introduce the defining motif with sparse drums and bass for the DJ opening."
            if s.name == "Intro"
            else f"Develop the existing instrumental motif, bass response and drum articulation; "
            f"vary density for the {s.name.lower()} without changing the musical style."
            for s in plan.sections
        },
    )


def _write_json(plan: TrackPlan, settings: dict, schema: dict, instruction: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": "You are Vexa's music writer. Return only the requested valid JSON object. "
            "The backend owns timing, track duration, section order and vocal configuration. "
            "Resolved genre, vocal_mode, language and explicit preferences override "
            "conflicting free-text hints in the theme. "
            "Never write timecodes, headings, tempo/key metadata or invent section IDs. "
            "Preserve the listener's genre, era, exclusions and language. Use concrete musical "
            "changes and original imagery. No artist names or copied lyrics. " + instruction,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "intent": plan.intent.model_dump(),
                    "role": plan.role,
                    "energy_target": plan.energy,
                    "track_number": plan.slot + 1,
                    "sections": [
                        {"id": s.section_id, "name": s.name, "bars": s.bars} for s in plan.sections
                    ],
                    "response_schema": schema,
                },
                ensure_ascii=False,
            ),
        },
    ]
    raw = ""
    try:
        stage = "creative" if "title" in schema["properties"] else "lyrics"
        raw = complete(settings, messages, schema, stage)
        value = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        _record_prompt_attempt(
            plan.intent.theme,
            settings["model"],
            raw,
            pipeline=2,
            status="received",
            requests=settings.get("_requests", []),
            slot=plan.slot,
        )
        return value
    except Exception as exc:
        _record_prompt_attempt(
            plan.intent.theme,
            settings["model"],
            getattr(exc, "response_text", raw),
            failure=str(exc),
            pipeline=2,
            requests=settings.get("_requests", []),
            slot=plan.slot,
        )
        raise


def _feedback_state(path: Path, identity: str) -> dict:
    try:
        previous = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    if not isinstance(previous, dict) or previous.get("identity") != identity:
        return {}
    if not isinstance(previous.get("error"), str):
        return {}
    count = previous.get("attempts", 1)
    previous["attempts"] = count if type(count) is int and 1 <= count <= 10000 else 1
    return previous


def _write_plan(plan: TrackPlan, settings: dict, cache_path: Path) -> TrackBrief:
    """Persist accepted creative fields before asking for lyrics or releasing the worker."""
    identity = settings_identity(plan, settings)
    cached = {}
    if cache_path.is_file():
        try:
            saved = json.loads(cache_path.read_text())
            if saved.get("identity") == identity:
                cached = saved
        except (ValueError, OSError):
            LOGGER.warning("Ignoring unreadable creative cache %s", cache_path)

    def persist():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(".tmp")
        temp.write_text(json.dumps({**cached, "identity": identity}, ensure_ascii=False) + "\n")
        temp.replace(cache_path)

    if settings["mode"] == "template":
        supplied = plan.intent.preferences.custom_lyrics if plan.intent.preferences else ""
        if plan.intent.vocal_mode != "instrumental" and not supplied:
            raise ValueError("original vocal lyrics require an available prompt writer")
        from .generation_settings import custom_lyrics

        return compile_brief(
            plan, template_draft(plan), custom_lyrics(plan, supplied) if supplied else None, "local"
        )
    feedback_path = cache_path.with_suffix(".feedback.json")
    feedback = ""
    previous = _feedback_state(feedback_path, identity)
    if previous:
        feedback = " Correct this previous validation failure: " + previous["error"][:1600]
    from .ace_planner import local_writer_server

    context = (
        local_writer_server(configured_url=settings["base"])
        if settings["mode"] == "local"
        else nullcontext(settings["base"])
    )
    with context as base:
        connection = {**settings, "base": base}
        connection["_requests"] = settings.get("_telemetry", [])
        connection["_keep_alive"] = (
            60 if not settings["base"] or settings.get("_lease_owned") else 0
        )
        connection["_identity"] = preflight(settings, base)
        connection["seed"] = (
            settings.get("seed", 42) + plan.slot + 1009 * previous.get("attempts", 0)
        ) % 2147483647
        if "creative" in cached:
            draft = validate_creative(CreativeDraft.model_validate(cached["creative"]), plan)
        else:
            schema = CreativeDraft.model_json_schema()
            schema["properties"]["section_descriptions"] = {
                "type": "object",
                "properties": {s.section_id: {"type": "string"} for s in plan.sections},
                "required": [s.section_id for s in plan.sections],
                "additionalProperties": False,
            }
            draft = validate_creative(
                CreativeDraft.model_validate(
                    _write_json(
                        plan,
                        connection,
                        schema,
                        "Write a title, production_notes, instrument roles "
                        "and musical descriptions "
                        "for exactly the supplied section IDs. "
                        "Do not include lyrics or vocal instructions. "
                        "Use 30-60 words for each section. "
                        "Keep rhythm active in DJ intro and outro. "
                        "Use a concrete theme-specific title, "
                        "avoiding generic Neon/Obsidian clichés. "
                        + (
                            "All described sounds must be instrumental."
                            if plan.intent.vocal_mode == "instrumental"
                            else "Describe the musical accompaniment to the requested voice."
                        )
                        + feedback,
                    )
                ),
                plan,
            )
            cached["creative"] = draft.model_dump()
            if plan.intent.vocal_mode == "instrumental":
                compile_brief(plan, draft, None, f"llm:{settings['model']}+v2")
            persist()
        lyrics = None
        supplied = plan.intent.preferences.custom_lyrics if plan.intent.preferences else ""
        if supplied:
            from .generation_settings import custom_lyrics

            lyrics = custom_lyrics(plan, supplied)
        elif plan.intent.vocal_mode != "instrumental":
            if "lyrics" in cached:
                lyrics = LyricsDraft.model_validate(cached["lyrics"])
            else:
                schema = LyricsDraft.model_json_schema()
                schema["properties"]["sections"] = {
                    "type": "object",
                    "properties": {
                        s.section_id: {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 180},
                            "minItems": 0
                            if s.name in {"Intro", "Outro", "Instrumental Break"}
                            else 16
                            if plan.intent.vocal_mode == "rap" and "Verse" in s.name
                            else 4,
                            "maxItems": 0
                            if s.name in {"Intro", "Outro", "Instrumental Break"}
                            else 16
                            if plan.intent.vocal_mode == "rap" and "Verse" in s.name
                            else 4
                            if plan.intent.vocal_mode == "rap"
                            else 12,
                        }
                        for s in plan.sections
                    },
                    "required": [s.section_id for s in plan.sections],
                    "additionalProperties": False,
                }
                lyrics = LyricsDraft.model_validate(
                    _write_json(
                        plan,
                        connection,
                        schema,
                        "Write complete original lyric lines by section ID, without tags/headings. "
                        "Use empty arrays for Intro, Outro and Instrumental Break. "
                        "Develop a coherent "
                        "narrator, setting and conflict in the requested language. "
                        "Use idiomatic language and believable concrete imagery. "
                        "Each verse must advance the story with new lines; never copy Verse 1 "
                        "into Verse 2. Repeated refrains are allowed. "
                        + (
                            "Two verses of 16 different lines "
                            "and two refrains of 4 written-out lines. "
                            "Rhythmic spoken rap, no sung pop chorus."
                            if plan.intent.vocal_mode == "rap"
                            else "Use two developed verses and written-out refrains, at least four "
                            "lines per vocal section. Avoid a tiny summary of the song."
                        )
                        + feedback,
                    )
                )
                # Validate lyrics before caching so a bad response cannot poison retries.
                compile_brief(plan, draft, lyrics, f"llm:{settings['model']}+v2")
                cached["lyrics"] = lyrics.model_dump()
                persist()
        brief = compile_brief(plan, draft, lyrics, f"llm:{settings['model']}+v2")
        brief = replace(
            brief,
            music_plan={
                **brief.music_plan,
                "writer": connection["_identity"],
                "writer_requests": connection.get("_requests", []),
                "writer_config": {
                    k: settings.get(k)
                    for k in (
                        "context_length",
                        "max_tokens",
                        "creativity",
                        "lyric_creativity",
                        "seed",
                    )
                },
            },
        )
    LOGGER.info(
        "Backend-owned plan ready slot=%d writer=%s; writer requests complete",
        plan.slot + 1,
        settings["model"],
    )
    return brief


def write_plan(plan: TrackPlan, settings: dict, cache_path: Path) -> TrackBrief:
    """Retry rejected fields with validation feedback while preserving accepted drafts."""
    feedback_path = cache_path.with_suffix(".feedback.json")
    try:
        brief = _write_plan(plan, settings, cache_path)
    except (httpx.HTTPError, FileNotFoundError) as exc:
        raise BackendUnavailable(f"prompt writer unavailable: {exc}") from exc
    except RuntimeError as exc:
        if "private lyric writer" in str(exc) or str(exc).startswith("Ollama:"):
            raise BackendUnavailable(f"prompt writer unavailable: {exc}") from exc
        raise
    except ValueError as exc:
        identity = settings_identity(plan, settings)
        attempts = _feedback_state(feedback_path, identity).get("attempts", 0) + 1
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        temp = feedback_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"identity": identity, "error": str(exc), "attempts": attempts}) + "\n"
        )
        temp.replace(feedback_path)
        _record_prompt_attempt(
            plan.intent.theme,
            settings["model"],
            "",
            pipeline=2,
            failure=str(exc),
            stage="validation",
        )
        raise
    feedback_path.unlink(missing_ok=True)
    return brief


def plan_track(
    theme: str,
    slot: int,
    generator: str,
    *,
    settings: dict,
    cache_path: Path,
    cancel=None,
    controls: dict | None = None,
) -> TrackBrief:
    from .background import run_job

    plan = build_plan(theme, slot, generator, controls=controls)
    if settings["mode"] == "template":
        return write_plan(plan, settings, cache_path)
    result = run_job(
        "plan_v2",
        {"plan": plan.model_dump(), "settings": settings, "cache_path": str(cache_path)},
        cancel=cancel,
        timeout_s=(controls or {}).get("planning_timeout_s", 600),
    )
    return TrackBrief(**result["brief"])
