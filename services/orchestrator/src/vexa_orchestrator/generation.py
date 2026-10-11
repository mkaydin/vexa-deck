"""Generate a themed YuE2 track, gate it, and add it to the local depot."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import soundfile as sf
from vexa_audio.cues import analyse_cues
from vexa_contracts import ApprovalState, Estimate, Provenance, SourceType
from vexa_yue2.backends import BackendUnavailable
from vexa_yue2.gates import analyse
from vexa_yue2.master import master
from vexa_yue2.runtime import native_environment

from .audio_evidence import EvidenceSession
from .depot import DepotAsset
from .music_plan import lyric_lines, validate_lyric_theme
from .track_titles import clean_title, fallback_title, unique_title
from .vocal_intent import has_vocal_delivery, instrumental_details, vocal_intent

ROOT = Path(__file__).resolve().parents[4]
LOGGER = logging.getLogger(__name__)
TRACK_COUNT = 10
TRACK_DURATIONS = tuple(165.0 + index * 1.5 for index in range(TRACK_COUNT))
ENERGY_ARC = (0.35, 0.42, 0.50, 0.57, 0.66, 0.75, 0.82, 0.72, 0.58, 0.43)
ROLES = (
    "inviting opener",
    "deeper groove",
    "melodic development",
    "rhythmic lift",
    "harmonic turn",
    "driving middle",
    "peak moment",
    "late-night release",
    "warm descent",
    "spacious closer",
)
DEFAULT_PROMPT_MODEL = "nemotron-3-super:cloud"


class PromptResponseError(ValueError):
    def __init__(self, message: str, response_text: str):
        super().__init__(message)
        self.response_text = response_text


def _llm_connection() -> tuple[str, str, str, str]:
    base = os.environ.get("VEXA_LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("VEXA_LLM_MODEL", "").strip() or DEFAULT_PROMPT_MODEL
    key = os.environ.get("VEXA_LLM_API_KEY", "")
    provider = os.environ.get("VEXA_LLM_PROVIDER", "").strip()
    if provider == "none":
        return "", "", "", "none"
    if not provider:
        provider = "ollama" if not base or "11434" in base or model.endswith(":cloud") else "openai"
    base = base or "http://127.0.0.1:11434"
    if provider == "ollama" and base.endswith("/v1"):
        base = base[:-3]
    return base, model, key, provider


def _ollama_prompts(
    base: str,
    model: str,
    messages: list[dict[str, str]],
    count: int,
    cancel: threading.Event | None,
    *,
    structured: bool = False,
    schema: dict | None = None,
    options: dict | None = None,
    metrics: dict | None = None,
    keep_alive: int | None = None,
    thinking: bool | None = False,
    schema_supported: bool | None = None,
) -> str:
    """Cloud models use requested JSON text; schema format is only supported locally."""
    request = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": int(os.environ.get("VEXA_LLM_KEEP_ALIVE", "0")),
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 20000, "num_ctx": 32768},
    }
    if structured:
        request["options"]["num_predict"] = min(20000, 5000 * count)
    if options:
        request["options"].update(options)
    if keep_alive is not None:
        request["keep_alive"] = keep_alive
    if thinking is None:
        request.pop("think")
    else:
        request["think"] = thinking
    if schema_supported if schema_supported is not None else not model.endswith(":cloud"):
        request["format"] = {
            "type": "object",
            "properties": {
                "tracks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"style": {"type": "string"}, "lyrics": {"type": "string"}},
                        "required": ["style", "lyrics"],
                    },
                    "minItems": count,
                    "maxItems": count,
                }
            },
            "required": ["tracks"],
        }
        if structured:
            item = request["format"]["properties"]["tracks"]["items"]
            item["properties"].update(
                {"structured_caption": {"type": "string"}, "title": {"type": "string"}}
            )
            item["required"].append("structured_caption")
        if schema is not None:
            request["format"] = schema
    chunks: list[str] = []
    text_chars = 0
    progress_at = time.monotonic()
    started = progress_at
    first_token_at = None
    finished = False
    with httpx.stream(
        "POST", f"{base}/api/chat", timeout=httpx.Timeout(300.0, read=120.0), json=request
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if cancel is not None and cancel.is_set():
                raise RuntimeError("prompt writing cancelled")
            if line:
                body = json.loads(line)
                if body.get("error"):
                    raise RuntimeError(f"Ollama: {body['error']}")
                chunk = body.get("message", {}).get("content", "")
                if chunk and first_token_at is None:
                    first_token_at = time.monotonic() - started
                chunks.append(chunk)
                text_chars += len(chunk)
                if structured and text_chars > 24000 * count:
                    raise PromptResponseError(
                        "prompt response exceeded the structured track size limit", "".join(chunks)
                    )
                if time.monotonic() - progress_at > 15:
                    progress_at = time.monotonic()
                    LOGGER.info("Prompt writer progress model=%s text_chars=%d", model, text_chars)
                if body.get("done"):
                    finished = True
                    if metrics is not None:
                        metrics.update(
                            {
                                key: body.get(key)
                                for key in (
                                    "prompt_eval_count",
                                    "eval_count",
                                    "total_duration",
                                    "load_duration",
                                    "prompt_eval_duration",
                                    "eval_duration",
                                    "done_reason",
                                )
                            }
                        )
                        metrics.update(wall_s=time.monotonic() - started, ttft_s=first_token_at)
                    if body.get("done_reason") == "length":
                        raise PromptResponseError("prompt response was truncated", "".join(chunks))
                    break
    if metrics is not None and not finished:
        raise PromptResponseError(
            "prompt stream ended without a completion marker", "".join(chunks)
        )
    return "".join(chunks)


@dataclass(frozen=True, slots=True)
class TrackBrief:
    style: str
    duration_s: float
    energy: float
    origin: str
    lyrics: str = ""
    music_plan: dict | None = None
    title: str = ""


def _theme_profile(theme: str) -> tuple[str, str, bool]:
    """Anchor known genres and vocal intent before any LLM embellishes the brief."""
    lowered = theme.lower()
    intent = vocal_intent(theme)
    instrumental = intent is False
    if re.search(r"\bphonk\b", lowered):
        if re.search(r"\bmemphis\b", lowered):
            return (
                "Memphis phonk",
                "pitched-down haunted minor-key piano and soul-sample chops, sparse swung "
                "Southern drum-machine beats, dry snares, deep rounded 808 bass, "
                "dusty sampler grit, low-pass filtering and warm cassette saturation",
                intent is True,
            )
        genre = "JDM drift phonk" if re.search(r"\bjdm\b|\bdrift\b", lowered) else "phonk"
        return (
            genre,
            "distorted 808 sub bass, crunchy syncopated trap drums, rapid hi-hats, "
            "ominous cowbell motifs, dark minor-key synths and gritty tape texture",
            intent is True,
        )
    rap = bool(re.search(r"\brap\b|hip[ -]?hop|boom[ -]?bap|\btrap\b|\bdrill\b", lowered))
    if rap:
        if re.search(r"\btrap\b|\bdrill\b", lowered):
            subgenre = "drill" if "drill" in lowered else "trap"
            return (
                f"{subgenre} hip hop; {theme.strip()}",
                "808 sub bass, syncopated electronic kick, crisp snare and hi-hat rolls, "
                "dark sparse melodic motifs"
                + (" and rhythmically precise rap delivery" if not instrumental else ""),
                not instrumental,
            )
        era = "1990s " if re.search(r"90s|90's|1990|doksan", lowered) else ""
        genre = (
            f"{era}underground boom bap hip hop" if era or "underground" in lowered else "hip hop"
        )
        palette = (
            "sample-chopped dusty jazz and soul loops, swung breakbeat drums, punchy kick, "
            "dry snare, warm sampled bass, sparse vinyl texture and turntable cuts"
        )
        return genre, palette, not instrumental
    vocal = intent is True
    return theme.strip(), "a genre-authentic rhythm section and era-appropriate instruments", vocal


def _parse_track_plan(
    raw: str,
    theme: str,
    count: int,
    model: str,
    *,
    generator: str = "yue2",
    track_index: int | None = None,
) -> list[TrackBrief]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    body = json.loads(raw)
    tracks = body.get("tracks") if isinstance(body, dict) else body
    if not isinstance(tracks, list) or len(tracks) != count:
        raise ValueError(f"expected exactly {count} tracks")
    genre, _palette, vocal = _theme_profile(theme)
    rap = "hip hop" in genre
    briefs = []
    seen = set()
    seen_titles = set()
    for index, row in enumerate(tracks):
        position = track_index if track_index is not None else index
        if not isinstance(row, (dict, str)):
            raise ValueError(f"track {index + 1}: expected a production brief object")
        style = row if isinstance(row, str) else row.get("style", "")
        lyrics = "" if isinstance(row, str) else row.get("lyrics", "")
        if not isinstance(style, str) or not 30 <= len(style.strip()) <= 1800:
            raise ValueError(f"track {index + 1}: invalid style length")
        if not isinstance(lyrics, str) or len(lyrics) > 5000:
            raise ValueError(f"track {index + 1}: invalid lyrics")
        if generator == "minimax":
            from .minimax_prompts import normalize_lyrics

            lyrics = normalize_lyrics(lyrics)
        lowered = style.lower()
        if lowered in seen:
            raise ValueError("duplicate track prompts")
        seen.add(lowered)
        if "phonk" in genre.lower() and not re.search(r"\bphonk\b", lowered):
            raise ValueError(f"track {index + 1}: missing requested phonk genre")
        if genre == "Memphis phonk":
            if "memphis" not in lowered:
                raise ValueError(f"track {index + 1}: missing requested Memphis phonk subgenre")
            if re.search(r"\bdrift[ -]+phonk\b|\bjdm\b|phonk\s+house|four.on.the.floor", lowered):
                raise ValueError(f"track {index + 1}: Memphis phonk prompt switched subgenre")
        if rap:
            if not re.search(r"\brap\b|hip[ -]?hop|boom[ -]?bap", lowered):
                raise ValueError(f"track {index + 1}: missing requested hip hop / rap genre")
            if ("underground" in genre or "boom bap" in genre) and re.search(
                r"\bpop\b|edm|four.on.the.floor|anthemic|melodic singing|sung chorus", lowered
            ):
                raise ValueError(
                    f"track {index + 1}: contradicts underground rap with pop/dance singing"
                )
        if vocal and re.search(r"no vocals|purely instrumental|^instrumental", lowered):
            raise ValueError(f"track {index + 1}: requested vocals were removed")
        if vocal and len(lyrics.strip()) < 100:
            raise ValueError(f"track {index + 1}: missing original vocal lyrics")
        if rap and vocal:
            lines = lyric_lines(lyrics)
            if len(lines) < 40:
                raise ValueError(f"track {index + 1}: rap needs 40 lyric lines for a full song")
        if vocal:
            validate_lyric_theme(theme, lyrics)
        if not vocal:
            if lyrics.strip():
                raise ValueError(f"track {index + 1}: instrumental tracks must have empty lyrics")
            if has_vocal_delivery(style):
                raise ValueError(f"track {index + 1}: instrumental style added vocal delivery")
            lyrics = ""
        voice = (
            "Rhythmic spoken rap verses, dry intimate MC delivery, spoken refrain."
            if rap and vocal
            else "Vocal delivery follows the listener's requested style."
            if vocal
            else "Instrumental, no vocals."
        )
        # The immutable theme/genre leads the conditioning, before track-specific details.
        style = f"{genre}. Listener theme: {theme.strip()}. {voice} {style.strip()}"
        musical_plan = None
        if generator == "minimax":
            from .minimax_prompts import FORMAT, retime_caption, validate_caption
            from .music_plan import arrangement, tempo_hint

            original_caption = row.get("structured_caption") if isinstance(row, dict) else None
            if not isinstance(original_caption, str):
                raise ValueError("MiniMax plan needs structured_caption text")
            timed_caption, retimed = retime_caption(original_caption, TRACK_DURATIONS[position])
            caption = validate_caption(
                timed_caption,
                style=style,
                lyrics=lyrics,
                genre=genre,
                vocal=vocal,
                duration_s=TRACK_DURATIONS[position],
            )
            if retimed:
                LOGGER.info(
                    "MiniMax caption timeline repaired track=%d target_s=%s; "
                    "original caption retained in music_plan",
                    position + 1,
                    TRACK_DURATIONS[position],
                )
            bpm = tempo_hint(theme, genre)
            musical_plan = {
                "minimax_caption": caption,
                "caption_format": FORMAT,
                "caption_origin": f"llm:{model}",
                "genre": genre,
                "bpm": bpm,
                "measured": False,
                "requested_duration_s": TRACK_DURATIONS[position],
                "caption_retimed": retimed,
                "original_minimax_caption": original_caption if retimed else None,
                "sections": arrangement(TRACK_DURATIONS[position], bpm, vocal=vocal, rap=rap),
            }
        briefs.append(
            TrackBrief(
                style,
                TRACK_DURATIONS[position],
                ENERGY_ARC[position],
                f"llm:{model}",
                lyrics.strip(),
                music_plan=musical_plan,
                title=unique_title(
                    (clean_title(row.get("title")) if isinstance(row, dict) else "")
                    or fallback_title(theme, style, str(index)),
                    seen_titles,
                ),
            )
        )
    return briefs


def _repair_instrumental_plan(
    raw: str, theme: str, count: int, track_index: int | None = None
) -> tuple[str, list[dict]]:
    """Project a valid JSON response onto the listener's explicit instrumental request.

    JSON shape and genre gates remain strict. Voice directions cannot cause a whole
    instrumental batch to fail: keep musical details and discard unwanted lyrics.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    body = json.loads(text)
    tracks = body.get("tracks") if isinstance(body, dict) else body
    if not isinstance(tracks, list) or len(tracks) != count:
        return raw, []
    genre, palette, vocal = _theme_profile(theme)
    if vocal:
        return raw, []
    repairs = []
    for index, row in enumerate(tracks):
        style = (
            row if isinstance(row, str) else row.get("style", "") if isinstance(row, dict) else None
        )
        if not isinstance(style, str) or not 30 <= len(style.strip()) <= 1800:
            continue  # Malformed fields still fail the normal parser.
        lyrics = row.get("lyrics", "") if isinstance(row, dict) else ""
        if not isinstance(lyrics, str):
            continue
        removed = []
        if has_vocal_delivery(style):
            details, removed = instrumental_details(style)
            position = track_index if track_index is not None else index
            if len(details) < 30:
                details = (
                    f"{palette}. {ROLES[position]}; variation {position + 1}: "
                    "develop the instrumental motif, bass groove and percussion through "
                    "a sparse rhythmic intro, evolving middle and mixable outro."
                )
            style = f"{genre}. {details}"
        if removed or lyrics.strip():
            tracks[index] = {**(row if isinstance(row, dict) else {}), "style": style, "lyrics": ""}
            repairs.append(
                {
                    "track": index + 1,
                    "removed_sentences": removed,
                    "cleared_lyrics": bool(lyrics.strip()),
                }
            )
    return json.dumps(body, ensure_ascii=False) if repairs else raw, repairs


def _record_prompt_attempt(theme: str, model: str, raw: str, **details) -> Path:
    """Keep the exact failed/repaired response so a generic error can be diagnosed."""
    path = ROOT / "var/plans/raw" / f"writer-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"theme": theme, "model": model, "response": raw, **details},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _plan_remote_theme(
    theme: str,
    *,
    count: int = TRACK_COUNT,
    cancel: threading.Event | None = None,
    track_index: int | None = None,
    generator: str = "yue2",
) -> list[TrackBrief]:
    """Preserve genre, era and vocal intent; validate and repair a ten-track production plan."""
    if not theme.strip() or not 1 <= count <= TRACK_COUNT:
        raise ValueError("theme and a count from 1 to 10 are required")
    genre, palette, vocal = _theme_profile(theme)
    base, model, key, provider = _llm_connection()
    if provider == "none":
        if vocal:
            raise RuntimeError("vocal tracks need the prompt LLM to write original lyrics")
        seen_titles = set()
        return [
            TrackBrief(
                style=(
                    f"Instrumental {theme.strip()}, {genre}, {ROLES[index]}; {palette}. "
                    f"Variation {index + 1}: change sample chops, drum fills and bass phrasing "
                    "within this genre, with a sparse DJ intro and outro. No vocals."
                ),
                duration_s=TRACK_DURATIONS[index],
                energy=ENERGY_ARC[index],
                origin="local",
                title=unique_title(fallback_title(theme, "", str(index)), seen_titles),
            )
            for index in range(count)
        ]
    rap = "hip hop" in genre
    voice = (
        "Write rhythmic spoken rap, not sung pop. Provide original lyrics for each track, "
        "with [Intro], [Verse 1], [Refrain], [Instrumental Break], [Verse 2], "
        "[Refrain], [Outro] sections. Each verse MUST have 16 different lines; each refrain "
        "MUST have 4 spoken lines, written out twice: at least 40 lyric lines in total. "
        "Keep the breakbeat and arrangement active for the entire 165-180 seconds; do not "
        "finish after one minute or pad the tail with silence. Use an 8-bar intro, 16-bar "
        "verse, 4-bar refrain, 4-bar instrumental break, 16-bar verse, 4-bar refrain and "
        "8-bar rhythmic outro. "
        "Use the requested language, otherwise English. Do not copy existing songs."
        if vocal and rap
        else "Provide original genre-appropriate lyrics in verse/refrain sections, in the "
        "requested language, otherwise English."
        if vocal
        else "Instrumental, no vocals; lyrics must be an empty string."
    )
    constraints = (
        "All tracks must be underground boom bap rap/hip hop. Preserve the 1990s "
        "era if requested: swung sampled breakbeats, dry snares, chopped samples, "
        "bass groove. "
        + ("Use dry MC delivery. " if vocal else "Keep all melodic parts instrumental. ")
        + "Avoid pop, EDM, four-on-the-floor drums, "
        "sung choruses and glossy synth anthems. Use positive style tags; do not "
        "put rejected genre words in the style field."
        if rap
        else "Keep every track in the listener's genre and era; do not default to house, "
        "techno or pop unless requested. Interpret ambiguous moods musically."
    )
    if "phonk" in genre.lower() and not vocal:
        constraints += (
            " This is instrumental Memphis phonk: use haunted pitched-down sample-like "
            "instrument motifs, sparse swung drum-machine beats, rounded 808 bass and "
            "cassette texture. Keep a slow head-nodding pocket and loop-driven arrangement. "
            "Use positive style tags describing this palette. "
            if genre == "Memphis phonk"
            else " This is instrumental phonk: perform rhythmic hooks with cowbells, synths "
            "and percussion. Replace Memphis rap samples and vocal chops with instrumental "
            "riffs. Describe only the instruments that will actually play."
        )
    story = (
        "For vocal tracks, choose a concrete setting, narrator and conflict, "
        "and a story that develops across verses; avoid interchangeable party/love slogans. "
        if vocal
        else "For instrumental tracks, express the theme through timbre, groove, harmony and "
        "evolving motifs. Describe instrumental sections only; no singer, MC, spoken words, "
        "humming or vocal samples. Keep lyrics empty. "
    )
    system = (
        f"You are Vexa's music producer. Write exactly {count} distinct "
        f"{'MiniMax Music 3.0' if generator == 'minimax' else 'YuE2'} production briefs. "
        "The listener's genre, era, language and vocal request take priority over energy arc. "
        f"Genre anchor: {genre}. Instrument palette: {palette}. {constraints} {voice} "
        "Tracks run 165-180 seconds. Each style is 45-100 words, starting with genre and era, "
        "then drum groove, timbre, bass, harmony, delivery and a concrete evolving arrangement. "
        "Keep a sparse rhythmic intro/outro for DJ mixing. Energy rises and falls through "
        "density and drum/sample variation, without changing genre. BPM can be a broad hint, "
        "not a guaranteed output. No artist names. Give each track a distinct, memorable "
        "2-6 word title reflecting its theme and musical mood, in the listener's language. "
        "Avoid generic track numbers. Return ONLY valid JSON, no markdown: "
        '{"tracks":[{"title":"memorable track name","style":"production description",'
        '"lyrics":"original lyrics or empty"}]}. '
        "Escape lyric newlines inside the JSON strings. "
        f"{story} Track roles in order: "
        f"{ROLES[track_index] if track_index is not None else ', '.join(ROLES[:count])}."
    )
    if track_index is not None:
        focus = (
            "setting and first impressions",
            "conflict and its cost",
            "a vivid memory",
            "an obstacle and response",
            "a change of perspective",
            "a decisive moment",
            "emotional climax",
            "consequences",
            "quiet reflection",
            "resolution",
        )[track_index]
        system += (
            f" This is track {track_index + 1} of a ten-track set; "
            f"energy {ENERGY_ARC[track_index]:.2f}. "
            f"Target {TRACK_DURATIONS[track_index]:g} seconds. Give this track its own motif, "
            + (
                f"narrator details and a story focusing on {focus} within the listener's theme. "
                "Count actual lyric lines: stage directions and headings DO NOT count. "
                "Write all refrain lines explicitly, never '(repeat chorus)'."
                if vocal
                else "bass pattern, instrumental phrasing and evolving arrangement. "
                "Do not write lyrics, vocal delivery or verse/refrain directions."
            )
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": theme.strip()}]
    if generator == "minimax":
        from .minimax_prompts import timeline_outline, writer_instructions
        from .music_plan import tempo_hint

        messages[0]["content"] = (
            system.replace(
                '"lyrics":"original lyrics or empty"}',
                '"lyrics":"original lyrics or empty","structured_caption":"full caption"}',
            )
            + writer_instructions()
        )
        positions = [track_index] if track_index is not None else range(count)
        for position in positions:
            target = TRACK_DURATIONS[position]
            messages[0]["content"] += (
                f" Track {position + 1} target length {int(target) // 60}:"
                f"{round(target) % 60:02} ({target:g} seconds). "
                "Use these section time anchors and section names for the caption and lyrics: "
                + timeline_outline(target, tempo_hint(theme, genre), vocal=vocal, rap=rap)
                + ". "
            )
    failure = ""
    raw = ""
    for attempt in range(3 if generator == "minimax" else 2):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("prompt writing cancelled")
        if attempt:
            messages.append(
                {
                    "role": "user",
                    "content": f"Previous plan failed validation: {failure}. Rewrite the complete "
                    f"{count}-track JSON. Keep the requested genre and vocal intent.",
                }
            )
        try:
            if provider == "ollama":
                raw = _ollama_prompts(
                    base,
                    model,
                    messages,
                    count,
                    cancel,
                    **({"structured": True} if generator == "minimax" else {}),
                )
            else:
                response = httpx.post(
                    f"{base}/chat/completions",
                    timeout=180,
                    headers={"Authorization": f"Bearer {key}"} if key else {},
                    json={
                        "model": model,
                        "max_completion_tokens": 24000 if generator == "minimax" else 12000,
                        "messages": messages,
                    },
                )
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]
            if not vocal:
                repaired, repairs = _repair_instrumental_plan(raw, theme, count, track_index)
                if repairs:
                    report = _record_prompt_attempt(
                        theme, model, raw, repairs=repairs, repaired_response=repaired
                    )
                    LOGGER.info(
                        "Prompt plan repaired instrumental tracks=%d report=%s",
                        len(repairs),
                        report,
                    )
                    raw = repaired
            briefs = _parse_track_plan(
                raw, theme, count, model, generator=generator, track_index=track_index
            )
            LOGGER.info(
                "Prompt plan accepted model=%s genre=%s vocals=%s tracks=%d",
                model,
                genre,
                vocal,
                count,
            )
            return briefs
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            failure = str(exc)
            if isinstance(exc, PromptResponseError):
                raw = exc.response_text
            report = _record_prompt_attempt(theme, model, raw, failure=failure, attempt=attempt + 1)
            LOGGER.warning(
                "Prompt plan rejected attempt=%d model=%s reason=%s report=%s",
                attempt + 1,
                model,
                failure,
                report,
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            raise RuntimeError(f"prompt model {model} failed: {exc}") from exc
    raise RuntimeError(f"prompt model {model} returned an invalid music plan: {failure}")


def plan_theme(
    theme: str,
    *,
    count: int = TRACK_COUNT,
    cancel: threading.Event | None = None,
    generator: str = "yue2",
) -> list[TrackBrief]:
    """Finish the local text plan in a child process before any YuE2 rendering."""
    if not theme.strip() or not 1 <= count <= TRACK_COUNT:
        raise ValueError("theme and a count from 1 to 10 are required")
    planner = os.environ.get("VEXA_MUSIC_PLANNER", "acestep").strip().lower()
    if os.environ.get("VEXA_LLM_PROVIDER") == "none" or planner == "ollama":
        return _plan_remote_theme(theme, count=count, cancel=cancel, generator=generator)
    if planner != "acestep":
        raise ValueError(f"unknown VEXA_MUSIC_PLANNER: {planner}")
    from .background import run_job

    result = run_job(
        "plan",
        {
            "theme": theme,
            "count": count,
            **({"generator": generator} if generator == "minimax" else {}),
        },
        cancel=cancel,
        timeout_s=float(os.environ.get("VEXA_ACE_TIMEOUT_S", "1800")),
    )
    seen_titles = set()
    briefs = []
    for index, row in enumerate(result["tracks"]):
        title = unique_title(
            clean_title(row.get("title")) or fallback_title(theme, row["style"], str(index)),
            seen_titles,
        )
        briefs.append(replace(TrackBrief(**row), title=title))
    if len(briefs) != count:
        raise ValueError("local planner returned an incomplete track list")
    LOGGER.info(
        "Music plan ready origin=%s tracks=%d report=%s; GPU released",
        briefs[0].origin,
        len(briefs),
        result["report"],
    )
    return briefs


def _gpu_index() -> str:
    """Return a stable GPU UUID; CUDA and nvidia-smi index orders differ here."""
    configured = (
        os.environ.get("VEXA_GENERATION_GPU") or os.environ.get("VEXA_YUE_GPU", "")
    ).strip()
    if configured.startswith("GPU-"):
        return configured
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        devices = [
            parts
            for line in result.stdout.splitlines()
            if len(parts := tuple(part.strip() for part in line.split(",", 2))) == 3
        ]
        if configured:
            for index, _name, uuid in devices:
                if index == configured:
                    return uuid
            raise RuntimeError(f"configured VEXA_YUE_GPU={configured} was not found")
        for _index, name, uuid in devices:
            if "RTX 5060 Ti" in name:
                return uuid
    except (OSError, subprocess.TimeoutExpired):
        pass
    raise RuntimeError("RTX 5060 Ti was not found; music generation cannot start")


def style_from_theme(theme: str) -> str:
    """Ask a configured LLM for a production brief; keep a useful local fallback."""
    return plan_theme(theme, count=1)[0].style


def generate_theme(
    theme: str,
    library: Path,
    *,
    duration_s: float = 165.0,
    energy: float = 0.5,
    cancel: threading.Event | None = None,
    style: str | None = None,
    prompt_origin: str = "local",
    lyrics: str = "",
    music_plan: dict | None = None,
    title: str = "",
    generator: str | None = None,
    preset: str | None = None,
    render_id: str | None = None,
    seed: int | None = None,
    render_timeout_s: int | None = None,
) -> DepotAsset:
    """Render through the selected backend and admit only a gated, cueable result."""
    from .generators import native_revision, render_alternative, selection

    generator, preset = selection(generator, preset)
    seed = secrets.randbelow(2**31) if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("render seed must be a nonnegative 31-bit integer")
    render_metadata = {"generator": generator, "preset": preset, "seed": seed}
    backend = ROOT / "third_party/yue2.cpp"
    binary = Path(os.environ.get("VEXA_YUE_BINARY", backend / "build/yue-synth"))
    model = Path(os.environ.get("VEXA_YUE_MODEL", backend / "models/YuE2-3B-Q8_0.gguf"))
    vae = Path(os.environ.get("VEXA_YUE_VAE", backend / "models/YuE2-Vae-F32.gguf"))
    if generator == "yue2":
        render_metadata["runtime_revision"] = native_revision(binary)
        for path in (binary, model, vae):
            if not path.exists():
                raise BackendUnavailable(f"YuE2 component missing: {path}")
    library.mkdir(parents=True, exist_ok=True)
    if render_id is not None and not re.fullmatch(r"live-[a-f0-9]{12}", render_id):
        raise ValueError("invalid reserved render ID")
    asset_id = render_id or f"live-{uuid.uuid4().hex[:12]}"
    wav = library / f"{asset_id}.wav"
    if any(path.exists() for path in (wav, wav.with_suffix(".raw.wav"), wav.with_suffix(".json"))):
        raise ValueError("reserved render ID already exists; refusing to overwrite audio")
    request = library / f"{asset_id}.request.json"
    if style is None:
        brief = plan_theme(theme, count=1, cancel=cancel, generator=generator)[0]
        style = brief.style
        lyrics = lyrics or brief.lyrics
        music_plan = music_plan or brief.music_plan
        title = title or brief.title
    if vocal_intent(theme) is False:
        if lyrics.strip():
            raise ValueError("instrumental generation requires empty lyrics")
        if has_vocal_delivery(style):
            raise ValueError("instrumental generation style contains vocal delivery")
    if generator != "yue2":
        render_metadata = render_alternative(
            generator,
            preset,
            wav,
            request,
            style=style,
            lyrics=lyrics,
            duration_s=duration_s,
            music_plan=music_plan,
            cancel=cancel,
            seed=seed,
            timeout_s=render_timeout_s,
        )
    else:
        # YuE2's default semantic minimum is only 200 frames (eight seconds). A duration
        # request otherwise caps generation without preventing an early end token.
        frames = round(duration_s * 25)
        request.write_text(
            json.dumps(
                {
                    "style": style,
                    "lyrics": lyrics,
                    "cot": "off",
                    "cfg_scale": 1.2,
                    "seed": seed,
                    "duration": duration_s,
                    "semantic_sampling": {
                        "min_tokens": max(200, frames - 25),
                        "max_tokens": frames,
                    },
                }
            ),
            encoding="utf-8",
        )
        cmd = [
            str(binary),
            "--model",
            str(model),
            "--vae",
            str(vae),
            "--request",
            str(request),
            "--out",
            str(wav),
            "--duration",
            str(duration_s),
            "--steps",
            "32",
            "--max-seq",
            "8192",
        ]
        log = wav.with_suffix(".render.log")
        try:
            gpu = _gpu_index()
        except RuntimeError as exc:
            raise BackendUnavailable(str(exc)) from exc
        # GPU kernels retain full access to the selected GPU. CPU helpers must leave room
        # for the playback callback, including BLAS/OpenMP pools inherited from the launcher.
        render_env = {
            **native_environment(binary),
            "CUDA_VISIBLE_DEVICES": gpu,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
        if shutil.which("nice"):
            cmd = ["nice", "-n", "10", *cmd]
        with log.open("w", encoding="utf-8") as output:
            LOGGER.info(
                "YuE2 render start asset=%s duration=%.1fs gpu=%s prompt_origin=%s log=%s",
                asset_id,
                duration_s,
                gpu,
                prompt_origin,
                log,
            )
            try:
                process = subprocess.Popen(
                    cmd, stdout=output, stderr=subprocess.STDOUT, env=render_env
                )
            except OSError as exc:
                raise BackendUnavailable(f"YuE2 could not launch: {exc}") from exc
            deadline = time.monotonic() + (render_timeout_s or max(900.0, duration_s * 15.0))
            try:
                while process.poll() is None:
                    if (cancel is not None and cancel.wait(0.25)) or time.monotonic() > deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise RuntimeError("YuE2 render cancelled or timed out")
                    if cancel is None:
                        time.sleep(0.25)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
        if process.returncode != 0:
            detail = log.read_text()[-1000:].strip()
            error = f"YuE2 exited {process.returncode}: {detail}"
            if process.returncode in (126, 127) or "error while loading shared libraries" in detail:
                raise BackendUnavailable(error)
            raise RuntimeError(error)
        if cancel is not None and cancel.is_set():
            raise RuntimeError("YuE2 render cancelled")
    return admit_render(
        wav,
        theme=theme,
        duration_s=duration_s,
        energy=energy,
        cancel=cancel,
        style=style,
        lyrics=lyrics,
        music_plan=music_plan,
        title=title,
        generator=generator,
        preset=preset,
        seed=seed,
        prompt_origin=prompt_origin,
        render_metadata=render_metadata,
    )


def admit_render(
    wav: Path,
    *,
    theme: str,
    duration_s: float,
    energy: float,
    cancel=None,
    style: str = "",
    lyrics: str = "",
    music_plan: dict | None = None,
    title: str = "",
    generator: str = "yue2",
    preset: str = "q8",
    seed: int = 42,
    prompt_origin: str = "audit",
    render_metadata: dict | None = None,
) -> DepotAsset:
    """Inspect/repair/admit a reserved render. Shared by generation and offline audits."""
    render_metadata = render_metadata or {}
    asset_id = wav.stem
    preferences = (music_plan or {}).get("generation_settings") or {}
    plan_intent = ((music_plan or {}).get("track_plan") or {}).get("intent") or {}
    evidence = EvidenceSession(
        wav,
        duration_s=duration_s,
        requested={
            **plan_intent,
            "energy_target": energy,
            "include_instruments": preferences.get("include_instruments", []),
            "exclude_instruments": preferences.get("exclude_instruments", []),
        },
    )
    try:
        evidence.begin()
        master_report = master(wav)
        evidence.mastered(master_report)
        if cancel is not None and cancel.is_set():
            raise RuntimeError("YuE2 render cancelled")
        rendered_duration = sf.info(str(wav)).duration
        LOGGER.info(
            "Music render completed asset=%s requested=%.1fs actual=%.1fs",
            asset_id,
            duration_s,
            rendered_duration,
        )
        if not duration_s - 2.0 <= rendered_duration <= duration_s + 2.0:
            raise ValueError(
                f"Music generator rendered {rendered_duration:.1f}s; requested {duration_s:.1f}s"
            )
        result = analyse(
            wav,
            asset_id=asset_id,
            family_id=f"family_{asset_id}",
            source_type=SourceType.YUE2_RENDER
            if generator == "yue2"
            else SourceType.GENERATED_RENDER,
            approval=ApprovalState.APPROVED,
            provenance=Provenance(source_prompt=style, seed=seed),
        )
        if not result.admitted or result.manifest is None:
            if result.manifest is not None:
                evidence.body["quality"] = result.manifest.quality.model_dump()
                quality = result.manifest.quality
                evidence.body["admission"]["playable"] = bool(
                    quality.decode_ok
                    and quality.duration_ok
                    and quality.loudness_ok
                    and quality.audio_quality_ok
                )
                evidence.body["admission"]["capabilities"]["full_track_playback"] = evidence.body[
                    "admission"
                ]["playable"]
            evidence.reject(
                ["quality_gate_failed"], detail=str(result.fatal or result.quality.flags)
            )
        manifest = result.manifest
        manifest.title = clean_title(title) or fallback_title(theme, style, asset_id)
        manifest.tags = {
            "generator": [Estimate(value=generator, confidence=1.0)],
            "generator_preset": [Estimate(value=preset, confidence=1.0)],
            "intent_energy": [Estimate(value=f"{energy:.2f}", confidence=1.0)],
            "intent_mood": [Estimate(value=theme[:80], confidence=1.0)],
            "intent_instrumental": [Estimate(value="false" if lyrics else "true", confidence=1.0)],
        }
        evidence.body["quality"] = manifest.quality.model_dump()
        evidence.body["admission"]["playable"] = True
        evidence.body["admission"]["capabilities"]["full_track_playback"] = True
        cue = analyse_cues(wav, expected_bpm=manifest.beat_grid.bpm)
        cue.save(wav.with_suffix(".cues.json"))
        if cue.entry_s is None or cue.exit_s is None:
            last_beat = cue.beat_times_s[-1] if cue.beat_times_s else None
            evidence.reject(
                ["transition_cues_unreliable"],
                detail=(
                    "generated audio passed gates but has no reliable transition cues; "
                    f"beat confidence={cue.beat_confidence}, last beat={last_beat}s, "
                    f"duration={rendered_duration:.1f}s"
                ),
            )
        target_bpm = (music_plan or {}).get("bpm")
        if preferences.get("tempo_policy") == "strict" and target_bpm:
            measured_bpm = manifest.beat_grid.bpm
            error = min(abs(measured_bpm * factor - target_bpm) for factor in (0.5, 1, 2))
            if error > 3:
                evidence.reject(
                    ["strict_tempo_mismatch"],
                    detail=(
                        f"measured tempo {measured_bpm:.1f} BPM differs from strict "
                        f"target {target_bpm:g} BPM (tolerance 3 BPM)"
                    ),
                )
        from .semantic_audio import observe

        observations = observe(wav, evidence.body["requested"], cancel=cancel)
        evidence.body["observations"] = observations
        if cancel is not None and cancel.is_set():
            evidence.body["status"] = "cancelled"
            evidence.save()
            raise RuntimeError("audio analysis cancelled")
        evidence.body["timing"] = {
            "bpm": manifest.beat_grid.bpm,
            "beat_confidence": manifest.beat_grid.confidence,
            "key": manifest.key.model_dump() if manifest.key else None,
            "entry_s": cue.entry_s,
            "exit_s": cue.exit_s,
        }
        evidence.body["admission"].update(
            mixable=True,
            technical_constraints_satisfied=True,
            constraints_satisfied=None,
            capabilities={
                "full_track_playback": True,
                "beat_synchronized_mix": True,
                "stem_mix": False,
                "verified_vocal_safe_overlap": False,
            },
            warnings=[
                "Genre, vocal activity and language are uncalibrated or unavailable; "
                "observation policy."
            ],
        )
        evidence.body["status"] = "admitted"
        evidence.save()
        LOGGER.info(
            "Music admitted asset=%s entry=%.2fs exit=%.2fs bpm=%.2f",
            asset_id,
            cue.entry_s,
            cue.exit_s,
            manifest.beat_grid.bpm,
        )
        wav.with_suffix(".brief.json").write_text(
            json.dumps(
                {
                    **render_metadata,
                    "title": manifest.title,
                    "theme": theme,
                    "style": style,
                    "lyrics": lyrics,
                    "prompt_origin": prompt_origin,
                    "music_plan": music_plan,
                    "requested_duration_s": duration_s,
                    "rendered_duration_s": rendered_duration,
                    "energy": energy,
                    "audio_evidence_version": evidence.body["evidence_version"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        published = wav.with_suffix(".json.tmp")
        published.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
        )
        published.replace(wav.with_suffix(".json"))
        return DepotAsset(manifest, wav, frozenset(theme.lower().split()), {})
    except Exception as exc:
        if evidence.body["status"] not in {"rejected", "cancelled"}:
            evidence.body["status"] = (
                "cancelled" if cancel and cancel.is_set() else "analysis_failed"
            )
            evidence.body["admission"]["reason_codes"] = [evidence.body["status"]]
            evidence.body["admission"]["detail"] = str(exc)
            evidence.save()
        raise


def isolated_generate_theme(theme: str, library: Path, *, cancel=None, **kwargs) -> DepotAsset:
    """Keep render mastering, librosa/JIT analysis and metadata building off the audio host."""
    from vexa_contracts import AssetManifest

    from .background import run_job

    body = run_job(
        "generate",
        {"theme": theme, "library": str(library), **kwargs},
        cancel=cancel,
        timeout_s=kwargs.get("render_timeout_s") or max(2000, kwargs.get("duration_s", 165) * 32),
    )
    manifest = AssetManifest.model_validate(body["manifest"])
    return DepotAsset(manifest, Path(body["path"]), frozenset(theme.lower().split()), {})
