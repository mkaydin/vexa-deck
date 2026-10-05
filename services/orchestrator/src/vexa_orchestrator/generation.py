"""Generate a themed YuE2 track, gate it, and add it to the local depot."""

from __future__ import annotations

import json
import logging
import os
import re
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
from vexa_contracts import ApprovalState, Estimate, Provenance
from vexa_yue2.backends import BackendUnavailable
from vexa_yue2.gates import analyse
from vexa_yue2.master import master
from vexa_yue2.runtime import native_environment

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
    if not model.endswith(":cloud"):
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
    chunks: list[str] = []
    text_chars = 0
    progress_at = time.monotonic()
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
                chunks.append(chunk)
                text_chars += len(chunk)
                if time.monotonic() - progress_at > 15:
                    progress_at = time.monotonic()
                    LOGGER.info("Prompt writer progress model=%s text_chars=%d", model, text_chars)
                if body.get("done"):
                    if body.get("done_reason") == "length":
                        raise ValueError("prompt response was truncated")
                    break
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


def _parse_track_plan(raw: str, theme: str, count: int, model: str) -> list[TrackBrief]:
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
        if not isinstance(row, (dict, str)):
            raise ValueError(f"track {index + 1}: expected a production brief object")
        style = row if isinstance(row, str) else row.get("style", "")
        lyrics = "" if isinstance(row, str) else row.get("lyrics", "")
        if not isinstance(style, str) or not 30 <= len(style.strip()) <= 1800:
            raise ValueError(f"track {index + 1}: invalid style length")
        if not isinstance(lyrics, str) or len(lyrics) > 5000:
            raise ValueError(f"track {index + 1}: invalid lyrics")
        lowered = style.lower()
        if lowered in seen:
            raise ValueError("duplicate track prompts")
        seen.add(lowered)
        if "phonk" in genre.lower() and not re.search(r"\bphonk\b", lowered):
            raise ValueError(f"track {index + 1}: missing requested phonk genre")
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
        briefs.append(
            TrackBrief(
                style, TRACK_DURATIONS[index], ENERGY_ARC[index], f"llm:{model}", lyrics.strip(),
                title=unique_title(
                    (clean_title(row.get("title")) if isinstance(row, dict) else "")
                    or fallback_title(theme, style, str(index)), seen_titles
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
            tracks[index] = {**(row if isinstance(row, dict) else {}),
                             "style": style, "lyrics": ""}
            repairs.append({"track": index + 1, "removed_sentences": removed,
                            "cleared_lyrics": bool(lyrics.strip())})
    return json.dumps(body, ensure_ascii=False) if repairs else raw, repairs


def _record_prompt_attempt(theme: str, model: str, raw: str, **details) -> Path:
    """Keep the exact failed/repaired response so a generic error can be diagnosed."""
    path = ROOT / "var/plans/raw" / f"writer-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"theme": theme, "model": model, "response": raw, **details},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _plan_remote_theme(
    theme: str,
    *,
    count: int = TRACK_COUNT,
    cancel: threading.Event | None = None,
    track_index: int | None = None,
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
            " This is instrumental phonk: perform rhythmic hooks with cowbells, synths "
            "and percussion. Replace Memphis rap samples and vocal chops with instrumental "
            "riffs. Describe only the instruments that will actually play."
        )
    story = (
        "For vocal tracks, choose a concrete setting, narrator and conflict, "
        "and a story that develops across verses; avoid interchangeable party/love slogans. "
        if vocal else
        "For instrumental tracks, express the theme through timbre, groove, harmony and "
        "evolving motifs. Describe instrumental sections only; no singer, MC, spoken words, "
        "humming or vocal samples. Keep lyrics empty. "
    )
    system = (
        f"You are Vexa's music producer. Write exactly {count} distinct YuE2 production briefs. "
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
                if vocal else
                "bass pattern, instrumental phrasing and evolving arrangement. "
                "Do not write lyrics, vocal delivery or verse/refrain directions."
            )
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": theme.strip()}]
    failure = ""
    raw = ""
    for attempt in range(2):
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
                raw = _ollama_prompts(base, model, messages, count, cancel)
            else:
                response = httpx.post(
                    f"{base}/chat/completions",
                    timeout=180,
                    headers={"Authorization": f"Bearer {key}"} if key else {},
                    json={"model": model, "max_completion_tokens": 12000, "messages": messages},
                )
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]
            if not vocal:
                repaired, repairs = _repair_instrumental_plan(raw, theme, count, track_index)
                if repairs:
                    report = _record_prompt_attempt(theme, model, raw, repairs=repairs,
                                                    repaired_response=repaired)
                    LOGGER.info("Prompt plan repaired instrumental tracks=%d report=%s",
                                len(repairs), report)
                    raw = repaired
            briefs = _parse_track_plan(raw, theme, count, model)
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
            report = _record_prompt_attempt(theme, model, raw, failure=failure, attempt=attempt + 1)
            LOGGER.warning(
                "Prompt plan rejected attempt=%d model=%s reason=%s report=%s",
                attempt + 1, model, failure, report,
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            raise RuntimeError(f"prompt model {model} failed: {exc}") from exc
    raise RuntimeError(f"prompt model {model} returned an invalid music plan: {failure}")


def plan_theme(
    theme: str, *, count: int = TRACK_COUNT, cancel: threading.Event | None = None
) -> list[TrackBrief]:
    """Finish the local text plan in a child process before any YuE2 rendering."""
    if not theme.strip() or not 1 <= count <= TRACK_COUNT:
        raise ValueError("theme and a count from 1 to 10 are required")
    planner = os.environ.get("VEXA_MUSIC_PLANNER", "acestep").strip().lower()
    if os.environ.get("VEXA_LLM_PROVIDER") == "none" or planner == "ollama":
        return _plan_remote_theme(theme, count=count, cancel=cancel)
    if planner != "acestep":
        raise ValueError(f"unknown VEXA_MUSIC_PLANNER: {planner}")
    from .background import run_job

    result = run_job(
        "plan",
        {"theme": theme, "count": count},
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
    configured = os.environ.get("VEXA_YUE_GPU", "").strip()
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
    raise RuntimeError("RTX 5060 Ti was not found; YuE2 generation cannot start")


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
) -> DepotAsset:
    """Run the host yue-synth executable and admit only a gated, cueable result."""
    backend = ROOT / "third_party/yue2.cpp"
    binary = Path(os.environ.get("VEXA_YUE_BINARY", backend / "build/yue-synth"))
    model = Path(os.environ.get("VEXA_YUE_MODEL", backend / "models/YuE2-3B-Q8_0.gguf"))
    vae = Path(os.environ.get("VEXA_YUE_VAE", backend / "models/YuE2-Vae-F32.gguf"))
    for path in (binary, model, vae):
        if not path.exists():
            raise BackendUnavailable(f"YuE2 component missing: {path}")
    library.mkdir(parents=True, exist_ok=True)
    asset_id = f"live-{uuid.uuid4().hex[:12]}"
    wav = library / f"{asset_id}.wav"
    request = library / f"{asset_id}.request.json"
    if style is None:
        brief = plan_theme(theme, count=1, cancel=cancel)[0]
        style = brief.style
        lyrics = lyrics or brief.lyrics
        music_plan = music_plan or brief.music_plan
        title = title or brief.title
    if vocal_intent(theme) is False:
        if lyrics.strip():
            raise ValueError("instrumental generation requires empty lyrics")
        if has_vocal_delivery(style):
            raise ValueError("instrumental generation style contains vocal delivery")
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
                "seed": -1,
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
            process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=render_env)
        except OSError as exc:
            raise BackendUnavailable(f"YuE2 could not launch: {exc}") from exc
        deadline = time.monotonic() + max(900.0, duration_s * 15.0)
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
    master(wav)
    if cancel is not None and cancel.is_set():
        raise RuntimeError("YuE2 render cancelled")
    rendered_duration = sf.info(str(wav)).duration
    LOGGER.info(
        "YuE2 render completed asset=%s requested=%.1fs actual=%.1fs",
        asset_id,
        duration_s,
        rendered_duration,
    )
    if not duration_s - 2.0 <= rendered_duration <= duration_s + 2.0:
        raise ValueError(f"YuE2 rendered {rendered_duration:.1f}s; requested {duration_s:.1f}s")
    result = analyse(
        wav,
        asset_id=asset_id,
        family_id=f"family_{asset_id}",
        approval=ApprovalState.APPROVED,
        provenance=Provenance(source_prompt=style),
    )
    if not result.admitted or result.manifest is None:
        raise ValueError(
            f"generated audio failed quality gates: {result.fatal or result.quality.flags}"
        )
    manifest = result.manifest
    manifest.title = clean_title(title) or fallback_title(theme, style, asset_id)
    manifest.tags = {
        "energy": [Estimate(value=f"{energy:.2f}", confidence=0.6)],
        "mood": [Estimate(value=theme[:80], confidence=1.0)],
        "instrumental": [Estimate(value="false" if lyrics else "true", confidence=0.8)],
    }
    cue = analyse_cues(wav, expected_bpm=manifest.beat_grid.bpm)
    cue.save(wav.with_suffix(".cues.json"))
    if cue.entry_s is None or cue.exit_s is None:
        last_beat = cue.beat_times_s[-1] if cue.beat_times_s else None
        raise ValueError(
            "generated audio passed gates but has no reliable transition cues; "
            f"beat confidence={cue.beat_confidence}, last beat={last_beat}s, "
            f"duration={rendered_duration:.1f}s"
        )
    LOGGER.info(
        "YuE2 admitted asset=%s entry=%.2fs exit=%.2fs bpm=%.2f",
        asset_id,
        cue.entry_s,
        cue.exit_s,
        manifest.beat_grid.bpm,
    )
    wav.with_suffix(".brief.json").write_text(
        json.dumps(
            {
                "title": manifest.title,
                "theme": theme,
                "style": style,
                "lyrics": lyrics,
                "prompt_origin": prompt_origin,
                "music_plan": music_plan,
                "requested_duration_s": duration_s,
                "rendered_duration_s": rendered_duration,
                "energy": energy,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    wav.with_suffix(".json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    return DepotAsset(manifest, wav, frozenset(theme.lower().split()), {})


def isolated_generate_theme(theme: str, library: Path, *, cancel=None, **kwargs) -> DepotAsset:
    """Keep render mastering, librosa/JIT analysis and metadata building off the audio host."""
    from vexa_contracts import AssetManifest

    from .background import run_job

    body = run_job(
        "generate",
        {"theme": theme, "library": str(library), **kwargs},
        cancel=cancel,
        timeout_s=max(1000, kwargs.get("duration_s", 165) * 16),
    )
    manifest = AssetManifest.model_validate(body["manifest"])
    return DepotAsset(manifest, Path(body["path"]), frozenset(theme.lower().split()), {})
