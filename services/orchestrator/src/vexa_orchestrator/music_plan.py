"""Text-only musical intent and a duration-aware arrangement for YuE2 conditioning."""

from __future__ import annotations

import math
import re


def vocal_language(theme: str) -> str:
    """Respect explicit language; English is the established default for vocal briefs."""
    for code, pattern in (
        ("tr", r"türkçe|turkish"),
        ("zh", r"chinese|mandarin|çince"),
        ("ja", r"japanese|japonca"),
        ("ko", r"korean|korece"),
        ("es", r"spanish|español|ispanyolca"),
        ("fr", r"french|français|frans\u0131zca"),
        ("de", r"german|deutsch|almanca"),
        ("en", r"english|ingilizce"),
    ):
        if re.search(pattern, theme, re.I):
            return code
    return "en"


def arrangement(duration_s: float, bpm: float, *, vocal: bool, rap: bool) -> list[dict]:
    # Quantize to complete 4/4 bars. The renderer still receives the exact target seconds.
    bars = max(20, round(duration_s * bpm / 240))
    mix_bars = 8 if bars >= 40 else 4
    interior = bars - 2 * mix_bars  # Eight bars of rhythmic intro and outro remain mixable.
    if vocal:
        hook = min(4 if rap else 8, max(2, interior // 5))
        bridge = min(4, max(2, interior // 6))
        verse1 = (interior - hook * 2 - bridge) // 2
        sections = [
            ("Intro", mix_bars),
            ("Verse 1", verse1),
            ("Refrain", hook),
            ("Instrumental Break", bridge),
            ("Verse 2", interior - verse1 - hook * 2 - bridge),
            ("Refrain", hook),
            ("Outro", mix_bars),
        ]
    else:
        first = interior // 3
        sections = [
            ("Intro", mix_bars),
            ("Groove", first),
            ("Development", first),
            ("Peak and Release", interior - 2 * first),
            ("Outro", mix_bars),
        ]
    cursor = 0
    result = []
    for name, length in sections:
        result.append(
            {
                "section": name,
                "bars": length,
                "start_s": round(cursor * 240 / bpm, 2),
                "end_s": round((cursor + length) * 240 / bpm, 2),
            }
        )
        cursor += length
    return result


def tempo_hint(theme: str, genre: str) -> float:
    explicit = re.search(r"\b(\d{2,3})\s*bpm\b", theme, re.I)
    if explicit:
        bpm = int(explicit[1])
        if not 40 <= bpm <= 220:
            raise ValueError("requested tempo must be between 40 and 220 BPM")
        return bpm
    for pattern, bpm in (
        (r"phonk", 150),
        (r"trap", 140),
        (r"hip hop|boom bap|rap", 92),
        (r"drum.?and.?bass|\bdnb\b|jungle", 172),
        (r"techno", 130),
        (r"house", 122),
        (r"ambient", 70),
        (r"jazz|lo.?fi|chill", 85),
    ):
        if re.search(pattern, genre, re.I):
            return bpm
    return 110


def validate_metadata(metadata: dict, theme: str, genre: str) -> dict:
    """These are composition targets, never measured features of rendered audio."""
    try:
        bpm = float(metadata["bpm"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("planner did not produce a valid BPM") from exc
    if not math.isfinite(bpm) or not 40 <= bpm <= 220:
        raise ValueError("planner BPM is outside 40-220")
    explicit = re.search(r"\b(\d{2,3})\s*bpm\b", theme, re.I)
    if explicit and abs(bpm - int(explicit[1])) > 1:
        raise ValueError("planner changed the requested tempo")
    if "boom bap" in genre and not 75 <= bpm <= 110:
        raise ValueError("boom bap plan lost its tempo range")
    key = str(metadata.get("keyscale", "")).strip()
    if not re.fullmatch(r"[A-G](?:#|b|♯|♭)?\s+(?:major|minor)", key, re.I):
        raise ValueError("planner did not produce a valid musical key")
    signature = str(metadata.get("timesignature", "4")).strip()
    if signature not in ("4", "4/4"):
        raise ValueError("current DJ arrangement requires 4/4")
    return {"bpm": bpm, "key": key, "time_signature": "4/4", "measured": False}


def track_query(
    theme: str,
    *,
    genre: str,
    palette: str,
    vocal: bool,
    duration_s: float,
    energy: float,
    index: int,
    role: str,
) -> str:
    rap = "hip hop" in genre
    bpm = tempo_hint(theme, genre)
    sections = arrangement(duration_s, bpm, vocal=vocal, rap=rap)
    form = ", ".join(f"{row['section']} {row['bars']} bars" for row in sections)
    focus = (
        "a concrete setting",
        "the narrator's conflict",
        "a vivid memory",
        "an obstacle and response",
        "a change of perspective",
        "a decisive moment",
        "the emotional climax",
        "its consequences",
        "a quieter reflection",
        "a resolution",
    )[index]
    voice = (
        "Dry rhythmic spoken rap by an MC, percussive phrasing and internal rhymes; "
        "two distinct 16-line verses and two written-out 4-line spoken refrains. "
        "At least 40 original lyric lines. Each line fits one bar; no singing."
        if rap and vocal
        else "Original sung lyrics, two developed verses and written-out refrains."
        if vocal
        else "Instrumental; do not write lyrics or introduce voices."
    )
    narrative = (
        "Interpret the listener's theme in the imagery, narrator and emotional tone; "
        f"focus on {focus}. "
        "When the theme only names a genre, choose an original genre-appropriate story. "
        "Use concrete images and a coherent narrative, not generic party/love slogans. "
        "Each verse develops the story and the refrain states its central idea. "
        if vocal else
        "Express the theme through instrumental timbre, harmony, groove and developing motifs. "
        "Do not introduce a singer, MC, vocal samples, spoken words or humming. "
    )
    return (
        f"Listener's exact theme: {theme.strip()}\n"
        f"Track {index + 1}: {role}; energy {energy:.2f}. Preserve genre: {genre}. "
        f"Instrument palette: {palette}. Target {bpm:g} BPM, 4/4, {duration_s:g} seconds. "
        f"Arrangement: {form}. {voice} "
        f"Vocal language: {vocal_language(theme) if vocal else 'unknown'}. {narrative}"
        "Give the track a distinct sample/motif, bass pattern and drum development. "
        "Use only era-appropriate instruments. Keep the rhythm active to the end, "
        "with sparse instrumental DJ intro and outro. "
        + ("Write metadata then complete lyrics." if vocal else "Write musical metadata only.")
    )


def lyric_lines(lyrics: str) -> list[str]:
    """Stage directions do not count as sung/spoken lines toward a full song."""
    return [
        line.strip()
        for line in lyrics.splitlines()
        if line.strip() and not line.strip().startswith(("[", "(", "#"))
    ]


def validate_lyric_theme(theme: str, lyrics: str) -> None:
    """Catch obvious narrative drift for explicit 'about …' topics; not a semantic score."""
    topic = re.search(r"\babout\s+(.+)", theme, re.I)
    if not topic:
        return
    stopwords = {
        "about",
        "with",
        "that",
        "this",
        "from",
        "through",
        "their",
        "into",
        "after",
        "before",
        "while",
        "returning",
        "some",
        "very",
        "being",
    }
    words = {
        word[:4] for word in re.findall(r"[a-z]{4,}", topic[1].lower()) if word not in stopwords
    }
    if not words:
        return
    matching = sum(stem in "\n".join(lyric_lines(lyrics)).lower() for stem in words)
    minimum = 2 if len(words) >= 4 else 1
    if matching < minimum:
        raise ValueError("lyrics lost the listener's explicit narrative topic")
