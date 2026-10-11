"""Music 3 caption contract; lyrics stay separate from production instructions."""

from __future__ import annotations

import re
from statistics import median

from .vocal_intent import has_vocal_delivery

FORMAT = "minimax-structured-v1"
HEADINGS = ("Global Metadata", "Vocal Details", "Arrangement")


def normalize_lyrics(lyrics: str) -> str:
    """Keep lyric words intact while standardizing section tags from the writer."""
    return re.sub(
        r"(?im)^[ \t]*\((intro|verse(?: \d+)?|pre[ -]chorus|chorus|refrain|bridge|"
        r"instrumental(?: break)?|solo|outro)\)[ \t]*",
        lambda match: f"[{match[1]}]\n",
        lyrics,
    )


def normalize_caption(text: str) -> str:
    text = re.sub(
        r"(?m)^\s*(?:#{1,3}\s*)?(?:\*\*)?"
        r"(Global Metadata|Vocal Details|Arrangement)(?:\*\*)?\s*:\s*",
        r"\1\n",
        text.strip(),
    )
    text = re.sub(
        r"(?m)^\s*(?:#{1,3}\s*)?(?:\*\*)?"
        r"(Global Metadata|Vocal Details|Arrangement)(?:\*\*)?\s*:?\s*$",
        r"\1",
        text.strip(),
    )
    # Some writers echo the template's M prefix before an actual clock value.
    text = re.sub(r"\bM:(\d{1,2}:\d{2})\b", r"\1", text)
    return re.sub(r"\bM:(\d{2})\b", r"0:\1", text)


def validate_caption(
    text: str,
    *,
    style: str,
    lyrics: str,
    genre: str,
    vocal: bool,
    duration_s: float | None = None,
) -> str:
    if not isinstance(text, str):
        raise ValueError("MiniMax caption must be text")
    text = normalize_caption(text)
    if not 500 <= len(text) <= 8000:
        raise ValueError("MiniMax needs a detailed structured caption (500-8000 characters)")
    found = re.findall(r"(?m)^(Global Metadata|Vocal Details|Arrangement)\s*$", text)
    if tuple(found) != HEADINGS or not text.startswith("Global Metadata\n"):
        raise ValueError(
            "MiniMax caption needs Global Metadata, Vocal Details and Arrangement in order"
        )
    lower = text.lower()
    tempo = re.search(r"\b(\d{2,3})\s*bpm\b", style, re.I)
    if tempo and not re.search(
        rf"\b{tempo[1]}\s*bpm\b|\bbpm\s*(?:is|:)?\s*{tempo[1]}\b", text, re.I
    ):
        raise ValueError("MiniMax caption lost the requested BPM")
    if "phonk" in genre.lower() and "phonk" not in lower:
        raise ValueError("MiniMax caption lost the phonk genre")
    if genre == "Memphis phonk" and (
        "memphis" not in lower
        or re.search(r"\bdrift[ -]+phonk\b|\bjdm\b|phonk\s+house|four.on.the.floor", lower)
    ):
        raise ValueError("MiniMax caption changed the Memphis subgenre")
    if "hip hop" in genre and not re.search(r"hip[ -]?hop|\brap\b|boom[ -]?bap", lower):
        raise ValueError("MiniMax caption lost the hip-hop genre")
    if not vocal:
        # Headings describe the schema, not a request for a singer.
        body = text.replace("\nVocal Details\n", "\n")
        if "instrumental" not in lower or has_vocal_delivery(body):
            raise ValueError("MiniMax instrumental caption introduced vocal delivery")
    tags = re.findall(r"\[([^\]\n]+)\]", lyrics)
    timeline = text.split("\nArrangement\n", 1)[1]
    if duration_s is not None:
        ranges = re.findall(r"(\d{1,2}:\d{2})\s*(?:-|\u2013|\u2014|to)\s*(\d{1,2}:\d{2})", timeline)

        def seconds(value):
            minutes, seconds_ = map(int, value.split(":"))
            if seconds_ >= 60:
                raise ValueError("MiniMax timeline has an invalid timecode")
            return minutes * 60 + seconds_

        if not ranges:
            raise ValueError("MiniMax arrangement needs chronological M:SS-M:SS section ranges")
        cursor = 0
        for start, end in ranges:
            start_s, end_s = seconds(start), seconds(end)
            if abs(start_s - cursor) > 2 or end_s <= start_s:
                raise ValueError("MiniMax timeline has gaps, overlaps or reversed sections")
            cursor = end_s
        if abs(cursor - duration_s) > 2:
            raise ValueError(
                f"MiniMax timeline ends at {cursor}s; required {duration_s:g}s. "
                "Rewrite time ranges to cover the full target duration."
            )
    canonical = re.sub(r"[^a-z0-9]", "", timeline.lower())
    for tag in tags:
        name = tag.split(":", 1)[0].split("\u2013", 1)[0].strip()
        if re.sub(r"[^a-z0-9]", "", name.lower()) not in canonical:
            raise ValueError(f"MiniMax arrangement omitted lyric section: {name}")
    return text


def writer_instructions() -> str:
    return (
        "This target is MiniMax Music 3.0. In addition to the short style and separate lyrics, "
        "return structured_caption for each track, approximately 250-450 English words. "
        "Use exactly three plain headings separated by newlines: Global Metadata, Vocal Details, "
        "Arrangement. Global Metadata describes the exact genre/subgenre, groove, tempo, emotional "
        "progression, production texture and stereo space. Use an exact tempo/key only when "
        "specified or supported by the musical plan; otherwise describe them conservatively. "
        "Vocal Details describes the requested lead, timbre, delivery, backing and effects; "
        "for instrumental tracks state Instrumental and name the instrumental lead instead. "
        "Arrangement must describe the sections chronologically: specify which instruments enter, "
        "recede or vary, how drums/bass and energy change, and how transitions connect. "
        "Every bracketed lyric section must appear in its matching arrangement section. "
        "Use square brackets for lyric section tags, on their own lines. "
        "Write each section once; combine its instrument changes in the same paragraph. "
        "Prefix each section with a plain clock range such as 0:00-0:15. Copy the supplied "
        "time anchors literally; do not add M: prefixes or split seconds digits with colons. "
        "Start at 0:00 and "
        "end at the exact target duration, with no gaps or overlapping time ranges. "
        "Keep percussion or drums active during both DJ intro and outro. Do not make "
        "the opening silent or drop every rhythm layer before the final bar. "
        "Keep the sparse rhythmic DJ intro/outro and full requested duration. Preserve explicit "
        "user instruments, exclusions, era, subgenre and vocal configuration. Keep instruments "
        "consistent across the song. Do not default every genre to club drops or pop choruses. "
        "Do not include a track title, lyric lines, copied examples, artist names or reasoning "
        "inside structured_caption. The lyrics field remains the complete original lyrics or "
        "empty for instrumental music. Use concrete musical changes rather than decorative prose. "
    )


def fallback_caption(style: str, lyrics: str, metadata: dict, duration_s: float) -> str:
    """Expand legacy briefs offline without inventing a key, vocalist or instrument."""
    from .music_plan import arrangement

    tempo = metadata.get("bpm")
    basic = f"Target duration {duration_s:g} seconds. "
    if tempo:
        basic += f"Tempo target {tempo:g} BPM. "
    if metadata.get("key"):
        basic += f"Key: {metadata['key']}. "
    form = metadata.get("sections")
    if not form:
        form = arrangement(
            duration_s,
            float(tempo or 110),
            vocal=bool(lyrics),
            rap=bool(re.search(r"hip[ -]?hop|\brap\b", style, re.I)),
        )
    changes = (
        "Introduce the defining instrumental motif with sparse drums; leave room for mixing.",
        "Bring in the main bass and full groove while retaining the established motif.",
        "Vary drum articulation and supporting layers; keep the core instruments identifiable.",
        "Build density through the existing layers, then thin the arrangement for contrast.",
        "Return to the main groove with a changed supporting response and clear lead.",
        "Resolve the developing motif and reduce layers smoothly.",
    )
    timeline = []
    for index, section in enumerate(form):
        change = (
            "Withdraw supporting layers gradually; keep a sparse instrumental drum/bass "
            "foundation active until the end for the next DJ transition."
            if index == len(form) - 1
            else changes[min(index, len(changes) - 1)]
        )
        timeline.append(f"{section['section']} ({section['bars']} bars): {change}")
    voice = (
        "Keep the vocal configuration and delivery specified in the brief. Perform the "
        "separate supplied lyrics and follow their section tags. Keep the instrumental "
        "DJ intro/outro free of the lead and use restrained space around the voice."
        if lyrics.strip()
        else "Instrumental. The described primary instrumental motif carries the lead. "
        "No vocals, singing, speech, humming or vocal samples."
    )
    return (
        "Global Metadata\n" + basic + style + "\n"
        "Keep the requested sonic texture and groove consistent. Develop the mood through "
        "changing layer density and articulation rather than changing the genre.\n"
        "Vocal Details\n" + voice + "\nArrangement\n" + "\n".join(timeline)
    )


def retime_caption(text: str, duration_s: float) -> tuple[str, bool]:
    """Own caption timing in code, preserving musical section order and descriptions.

    Adjacent notes for the same named interval belong to one section. Other timing
    mistakes are repaired from section duration weights, in the writer's text order.
    Genre, vocal and lyric validation still runs on the resulting complete caption.
    This edits conditioning text only; it never stretches or repeats rendered audio.
    """
    text = normalize_caption(text)
    if "\nArrangement\n" not in text:
        return text, False
    prefix, timeline = text.split("\nArrangement\n", 1)
    clock = r"\d{1,2}(?::\d{1,2}){1,2}"
    pattern = (rf"(?<![\w:])(?:M:)?({clock})\s*(?:-|\u2013|\u2014|to)\s*"
               rf"(?:M:)?({clock})(?!\d)")
    ranges = list(re.finditer(pattern, timeline))
    if not ranges:
        return text, False

    def seconds(value):
        pieces = value.split(":")
        # Gemma sometimes splits the seconds' digits: 1:4:7 means 1:47.
        return int(pieces[0]) * 60 + int("".join(pieces[1:]))

    sections = []
    malformed = False
    for index, match in enumerate(ranges):
        malformed |= any(
            not re.fullmatch(r"\d{1,2}:[0-5]\d", value) for value in (match[1], match[2])
        )
        start, end = seconds(match[1]), seconds(match[2])
        stop = ranges[index + 1].start() if index + 1 < len(ranges) else len(timeline)
        body = timeline[match.end():stop].strip().lstrip(":").lstrip()
        label = re.match(r"([A-Za-z][A-Za-z0-9 /&()_-]{0,70}):\s*", body)
        if not label:
            return text, False  # No reliable section boundary: let the writer correct it.
        name = label[1].strip()
        description = body[label.end():].strip()
        if sections and (start, end, name.casefold()) == (
            sections[-1]["start"], sections[-1]["end"], sections[-1]["name"].casefold()
        ):
            sections[-1]["description"] += " " + description
        else:
            sections.append({"start": start, "end": end, "name": name,
                             "description": description})
    cursor = 0
    continuous = True
    for section in sections:
        continuous &= section["start"] == cursor and section["end"] > section["start"]
        cursor = section["end"]
    if (continuous and not malformed and len(sections) == len(ranges)
            and abs(cursor - duration_s) <= 1):
        return text, False

    positive = [s["end"] - s["start"] for s in sections if s["end"] > s["start"]]
    default_weight = median(positive) if positive else 1
    weights = [s["end"] - s["start"] if s["end"] > s["start"] else default_weight
               for s in sections]
    scale = duration_s / sum(weights)

    def timecode(value):
        value = round(value)
        return f"{value // 60}:{value % 60:02}"

    timed = []
    cursor = 0
    for section, weight in zip(sections, weights, strict=True):
        end = cursor + weight * scale
        timed.append(f"{timecode(cursor)}-{timecode(end)} {section['name']}: "
                     f"{section['description']}")
        cursor = end
    lead = timeline[:ranges[0].start()]
    return prefix + "\nArrangement\n" + lead + "\n".join(timed), True


def timeline_outline(duration_s: float, bpm: float, *, vocal: bool, rap: bool) -> str:
    from .music_plan import arrangement

    form = arrangement(duration_s, bpm, vocal=vocal, rap=rap)
    scale = duration_s / form[-1]["end_s"]

    def clock(seconds):
        seconds = round(seconds * scale)
        return f"{seconds // 60}:{seconds % 60:02}"

    return "; ".join(
        f"{clock(row['start_s'])}-{clock(row['end_s'])} {row['section']}" for row in form
    )
