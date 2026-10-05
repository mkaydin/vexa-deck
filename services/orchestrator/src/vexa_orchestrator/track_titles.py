"""Presentation titles; audio identity, filenames and measured tags stay independent."""

from __future__ import annotations

import hashlib
import re
import unicodedata

MAX_TITLE = 80


def clean_title(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value)
    value = "".join(char for char in value if not unicodedata.category(char).startswith("C"))
    value = re.sub(r"\s+", " ", value).strip(" \"'`#*")
    if not value or re.fullmatch(r"(?:untitled|track|song)(?:[\s#_-]*\d+)?", value, re.I):
        return ""
    if len(value) > MAX_TITLE:
        shortened = value[:MAX_TITLE]
        value = shortened.rsplit(" ", 1)[0] if " " in shortened else shortened
    return value


def fallback_title(theme: str, style: str, key: str) -> str:
    """Stable names for old tracks and writers that omit a title. No model or file I/O."""
    text = f"{theme} {style}".lower()
    prefixes = (
        "Neon",
        "Midnight",
        "Hidden",
        "Afterhours",
        "Silent",
        "Electric",
        "Distant",
        "Velvet",
    )
    if re.search(r"phonk|drift|jdm", text):
        images = (
            "Redline",
            "Asphalt",
            "Chrome",
            "Tunnel",
            "Nitro",
            "Overpass",
            "Throttle",
            "Street",
        )
    elif re.search(r"rap|hip.?hop|boom.?bap", text):
        images = (
            "Concrete",
            "Vinyl",
            "Basement",
            "Block",
            "Sidewalk",
            "Tape",
            "Backstreet",
            "Rooftop",
        )
    elif "jazz" in text:
        images = ("Blue", "Rain", "Brass", "Smoke", "Piano", "Lounge", "Window", "Boulevard")
    elif "techno" in text:
        images = (
            "Circuit",
            "Signal",
            "Machine",
            "Pulse",
            "Voltage",
            "Warehouse",
            "Grid",
            "Frequency",
        )
    elif "house" in text:
        images = ("Disco", "Groove", "Sunrise", "Bassline", "Soul", "Starlight", "Club", "Terrace")
    else:
        images = ("Skyline", "Moon", "Shadow", "City", "Horizon", "Dream", "Orbit", "Echo")
    endings = (
        "Run",
        "Mirage",
        "Memory",
        "Drift",
        "Ritual",
        "Bloom",
        "Passage",
        "Reverie",
        "Motion",
        "Trace",
        "Voyage",
        "Current",
        "Return",
        "Glow",
        "Escape",
        "Reflections",
    )
    digest = hashlib.sha256(f"{theme}|{style}|{key}".encode()).digest()
    return (
        f"{prefixes[digest[0] % len(prefixes)]} {images[digest[1] % len(images)]} "
        f"{endings[digest[2] % len(endings)]}"
    )


def unique_title(title: str, seen: set[str]) -> str:
    candidate = title
    index = 2
    while candidate.casefold() in seen:
        suffix = f" / {index:02d}"
        candidate = title[: MAX_TITLE - len(suffix)].rstrip() + suffix
        index += 1
    seen.add(candidate.casefold())
    return candidate


def title_for_asset(asset) -> str | None:
    if asset is None:
        return None
    manifest = asset.manifest
    title = clean_title(getattr(manifest, "title", None))
    if title:
        return title
    identity = manifest.asset_id
    if not identity.startswith("live-"):
        return identity.replace("-", " ").replace("_", " ")
    tags = getattr(manifest, "tags", {})
    moods = tags.get("mood", [])
    theme = moods[0].value if moods else ""
    style = getattr(getattr(manifest, "provenance", None), "source_prompt", "") or ""
    return fallback_title(theme, style, identity)


def hydrate_title(manifest, path):
    """Read title sidecars only when admitting/reloading an asset, never during playback."""
    import json

    title = clean_title(manifest.title)
    if not title and manifest.asset_id.startswith("live-"):
        try:
            brief = json.loads(path.with_suffix(".brief.json").read_text())
        except (OSError, ValueError):
            brief = {}
        if not isinstance(brief, dict):
            brief = {}
        title = clean_title(brief.get("title")) or fallback_title(
            str(brief.get("theme") or ""),
            manifest.provenance.source_prompt or "",
            manifest.asset_id,
        )
    if title:
        manifest.title = title
