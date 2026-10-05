"""Shared vocal intent for listener requests, generation and depot retrieval."""

from __future__ import annotations

import re

# Negations must be consumed before looking for words such as "lyrics" or "singing".
_SILENT = re.compile(
    r"\b(?:no|without|zero)\s+(?:any\s+)?(?:lyrics?|vocals?|singing|voice|voices)\b"
    r"|\b(?:lyric|lyrics|vocal|vocals)[ -]free\b"
    r"|\b(?:vokalsiz|sözsüz|soz[sş]uz|wordless)\b",
    re.I,
)
_INSTRUMENTAL = re.compile(r"\binstrumental\b(?!\s+(?:break|intro|outro|section)\b)", re.I)
_NOT_INSTRUMENTAL = re.compile(r"\b(?:not|non)[ -]instrumental\b", re.I)
_VOCAL = re.compile(
    r"\b(?:vocals?|singing|singer|lyrics?|song|vokal|sözlü|sozlu|şark\u0131|sarki)\b", re.I
)
_VOICE_TERM = (
    r"(?:lyrics?|vocals?|vocalists?|singers?|singing|sung(?:\s+(?:choruses?|melodies?))?"
    r"|rapping|humming|chants?|chanting|voices?|spoken\s+(?:words|verses?|refrains?)"
    r"|rap\s+(?:verses?|refrains?|delivery|samples?|chops?)|MC\s+delivery|ad[ -]?libs?"
    r"|vocali[sz](?:ing|ations?))"
)
_DELIVERY = re.compile(r"\b" + _VOICE_TERM + r"\b|\bvocal\s+performance\b", re.I)
# Consume a complete negative list before checking for positive vocal directions.
# 'No vocals, singing or humming' is one exclusion, not three independent requests.
_NEGATED_ITEM = (
    r"(?:(?:male|female|any|human|chopped|sampled)\s+)*" + _VOICE_TERM
    + r"(?:\s+(?:samples?|chops?|hooks?|melody|delivery|performance))?"
)
_NEGATED_DELIVERY = re.compile(
    r"\b(?:no|without|avoid|exclude|excluding)\s+" + _NEGATED_ITEM
    + r"(?:\s*(?:,|and|or|/)\s*" + _NEGATED_ITEM + r")*\b", re.I,
)


def instrumental_details(style: str) -> tuple[str, list[str]]:
    """Keep complete instrumental sentences; remove sentences naming vocal material.

    This is prompt repair, not audio separation. Removing the whole sentence avoids
    leaving 'chopped samples' after simply deleting 'vocal' from 'chopped vocal samples'.
    """
    kept, removed = [], []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", style.strip()):
        (removed if _DELIVERY.search(sentence) else kept).append(sentence)
    return " ".join(kept).strip(), removed


def vocal_intent(theme: str) -> bool | None:
    """False means explicitly instrumental, True explicitly vocal, None unspecified."""
    if _SILENT.search(theme):
        return False
    if _NOT_INSTRUMENTAL.search(theme):
        return True
    if _INSTRUMENTAL.search(theme):
        return False
    if _VOCAL.search(theme):
        return True
    return None


def apply_instrumental_mode(theme: str, instrumental: bool | None) -> str:
    """Turn an explicit UI/API override into one consistent conditioning theme.

    LiveSet carries the resulting theme through its generation queue and every depot search,
    so the setting cannot disappear during replenishment or a later deck transition.
    """
    theme = theme.strip()
    if instrumental is None:
        return theme
    cleaned = _NOT_INSTRUMENTAL.sub(" ", theme)
    cleaned = _SILENT.sub(" ", cleaned)
    cleaned = _INSTRUMENTAL.sub(" ", cleaned)
    cleaned = _VOCAL.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    instruction = (
        "Instrumental, no vocals, no lyrics" if instrumental else "With vocals and original lyrics"
    )
    return f"{cleaned}. {instruction}" if cleaned else instruction


def has_vocal_delivery(style: str) -> bool:
    """Detect audible vocal directions without mistaking 'no vocals' for one."""
    return bool(_DELIVERY.search(_SILENT.sub(" ", _NEGATED_DELIVERY.sub(" ", style))))
