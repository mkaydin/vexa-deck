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
    r"|rapping|humming|chants?|chanting|voices?|speech|spoken\s+(?:words|verses?|refrains?)"
    r"|rap\s+(?:verses?|refrains?|delivery|samples?|chops?)|MC\s+delivery|ad[ -]?libs?"
    r"|vocali[sz](?:ing|ations?))"
)
_DELIVERY = re.compile(r"\b" + _VOICE_TERM + r"\b|\bvocal\s+performance\b", re.I)
_INSTRUMENT_VOICE = re.compile(r"\binstrumental\s+voices?\b", re.I)
_INSTRUMENT_ROLE = re.compile(
    r"(?P<context>\b(?:synthesizer|synth|piano|guitar|bass|saxophone|strings|"
    r"instrumental\s+(?:element|motif|lead|part))\b"
    r"[^.!?\n]{0,100}?\b(?:(?:acts?|serves?|functions?|acting|serving|functioning)"
    r"\s+as|is)\s+(?:the\s+)?"
    r"(?:(?:primary|main|lead|melodic)\s+){0,2})voices?\b",
    re.I,
)
# Consume a complete negative list before checking for positive vocal directions.
# 'No vocals, singing or humming' is one exclusion, not three independent requests.
_NEGATED_ITEM = (
    r"(?:(?:male|female|any|human|chopped|sampled)\s+)*"
    + r"(?:"
    + _VOICE_TERM
    + r"|backing(?:\s+vocals?)?|effects\s+applied\s+to\s+vocals?"
    + r"|(?:EDM|pop|melodic)\s+(?:drops?|breaks?))"
    + r"(?:\s+(?:samples?|chops?|hooks?|melody|delivery|performance|elements?|effects|"
    r"parts?|layers?|material))?"
)
_NEGATED_DELIVERY = re.compile(
    r"\b(?:no|without|avoid(?:ing)?|exclude|excluding)\s+"
    + r"(?:(?:introducing|adding|using|including)\s+)?"
    + _NEGATED_ITEM
    + r"(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or|/)\s*)"
    + _NEGATED_ITEM
    + r")*\b",
    re.I,
)


def instrumental_details(style: str) -> tuple[str, list[str]]:
    """Keep complete instrumental sentences; remove sentences naming vocal material.

    This is prompt repair, not audio separation. Removing the whole sentence avoids
    leaving 'chopped samples' after simply deleting 'vocal' from 'chopped vocal samples'.
    """
    kept, removed = [], []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", style.strip()):
        (removed if _DELIVERY.search(_instrument_voice_context(sentence)) else kept).append(
            sentence
        )
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
    body = _instrument_voice_context(_SILENT.sub(" ", _NEGATED_DELIVERY.sub(" ", style)))
    return bool(_DELIVERY.search(body))


def _instrument_voice_context(text: str) -> str:
    # Replace only the metaphorical word, retaining the rest of the sentence so
    # an actual singer or vocal sample in the same sentence is still detected.
    return _INSTRUMENT_ROLE.sub(r"\g<context>instrumental lead", _INSTRUMENT_VOICE.sub(" ", text))
