"""Generate YuE2 briefs with an LLM, falling back to a rule-based layout.

``tools/build_yue2_depot.py`` writes 12 hand-written palettes. That was a deliberate first pass and
it has a measured ceiling: 210 renders from 12 style prompts, and the resulting library concentrates
in a handful of keys -- the top 4 hold 41%. More renders from the same 12 prompts deepen the skew
rather than filling it.

An LLM writes briefs that are musically specific and mutually distinct in a way a fixed table
cannot, which is exactly the axis that is short. Tempo is deliberately *not* the axis it varies:
YuE2 was asked for 90 and produced 85, asked for 100 and produced 84. Hints move the neighbourhood,
not the number, so briefs ask for genre and character and let tempo fall where it falls.

**OpenAI-compatible**, so it works against whatever the user runs. ``README.md`` calls GPT-6 Sol an
optional planner, and this follows the same rule: with no endpoint configured it uses the rule
layout and nothing fails. A generation run must not depend on a network service being up.

**Provenance is explicit.** LLM briefs are written to their own file with the model recorded, never
passed straight to the renderer, so a bad batch can be read and discarded without spending 17
seconds per track. Nobody has listened to them, so they may fill the depot but must never be
mistaken for labelled preference.

Usage::

    uv run --no-sync python tools/generate_prompts.py --count 40
    uv run --no-sync python tools/generate_prompts.py --count 40 --no-llm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Written briefs land here rather than going straight to the renderer, so a batch can be argued
#: with and discarded without spending render time on it.
DEFAULT_OUT = ROOT / "data" / "briefs" / "llm_briefs.json"

SYSTEM_PROMPT = (
    "You write one-line prompts for a music generator that produces instrumental DJ tracks.\n"
    "\n"
    "Each prompt must:\n"
    "- name a specific genre and its sub-style\n"
    "- name 2-4 concrete instruments or sound elements\n"
    "- state the energy in one descriptive word (intimate, unhurried, propulsive, relentless)\n"
    '- say "instrumental, no vocals"\n'
    "- stay under 25 words\n"
    "\n"
    "Vary the GENRE as widely as you can: techno, house, ambient, jazz, downtempo, trip-hop,\n"
    "drum and bass, lo-fi, synthwave, breaks, dub, drone, bossa, highlife, gqom, footwork,\n"
    "garage, deep listening, library music, balearic, tribal, progressive, plus less common\n"
    "regional and micro-genres.\n"
    "\n"
    "Do NOT try to control tempo. The generator ignores tempo instructions, so mentioning BPM\n"
    'only wastes words. Describe rhythm and feel instead ("driving", "half-time", "sparse").\n'
    "\n"
    "Return ONLY a JSON array of strings. No prose, no markdown fence, no explanation."
)


@dataclass(frozen=True, slots=True)
class Brief:
    name: str
    style: str
    energy: float
    mood: str
    origin: str
    model: str = ""


#: Fallback layout, in the spirit of build_yue2_depot.py's hand-written palettes but reaching for
#: genres the first pass missed. Used when no endpoint is configured or the call fails.
FALLBACK_STYLES: tuple[tuple[str, float, str], ...] = (
    ("instrumental ambient, long pads, no percussion, weightless", 0.10, "calm"),
    ("instrumental dub techno, chords with tape delay, sparse kick, deep", 0.25, "dark"),
    ("instrumental deep listening, granular textures, bowed metal, still", 0.15, "calm"),
    ("instrumental highlife, interlocking guitars, bright kit, joyful", 0.70, "bright"),
    ("instrumental gqom, polyrhythmic drums, call and response, raw", 0.75, "dark"),
    ("instrumental footwork, clipped percussion, rolling bassline, tense", 0.80, "driving"),
    ("instrumental drone, single sustained tone, bowed strings, still", 0.20, "dark"),
    ("instrumental bossa nova, nylon guitar, brushed kit, relaxed", 0.35, "warm"),
    ("instrumental trip-hop, vinyl texture, muted beat, smoky", 0.40, "dark"),
    ("instrumental library music, tape wobble, sparse piano, warm", 0.30, "warm"),
    ("instrumental garage, bleepy percussion, shuffling hats, loose", 0.65, "driving"),
    ("instrumental breaks, syncopated break, jazzy keys, restless", 0.60, "driving"),
)


def rule_briefs(count: int) -> list[Brief]:
    """Deterministic fallback layout. Used when no endpoint is configured or the call fails."""
    briefs: list[Brief] = []
    for index in range(count):
        style, energy, mood = FALLBACK_STYLES[index % len(FALLBACK_STYLES)]
        # Enforced in code rather than trusted from each literal: the filter only works if every
        # brief is instrumental, and a future edit that drops the clause should not silently
        # produce a depot of vocal tracks.
        if "no vocals" not in style.lower():
            style = f"{style}, no vocals"
        variant = index // len(FALLBACK_STYLES)
        briefs.append(
            Brief(
                name=f"brief-{mood}-e{str(energy).replace('.', '_')}-v{variant}",
                style=style,
                energy=energy,
                mood=mood,
                origin="rule",
            )
        )
    return briefs


def llm_briefs(count: int, *, base_url: str, model: str, api_key: str,
               timeout: float) -> list[Brief]:
    """Ask an OpenAI-compatible endpoint for briefs.

    Raises on any failure so the caller can fall back. A generation run must not stop because a
    network service is down, and must never silently accept a half-parsed reply.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Write {count} distinct prompts."},
        ],
        "temperature": 1.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _parse(body["choices"][0]["message"]["content"], model=model)


def _parse(content: str, *, model: str) -> list[Brief]:
    """Pull the array out of a reply.

    Models wrap JSON in fences and add prose despite being told not to, so the array is located
    rather than assumed. Anything unparseable raises; it is never a partial result.
    """
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON array in reply: {content[:200]!r}")
    styles = [
        s.strip()
        for s in json.loads(content[start:end + 1])
        if isinstance(s, str) and s.strip()
    ]
    if not styles:
        raise ValueError("reply contained no usable prompts")

    briefs: list[Brief] = []
    for style in styles:
        if "no vocals" not in style.lower():
            style = f"{style.rstrip('.')}, instrumental, no vocals"
        digest = hashlib.sha256(style.encode("utf-8")).hexdigest()[:6]
        briefs.append(
            Brief(
                name=f"llm-{digest}",
                style=style,
                # Inferred, not measured. The prompt describes character in words while the
                # filters need numbers, so this is a crude prior and is not presented as an
                # estimate.
                energy=0.5,
                mood="unknown",
                origin="llm",
                model=model,
            )
        )
    return briefs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.environ.get("VEXA_PLANNER_MODEL", "gpt-6-sol"))
    parser.add_argument(
        "--base-url", default=os.environ.get("VEXA_PLANNER_URL", ""),
        help="OpenAI-compatible base URL; empty means use the rule layout",
    )
    parser.add_argument("--api-key", default=os.environ.get("VEXA_PLANNER_KEY", ""))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM, emit the rule layout")
    args = parser.parse_args()

    briefs: list[Brief]
    source = "rule"
    if args.no_llm or not args.base_url:
        if not args.no_llm:
            print("no --base-url configured; using the rule layout", file=sys.stderr)
        briefs = rule_briefs(args.count)
    else:
        try:
            briefs = llm_briefs(
                args.count,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            source = f"llm:{args.model}"
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError, OSError) as exc:
            print(f"LLM call failed ({exc}); falling back to the rule layout", file=sys.stderr)
            briefs = rule_briefs(args.count)

    unique: dict[str, Brief] = {}
    for brief in briefs:
        unique.setdefault(brief.style.lower(), brief)
    kept = list(unique.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": source,
                "model": args.model if source.startswith("llm") else "",
                "note": (
                    "LLM-written briefs. Nobody has listened to these, so they may fill the depot "
                    "but must never be treated as labelled preference."
                ),
                "briefs": [asdict(brief) for brief in kept],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"source={source}  requested={args.count}  unique={len(kept)}  -> {args.out}")
    for brief in kept[:5]:
        print(f"  {brief.name}: {brief.style}")
    if len(kept) > 5:
        print(f"  ... and {len(kept) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
