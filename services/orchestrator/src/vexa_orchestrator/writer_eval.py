"""Reproducible prompt evaluation. Text proxies never certify generated audio quality."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

CORPUS = (
    (
        "techno",
        "dark melodic techno instrumental, 128 BPM",
        (r"kick|four.on.the.floor", r"arpegg|sequence", r"minor|dark|brooding"),
    ),
    (
        "house",
        "warm house instrumental, 122 BPM, soulful Rhodes and shuffled hats",
        (r"rhodes|electric piano", r"shuffl|swing", r"bass|low.end"),
    ),
    (
        "memphis",
        "Memphis phonk instrumental, 88 BPM, dusty cassette, no EDM drop",
        (r"cassette|tape|sampler", r"808|sub.bass", r"southern|memphis|lo.fi|lofi"),
    ),
    (
        "drift",
        "JDM drift phonk instrumental, 150 BPM, distorted 808 and cowbell lead",
        (r"cowbell", r"808|sub.bass", r"distort|saturat"),
    ),
    (
        "rap_en",
        "90s underground rap in English about factory workers, unpaid rent and night buses",
        (r"boom.bap|breakbeat|sampled.drums", r"vinyl|sampler|dusty", r"bass|low.end"),
    ),
    (
        "rap_tr",
        "90s underground boom bap rap in Turkish about Istanbul ferries and unpaid rent",
        (r"boom.bap|breakbeat|sampled.drums", r"vinyl|sampler|dusty", r"bass|low.end"),
    ),
    (
        "jazz_vocal",
        "rain-soaked jazz with female vocals in English about a midnight train journey",
        (r"piano|rhodes", r"brush|upright|double bass", r"swing|jazz|sax|trumpet"),
    ),
    (
        "ambient",
        "dark ambient instrumental, 70 BPM, spacious drones and granular textures",
        (r"drone|sustain", r"granular|texture", r"spacious|reverb|space"),
    ),
    (
        "mixed",
        "jazzy downtempo instrumental, 92 BPM, mellow broken beats and soft trumpet",
        (r"broken|syncopat|downtempo", r"trumpet", r"mellow|soft|warm"),
    ),
    (
        "unusual",
        "winter greenhouse instrumental electronica, 100 BPM, glass bells and icy textures",
        (r"glass|bell", r"icy|ice|winter|cold", r"texture|pad|synth"),
    ),
)
CORPUS_HASH = hashlib.sha256(
    json.dumps(CORPUS).encode()
    + b"".join(
        Path(__file__).with_name(name).read_bytes()
        for name in (
            "generation_v2.py",
            "writers.py",
            "vocal_intent.py",
            "music_plan.py",
            "generation_settings.py",
        )
    )
    + (
        Path(__file__).resolve().parents[4] / "packages/contracts/src/vexa_contracts/music.py"
    ).read_bytes()
).hexdigest()


def text_evidence(brief, groups: tuple[str, ...]) -> dict:
    draft = brief.music_plan["creative_draft"]
    body = " ".join(
        [draft["production_notes"], *draft["instruments"], *draft["section_descriptions"].values()]
    )
    hit = [bool(re.search(group, body, re.I)) for group in groups]
    descriptions = list(draft["section_descriptions"].values())
    words = [re.findall(r"\w+", text.lower()) for text in descriptions]
    trigrams = [set(zip(w, w[1:], w[2:], strict=False)) for w in words]
    overlaps = [
        len(a & b) / max(1, len(a | b)) for i, a in enumerate(trigrams) for b in trigrams[i + 1 :]
    ]
    verses = []
    inside = False
    for line in brief.lyrics.splitlines():
        if line.startswith("["):
            inside = line.startswith("[Verse")
        elif inside and line.strip():
            verses.append(line.strip().casefold())
    language = brief.music_plan["track_plan"]["intent"]["language"]
    lyric_body = " ".join(verses)
    turkish_hits = re.findall(
        r"\b(?:bir|ben|bu|ve|gece|sokak|yol|istanbul|kira|gibi)\b", lyric_body, re.I
    )
    return {
        "semantic_proxy_coverage": sum(hit) / len(hit),
        "concept_hits": hit,
        "section_trigram_overlap": statistics.mean(overlaps) if overlaps else 0,
        "verse_unique_line_ratio": len(set(verses)) / len(verses) if verses else None,
        "turkish_language_proxy": len(turkish_hits) >= 3 if language == "tr" else None,
        "scope": "lexical text proxies; no audio or human genre assessment",
    }


class DeviceMemory:
    def __init__(self):
        from .generation import _gpu_index

        self.gpu = _gpu_index()
        self.values = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        self.gpu,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    self.values.append(int(result.stdout.strip()))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=4)


def benchmark_wave(payload: dict) -> dict:
    """At most three cases under one GPU lease; separate waves release the GPU."""
    from .ace_planner import local_writer_server
    from .generation_v2 import build_plan, write_plan

    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    profile = payload["profile"]
    cases = payload["cases"]
    if not 1 <= len(cases) <= 3:
        raise ValueError("writer benchmark waves must contain 1-3 cases")
    from contextlib import nullcontext

    context = (
        local_writer_server(configured_url=profile["base"])
        if profile["mode"] == "local"
        else nullcontext(profile["base"])
    )
    rows = []
    with context as base:
        settings = {**profile, "base": base, "_lease_owned": profile["mode"] == "local"}
        for case, variation in cases:
            key, theme, groups = CORPUS[case]
            plan = build_plan(theme, variation * 3, "minimax")
            path = output / f"{key}-{variation}.json"
            cache = output / f"{key}-{variation}.cache.json"
            started = time.monotonic()
            row = {
                "case": key,
                "theme": theme,
                "variation": variation,
                "profile": profile,
                "corpus_hash": CORPUS_HASH,
                "attempts": [],
                "audio_generated": False,
                "audio_played": False,
            }
            with DeviceMemory() as memory:
                for _attempt in range(3):
                    before = time.monotonic()
                    settings["_telemetry"] = []
                    try:
                        brief = write_plan(plan, settings, cache)
                        row.update(
                            accepted=True,
                            brief=asdict(brief),
                            evidence=text_evidence(brief, groups),
                        )
                        row["attempts"].append(
                            {
                                "accepted": True,
                                "wall_s": time.monotonic() - before,
                                "requests": settings["_telemetry"],
                            }
                        )
                        break
                    except Exception as exc:
                        row["attempts"].append(
                            {
                                "accepted": False,
                                "error": str(exc),
                                "wall_s": time.monotonic() - before,
                                "requests": settings["_telemetry"],
                            }
                        )
                        row["accepted"] = False
                        from vexa_yue2.backends import BackendUnavailable

                        if isinstance(exc, BackendUnavailable):
                            row["unavailable"] = True
                            break
            row.update(
                wall_s=time.monotonic() - started,
                device_peak_mib=max(memory.values) if memory.values else None,
                device_baseline_mib=memory.values[0] if memory.values else None,
                gpu_uuid=memory.gpu,
            )
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
            temp.replace(path)
            rows.append(row)
            if row.get("unavailable"):
                break
    return {"rows": rows}


def scorecard(rows: list[dict]) -> dict:
    accepted = [r for r in rows if r["accepted"]]
    latencies = sorted(r["wall_s"] for r in rows)
    evidence = [r["evidence"] for r in accepted]
    peaks = [r["device_peak_mib"] for r in rows if r.get("device_peak_mib") is not None]
    return {
        "requests": len(rows),
        "accepted": len(accepted),
        "first_pass": sum(r["attempts"][0]["accepted"] for r in rows),
        "attempts": sum(len(r["attempts"]) for r in rows),
        "p50_wall_s": statistics.median(latencies) if latencies else None,
        "p95_wall_s": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)]
        if latencies
        else None,
        "peak_device_mib": max(peaks) if peaks else None,
        "semantic_proxy_coverage": statistics.mean(e["semantic_proxy_coverage"] for e in evidence)
        if evidence
        else 0,
        "section_trigram_overlap": statistics.mean(e["section_trigram_overlap"] for e in evidence)
        if evidence
        else 0,
        "unique_titles": len({r["brief"]["title"].casefold() for r in accepted}),
        "failures": [
            {
                "case": r["case"],
                "variation": r["variation"],
                "errors": [a.get("error") for a in r["attempts"] if a.get("error")],
            }
            for r in rows
            if not r["accepted"]
        ],
    }
