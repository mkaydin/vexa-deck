"""Build a YuE2 depot big enough for the selector to have real choices.

**Why this exists.** Ten YuE2 renders give roughly 30-40 genuinely transitionable pairs
(``FINETUNE-DECISION.md``), which is too few to fine-tune on. The Strudel depot solved the count
and failed the product: 125 programmatic loops were audibly wrong for the project's sound, so they
were removed. Scale has to come from YuE2.

**The obstacle is tempo.** YuE2 was asked for a 92 BPM jazz bed and produced 144. Tempo cannot be
requested, only *hoped for and measured*. At ``max_tempo_ratio`` 1.10, a tempo scattered across
88-176 leaves most tracks with no transitionable partner -- which is exactly what ten renders did:
only the 117-134 band had company.

So the depot is built by **volume, not control**. Enough briefs that measured tempo falls into
dense enough buckets for every track to have partners. ``--count`` defaults to 200 because the
arithmetic is unforgiving: over a factor-of-two tempo range, each transitionable band holds roughly
1/7 of the library, so each track's partners scale linearly with N.

Runs in the background; each render is its own subprocess so the GPU is released between them.

Usage::

    uv run --no-sync python tools/build_yue2_depot.py --count 200
    uv run --no-sync python tools/build_yue2_depot.py --count 200 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers/yue2/src"))

from vexa_contracts import ApprovalState, AssetManifest, Provenance  # noqa: E402
from vexa_yue2.backends import primary_gpu  # noqa: E402
from vexa_yue2.gates import analyse  # noqa: E402
from vexa_yue2.master import master  # noqa: E402

BACKEND = ROOT / "third_party" / "yue2.cpp"
BINARY = BACKEND / "build" / "yue-synth"
MODEL = BACKEND / "models" / "YuE2-3B-Q8_0.gguf"
VAE = BACKEND / "models" / "YuE2-Vae-F32.gguf"

#: (tempo intent, style, mood). Tempo is a *hint* -- YuE2 lands where it lands -- so these are
#: written to nudge the model into a neighbourhood rather than to hit a number.
#:
#: Styles are kept deliberately musical. The Strudel depot proved that a depot can be large,
#: exactly measurable and still wrong for the product; nothing here is a drum machine.
PALETTES: tuple[tuple[int, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        90, "instrumental downtempo, dusty drums, warm Rhodes, sub bass, laid-back groove, "
            "no vocals",
        ("dark", "calm", "warm"),
        ("muted Rhodes", "dusty breakbeat", "warm sub bass", "vintage tape hiss"),
    ),
    (
        100, "instrumental chillout, soft pads, brushed percussion, mellow electric piano, "
             "unhurried, no vocals",
        ("calm", "warm", "rain-soaked"),
        ("soft analog pad", "brushed snare", "mellow Rhodes", "tape saturation"),
    ),
    (
        118, "instrumental deep house, warm sub bass, Rhodes chords, soft claps, "
             "steady four to the floor, no vocals",
        ("warm", "bright", "driving"),
        ("warm sub bass", "Rhodes chord stab", "soft clap", "shaker"),
    ),
    (
        124, "instrumental house, rolling bassline, filtered chords, crisp hats, "
             "clean and confident, no vocals",
        ("bright", "driving", "warm"),
        ("rolling bassline", "filtered chord", "crisp closed hat", "tight kick"),
    ),
    (
        128, "instrumental melodic techno, evolving arpeggio, deep kick, wide pads, "
             "hypnotic, no vocals",
        ("dark", "driving", "warm"),
        ("evolving arpeggio", "deep kick", "wide analog pad", "metallic pluck"),
    ),
    (
        132, "instrumental peak-time techno, relentless kick, acid lines, hypnotic, no vocals",
        ("peak", "driving", "dark"),
        ("acid 303 line", "relentless kick", "stab sequence", "noise sweep"),
    ),
    (
        138, "instrumental hard groove, heavy bass, syncopated drums, tense, no vocals",
        ("driving", "peak", "dark"),
        ("heavy syncopated bass", "tight snare", "stab chord", "industrial texture"),
    ),
    (
        145, "instrumental progressive trance, soaring lead, rolling bass, euphoric build, "
             "no vocals",
        ("peak", "bright", "warm"),
        ("soaring saw lead", "rolling bass", "reverse swell", "wide reverb"),
    ),
    (
        150, "instrumental hardstyle, distorted kick, relentless energy, dark, no vocals",
        ("peak", "dark"),
        ("distorted kick", "screaming lead", "hard bass", "dark atmosphere"),
    ),
    (
        170, "instrumental drum and bass, breakbeat, reese bass, fast and rolling, no vocals",
        ("driving", "peak", "dark"),
        ("amen breakbeat", "reese bass", "stab", "fast rolling drums"),
    ),
    (
        82, "instrumental lo-fi hip hop, dusty drums, mellow keys, vinyl crackle, "
            "late night, no vocals",
        ("calm", "warm", "rain-soaked"),
        ("dusty boom-bap kick", "mellow electric keys", "vinyl crackle", "soft snare"),
    ),
    (
        96, "instrumental jazz, brushed drums, upright bass, warm Rhodes, rainy and "
            "intimate, no vocals",
        ("calm", "rain-soaked", "warm"),
        ("brushed snare", "upright bass", "warm Rhodes", "intimate room"),
    ),
)

#: Energy is what the retrieval filters match on, so it is spread rather than clustered.
ENERGY_BANDS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.15, ("intimate", "laid-back", "sparse", "unhurried", "quiet")),
    (0.35, ("mellow", "smooth", "gentle", "restrained", "soft")),
    (0.55, ("steady", "confident", "even", "measured", "flowing")),
    (0.75, ("forceful", "driving", "energetic", "propulsive", "urgent")),
    (0.90, ("relentless", "hypnotic", "euphoric", "overwhelming", "ferocious")),
)

MOODS: tuple[str, ...] = ("dark", "calm", "warm", "bright", "driving", "peak", "rain-soaked")


@dataclass(frozen=True, slots=True)
class DepotBrief:
    name: str
    style: str
    energy: float
    mood: str
    duration_s: float = 45.0


def build_briefs(count: int) -> list[DepotBrief]:
    """Deterministic layout across palettes, moods and energy bands.

    Deterministic so the depot is reproducible and the manifest does not change under us.
    """
    briefs: list[DepotBrief] = []
    index = 0
    while len(briefs) < count:
        tempo, style, _tastes, _moods = PALETTES[index % len(PALETTES)]
        energy, adverbs = ENERGY_BANDS[index % len(ENERGY_BANDS)]
        mood = MOODS[index % len(MOODS)]
        adverb = adverbs[index % len(adverbs)]
        briefs.append(
            DepotBrief(
                name=f"depot-{tempo}-{mood}-e{str(energy).replace('.', '_')}",
                style=f"{adverb} {style}",
                energy=energy,
                mood=mood,
            )
        )
        index += 1
    return briefs


def render(brief: DepotBrief, wav: Path, gpu: int, *, max_seq: int, steps: int) -> float:
    """One render. Subprocess so the GPU context is released on exit and memory stays flat."""
    started = time.time()
    request = wav.with_suffix(".request.json")
    request.write_text(
        json.dumps({"style": brief.style, "lyrics": "", "cot": "off",
                    "cfg_scale": 1.2, "seed": -1}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(BINARY), "--model", str(MODEL), "--vae", str(VAE),
         "--request", str(request), "--out", str(wav),
         "--duration", str(brief.duration_s), "--steps", str(steps),
         "--max-seq", str(max_seq)],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"backend exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-500:]}"
        )
    return time.time() - started


def admit(brief: DepotBrief, wav: Path, *, approve: bool) -> tuple[bool, str]:
    """Master, gate and write the manifest, exactly as the existing YuE2 library was built."""
    master(wav)
    outcome = analyse(
        wav,
        asset_id=brief.name,
        family_id=f"family_{brief.name}",
        approval=ApprovalState.APPROVED if approve else ApprovalState.PENDING,
        provenance=Provenance(source_prompt=brief.style),
    )
    if outcome.fatal or outcome.manifest is None:
        return False, f"unreadable: {outcome.fatal or 'no manifest'}"
    if not outcome.admitted:
        flags = ", ".join(outcome.quality.flags) or "unnamed"
        return False, f"gates failed: {flags}"
    _write_manifest(outcome.manifest, wav.parent)
    key = outcome.manifest.key.value if outcome.manifest.key else "?"
    return True, f"{outcome.manifest.beat_grid.bpm:.1f} BPM  key={key}"


def approve_existing(directory: Path) -> int:
    """Mark already-written, gate-passing manifests approved.

    Needed because ``admit`` records approval at write time, so a run that did not pass
    ``--approve`` leaves every track pending and ``FeasibilityFilter`` refuses to schedule any of
    them -- an entire generation run that exists but cannot be played. Only manifests already in the
    library are touched, which means each one has passed every gate; this records that fact rather
    than bypassing it. Each is re-validated before and after so a bad edit cannot slip through.
    """
    from vexa_contracts import ApprovalState, ReadinessState

    approved = 0
    for path in sorted(directory.glob("*.json")):
        if ".request." in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("approval") == ApprovalState.APPROVED.value:
            continue
        AssetManifest.model_validate(data)  # must load before we touch it
        quality = data["quality"]
        if not all(
            quality[flag]
            for flag in (
                "decode_ok", "duration_ok", "loudness_ok",
                "beat_grid_ok", "loop_boundary_ok", "audio_quality_ok",
            )
        ):
            print(f"  skipped {path.name}: gates did not pass", file=sys.stderr)
            continue
        data["approval"] = ApprovalState.APPROVED.value
        data["readiness"] = ReadinessState.READY.value
        AssetManifest.model_validate(data)  # and still load afterwards
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        approved += 1
    return approved


def _write_manifest(manifest: AssetManifest, directory: Path) -> None:
    (directory / f"{manifest.asset_id}.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "library")
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--approve", action="store_true",
        help="mark admitted tracks approved. Without it every asset stays pending, and "
             "FeasibilityFilter refuses to schedule anything that is not approved -- which is "
             "how the first depot run produced 200 tracks none of which could be played.",
    )
    parser.add_argument(
        "--approve-existing", action="store_true",
        help="approve manifests already written by an earlier run and exit",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip briefs that already have a manifest (a full run is ~45 min)",
    )
    args = parser.parse_args()

    if args.approve_existing:
        count = approve_existing(args.out)
        print(f"approved {count} existing manifest(s) in {args.out}")
        return 0

    if not BINARY.exists():
        print(f"backend missing: {BINARY}", file=sys.stderr)
        return 2

    gpu = primary_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    briefs = build_briefs(args.count)
    if args.resume:
        pending = [b for b in briefs if not (args.out / f"{b.name}.json").exists()]
        print(f"{len(briefs) - len(pending)} already admitted, {len(pending)} to do")
        briefs = pending

    print(f"library={args.out}  gpu={gpu}  briefs={len(briefs)}")
    admitted = rejected = 0
    started = time.time()
    for index, brief in enumerate(briefs, 1):
        wav = args.out / f"{brief.name}.wav"
        try:
            elapsed = render(brief, wav, gpu, max_seq=args.max_seq, steps=args.steps)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"[{index}/{len(briefs)}] {brief.name} RENDER FAIL: {exc}", file=sys.stderr)
            rejected += 1
            continue
        ok, detail = admit(brief, wav, approve=args.approve)
        if ok:
            admitted += 1
            print(f"[{index}/{len(briefs)}] {brief.name} {elapsed:.0f}s — {detail}", flush=True)
        else:
            rejected += 1
            print(f"[{index}/{len(briefs)}] {brief.name} REJECT: {detail}", flush=True)

    print(f"\nadmitted={admitted} rejected={rejected} in {(time.time() - started) / 60:.1f} min")
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())