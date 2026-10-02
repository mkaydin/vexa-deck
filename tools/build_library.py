"""Generate the starter library.

Every asset is rendered, analysed, gated, and written with its manifest — the same path a live
generation job takes, just run once per brief instead of from a queue.

This is **inference only**. No training, no fine-tuning, no weight updates.

The briefs cover the ground ``PLAN.md`` Q3 asks for: at least three moods across an energy range,
each with an explicit instrumental (no-vocal) variant so the request filters have something real
to act on.

Usage::

    uv run --no-sync python tools/build_library.py --count 1     # one render, as a smoke check
    uv run --no-sync python tools/build_library.py               # the full set
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services/audio/src"))
sys.path.insert(0, str(ROOT / "workers/yue2/src"))

from vexa_contracts import ApprovalState, AssetManifest, Estimate  # noqa: E402
from vexa_yue2.backends import primary_gpu  # noqa: E402
from vexa_yue2.gates import GateThresholds, analyse  # noqa: E402
from vexa_yue2.master import master  # noqa: E402

BACKEND = ROOT / "third_party" / "yue2.cpp"
BINARY = BACKEND / "build" / "yue-synth"
MODEL = BACKEND / "models" / "YuE2-3B-Q8_0.gguf"
VAE = BACKEND / "models" / "YuE2-Vae-F32.gguf"
OUT = ROOT / "assets" / "library"


@dataclass(frozen=True, slots=True)
class BriefSpec:
    """One library entry. ``energy`` is the target the retrieval filters will match against."""

    name: str
    style: str
    lyrics: str
    energy: float
    mood: str = ""
    cot: str = "off"
    duration_s: float = 40.0


#: `cot="off"` is deliberate for a first library: symbolic planning roughly doubles render time,
#: and these exist to prove the pipeline and give the filters something real to select against.
#: Plan-then-edit is exercised separately once the basics are green.
#: Tempo *clusters* rather than a spread. A selector can only usefully choose between
#: assets that are close enough to transition into, so five briefs at 92-125 BPM produced
#: mostly-empty menus. These cluster around the tempi a DJ would actually move between.
BRIEFS: list[BriefSpec] = [
    BriefSpec(
        "calm-jazz-bed", "instrumental jazz, brushed drums, upright bass, warm Rhodes, "
        "slow and rain-soaked, no vocals, no guitar",
        "", energy=0.15, mood="rain-soaked", duration_s=45,
    ),
    BriefSpec(
        "moody-downtempo", "instrumental downtempo, muted Rhodes, sub bass, brushed "
        "percussion, dark and sparse, no vocals",
        "", energy=0.25, mood="dark", duration_s=45,
    ),
    BriefSpec(
        "late-night-drive", "instrumental synthwave, analog pads, gated drums, warm bass, "
        "night drive, no vocals",
        "", energy=0.60, mood="driving", duration_s=45,
    ),
    BriefSpec(
        "warm-deep-house", "instrumental deep house, warm sub bass, Rhodes chords, soft "
        "claps, steady four to the floor, no vocals",
        "", energy=0.45, mood="warm", duration_s=45,
    ),
    BriefSpec(
        "deep-house-b", "instrumental deep house, deep bass, muted chords, steady drums, "
        "no vocals", "", energy=0.50, mood="warm", duration_s=45,
    ),
    BriefSpec(
        "house-a", "instrumental house, bright plucks, four on the floor, no vocals",
        "", energy=0.58, mood="bright", duration_s=45,
    ),
    BriefSpec(
        "house-b", "instrumental house, rolling bassline, soft percussion, no vocals",
        "", energy=0.62, mood="bright", duration_s=45,
    ),
    BriefSpec(
        "bright-peak-house", "instrumental peak-time house, driving drums, bright synth "
        "stabs, relentless energy, no vocals",
        "", energy=0.80, mood="peak", duration_s=45,
    ),
    BriefSpec(
        "peak-b", "instrumental peak-time techno, relentless kick, acid lines, no vocals",
        "", energy=0.85, mood="peak", duration_s=45,
    ),
    BriefSpec(
        "downtempo-b", "instrumental downtempo, dusty drums, warm pads, no vocals",
        "", energy=0.30, mood="dark", duration_s=45,
    ),
]


def _write_manifest(manifest: AssetManifest) -> None:
    """Persist the manifest next to its audio, so the library is self-describing."""
    (OUT / f"{manifest.asset_id}.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def render(spec: BriefSpec, *, gpu: int, max_seq: int, steps: int) -> tuple[Path, float]:
    """One render via the C++ backend. Subprocess: cancellable, isolated, GPU released on exit."""
    import subprocess

    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / f"{spec.name}.wav"
    request = OUT / f"{spec.name}.request.json"
    score = OUT / f"{spec.name}.abc"

    request.write_text(
        json.dumps({
            "style": spec.style,
            "lyrics": spec.lyrics,
            "cot": spec.cot,
            "cfg_scale": 1.2,
            "seed": -1,
        }),
        encoding="utf-8",
    )

    started = time.time()
    cmd = [
        str(BINARY), "--model", str(MODEL), "--vae", str(VAE),
        "--request", str(request), "--out", str(wav),
        "--duration", str(spec.duration_s), "--steps", str(steps),
        "--score", str(score), "--max-seq", str(max_seq),
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    import os

    completed = subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ, **env}, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"backend exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-600:]}"
        )
    return wav, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=len(BRIEFS), help="how many briefs to render")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="CUDA device index; defaults to the primary (RTX 5060 Ti)",
    )
    parser.add_argument("--max-seq", type=int, default=8192,
                        help="KV cache size; trades song length for VRAM")
    parser.add_argument("--steps", type=int, default=32, help="flow matching steps")
    parser.add_argument("--approve", action="store_true",
                        help="mark passing assets approved so they are immediately schedulable")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"error: backend not built at {BINARY}", file=sys.stderr)
        return 1
    for path in (MODEL, VAE):
        if not path.exists():
            print(f"error: weights missing at {path}", file=sys.stderr)
            return 1

    gpu = args.gpu if args.gpu is not None else primary_gpu()
    print(f"backend : {BINARY.name}")
    print("weights : Q8_0 backbone + F32 VAE")
    print(f"device  : cuda:{gpu}   max-seq {args.max_seq}   steps {args.steps}")
    print(f"briefs  : {args.count} of {len(BRIEFS)}\n")

    admitted, quarantined, total_time = 0, 0, 0.0
    for spec in BRIEFS[: args.count]:
        print(f"[{spec.name}] energy {spec.energy:.2f}  cot={spec.cot}")
        try:
            wav, elapsed = render(spec, gpu=gpu, max_seq=args.max_seq, steps=args.steps)
        except RuntimeError as exc:
            print(f"  [FAIL] {exc}")
            quarantined += 1
            continue
        total_time += elapsed

        # Master before analysing. A raw render peaks at or above 0 dBTP with 30-65 LU of range,
        # which leaves no crossfade headroom and makes every transition a loudness jump.
        report = master(wav)
        print(f"  mastered {report.input_lufs:.1f} -> {report.output_lufs:.1f} LUFS, "
              f"peak {report.input_peak_dbtp:.2f} -> {report.output_peak_dbtp:.2f} dBTP"
              f"{' (limited)' if report.limited else ''}")

        outcome = analyse(
            wav,
            asset_id=spec.name,
            family_id=f"family_{spec.name}",
            approval=ApprovalState.APPROVED if args.approve else ApprovalState.PENDING,
            thresholds=GateThresholds(),
        )
        print(f"  rendered in {elapsed:.1f}s")
        if outcome.fatal:
            print(f"  [FAIL] {outcome.fatal}")
            quarantined += 1
            continue

        manifest = outcome.manifest
        ok = outcome.quality.passed
        if manifest is not None and ok:
            # Measurements come from the analyser; *intent* comes from the brief. Without these
            # tags every energy filter is inert, because nothing knows what a track is for.
            tagged = manifest.model_copy(update={
                "tags": {
                    "energy": [Estimate(value=f"{spec.energy}", confidence=1.0)],
                    "mood": [Estimate(value=spec.mood, confidence=0.9)],
                    "instrumental": [Estimate(value="true", confidence=1.0)],
                }
            })
            _write_manifest(tagged)
            manifest = tagged
        if ok:
            admitted += 1
        else:
            quarantined += 1
        flags = "; ".join(outcome.quality.flags) or "none"
        print(f"  [{'PASS' if ok else 'FAIL'}] gates — {flags}")
        if manifest.beat_grid:
            print(f"         {manifest.audio.duration_s:.1f}s @ "
                  f"{manifest.beat_grid.bpm:.1f} BPM "
                  f"(confidence {manifest.beat_grid.confidence:.2f}), "
                  f"bars={manifest._bars()}, {len(manifest.loops)} loop(s)")
        print(f"         loudness {manifest.audio.integrated_lufs} LUFS, "
              f"peak {manifest.audio.true_peak_dbtp} dBTP")
        print(f"         {manifest.readiness.value} / {manifest.approval.value}  "
              f"schedulable={manifest.admissible()}")

    print(f"\n{'=' * 64}")
    print(f"admitted {admitted}   quarantined {quarantined}   "
          f"total render time {total_time:.1f}s")
    print(f"{'=' * 64}")
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())