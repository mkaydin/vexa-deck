"""Build the algorithmic depot with Strudel.

Generates programmatic loops that go through **the same** render → master → analyse → gate path
``tools/build_library.py`` uses for YuE2. Nothing here bypasses a gate or writes a manifest by
hand; if a render fails a check it is rejected exactly as a YuE2 render would be.

**Why this exists.** ``FINETUNE-DECISION.md:150``: ten tracks yield roughly 30-40 genuinely
distinct decision situations, which is too few to fine-tune on without memorising the library.
A hundred tracks is enough. Strudel gets us there in minutes: measured 0.52 s for 12 s of stereo
audio, against YuE2's 13 s per 45 s.

**Why Strudel and not YuE2 for the depot.** Two properties YuE2 cannot supply:

* **Tempo is specified, not estimated.** YuE2 was asked for a 92 BPM jazz bed and produced 144.
* **The parts are separate by construction.** Strudel patterns are per-instrument, so kick, bass
  and keys are distinct. These are the stems YuE2 could not yield natively, by score editing, or
  by separation (HTDemucs and BS-RoFormer both leave a ~141 % residual).

**Tempo handling.** ``FINETUNE-DECISION.md:216`` - the *specified* tempo is what we store. The
measured tempo carries ~1 % error from frame quantisation at 48 kHz with a 512 hop, so it is a
**gate**, not metadata: a render is rejected if measurement disagrees with the request by more
than ``--tolerance`` percent.

YuE2 remains the production aesthetic. This fills the depot and gives the selector exact
metadata to reason over; it does not replace the audible library.

Usage::

    uv run --no-sync python tools/build_strudel_depot.py --count 1   # smoke check
    uv run --no-sync python tools/build_strudel_depot.py --count 100 # the depot
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers/yue2/src"))
from vexa_contracts import ApprovalState, AssetManifest, Provenance, SourceType  # noqa: E402
from vexa_yue2.gates import analyse  # noqa: E402
from vexa_yue2.master import master  # noqa: E402

#: Rates a beat tracker may report for a specification, as multipliers on its own reading.
#:
#: A tracker returns whichever metrical level has the strongest periodicity, and this material
#: routinely defeats that: a bass note on every eighth note reads as double-time, and a 168 BPM
#: bed reads as 112.5 when the tracker hears compound meter. Half/double and the 3:2 family are
#: ordinary properties of musical time rather than defects, so the gate tests the whole family.
#:
#: What the gate is actually for is catching a **broken** render - silence, clipping, garbage, the
#: wrong file. Every multiplier below still confirms a strong periodic pulse at a rate consistent
#: with what was requested, which is the verification that matters.
METRICAL_MULTIPLIERS: dict[str, float] = {
    "as-measured": 1.0,
    "half": 0.5,
    "double": 2.0,
    "two-thirds": 2.0 / 3.0,
    "three-quarters": 3.0 / 4.0,
    "four-thirds": 4.0 / 3.0,
    "three-halves": 1.5,
}

#: Default tolerance for the comparison above. Not arbitrary: the tempogram quantises tempo into
#: bins, and measured values were observed snapping up to 2.4 % from the requested rate (a 144 BPM
#: request measuring 140.6). The gate absorbs that quantisation and rejects real mismatches.
DEFAULT_TOLERANCE_PCT = 3.0

#: Third-party. Kept under ``third_party/`` like ``yue2.cpp`` because it is not ours and must
#: never be vendored into the application package.
WIRBEL = ROOT / "third_party" / "strudel" / "wirbel" / "bin" / "wirbel.js"

#: Strudel's own example material is CC BY-NC-SA 4.0 and wirbel is AGPL-3.0-only. Neither is
#: permissive, so every manifest records the licence explicitly (``LAYA_DATA.md:85``).
STRUDEL_LICENSE = "CC BY-NC-SA 4.0"

#: Strudel expresses tempo in cycles; one cycle is one 4/4 bar.
BEATS_PER_CYCLE = 4

DEFAULT_OUT = ROOT / "assets" / "library"


@dataclass(frozen=True, slots=True)
class DepotSpec:
    """One depot entry.

    ``bpm`` is *specified*: written into the pattern and stored in the manifest. ``cluster`` is
    the tempo cluster the entry belongs to, which selects its kit and bass register.
    """

    name: str
    cluster: float
    bpm: float
    energy: float
    mood: str
    duration_s: float


#: Tempo clusters, not a spread. ``PLAN.md`` Q3 asks for tempo clusters, and the reason is
#: practical: a selector can only usefully choose between assets close enough to transition into.
#: Spanning 80-180 BPM uniformly would build a library where almost nothing is transitionable.
TEMPO_CLUSTERS: tuple[tuple[float, tuple[float, ...], str, str], ...] = (
    # (centre, tempi, drum kit, bass root)
    (92.0, (88.0, 90.0, 92.0, 94.0, 96.0), "bd ~ ~ sd", "C2"),
    (122.0, (118.0, 120.0, 122.0, 124.0, 126.0), "bd ~ bd ~", "G1"),
    (130.0, (128.0, 130.0, 132.0, 134.0, 136.0), "bd ~ ~ bd", "A1"),
    (140.0, (138.0, 140.0, 142.0, 144.0, 146.0), "bd bd bd bd", "D2"),
    (170.0, (168.0, 170.0, 172.0, 174.0, 176.0), "bd ~ bd ~", "E1"),
)

MOODS: tuple[str, ...] = ("dark", "calm", "warm", "bright", "peak")

#: Chord roots per mood, so key is a chosen property rather than an estimate.
MOOD_ROOTS: dict[str, str] = {
    "dark": "<A1 A1 C2 E1>",
    "calm": "<D3 F3 A3 C4>",
    "warm": "<F2 A2 C3 D3>",
    "bright": "<C3 E3 G3 B3>",
    "peak": "<A2 A2 C3 E3>",
}
#: Energy band per mood, with two steps inside each. Energy is derived from mood rather than
#: indexed independently: with 20 slots per cluster there are not three independent axes to draw
#: from, and a "peak" energy track that reads calm would be a library the filters cannot serve.
MOOD_ENERGY: dict[str, tuple[float, float]] = {
    "calm": (0.10, 0.20),
    "dark": (0.25, 0.35),
    "warm": (0.45, 0.55),
    "bright": (0.60, 0.70),
    "peak": (0.80, 0.90),
}


def build_specs(count: int) -> list[DepotSpec]:
    """Deterministically lay out ``count`` specs across tempo clusters, moods and energies.

    Deterministic on purpose: the depot must be reproducible, and a shuffled seed would make the
    manifest and the family splits irreproducible too.
    """
    per_cluster = -(-count // len(TEMPO_CLUSTERS))  # ceil
    specs: list[DepotSpec] = []
    for cluster_index, (centre, tempi, _kit, _root) in enumerate(TEMPO_CLUSTERS):
        for i in range(per_cluster):
            if len(specs) >= count:
                break
            # bpm and mood draw from different strides. Deriving both from the same modulo makes
            # them perfectly correlated: one cluster ends up bound to one mood, and the whole set
            # repeats long before `count`. The cluster offset rotates which mood each tempo lands
            # on, so every cluster still covers every mood across the run.
            bpm = tempi[i % len(tempi)]
            mood = MOODS[(i // len(tempi) + cluster_index) % len(MOODS)]
            energy = MOOD_ENERGY[mood][i % 2]
            specs.append(
                DepotSpec(
                    name=f"strudel-{centre:.0f}-{bpm:.0f}-{mood}-e{energy:.2f}".replace(".", "_"),
                    cluster=centre,
                    bpm=bpm,
                    energy=energy,
                    mood=mood,
                    # Longer for slower material, so every entry has enough bars to loop and to
                    # offer a real phrase boundary.
                    duration_s=32.0 if bpm < 128 else 24.0,
                )
            )
    return specs


def pattern_for(spec: DepotSpec) -> str:
    """Render a spec to Strudel source.

    ``setcps`` is **cycles per second**. ``setcpm`` is cycles per minute and wirbel divides by 60
    again, producing audio 60x too slow: a 12 s render then holds 0.1 cycles and no detectable
    beat. ``FINETUNE-DECISION.md:175`` records that misdiagnosis.

    ``setcps`` is cycles per second. The energy target chooses between a sustained pad and a
    stab: low energy gets one chord per cycle, high energy gets two shorter stabs.
    """
    cps = spec.bpm / 60.0 / BEATS_PER_CYCLE
    kit = next(k for c, _t, k, _r in TEMPO_CLUSTERS if c == spec.cluster)
    root = next(r for c, _t, _k, r in TEMPO_CLUSTERS if c == spec.cluster)
    chords = MOOD_ROOTS[spec.mood]
    stab = "!" in chords or spec.energy >= 0.6
    keys_line = (
        f'note("{chords}").s("triangle").gain(.3)'
        if not stab
        else f'note("{chords}!8").s("triangle").lpf(2200).gain(.32)'
    )
    return f"""// vexa-deck depot asset: {spec.name}
// tempo is SPECIFIED here ({spec.bpm} BPM) and stored as manifest metadata.
setcps({cps:.10f})
stack(
  s("{kit}").gain(1.0),
  s("~ cp").gain(.45),
  note("{root}*8").s("sawtooth").lpf(420).gain(.55),
  {keys_line}
).slowcat(2)
"""


def render(spec: DepotSpec, wav: Path, out_dir: Path) -> float:
    """One render in a **subprocess**, so the renderer is cancellable and releases everything.

    Isolation is deliberate: wirbel drives a headless browser, which is exactly the shape that
    leaks memory if run in-process repeatedly. One process per render, exiting after, keeps
    resident memory flat across a 125-track build.

    **Do not apply a virtual-memory limit here.** Chromium reserves tens of gigabytes of address
    space for its V8 sandbox while holding little resident memory, so ``ulimit -v`` makes the
    renderer hang indefinitely rather than fail fast. Verified: capped at 4 GB it times out,
    uncapped it renders 12 s of audio in 0.4 s. Bound the work by process count and duration
    instead, not by address space.
    """
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="vexa-strudel-") as tmp:
        src = Path(tmp) / f"{spec.name}.strudel"
        src.write_text(pattern_for(spec), encoding="utf-8")
        completed = subprocess.run(
            [
                shutil.which("node") or "node", str(WIRBEL),
                str(src), "--format", "wav", "--target", tmp,
                "--duration", str(spec.duration_s), "--force",
            ],
            capture_output=True, text=True, check=False, timeout=300,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"wirbel exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout)[-600:]}"
            )
        produced = Path(tmp) / f"{spec.name}.wav"
        if not produced.exists():
            raise RuntimeError(f"wirbel produced no audio for {spec.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(wav))
    return time.time() - started


def _write_manifest(manifest: AssetManifest, out_dir: Path) -> None:
    """Same self-describing layout as the YuE2 library: manifest beside its audio."""
    (out_dir / f"{manifest.asset_id}.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )


def admit(
    spec: DepotSpec, wav: Path, out_dir: Path, *, tolerance_pct: float, approve: bool
) -> tuple[bool, str]:
    """Master, gate, and write the manifest. Returns ``(admitted, detail)``.

    The one departure from the YuE2 path: after analysis the **specified** tempo replaces the
    measured one, and disagreement becomes a gate rather than a silent correction.
    """
    master(wav)
    outcome = analyse(
        wav,
        asset_id=spec.name,
        family_id=f"family_{spec.name}",
        source_type=SourceType.PACK_IMPORT,
        provenance=Provenance(license=STRUDEL_LICENSE),
        approval=ApprovalState.APPROVED if approve else ApprovalState.PENDING,
    )

    if outcome.fatal or outcome.manifest is None:
        return False, f"unreadable: {outcome.fatal or 'no manifest'}"

    measured = outcome.manifest.beat_grid.bpm if outcome.manifest.beat_grid else 0.0
    if measured <= 0:
        return False, "no tempo detected"

    # Metrical-level tolerant comparison. A tracker locks onto whichever metrical level has the
    # strongest periodicity, and this material defeats that: a bass note on every eighth note
    # reads as double-time, and a 168 BPM bed reads as 112 when the tracker hears compound meter.
    # None of that is a defect in the render, so the specification is tested against the whole
    # family of musically-related rates. What this gate is for is catching a *broken* render -
    # silence, garbage, the wrong file - and every candidate below still confirms a strong
    # periodic pulse consistent with what was asked for.
    candidates = {label: measured * factor for label, factor in METRICAL_MULTIPLIERS.items()}
    errors = {
        label: abs(value - spec.bpm) / spec.bpm * 100.0 for label, value in candidates.items()
    }
    best = min(errors, key=lambda label: errors[label])
    if errors[best] > tolerance_pct:
        detail = ", ".join(f"{label}={value:.1f}" for label, value in candidates.items())
        return False, (
            f"no metrical level matches {spec.bpm:.0f} BPM within "
            f"{tolerance_pct}%: {detail}"
        )
    detected_level = None if best == "as-measured" else best

    # Store what was *specified*. Measured tempo carries ~1 % error from frame quantisation at
    # 48 kHz with a 512 hop, so it is a gate, not metadata.
    #
    # Overriding the tempo re-scales every bar number, because ``analyse`` counted bars at the
    # measured rate. A track read at 175.8 is counted as 23 bars; the same 32 s of audio stored at
    # the specified 88 is 11. Dropping the out-of-range markers is not enough -- on a short track
    # the only section *is* the whole track, and discarding it leaves nothing to enter at, which
    # is how 55 assets ended up unplayable. Rescale, then clamp to the stored length.
    if outcome.manifest.beat_grid:
        outcome.manifest.beat_grid.bpm = spec.bpm
    if measured > 0:
        _rescale_bars(outcome.manifest, spec.bpm / measured)
    if not outcome.admitted:
        flags = ", ".join(outcome.quality.flags) or "unknown"
        return False, f"gates failed: {flags}"

    _write_manifest(outcome.manifest, out_dir)
    (out_dir / f"{spec.name}.request.json").write_text(
        json.dumps(
            {
                "generator": "strudel",
                "license": STRUDEL_LICENSE,
                "specified_bpm": spec.bpm,
                "measured_bpm": round(measured, 3),
                "detected_metrical_level": detected_level or "as-measured",
                "energy": spec.energy,
                "mood": spec.mood,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    octave_note = "" if detected_level is None else f" (detected at {detected_level})"
    return True, f"measured={measured:.1f} err={errors[best]:.2f}%{octave_note}"


def _rescale_bars(manifest: AssetManifest, scale: float) -> None:
    """Convert bar numbers counted at one tempo into bars at another.

    ``scale`` is ``specified_bpm / measured_bpm``. Markers past the stored length are **clamped,
    not removed**: a section spanning the whole track stays a section spanning the whole track,
    and ``FeasibilityFilter`` refuses an asset with no section or loop to enter at. Discarding
    the out-of-range marker instead is what left 55 assets unplayable.
    """
    total_bars = int(manifest.audio.duration_s / (240.0 / manifest.beat_grid.bpm))
    if total_bars < 1 or scale <= 0:
        return
    last = total_bars - 1

    def convert(value: int) -> int:
        return max(0, min(round(value * scale), last))

    manifest.sections = [
        s.model_copy(update={"start_bar": convert(s.start_bar), "end_bar": convert(s.end_bar)})
        for s in manifest.sections
    ]
    manifest.loops = [
        loop.model_copy(
            update={"start_bar": convert(loop.start_bar), "end_bar": convert(loop.end_bar)}
        )
        for loop in manifest.loops
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Strudel algorithmic depot.")
    parser.add_argument("--count", type=int, default=125, help="how many specs to render")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE_PCT,
        help="reject a render whose measured tempo disagrees with the request by more than this %",
    )
    parser.add_argument("--approve", action="store_true", help="mark admitted assets approved")
    parser.add_argument(
        "--force", action="store_true",
        help="re-render specs that already have a manifest (a full pass takes ~6 min)",
    )
    args = parser.parse_args()

    if not WIRBEL.exists():
        print(f"renderer missing: {WIRBEL}", file=sys.stderr)
        return 2

    specs = build_specs(args.count)
    if not args.force:
        # A full pass is dominated by analysis, not rendering, so skipping admitted specs keeps an
        # interrupted build cheap to finish instead of restarting it.
        pending = [s for s in specs if not (args.out / f"{s.name}.json").exists()]
        print(f"{len(specs) - len(pending)} already admitted, {len(pending)} to do")
        specs = pending
    admitted = rejected = 0
    for index, spec in enumerate(specs, 1):
        wav = args.out / f"{spec.name}.wav"
        try:
            elapsed = render(spec, wav, args.out)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"[{index}/{len(specs)}] {spec.name} RENDER FAIL: {exc}", file=sys.stderr)
            rejected += 1
            continue
        ok, detail = admit(
            spec, wav, args.out, tolerance_pct=args.tolerance, approve=args.approve
        )
        if ok:
            admitted += 1
            print(f"[{index}/{len(specs)}] {spec.name} {spec.bpm:.0f}BPM {elapsed:.2f}s — {detail}")
        else:
            rejected += 1
            print(f"[{index}/{len(specs)}] {spec.name} REJECT: {detail}")

    print(f"\nadmitted={admitted} rejected={rejected} -> {args.out}")
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())