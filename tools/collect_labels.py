"""Collect the labels the fine-tune needs.

This is the missing link. We have a library, a renderer, and a training loop — and no labels.
No public dataset measures DJ preferences, so they have to be heard.

The loop, from ``LAYA_DATA.md:61`` and ``IDEAS.md:11``:

1. Pick a decision point: a session state plus the genuinely feasible options.
2. Render an identical-length preview of each.
3. A listener picks the best and flags the unacceptable ones.
4. Write a canonical :class:`~vexa_laya.dataset.Annotation`.

Design rules that are not negotiable, because violating them produces a model that looks trained
and is not:

* ``continue_current`` and a safe fallback are **always offered**. Without them the model learns
  to over-transition (``LAYA_DATA.md:57``).
* Options are the ones the **feasibility filter** admitted. The listener ranks; the filter
  decides what is even possible.
* Disagreements are kept. Annotator spread is signal about subjectivity, not noise
  (``LAYA_DATA.md:59``).
* Nothing plays through a device unless ``--audible`` is passed.

Usage::

    uv run --no-sync python tools/collect_labels.py --annotator me --count 20
    uv run --no-sync python tools/collect_labels.py --annotator me --audible
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services/audio/src"))
sys.path.insert(0, str(ROOT / "services/laya/src"))
sys.path.insert(0, str(ROOT / "services/orchestrator/src"))
sys.path.insert(0, str(ROOT / "workers/yue2/src"))

from vexa_audio.loader import load_track  # noqa: E402
from vexa_audio.preview import PreviewRenderer  # noqa: E402
from vexa_contracts import ApprovalState, AssetManifest  # noqa: E402
from vexa_laya.dataset import (  # noqa: E402
    Annotation,
    CandidateType,
    DecisionContext,
    LabelSource,
    StateSnapshot,
    read_jsonl,
    write_jsonl,
)
from vexa_orchestrator.feasibility import FeasibilityFilter  # noqa: E402
from vexa_yue2.gates import analyse  # noqa: E402

LIBRARY = ROOT / "assets" / "library"
OUT = ROOT / "data" / "sessions" / "labels.jsonl"


def load_library() -> dict[str, object]:
    """Every admitted asset, keyed by id.

    The committed manifest is the source of truth, not a fresh analysis. Re-running the analyser
    rebuilds the manifest from the audio alone and throws away the descriptive tags the brief
    carried -- so every energy and mood filter silently becomes inert.
    """
    manifests = {}
    for path in sorted(LIBRARY.glob("*.wav")):
        manifest_path = path.with_suffix(".json")
        if manifest_path.exists():
            manifest = AssetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        else:
            outcome = analyse(path, asset_id=path.stem, approval=ApprovalState.APPROVED)
            manifest = outcome.manifest
        if manifest is not None and manifest.admissible():
            manifests[manifest.asset_id] = manifest
    return manifests


def _session_id(index: int, theme: str) -> str:
    """Stable id for a decision point, including a digest of the request that shaped it."""
    import hashlib

    digest = hashlib.sha256(theme.encode("utf-8")).hexdigest()[:8]
    return f"collect-{index}-{digest}"


def _stratified_order(manifests: dict[str, object]) -> list[str]:
    """Asset ids ordered so early indices span tempo and mood rather than one palette.

    Assets are bucketed by (tempo decade, mood) and dealt round-robin, so consecutive indices land
    in different regions of the library. Deterministic, so every annotator who asks for index *n*
    is shown the same decision -- which is what makes their answers comparable at all.
    """
    buckets: dict[tuple[int, str], list[str]] = {}
    for asset_id, manifest in manifests.items():
        bpm = manifest.beat_grid.bpm if manifest.beat_grid else 0.0
        mood = "?"
        for entry in manifest.tags.get("mood", []):
            mood = entry.value
            break
        buckets.setdefault((int(bpm // 20), mood), []).append(asset_id)

    for members in buckets.values():
        members.sort()
    order: list[str] = []
    keys = sorted(buckets)
    depth = max((len(v) for v in buckets.values()), default=0)
    for step in range(depth):
        for key in keys:
            if step < len(buckets[key]):
                order.append(buckets[key][step])
    return order


def _has_transitions(manifests: dict[str, object], asset_id: str) -> bool:
    """Whether this deck can offer anything other than "continue"."""
    from vexa_contracts import SessionState

    asset = manifests[asset_id]
    state = SessionState(session_id="probe", theme="probe")
    menu = FeasibilityFilter().build(
        state=state, library=list(manifests.values()), current_asset=asset
    )
    return bool(menu.transitions)


def build_decision(
    manifests: dict[str, object], *, index: int, theme: str
) -> tuple[DecisionContext, list, object] | None:
    """One decision point: a session state, the feasible options, and the current track.

    The deck is chosen by **stratified round-robin**, not alphabetically. The library is 200 depot
    renders whose names start with their tempo palette, so ``sorted(ids)[0:20]`` was three originals
    plus seventeen tracks from the single ``depot-100`` palette. An annotator working through that
    would have labelled the same prompt seventeen times and called it judgement.
    """
    if not manifests:
        return None
    order = _stratified_order(manifests)
    # Advance to the first deck that can actually offer a transition. The slowest and fastest
    # tracks are outliers by construction -- at 61 BPM almost nothing in the library sits inside a
    # 10 % tempo ratio -- and a menu of nothing but "continue" teaches a selector that holding is
    # the only move. Deterministic given the same library, so annotators still agree on which
    # decision index *n* refers to.
    current_id = None
    for step in range(len(order)):
        candidate = order[(index + step) % len(order)]
        if _has_transitions(manifests, candidate):
            current_id = candidate
            break
    if current_id is None:
        return None

    from vexa_contracts import SessionState

    state = SessionState(
        # The theme is part of the session id. Five annotators must see the *same* request for
        # their answers to be comparable, and without this a divergent --theme produced two
        # annotations with an identical session_id and family_id but different request text --
        # silently corrupting the agreement measurement the whole exercise exists to produce.
        session_id=_session_id(index, theme),
        theme=theme,
        clock=__import__("vexa_contracts").MusicalClock(
            bpm=manifests[current_id].beat_grid.bpm if manifests[current_id].beat_grid else 120.0
        ),
        fallback_asset_id=current_id,
    )
    # Pass the current asset so the menu cannot offer a transition to itself.
    menu = FeasibilityFilter().build(
        state=state, library=list(manifests.values()), current_asset=manifests[current_id]
    )
    if not menu.transitions:
        return None

    candidates = [
        CandidateType(
            # The canonical safe-action id, not a local alias. ``FeasibilityFilter`` emits
            # ``continue_current``, the adapter recognises only that spelling, and shadow mode
            # logs it. Renaming it here made every "hold" label unjoinable to a shadow
            # observation -- and holding is exactly the choice Laya gets wrong, so the cases
            # that matter most would have been the ones dropped.
            id="continue_current",
            type="continue",
            energy=manifests[current_id].energy(),
            vocals=False,
        )
    ]
    for action in menu.transitions[:5]:
        target = manifests[action.asset_id]
        candidates.append(
            CandidateType(
                id=action.action_id.replace("transition_to_", ""),
                type="transition",
                asset=action.asset_id,
                bpm=target.beat_grid.bpm if target.beat_grid else None,
                energy=target.energy(),
                vocals=False,
            )
        )

    ctx = DecisionContext(
        request=theme,
        current=StateSnapshot(
            bpm=state.clock.bpm,
            key=manifests[current_id].key.value if manifests[current_id].key else None,
            section="outro",
            bars_to_boundary=8,
            energy=manifests[current_id].energy() or 0.5,
            vocals=False,
            last_action="continue",
            recent_families=[current_id],
        ),
        candidates=candidates,
    )
    return ctx, [action for action in menu.transitions[:5]], manifests[current_id]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True, help="who is listening")
    parser.add_argument("--count", type=int, default=10, help="decision points to collect")
    parser.add_argument("--theme", default="a warm, unhurried set that keeps building")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="where to append labels. Defaults to data/sessions/labels.jsonl. Give each "
             "annotator their own file; agreement is measured across files, not within one.",
    )
    parser.add_argument("--audible", action="store_true",
                        help="actually play the previews. Off by default.")
    args = parser.parse_args()

    print(f"library: {LIBRARY}")
    manifests = load_library()
    print(f"admitted assets: {len(manifests)}")
    if len(manifests) < 2:
        print("\nneed at least two schedulable assets to compare a transition", file=sys.stderr)
        return 1

    global OUT
    if args.out:
        OUT = args.out
    existing = list(read_jsonl(OUT)) if OUT.exists() else []
    print(f"existing labels: {len(existing)}")
    if existing:
        sessions = {a.session_id for a in existing}
        if args.theme and not any(s.startswith("collect-0-") or "-" in s for s in sessions):
            pass
        this_theme = _session_id(0, args.theme).rsplit("-", 1)[-1]
        foreign = {s.rsplit("-", 1)[-1] for s in sessions if "-" in s} - {this_theme}
        if foreign:
            print(
                f"\n  WARNING: labels already exist for a different request theme "
                f"({sorted(foreign)}).\n  Annotators only produce comparable agreement on the "
                f"same theme.\n  Either keep --theme as-is, or point --out at a fresh file.",
                file=sys.stderr,
            )

    renderer = PreviewRenderer(sample_rate=44100, out_dir=ROOT / "var" / "previews")
    collected: list[Annotation] = []
    failed = 0

    for index in range(args.count):
        built = build_decision(manifests, index=index, theme=args.theme)
        if built is None:
            print(f"[{index}] no feasible options here; skipping")
            failed += 1
            continue
        ctx, _transitions, current = built

        candidate_ids = [c.id for c in ctx.candidates]
        tracks = []
        for candidate in ctx.candidates:
            if candidate.type == "transition":
                tracks.append((candidate.id, load_track(LIBRARY / f"{candidate.asset}.wav",
                                                        sample_rate=44100)))
            else:
                tracks.append((candidate.id, load_track(LIBRARY / f"{current.asset_id}.wav",
                                                        sample_rate=44100)))
        previews = renderer.render_comparison(
            load_track(LIBRARY / f"{current.asset_id}.wav", sample_rate=44100),
            tracks,
            name=f"decision{index:03d}",
        )
        renderer.render_manifest(previews, Path(f"var/previews/decision{index:03d}.json"))

        print(f"\n[{index}] theme: {args.theme}")
        print(f"     playing: {current.asset_id}")
        for label, preview in previews.items():
            print(f"     {label:26s} {preview.duration_s:5.1f}s  {preview.path.name}")

        if args.audible:
            _play(list(previews.values()))
            print("     (played)")
        else:
            print(f"     previews in {renderer.out_dir} — pass --audible to play")

        try:
            picked = input("     best option (blank to skip): ").strip()
        except EOFError:
            print("\nno input available; stopping")
            break
        if not picked:
            print("     skipped")
            failed += 1
            continue
        # Accept either the printed index or the full label. Typing a 30-character asset id
        # several hundred times per annotator is a good way to get bad labels out of hurry, so the
        # index is the obvious thing to try first and the label stays as a fallback.
        if picked.isdigit() and 0 <= int(picked) < len(previews):
            picked = list(previews)[int(picked)]
        if picked not in previews:
            print(f"     '{picked}' is not one of the options; skipped")
            failed += 1
            continue

        rejected_raw = input("     unacceptable ones, comma separated (blank none): ").strip()
        rejected = [r.strip() for r in rejected_raw.split(",") if r.strip() in previews]
        rejected = [r for r in rejected if r != picked]
        others = [c for c in candidate_ids if c != picked and c not in rejected]

        annotation = Annotation(
            state=ctx,
            preferred_action_id=picked,
            also_acceptable=others,
            rejected_action_ids=rejected,
            family_id=current.asset_id,
            session_id=_session_id(index, args.theme),
            label_source=LabelSource(
                kind="human_pairwise_review",
                annotators=[args.annotator],
                agreement=1.0 / max(1, len(candidate_ids) - len(rejected) - 1),
            ),
            notes=f"previews in var/previews/decision{index:03d}.json",
        )
        collected.append(annotation)
        print(f"     recorded {picked}" + (f", rejected {rejected}" if rejected else ""))

    if collected:
        write_jsonl(OUT, [*existing, *collected])
        print(f"\nwrote {len(collected)} new labels to {OUT}")
        print(f"total now: {len(existing) + len(collected)}")
        print("\nnext: split by family, convert, fine-tune")
    else:
        print(f"\nnothing collected ({failed} skipped)")
    return 0 if collected else 1


def _play(previews) -> None:
    import sounddevice as sd

    for preview in previews:
        import soundfile as sf

        data, sr = sf.read(str(preview.path), dtype="float32")
        sd.play(data, sr, blocking=True)


if __name__ == "__main__":
    raise SystemExit(main())
