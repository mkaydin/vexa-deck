"""Render and measure depot transitions, retaining the rule gate and preview paths.

Usage: uv run --no-sync python tools/evaluate_transitions.py --theme 'dark melodic techno'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compatibility import compatibility, load_profiles
from vexa_audio.cues import CueMap
from vexa_audio.engine import AudioEngine
from vexa_audio.preview import PreviewRenderer
from vexa_contracts import MusicalClock, SessionState
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.feasibility import FeasibilityFilter, FilterConfig
from vexa_orchestrator.prepare import prepare_for_tempo
from vexa_orchestrator.transitions import score_rendered_transition

ROOT = Path(__file__).resolve().parent.parent


def evaluate(theme: str, library: Path, out: Path) -> dict[str, object]:
    depot = Depot(library)
    matches = depot.search(theme, limit=80)
    if not matches:
        raise ValueError(f"no depot match for {theme!r}")
    engine = AudioEngine()
    profiles = load_profiles(library)
    renderer = PreviewRenderer(out_dir=out.parent / "previews")
    source = next((asset for _, asset in matches if _cueable(asset.path)), None)
    if source is None:
        raise ValueError("no theme match has valid measured cues")
    source_track = engine.preload(source.path, bpm=source.manifest.beat_grid.bpm,
                                  expected_hash=source.manifest.content_sha256)
    source_cue = CueMap.load(source.path.with_suffix(".cues.json"))
    state = SessionState(session_id="evaluation", theme=theme,
                         clock=MusicalClock(bpm=source.manifest.beat_grid.bpm))
    shortlist = [asset for _, asset in matches
                 if asset.manifest.asset_id != source.manifest.asset_id]
    result = FeasibilityFilter(FilterConfig(max_tempo_ratio=1.05)).build(
        state=state, library=[asset.manifest for asset in shortlist],
        current_asset=source.manifest,
        recent_family_ids=(source.manifest.family_id,))
    allowed = {action.asset_id for action in result.transitions}
    rows = []
    for asset in shortlist:
        if asset.manifest.asset_id not in allowed or not _cueable(asset.path):
            continue
        cue = CueMap.load(asset.path.with_suffix(".cues.json"))
        track, cue = prepare_for_tempo(engine, asset, cue, state.clock.bpm,
                                       out.parent / "prepared")
        measured = score_rendered_transition(renderer, source_track, track,
            source_cue, cue, name=f"{source.manifest.asset_id}_to_{asset.manifest.asset_id}")
        heuristic = (compatibility(profiles[source.manifest.asset_id],
                                   profiles[asset.manifest.asset_id])
                     if source.manifest.asset_id in profiles
                     and asset.manifest.asset_id in profiles else None)
        rows.append({"asset_id": asset.manifest.asset_id,
                     "measured_score": measured.score,
                     "profile_score": heuristic,
                     "rms_jump_db": measured.rms_jump_db,
                     "spectral_distance": measured.spectral_distance,
                     "peak_dbfs": measured.peak_dbfs,
                     "preview": str(measured.preview_path)})
    rows.sort(key=lambda row: (-row["measured_score"], row["asset_id"]))
    payload = {"theme": theme, "source": source.manifest.asset_id,
               "rule_allowed": len(allowed), "rule_rejected": len(result.rejections),
               "rendered": len(rows), "rule_choice": rows[0]["asset_id"] if rows else None,
               "candidates": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _cueable(path: Path) -> bool:
    try:
        cue = CueMap.load(path.with_suffix(".cues.json"))
        return cue.valid_for(path) and cue.entry_s is not None and cue.exit_s is not None
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default="dark melodic techno")
    parser.add_argument("--library", type=Path, default=ROOT / "assets/library")
    parser.add_argument("--out", type=Path, default=ROOT / "var/reports/transitions.json")
    args = parser.parse_args()
    report = evaluate(args.theme, args.library, args.out)
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
