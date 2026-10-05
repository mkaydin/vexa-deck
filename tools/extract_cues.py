"""Build hash-bound beat and cue reports for the prepared depot.

Run after audio ingestion: uv run --no-sync python tools/extract_cues.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vexa_audio.cues import CueMap, analyse_cues

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "assets" / "library"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=LIBRARY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.library.glob("*.wav"))
    if args.limit:
        paths = paths[: args.limit]
    counts = {"analysed": 0, "cached": 0, "beat": 0, "downbeat": 0, "unknown": 0}
    errors: list[str] = []
    for index, wav in enumerate(paths, 1):
        cue_path = wav.with_suffix(".cues.json")
        try:
            if cue_path.exists() and not args.refresh:
                existing = CueMap.load(cue_path)
                if existing.valid_for(wav):
                    counts["cached"] += 1
                    counts[existing.cue_kind] += 1
                    continue
            manifest = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
            bpm = (manifest.get("beat_grid") or {}).get("bpm")
            cues = analyse_cues(wav, expected_bpm=bpm)
            cues.save(cue_path)
            counts["analysed"] += 1
            counts[cues.cue_kind] += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{wav.name}: {exc}")
        if index % 25 == 0:
            print(f"{index}/{len(paths)}", flush=True)
    print(json.dumps({"tracks": len(paths), **counts, "errors": errors[:20]}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
