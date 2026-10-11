"""Evaluate human vocal annotations against pinned, stored CLAP margins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/orchestrator/src"))


def main():
    from vexa_orchestrator.audio_calibration import evaluate_annotations

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "var/reports/audio-evidence/calibration.json"
    )
    args = parser.parse_args()
    rows = json.loads(args.annotations.read_text())
    # Annotation paths are resolved relative to the annotation file, not an arbitrary cwd.
    for row in rows:
        row["evidence_path"] = str((args.annotations.parent / row["evidence_path"]).resolve())
    result = evaluate_annotations(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
