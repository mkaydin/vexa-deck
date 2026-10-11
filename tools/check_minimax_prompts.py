"""Write and inspect real Music 3 structured prompts without rendering or playing audio."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
from backend_runtime import workspace_pythonpath  # noqa: E402

for path in workspace_pythonpath(ROOT).split(":"):
    sys.path.insert(0, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default="Memphis phonk instrumental, 88 BPM")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "var/reports/minimax-prompts")
    args = parser.parse_args()
    from dotenv import load_dotenv
    from vexa_orchestrator.generation import plan_theme
    from vexa_orchestrator.generators import minimax_caption

    load_dotenv(ROOT / ".env", override=False)
    briefs = plan_theme(args.theme, count=args.count, generator="minimax")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, brief in enumerate(briefs, 1):
        caption = minimax_caption(brief.style, brief.lyrics, brief.music_plan,
                                  duration_s=brief.duration_s)
        target = args.output / f"track-{index:02}.caption.txt"
        target.write_text(caption + "\n")
        rows.append({**asdict(brief), "caption_path": str(target),
                     "caption_words": len(caption.split())})
    report = {"theme": args.theme, "generator": "minimax", "tracks": rows,
              "audio_generated": False, "audio_played": False}
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(path), "tracks": len(rows),
                      "origin": briefs[0].origin}, indent=2))


if __name__ == "__main__":
    main()
