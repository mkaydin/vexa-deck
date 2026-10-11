"""Check real per-track writer output and native compilation without rendering audio."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
from backend_runtime import workspace_pythonpath  # noqa: E402

for source in workspace_pythonpath(ROOT).split(":"):
    sys.path.insert(0, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default="dark melodic techno instrumental")
    parser.add_argument("--count", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--generator", choices=("yue2", "acestep", "minimax"), default="minimax")
    parser.add_argument("--output", type=Path, default=ROOT / "var/reports/generation-v2")
    args = parser.parse_args()
    from dotenv import load_dotenv
    from vexa_orchestrator.generation_v2 import plan_track, writer_settings

    load_dotenv(ROOT / ".env", override=False)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = writer_settings()
    rows = []
    for slot in range(args.count):
        brief = plan_track(
            args.theme,
            slot,
            args.generator,
            settings=settings,
            cache_path=args.output / f"track-{slot + 1:02}.cache.json",
        )
        rows.append(asdict(brief))
        print(f"{slot + 1}/{args.count}: {brief.title} ({brief.duration_s:g}s)", flush=True)
        report = {
            "theme": args.theme,
            "generator": args.generator,
            "writer": settings["model"],
            "tracks": rows,
            "audio_generated": False,
            "audio_played": False,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
