"""Audit existing renders on protected copies in the isolated, low-priority worker."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in ("packages/contracts", "services/audio", "services/orchestrator", "workers/yue2"):
    sys.path.insert(0, str(ROOT / source / "src"))


def main():
    from vexa_orchestrator.background import run_job

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--theme", default="audio audit")
    parser.add_argument(
        "--duration", type=float, help="Expected duration; defaults to file duration"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "var/reports/audio-evidence")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for source in args.audio:
        import soundfile as sf

        output = args.out / f"audit-{uuid.uuid4().hex[:12]}.wav"
        body = run_job(
            "audit_audio",
            {
                "source": str(source.resolve()),
                "output": str(output.resolve()),
                "theme": args.theme,
                "duration_s": args.duration or sf.info(source).duration,
            },
            timeout_s=900,
        )
        evidence = body.get("evidence") or {}
        summary.append(
            {
                "source": str(source),
                "report": str(output.with_suffix(".evidence.json")),
                "status": evidence.get("status"),
                "admission": evidence.get("admission"),
                "corrections": evidence.get("corrections"),
                "error": body.get("error"),
            }
        )
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    (args.out / "audit-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
