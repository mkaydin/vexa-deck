"""Install a local writer in Vexa's private Ollama depot without running inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
from backend_runtime import workspace_pythonpath  # noqa: E402

for source in workspace_pythonpath(ROOT).split(":"):
    sys.path.insert(0, source)


def main():
    import httpx
    from dotenv import load_dotenv
    from vexa_orchestrator.ace_planner import local_writer_server
    from vexa_orchestrator.writers import WriterProfile, installed_local

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("VEXA_WRITER_MODEL")
        or os.environ.get("VEXA_ACE_WRITER_MODEL")
        or "gemma4:e4b",
    )
    args = parser.parse_args()
    WriterProfile(mode="local", model=args.model)
    logging.basicConfig(level=logging.INFO)
    with local_writer_server(configured_url="") as base:
        print(f"Installing local writer {args.model}", flush=True)
        previous, reported = "", 0
        with httpx.stream(
            "POST",
            f"{base}/api/pull",
            json={"model": args.model, "stream": True},
            timeout=httpx.Timeout(120, read=120),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                row = json.loads(line)
                if row.get("error"):
                    raise RuntimeError(row["error"])
                status, completed = row.get("status", ""), row.get("completed", 0)
                if status != previous or completed - reported > 256 * 1024 * 1024:
                    print(
                        f"{status} {completed / 1024**2:.0f}/"
                        f"{row.get('total', 0) / 1024**2:.0f} MiB",
                        flush=True,
                    )
                    previous, reported = status, completed
        manifest = installed_local(args.model)
        if manifest is None:
            raise RuntimeError("pull completed but the private model manifest is missing")
        print(json.dumps({"model": args.model, **manifest, "inference_run": False}), flush=True)


if __name__ == "__main__":
    main()
