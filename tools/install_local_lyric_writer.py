"""Install Gemma in the private Ollama depot used by the single-GPU lyric writer."""

import json
import logging
import os

import httpx
from vexa_orchestrator.ace_planner import local_writer_server

logging.basicConfig(level=logging.INFO)
model = os.environ.get("VEXA_ACE_WRITER_MODEL") or "gemma4:e4b"
with local_writer_server() as base:
    print(f"Installing local lyric writer {model}", flush=True)
    previous = ""
    reported = 0
    with httpx.stream(
        "POST",
        f"{base}/api/pull",
        json={"model": model, "stream": True},
        timeout=httpx.Timeout(120, read=120),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            row = json.loads(line)
            if row.get("error"):
                raise RuntimeError(row["error"])
            status = row.get("status", "")
            completed = row.get("completed", 0)
            if status != previous or completed - reported > 256 * 1024 * 1024:
                print(
                    f"{status} {completed / 1024**2:.0f}/{row.get('total', 0) / 1024**2:.0f} MiB",
                    flush=True,
                )
                previous, reported = status, completed
    print("Local writer ready", flush=True)
