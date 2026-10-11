"""Expiring process leases defer optional observers while decks need CPU time."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = ROOT / "var/audio-priority"


def publish(reason: str | None):
    path = DIRECTORY / f"{os.getpid()}.json"
    if reason is None:
        path.unlink(missing_ok=True)
        return
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"reason": reason, "expires": time.time() + 10}))
    temporary.replace(path)


def deferred_reason() -> str | None:
    for path in DIRECTORY.glob("*.json"):
        try:
            lease = json.loads(path.read_text())
            if time.time() < lease["expires"] <= time.time() + 11:
                return str(lease["reason"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None
