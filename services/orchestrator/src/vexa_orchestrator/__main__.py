"""Run the orchestrator.

``python -m vexa_orchestrator`` — the container entrypoint.

Configuration comes from the environment so one image serves every deployment. The defaults are
chosen so the service starts and answers ``/health`` with no configuration at all, because a
container that will not boot without a secret is a container nobody can debug.
"""

from __future__ import annotations

import logging
import os


def main() -> int:
    import uvicorn

    from .api import create_app
    from .depot import Depot
    from .live import LiveSet
    from .policy import RulePolicy

    host = os.environ.get("VEXA_HOST", "0.0.0.0")
    port = int(os.environ.get("VEXA_PORT", "8000"))
    log_level = os.environ.get("VEXA_LOG_LEVEL", "info")
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # The rule policy is the default and the permanent fallback. A learned policy is added
    # behind the same interface once it has passed shadow mode; nothing here changes when that
    # happens, which is the point of keeping the policy pluggable.
    from pathlib import Path

    library = Path(os.environ.get("VEXA_LIBRARY", "assets/library"))
    live = (LiveSet(Depot(library), min_track_duration_s=float(
        os.environ.get("VEXA_MIN_TRACK_DURATION_S", "148")))
        if os.environ.get("VEXA_LIVE", "1") == "1" else None)
    app = create_app(policy=RulePolicy(), live=live)

    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
