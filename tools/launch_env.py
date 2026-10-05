"""Load application .env values as data, then re-exec the launcher with inherited overrides."""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

script = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[1]
values = dotenv_values(root / ".env")
environment = {
    key: value
    for key, value in values.items()
    if value is not None and key.startswith(("VEXA_", "PLANNER_"))
}
environment.update(os.environ)
environment["VEXA_ENV_LOADED"] = "1"
os.execve("/usr/bin/bash", ["/usr/bin/bash", str(script), *sys.argv[2:]], environment)
