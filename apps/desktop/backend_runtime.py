"""Resolve local backend packages from the checkout, including after drive remounts."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def workspace_pythonpath(root: Path, inherited: str = "") -> str:
    """Prepend each declared workspace's src directory to Python's search path."""
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        members = project["tool"]["uv"]["workspace"]["members"]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"Cannot read backend workspace configuration: {error}") from error
    sources = [str((root / member / "src").resolve()) for member in members]
    missing = [source for source in sources if not Path(source).is_dir()]
    if missing:
        raise RuntimeError(f"Backend source directories missing: {', '.join(missing)}")
    if inherited:
        sources.extend(inherited.split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(sources))


if __name__ == "__main__":
    print(workspace_pythonpath(Path(__file__).resolve().parents[2]))
