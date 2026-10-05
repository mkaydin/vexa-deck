"""Environment for relocatable local GGML executables."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path


def native_environment(binary: str | Path,
                       inherited: Mapping[str, str] | None = None) -> dict[str, str]:
    """Find bundled shared libraries beside the executable after a checkout moves."""
    environment = dict(os.environ if inherited is None else inherited)
    executable = Path(shutil.which(str(binary)) or binary).resolve()
    directories = [executable.parent, executable.parent / "lib", executable.parent / "lib64"]
    paths = [str(directory) for directory in directories if directory.is_dir()]
    paths.extend(part for part in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                 if part)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(paths))
    return environment
