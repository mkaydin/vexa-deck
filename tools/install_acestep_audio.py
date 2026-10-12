"""Install pinned ACE-Step Turbo audio weights and its separate upstream environment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/ace-step/ACE-Step-1.5.git"
SOURCE_REVISION = "ca1e85fe9430179831e6bc6be790c332190a3866"
MODEL_REPO = "ACE-Step/Ace-Step1.5"
MODEL_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "third_party/ACE-Step-1.5")
    parser.add_argument("--models", type=Path, default=ROOT / "models/acestep-audio")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.directory.resolve()
    steps = [
        ["git", "clone", "--no-checkout", SOURCE, str(source)],
        ["git", "-C", str(source), "checkout", "--detach", SOURCE_REVISION],
    ]
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    sync = [uv, "sync", "--locked", "--no-dev", "--python", "3.12"]
    patterns = ["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "LICENSE*"]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "commands": steps,
                    "sync_in_source": sync,
                    "patterns": patterns,
                    "model_repo": MODEL_REPO,
                    "model_revision": MODEL_REVISION,
                },
                indent=2,
            )
        )
        return
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        for step in steps:
            subprocess.run(step, check=True)
    else:
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if revision != SOURCE_REVISION or dirty:
            raise RuntimeError("Preserve the existing ACE-Step checkout; choose --directory")
    subprocess.run(sync, cwd=source, check=True)
    os.environ.setdefault("HF_HOME", str(ROOT / "models/hf-cache"))
    from huggingface_hub import snapshot_download

    shared = ROOT / "models/acestep-5Hz-lm-1.7B"
    link = args.models / "acestep-5Hz-lm-1.7B"
    args.models.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        if link.is_symlink():
            link.unlink()
        if shared.is_dir():
            link.symlink_to(os.path.relpath(shared, args.models), target_is_directory=True)
        else:
            patterns.append("acestep-5Hz-lm-1.7B/*")
    if args.runtime_only:
        return
    snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, local_dir=args.models, allow_patterns=patterns
    )
    (args.models / "download-revision.json").write_text(
        json.dumps(
            {"repo": MODEL_REPO, "revision": MODEL_REVISION, "source_revision": SOURCE_REVISION},
            indent=2,
        )
        + "\n"
    )
    print("ACE-Step audio environment and Turbo weights ready")


if __name__ == "__main__":
    main()
