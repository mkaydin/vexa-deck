"""Install pinned native YuE2 CUDA source and the two required GGUF models separately."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/ServeurpersoCom/yue2.cpp.git"
SOURCE_REVISION = "11c1ecb084329200e22fcb286e252b847442ea5c"
MODEL_REPO = "Serveurperso/YuE2-GGUF"
MODEL_REVISION = "64b030e3deb6e8150d2b7c0db641ef5a17eca8a3"
WEIGHTS = ("YuE2-3B-Q8_0.gguf", "YuE2-Vae-F32.gguf")


def commands(destination: Path, jobs: int):
    return [
        ["git", "clone", "--no-checkout", SOURCE_URL, str(destination)],
        ["git", "-C", str(destination), "checkout", "--detach", SOURCE_REVISION],
        ["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"],
        [
            "cmake",
            "-S",
            str(destination),
            "-B",
            str(destination / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=native",
        ],
        ["cmake", "--build", str(destination / "build"), "--target", "yue-synth", "-j", str(jobs)],
    ]


def install(destination: Path, jobs: int, dry_run: bool):
    steps = commands(destination, jobs)
    if dry_run:
        print(
            json.dumps(
                {
                    "commands": steps,
                    "models": WEIGHTS,
                    "repo": MODEL_REPO,
                    "revision": MODEL_REVISION,
                    "license": "CC-BY-NC-4.0",
                    "directory": str(destination / "models"),
                },
                indent=2,
            )
        )
        return
    for dependency in ("git", "cmake", "nvcc"):
        if shutil.which(dependency) is None:
            raise RuntimeError(
                f"{dependency} is required; install the native CUDA build prerequisites"
            )
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(steps[0], check=True)
        subprocess.run(steps[1], check=True)
    else:
        revision = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != SOURCE_REVISION:
            raise RuntimeError(
                "Existing YuE2 checkout differs from pinned revision; use --directory"
            )
        dirty = subprocess.check_output(
            ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        if dirty:
            raise RuntimeError(
                "Existing YuE2 checkout has local edits; preserve them before building"
            )
    for step in steps[2:]:
        subprocess.run(step, check=True)
    from huggingface_hub import hf_hub_download

    paths = []
    for filename in WEIGHTS:
        paths.append(
            hf_hub_download(
                MODEL_REPO, filename, revision=MODEL_REVISION, local_dir=destination / "models"
            )
        )
    # No transcriber or alternate quantizations are needed for text-to-song generation.
    (destination / "models/vexa-download.json").write_text(
        json.dumps(
            {
                "source_revision": SOURCE_REVISION,
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "files": paths,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"YuE2 ready: {destination / 'build/yue-synth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "third_party/yue2.cpp")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        parser.error("--jobs must be between 1 and 64")
    install(args.directory.resolve(), args.jobs, args.dry_run)
