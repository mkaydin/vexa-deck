"""Install pinned minimaxmusic.cpp and a selected five-component GGUF set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/ServeurpersoCom/minimaxmusic.cpp.git"
SOURCE_REVISION = "06668b14ab87c106127c9012411aecde92d2c03b"
MODEL_REPO = "Serveurperso/MiniMax-Music3-GGUF"
MODEL_REVISION = "d4f7a6402a4ca10c8b3ca2f648c8a46c103558bc"
PATCH = ROOT / "tools/patches/minimax-min-duration.patch"
PRESETS = {
    "balanced": ("Q6_K", "Q8_0"),
    "quality": ("Q8_0", "Q8_0"),
    "compact": ("Q5_K_M", "Q5_K_M"),
}


def weights(preset):
    lm, dit = PRESETS[preset]
    return [
        f"MiniMax-Music3-language_model-{lm}.gguf",
        "MiniMax-Music3-rvq_depth_decoder-Q8_0.gguf",
        f"MiniMax-Music3-transformer-{dit}.gguf",
        "MiniMax-Music3-condition_encoder-F32.gguf",
        "MiniMax-Music3-vocoder-F32.gguf",
        "LICENSE",
    ]


def install(args):
    source = args.directory.resolve()
    models = args.models.resolve()
    steps = [
        ["git", "clone", "--no-checkout", SOURCE_URL, str(source)],
        ["git", "-C", str(source), "checkout", "--detach", SOURCE_REVISION],
        ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(source / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=120",
            "-DCMAKE_CUDA_COMPILER=" + str(shutil.which("nvcc") or "nvcc"),
        ],
        ["cmake", "--build", str(source / "build"), "--target", "mm-synth", "-j", str(args.jobs)],
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "commands": steps,
                    "source_patch": str(PATCH),
                    "model_repo": MODEL_REPO,
                    "revision": MODEL_REVISION,
                    "files": weights(args.preset),
                    "models": str(models),
                },
                indent=2,
            )
        )
        return
    for command in ("git", "cmake", "nvcc"):
        if not shutil.which(command):
            raise RuntimeError(f"{command} is required for the RTX 5060 Ti native build")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        for step in steps[:2]:
            subprocess.run(step, check=True)
    else:
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(source), "diff", "HEAD", "--"], text=True
        )
        if revision != SOURCE_REVISION or (diff and diff != PATCH.read_text()):
            raise RuntimeError(
                "Preserve the existing native checkout; choose a separate --directory"
            )
    subprocess.run(steps[2], check=True)
    # Accept only the pinned source or this exact patch, preserving other checkouts.
    applied = subprocess.run(
        ["git", "-C", str(source), "apply", "--reverse", "--check", str(PATCH)],
        capture_output=True,
    ).returncode == 0
    if not applied:
        subprocess.run(["git", "-C", str(source), "apply", str(PATCH)], check=True)
    for step in steps[3:]:
        subprocess.run(step, check=True)
    patch_hash = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    (source / "build/mm-synth.vexa.json").write_text(
        json.dumps({"runtime_patch_sha256": patch_hash, "source_revision": SOURCE_REVISION}) + "\n"
    )
    os.environ.setdefault("HF_HOME", str(ROOT / "models/hf-cache"))
    from huggingface_hub import hf_hub_download

    models.mkdir(parents=True, exist_ok=True)
    for filename in weights(args.preset):
        print("Downloading " + filename, flush=True)
        hf_hub_download(MODEL_REPO, filename, revision=MODEL_REVISION, local_dir=models)
    (models / "vexa-download.json").write_text(
        json.dumps(
            {
                "source_revision": SOURCE_REVISION,
                "runtime_patch_sha256": patch_hash,
                "model_revision": MODEL_REVISION,
                "model_repo": MODEL_REPO,
                "preset": args.preset,
                "files": weights(args.preset),
            },
            indent=2,
        )
        + "\n"
    )
    print("MiniMax Music 3 ready: " + str(source / "build/mm-synth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "third_party/minimaxmusic.cpp")
    parser.add_argument("--models", type=Path, default=ROOT / "models/minimax-music3")
    parser.add_argument("--preset", choices=PRESETS, default="balanced")
    parser.add_argument("--jobs", type=int, choices=range(1, 17), default=2)
    parser.add_argument("--dry-run", action="store_true")
    install(parser.parse_args())
