"""Build a reproducible Linux source/runtime release without local media or model weights."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Include current workspace source (including newly authored, uncommitted files).
TREES = (
    "apps/desktop",
    "packages",
    "services",
    "workers",
    "tools",
    "scripts",
    "packaging",
    "tests",
    "vexa-deck-docs",
    "assets/icons",
    "assets/docs",
    "data/fixtures",
    "docker",
    "third_party/video-art",
)
FILES = (
    "README.md",
    "LICENSE",
    ".gitignore",
    ".dockerignore",
    ".env.example",
    "pyproject.toml",
    "uv.lock",
    "docker-compose.yml",
    "idea1.md",
)
BLOCKED_PARTS = {
    "__pycache__",
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "models",
    "weights",
    "checkpoints",
    "vexa-videos",
    "node_modules",
    "build",
    "dist",
}
BLOCKED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".gguf",
    ".ggml",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".tflite",
    ".h5",
    ".hdf5",
    ".bin",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".log",
    ".partial",
    ".part",
}
MAX_SOURCE_BYTES = 16 * 1024 * 1024


def release_files(root: Path = ROOT) -> list[Path]:
    paths = [root / name for name in FILES]
    for tree in TREES:
        base = root / tree
        if base.exists():
            paths.extend(base.rglob("*"))
    result = []
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        if str(relative) == "packaging/release.json":
            continue
        if set(relative.parts) & BLOCKED_PARTS or path.suffix.lower() in BLOCKED_SUFFIXES:
            continue
        if any(
            part == ".env" or (part.startswith(".env.") and part != ".env.example")
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"Release source must not contain symlinks: {relative}")
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError(
                f"Unexpected large source file: {relative}; keep artifacts outside source"
            )
        result.append(path)
    for required in (
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "apps/desktop/main.py",
        "apps/desktop/run.sh",
        "scripts/install.sh",
        "tools/install_app.py",
        "apps/desktop/assets/video/renderer.js",
        "assets/icons/vexa-icon-256.png",
    ):
        if root / required not in result:
            raise ValueError(f"Release is missing {required}")
    return result


def inventory(root: Path, paths: list[Path]) -> dict:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    return {
        "name": "vexa-deck",
        "version": version,
        "format": "linux-source-v1",
        "model_weights_included": False,
        "user_media_included": False,
        "files": {
            str(path.relative_to(root)): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
    }


def build(root: Path, output: Path, epoch: int = 0) -> Path:
    paths = release_files(root)
    receipt = inventory(root, paths)
    name = f"vexa-deck-{receipt['version']}-linux"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.tar.gz"
    temporary = archive.with_suffix(".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=epoch, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w") as bundle,
    ):
        for path in paths:
            info = tarfile.TarInfo(f"{name}/{path.relative_to(root)}")
            payload = path.read_bytes()
            info.size = len(payload)
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            info.mtime = epoch
            bundle.addfile(info, io.BytesIO(payload))
        payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo(f"{name}/packaging/release.json")
        info.size, info.mode, info.mtime = len(payload), 0o644, epoch
        bundle.addfile(info, io.BytesIO(payload))
    temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(
        f"Built {archive} ({archive.stat().st_size / 1024**2:.2f} MiB; {len(paths)} source files)"
    )
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    build(ROOT, args.output, int(os.environ.get("SOURCE_DATE_EPOCH", "0")))
