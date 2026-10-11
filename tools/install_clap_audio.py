"""Install pinned local CLAP observation weights; never included in application packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = "laion/clap-htsat-unfused"
ROOT = Path(__file__).resolve().parents[1]


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models/audio-observers/clap-htsat-unfused"
    )
    args = parser.parse_args()
    # Keep all cache writes within the workspace, including hub metadata.
    os.environ.setdefault("HF_HOME", str(ROOT / "models/.cache/huggingface"))
    info = HfApi().model_info(REPO, revision=args.revision)
    snapshot_download(
        REPO,
        revision=info.sha,
        local_dir=args.output,
        cache_dir=ROOT / "models/.cache/huggingface/hub",
        allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin", "README.md"],
        max_workers=2,
    )
    files = sorted(path for path in args.output.glob("*") if path.is_file())
    if (
        not list(args.output.glob("*.safetensors"))
        and not (args.output / "pytorch_model.bin").is_file()
    ):
        raise RuntimeError("Model weights were not downloaded")
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    receipt = {
        "repo": REPO,
        "revision": info.sha,
        "files": hashes,
        "model_card": f"https://huggingface.co/{REPO}",
        "license": "apache-2.0",
    }
    (args.output / "vexa-install.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"installed": str(args.output), "revision": info.sha}))
    print(f"Set VEXA_CLAP_MODEL={args.output.resolve()} for optional CPU observation.")


if __name__ == "__main__":
    main()
