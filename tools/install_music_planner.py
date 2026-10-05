#!/usr/bin/env python3
"""Download only ACE-Step's text planner, never its DiT/VAE/audio weights."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
REPO = "ACE-Step/Ace-Step1.5"
REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
COMPONENT = "acestep-5Hz-lm-1.7B"


def main() -> None:
    destination = Path(os.environ.get("VEXA_ACE_MODEL") or ROOT / "models" / COMPONENT)
    destination.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=120)) as client:
        response = client.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}")
        response.raise_for_status()
        files = [
            row["rfilename"]
            for row in response.json()["siblings"]
            if row["rfilename"].startswith(f"{COMPONENT}/")
        ]
        receipt = {"repo": REPO, "revision": REVISION, "files": {}}
        for remote in files:
            target = destination / Path(remote).name
            if target.exists():
                print(f"Already present: {target.name}", flush=True)
            else:
                print(f"Downloading {target.name}", flush=True)
                partial = target.with_suffix(target.suffix + ".partial")
                total = 0
                last_report = 0
                with client.stream(
                    "GET", f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
                ) as stream:
                    stream.raise_for_status()
                    with partial.open("wb") as output:
                        for block in stream.iter_bytes(1024 * 1024):
                            output.write(block)
                            total += len(block)
                            if total - last_report >= 256 * 1024 * 1024:
                                print(f"  {target.name}: {total / 1024**2:.0f} MiB", flush=True)
                                last_report = total
                partial.replace(target)
            with target.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            receipt["files"][target.name] = {"sha256": digest, "bytes": target.stat().st_size}
        # Fail early on interrupted/partial installations.
        for required in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            if required not in receipt["files"]:
                raise RuntimeError(f"Required model file missing: {required}")
        (destination / "vexa-download.json").write_text(json.dumps(receipt, indent=2))
        print(f"Planner ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
