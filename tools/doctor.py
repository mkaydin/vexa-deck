"""Read-only installation checks. Does not download weights or open an audio device."""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(generation: bool = False) -> dict:
    rows = []
    # dotenv is installed with backend dependencies; exported variables retain precedence.
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(ROOT / ".env", override=False)

    def record(name, ready, detail, required=True):
        rows.append({"check": name, "ok": bool(ready), "required": required, "detail": str(detail)})

    python = ROOT / ".venv/bin/python"
    record("Python runtime", python.is_file(), python)
    if python.is_file():
        environment = dict(os.environ)
        result = subprocess.run(
            [str(python), str(ROOT / "apps/desktop/backend_runtime.py")],
            capture_output=True,
            text=True,
        )
        environment["PYTHONPATH"] = result.stdout.strip()
        probe = subprocess.run(
            [
                str(python),
                "-c",
                "import vexa_orchestrator, vexa_audio, sounddevice, soundfile, fastapi, uvicorn",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        record("Backend imports", probe.returncode == 0, probe.stderr.strip() or "ready")
    modules = "import PySide6.QtWidgets, PySide6.QtMultimedia, PySide6.QtWebEngineWidgets"
    qt = None
    for candidate in (str(python), "/usr/bin/python3"):
        if Path(candidate).is_file():
            probe = subprocess.run([candidate, "-c", modules], capture_output=True, text=True)
            if probe.returncode == 0:
                qt = candidate
                break
    record("Qt Widgets/Multimedia/WebEngine", qt is not None, qt or "install the desktop extra")
    record("PortAudio library", ctypes.util.find_library("portaudio"), "system PortAudio")
    record("FFmpeg", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "not found")
    record(
        "Desktop session",
        os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"),
        os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY") or "headless",
        False,
    )
    for name, path in (
        (
            "YuE2 renderer",
            Path(
                os.environ.get("VEXA_YUE_BINARY") or ROOT / "third_party/yue2.cpp/build/yue-synth"
            ),
        ),
        (
            "YuE2 backbone",
            Path(
                os.environ.get("VEXA_YUE_MODEL")
                or ROOT / "third_party/yue2.cpp/models/YuE2-3B-Q8_0.gguf"
            ),
        ),
        (
            "YuE2 VAE",
            Path(
                os.environ.get("VEXA_YUE_VAE")
                or ROOT / "third_party/yue2.cpp/models/YuE2-Vae-F32.gguf"
            ),
        ),
        (
            "ACE text planner",
            Path(
                os.environ.get("VEXA_ACE_MODEL")
                or ROOT / "models/acestep-5Hz-lm-1.7B/model.safetensors"
            ),
        ),
    ):
        if name == "ACE text planner" and path.is_dir():
            path /= "model.safetensors"
        record(name, path.is_file(), path, generation)
    record("Ollama", shutil.which("ollama"), "local lyric writer", generation)
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        probe = subprocess.run(
            [nvidia, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
        )
        record("RTX 5060 Ti", "RTX 5060 Ti" in probe.stdout, probe.stdout.strip(), generation)
    else:
        record("RTX 5060 Ti", False, "nvidia-smi unavailable", generation)
    return {"ok": all(row["ok"] for row in rows if row["required"]), "checks": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation", action="store_true", help="require local generation components"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check(args.generation)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for row in report["checks"]:
            state = "OK" if row["ok"] else "MISSING" if row["required"] else "OPTIONAL"
            print(f"[{state}] {row['check']}: {row['detail']}")
    raise SystemExit(0 if report["ok"] else 1)
