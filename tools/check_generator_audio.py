"""Benchmark an isolated alternative generator without opening audio or adding depot tracks."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
from backend_runtime import workspace_pythonpath  # noqa: E402

for path in workspace_pythonpath(ROOT).split(":"):
    sys.path.insert(0, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("minimax", "acestep"), default="minimax")
    parser.add_argument("--preset", default=None)
    parser.add_argument(
        "--case", choices=("phonk", "memphis", "hiphop", "techno", "vocal"), default="phonk"
    )
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "var/reports/generator-audio")
    args = parser.parse_args()
    if not 5 <= args.duration <= 180:
        parser.error("duration must be between 5 and 180 seconds")
    import subprocess

    import numpy as np
    import soundfile as sf
    from check_acestep_audio import CASES, LYRICS
    from vexa_orchestrator.background import gpu_job_lock
    from vexa_orchestrator.generation import _gpu_index
    from vexa_orchestrator.generators import render_alternative, selection

    generator, preset = selection(args.generator, args.preset, require_available=True)
    gpu = _gpu_index()
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{generator}-{preset}-{args.case}-{args.duration:g}s-{int(time.time())}"
    wav = args.output / f"{stem}.wav"
    report_path = args.output / f"{stem}.report.json"
    report = {
        "status": "running",
        "generator": generator,
        "preset": preset,
        "case": args.case,
        "requested_duration_s": args.duration,
        "gpu_uuid": gpu,
        "listening_review": "pending",
        "live_audio_test": "pending",
        "added_to_depot": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    stop = threading.Event()
    memory = []

    def monitor():
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    gpu,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            with suppress(ValueError):
                memory.append(float(result.stdout.strip()))
            stop.wait(1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        bpm, key, style = CASES[args.case]
        with gpu_job_lock():
            started = time.monotonic()
            meta = render_alternative(
                generator,
                preset,
                wav,
                wav.with_suffix(".request.json"),
                style=style,
                lyrics=LYRICS if args.case == "vocal" else "",
                duration_s=args.duration,
                music_plan={"bpm": bpm, "key": key},
            )
            report.update(meta, wall_s=time.monotonic() - started)
        samples, sr = sf.read(wav, always_2d=True)
        report.update(
            path=str(wav),
            duration_s=len(samples) / sr,
            sample_rate=sr,
            channels=samples.shape[1],
            finite=bool(np.isfinite(samples).all()),
            peak_dbfs=float(20 * np.log10(max(np.abs(samples).max(), 1e-12))),
            rms_dbfs=float(20 * np.log10(max(np.sqrt(np.mean(samples**2)), 1e-12))),
        )
        report["duration_matches"] = abs(report["duration_s"] - args.duration) <= 2
        if not report["finite"] or not report["duration_matches"] or report["channels"] != 2:
            raise RuntimeError("Rendered audio failed benchmark integrity/duration checks")
        report["status"] = "generated"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        stop.set()
        watcher.join(6)
        report["peak_device_memory_mib"] = max(memory) if memory else None
        report["memory_note"] = "Total device usage sampled every second; includes other processes."
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
