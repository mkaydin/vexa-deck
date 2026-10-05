"""Measure the real local ten-track planner, and optionally render its first track.

Usage: .venv/bin/python tools/check_music_planner.py [--render]
Plays only a quiet test tone; leaves existing app sessions untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vexa_audio.engine import AudioEngine
from vexa_audio.loader import PreloadedTrack
from vexa_orchestrator.generation import isolated_generate_theme, plan_theme

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--theme",
        default="90s underground rap about a factory worker returning home through rainy streets",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = AudioEngine()
    t = np.arange(engine.sample_rate * 2, dtype=np.float32) / engine.sample_rate
    samples = np.repeat((0.005 * np.sin(t * 2 * np.pi * 220))[:, None], 2, axis=1)
    tone = PreloadedTrack("planner-check", samples, engine.sample_rate, 2, duration_s=2)
    engine.load_and_play(tone)
    engine.set_loop(True)
    engine.start()
    time.sleep(1)
    before = engine.snapshot()["stats"]
    cancel = threading.Event()
    started = time.monotonic()
    report = {"theme": args.theme, "errors": []}
    gpu_snapshots = []
    monitor_stop = threading.Event()

    def monitor():
        while not monitor_stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            gpu_snapshots.append(
                {
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "devices": result.stdout.strip(),
                }
            )
            monitor_stop.wait(2)

    monitoring = threading.Thread(target=monitor, daemon=True)
    monitoring.start()
    try:
        briefs = plan_theme(args.theme, count=args.count, cancel=cancel)
        report["planning_s"] = time.monotonic() - started
        report["tracks"] = [asdict(brief) for brief in briefs]
        if args.render:
            brief = briefs[0]
            asset = isolated_generate_theme(
                args.theme,
                ROOT / "assets/library",
                style=brief.style,
                lyrics=brief.lyrics,
                duration_s=brief.duration_s,
                energy=brief.energy,
                prompt_origin=brief.origin,
                music_plan=brief.music_plan,
                cancel=cancel,
            )
            report["admitted_audio"] = str(asset.path)
    except BaseException as exc:
        report["errors"].append(str(exc) or type(exc).__name__)
        raise
    finally:
        monitor_stop.set()
        monitoring.join(timeout=6)
        report["gpu_snapshots"] = gpu_snapshots
        after = engine.snapshot()["stats"]
        report.update(
            elapsed_s=time.monotonic() - started,
            before=before,
            after=after,
            underflows=after["output_underflows"] - before["output_underflows"],
            callback_errors=after["underruns"] - before["underruns"],
        )
        path = ROOT / "var/reports/local-music-planner-qa.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        engine.stop()
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("tracks", "gpu_snapshots")
                },
                indent=2,
            )
        )
        if report["underflows"] or report["callback_errors"]:
            raise RuntimeError("Audio glitch during local planning; see report")


if __name__ == "__main__":
    main()
