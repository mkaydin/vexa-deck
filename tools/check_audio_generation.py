"""Measure radio playback during real GPU load and isolated candidate preparation.

Plays a quiet tone and swaps silent prepared decks. Does not alter an existing set.
Report: var/reports/audio-generation-load.json. Existing YuE2 GPU work is recorded if present.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from vexa_audio.engine import AudioEngine
from vexa_audio.loader import PreloadedTrack
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.prepared_audio import persist_pcm, prepare_candidate

ROOT = Path(__file__).resolve().parents[1]


def gpu_status() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    ).stdout.strip()


def main() -> None:
    engine = AudioEngine()
    t = np.arange(engine.sample_rate * 2, dtype=np.float32) / engine.sample_rate
    samples = np.repeat((0.008 * np.sin(t * 2 * np.pi * 220))[:, None], 2, axis=1)
    tone = PreloadedTrack("load-check", samples, engine.sample_rate, 2, duration_s=2)
    engine.start()
    engine.load_and_play(tone)
    engine.set_loop(True)
    results, errors, snapshots = [], [], []
    cancel = threading.Event()
    try:
        time.sleep(2)
        before = engine.snapshot()
        gpu_before = gpu_status()
        depot = Depot(ROOT / "assets/library")
        current = depot.assets["live-7ff5e4ae3dd8"]
        from vexa_audio.cues import CueMap
        from vexa_audio.loader import load_track

        current_track = load_track(
            current.path, sample_rate=engine.sample_rate, bpm=current.manifest.beat_grid.bpm
        )
        cue = CueMap.load(current.path.with_suffix(".cues.json"))
        cache = ROOT / "var/reports/audio-load-prepared"
        persist_pcm(current_track, cache)

        def prepare() -> None:
            try:
                for index in range(3):
                    track, _cue, report = prepare_candidate(
                        current,
                        current=current,
                        current_track=current_track,
                        current_cue=cue,
                        target_bpm=current.manifest.beat_grid.bpm * 1.01,
                        cache_dir=cache,
                        preview_dir=ROOT / "var/reports/audio-load-previews",
                        name=f"load-check-{index}",
                        cancel=cancel,
                    )
                    engine.prepare_deck(track, 1)
                    engine.set_gain(0.005, 1)
                    engine.crossfade_to(track, 1, fade_frames=engine.sample_rate)
                    time.sleep(1.2)
                    engine.crossfade_to(tone, 0, fade_frames=engine.sample_rate)
                    time.sleep(1.2)
                    results.append(
                        {
                            "score": report.score,
                            "pcm_frames": track.frames,
                            "memory_mapped": isinstance(track.samples, np.memmap),
                        }
                    )
            except Exception as exc:
                errors.append(str(exc))

        worker = threading.Thread(target=prepare)
        started = time.monotonic()
        worker.start()
        while worker.is_alive():
            snapshots.append(
                {
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "audio": engine.snapshot()["stats"],
                    "gpu": gpu_status(),
                }
            )
            worker.join(timeout=1)
        time.sleep(2)
        after = engine.snapshot()
        report = {
            "before": before,
            "after": after,
            "gpu_before": gpu_before,
            "preparations": results,
            "errors": errors,
            "snapshots": snapshots,
            "elapsed_s": round(time.monotonic() - started, 1),
            "underflows_during_work": (
                after["stats"]["output_underflows"] - before["stats"]["output_underflows"]
            ),
            "callback_errors_during_work": (
                after["stats"]["underruns"] - before["stats"]["underruns"]
            ),
        }
        path = ROOT / "var/reports/audio-generation-load.json"
        path.write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "snapshots"}, indent=2
            )
        )
        if errors or len(results) != 3 or report["underflows_during_work"]:
            raise RuntimeError("audio load verification failed; see report")
    finally:
        cancel.set()
        engine.stop()


if __name__ == "__main__":
    main()
