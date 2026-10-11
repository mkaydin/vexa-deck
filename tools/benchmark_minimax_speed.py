"""Compare denoising speed on one saved song, with full-track admission on a new copy.

The replay skips fresh autoregressive sampling. Estimated fresh-render savings use the
source's measured AR/native timing and the measured denoising difference; replay runtime
itself is never presented as fresh text-to-music generation speed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for member in ("packages/contracts", "services/audio", "services/orchestrator", "workers/yue2"):
    sys.path.insert(0, str(ROOT / member / "src"))


def main():
    from vexa_orchestrator.background import gpu_job_lock
    from vexa_orchestrator.generation import admit_render
    from vexa_orchestrator.generators import (
        MINIMAX_STEPS,
        components,
        minimax_timings,
        run_renderer,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--speed", choices=MINIMAX_STEPS, default="fast")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out.resolve() / (source.stem + "-" + args.speed + ".wav")
    if output.exists():
        raise ValueError("benchmark output already exists")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    brief = json.loads(source.with_suffix(".brief.json").read_text())
    request = json.loads(source.with_suffix(".replay.json").read_text())
    if not request.get("audio_codes"):
        raise ValueError("a saved native audio-code replay is required")
    request["steps"] = MINIMAX_STEPS[args.speed]
    request_path = output.with_suffix(".request.json")
    request_path.write_text(json.dumps(request) + "\n")
    baseline = minimax_timings(source.with_suffix(".render.log"))
    report = {
        "state": "waiting_for_gpu",
        "source": str(source),
        "steps": request["steps"],
        "baseline_native_timings": baseline,
        "method": "same-code full-length denoising replay",
    }
    report_path = args.out / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print("Waiting for the shared generation GPU lock", flush=True)
    os.nice(10)
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) > 2:
            os.sched_setaffinity(0, cpus[2:])
    paths = components("minimax", brief.get("preset", "balanced"))
    started = time.monotonic()
    with gpu_job_lock():
        report["queue_wait_s"] = round(time.monotonic() - started, 2)
        report["state"] = "rendering"
        save()
        run_renderer(
            [
                str(paths["binary"]),
                "--models",
                str(paths["lm_model"].parent),
                "--request",
                str(request_path),
                "--out",
                str(output),
                "--max-seq",
                "8192",
            ],
            paths["binary"],
            output.with_suffix(".render.log"),
            None,
            request["duration"],
        )
    timings = minimax_timings(output.with_suffix(".render.log"))
    report["replay_timings"] = timings
    if "total_s" in baseline and "dit_s" in baseline and "dit_s" in timings:
        estimate = baseline["total_s"] - baseline["dit_s"] + timings["dit_s"]
        report["estimated_fresh_native_s"] = round(estimate, 2)
        report["estimated_fresh_speedup"] = round(baseline["total_s"] / estimate, 3)
    plan = copy.deepcopy(brief.get("music_plan") or {})
    plan["generation_settings"] = {
        **(plan.get("generation_settings") or {}),
        "render_speed": args.speed,
    }
    report["state"] = "analyzing"
    save()
    try:
        asset = admit_render(
            output,
            theme=brief["theme"],
            duration_s=brief["requested_duration_s"],
            energy=brief["energy"],
            style=brief["style"],
            lyrics=brief["lyrics"],
            music_plan=plan,
            title=brief["title"],
            generator="minimax",
            preset=brief.get("preset", "balanced"),
            seed=request["seed"],
            render_metadata={
                "render_speed": args.speed,
                "denoising_steps": request["steps"],
                "render_timings": timings,
                "benchmark_source": str(source),
            },
        )
        report["admission"] = "admitted"
        report["duration_s"] = asset.manifest.audio.duration_s
    except ValueError as exc:
        report.update(admission="rejected", error=str(exc))
    report.update(state="completed", output=str(output), audible_review="pending")
    report["source_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == before
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
