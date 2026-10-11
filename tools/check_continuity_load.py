"""Real device soak with production scheduling, native generation and an optional Wayland GUI.

Uses an isolated report/generation directory and quiet deck gains. Existing depot audio is
read-only. Theme changes exercise a wider prepared inventory; no repeat policy is bypassed.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for member in (
    "packages/contracts",
    "services/audio",
    "services/orchestrator",
    "workers/yue2",
    "services/planner",
    "services/laya",
):
    sys.path.insert(0, str(ROOT / member / "src"))


def atomic(path, body):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, indent=2, default=str))
    temporary.replace(path)


def viewer(status_path):
    sys.path.insert(0, str(ROOT / "apps/desktop"))
    from main import VexaWindow
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    if app.platformName() != "wayland":
        raise RuntimeError("Wayland platform required for the GUI soak")
    window = VexaWindow(preview=True)
    window.api_label.setText("VALIDATION / HOST AUDIO")
    window.setWindowTitle("VEXA // P6 playback validation (isolated, quiet)")
    window.resize(960, 1000)
    window.show()
    captured = False

    def update():
        nonlocal captured
        try:
            body = json.loads(status_path.read_text())
            window.on_status(body["live"])
            if body["elapsed_s"] >= 30 and not captured:
                window.grab().save(str(status_path.parent / "wayland-playback.png"))
                captured = True
            if body["state"] != "running":
                window.close()
                app.quit()
        except (OSError, ValueError, KeyError):
            pass

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(500)
    app.exec()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--viewer", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "var/reports/continuity-load")
    parser.add_argument("--generator", default="minimax", choices=["minimax", "yue2", "acestep"])
    parser.add_argument("--renders", type=int, default=2)
    parser.add_argument("--theme", default="dark")
    parser.add_argument("--no-steer", action="store_true")
    parser.add_argument("--seed-library", type=Path, action="append", default=[])
    parser.add_argument("--production-generation", action="store_true")
    args = parser.parse_args()
    if args.viewer:
        viewer(args.viewer)
        return
    if args.duration <= 0 or not 0 <= args.renders <= 10:
        parser.error("duration must be positive and renders must be between 0 and 10")
    if args.production_generation and args.renders == 0:
        parser.error("production generation needs at least one requested render")
    from vexa_audio.engine import AudioEngine, EngineConfig
    from vexa_contracts import GenerationSettings
    from vexa_orchestrator.depot import Depot
    from vexa_orchestrator.generation import _gpu_index, isolated_generate_theme
    from vexa_orchestrator.generation_v2 import plan_track, writer_settings
    from vexa_orchestrator.generators import selection
    from vexa_orchestrator.live import LiveSet

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out.resolve()
    engine = AudioEngine(EngineConfig(device="pipewire"))
    for deck in engine.mixer.decks:
        deck.gain = 0.015
    live = LiveSet(Depot(ROOT / "assets/library"), engine, preview_dir=output / "previews")
    for directory in args.seed_library:
        for asset in Depot(directory).assets.values():
            live.depot.add(asset.manifest, asset.path)
    # The harness owns generation; the normal scheduler and admission/deck rules still run.
    if not args.production_generation:
        live._queue_generation = lambda *a, **k: False
    else:
        live.depot.library = output / "library"
        live.configure_generator(args.generator)
    controls = GenerationSettings(track_count=1, duration_s=150, vocal_mode="instrumental", bpm=128)
    generator, preset = selection(args.generator, require_available=True)
    gpu = _gpu_index()
    settings = writer_settings()
    cancel = threading.Event()
    generation = []

    def generate():
        for index in range(args.renders):
            if cancel.is_set():
                break
            start = time.monotonic()
            row = {"index": index, "generator": generator, "preset": preset, "state": "planning"}
            generation.append(row)
            try:
                brief = plan_track(
                    "dark melodic techno instrumental",
                    0,
                    generator,
                    settings=settings,
                    controls=controls.model_dump(),
                    cache_path=output / f"draft-{index}.json",
                    cancel=cancel,
                )
                row.update(planning_s=time.monotonic() - start, state="rendering")
                asset = isolated_generate_theme(
                    "dark melodic techno instrumental",
                    output / "library",
                    duration_s=brief.duration_s,
                    energy=brief.energy,
                    style=brief.style,
                    lyrics=brief.lyrics,
                    title=brief.title,
                    prompt_origin=brief.origin,
                    music_plan=brief.music_plan,
                    generator=generator,
                    preset=preset,
                    cancel=cancel,
                    seed=900 + index,
                    render_id=f"live-{uuid.uuid4().hex[:12]}",
                    render_timeout_s=1800,
                )
                with live._lock:
                    live.depot.add(asset.manifest, asset.path)
                    live._prepared_for = None
                row.update(state="admitted", asset_id=asset.manifest.asset_id, path=str(asset.path))
            except Exception as exc:
                row.update(state="cancelled" if cancel.is_set() else "failed", error=str(exc))
            finally:
                row["total_s"] = round(time.monotonic() - start, 2)

    worker = threading.Thread(target=generate, daemon=True)
    gui = None
    samples = []
    started = time.monotonic()
    status_path = output / "status.json"
    report = {
        "state": "running",
        "requested_s": args.duration,
        "gpu": gpu,
        "generator": generator,
        "preset": preset,
        "gui_wayland": args.gui,
        "quiet_gain": 0.015,
        "source_depot_modified": False,
        "audible_review": "pending",
        "samples": samples,
        "generation": generation,
        "production_generation": args.production_generation,
        "theme": args.theme,
        "seed_libraries": [str(p.resolve()) for p in args.seed_library],
    }
    try:
        starts = sorted(
            live.depot.search(args.theme, limit=80),
            key=lambda row: (
                row[1].manifest.audio.duration_s < 148,
                abs(row[1].manifest.beat_grid.bpm - 128),
                "minor" not in (row[1].manifest.key.value if row[1].manifest.key else ""),
                row[1].manifest.asset_id,
            ),
        )
        live.start(
            args.theme,
            session_id="p6-soak",
            controls=GenerationSettings(
                start_policy="depot", track_count=args.renders or 1, duration_s=150, bpm=128
            )
            if args.production_generation
            else GenerationSettings(start_policy="depot"),
            prepared_ids=[asset.manifest.asset_id for _, asset in starts],
        )
        if not live.state:
            raise RuntimeError("no cueable starting track")
        time.sleep(2)
        baseline = engine.snapshot()["stats"]
        report["baseline"] = baseline
        started = time.monotonic()
        if not args.production_generation:
            worker.start()
        if args.gui:
            gui = subprocess.Popen(
                ["/usr/bin/python3", str(Path(__file__).resolve()), "--viewer", str(status_path)],
                env={
                    **os.environ,
                    "QT_QPA_PLATFORM": "wayland",
                    "VEXA_LOG_DIR": str(output / "logs"),
                    "VEXA_RUN_ID": "p6-soak",
                },
            )
        themes = ["dark", "house", "warm", "driving", "peak", "calm"]
        theme_index = 0
        previous_cpu = resource.getrusage(resource.RUSAGE_SELF)
        previous_at = started
        while time.monotonic() - started < args.duration:
            now = time.monotonic()
            elapsed = now - started
            status = live.status()
            # Explicit new direction, never implicit reuse. Expose any silence/misses.
            if (
                not args.no_steer
                and (elapsed >= (theme_index + 1) * 300 or status["continuity"]["waiting"])
                and theme_index < 5
            ):
                theme_index += 1
                live.steer(themes[theme_index])
            cpu = resource.getrusage(resource.RUSAGE_SELF)
            load = (
                (cpu.ru_utime + cpu.ru_stime) - (previous_cpu.ru_utime + previous_cpu.ru_stime)
            ) / (now - previous_at)
            previous_cpu, previous_at = cpu, now
            gpu_result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    gpu,
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            samples.append(
                {
                    "elapsed_s": round(elapsed, 2),
                    "audio": status["audio"],
                    "continuity": status["continuity"],
                    "current": status["current_asset_id"],
                    "host_cpu_cores": round(load, 3),
                    "load_average": os.getloadavg(),
                    "gpu": gpu_result,
                }
            )
            atomic(status_path, {"state": "running", "elapsed_s": elapsed, "live": status})
            if len(samples) % 12 == 0:
                atomic(output / "report.json", report)
                print(
                    json.dumps(
                        {
                            "elapsed_s": round(elapsed),
                            "transitions": sum(e["event"] == "transition" for e in live.history),
                            "stats": status["audio"]["stats"],
                            "waiting": status["continuity"]["waiting"],
                            "generation": [r["state"] for r in generation],
                        }
                    ),
                    flush=True,
                )
            time.sleep(5)
        final = live.status()
        stats = final["audio"]["stats"]
        report.update(
            state="completed",
            elapsed_s=round(time.monotonic() - started, 2),
            final=final,
            events=live.history,
            deltas={
                key: stats[key] - baseline[key]
                for key in (
                    "callbacks",
                    "underruns",
                    "output_underflows",
                    "deadline_misses",
                    "queue_dropped",
                    "silence_blocks",
                    "fade_blocks",
                )
            },
        )
        report["health_passed"] = not any(
            report["deltas"][key]
            for key in ("underruns", "output_underflows", "deadline_misses", "queue_dropped")
        )
        report["continuous_music_passed"] = report["deltas"]["silence_blocks"] == 0 and not any(
            e["event"] == "waiting_for_music" for e in live.history
        )
        report["silent_input_s"] = round(
            report["deltas"]["silence_blocks"] * engine.config.blocksize / engine.sample_rate, 3
        )
        report["active_input_s"] = round(max(0, report["elapsed_s"] - report["silent_input_s"]), 3)
        report["passed"] = (
            not any(
                report["deltas"][key]
                for key in ("underruns", "output_underflows", "deadline_misses", "queue_dropped")
            )
            and any(e["event"] == "transition" for e in live.history)
            and (
                args.renders == 0
                or any(r["state"] == "admitted" for r in generation)
                or any(e["event"] == "generated" for e in live.history)
            )
            and (not args.gui or (gui is not None and gui.poll() is None))
        )
    except Exception as exc:
        report.update(state="failed", error=str(exc), elapsed_s=time.monotonic() - started)
        raise
    finally:
        cancel.set()
        live.stop()
        if worker.ident is not None:
            worker.join(10)
        report.setdefault("elapsed_s", time.monotonic() - started)
        atomic(output / "report.json", report)
        atomic(
            status_path,
            {"state": report["state"], "elapsed_s": report["elapsed_s"], "live": live.status()},
        )
        if gui:
            try:
                gui.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gui.terminate()
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("samples", "events", "final")},
                default=str,
            ),
            flush=True,
        )
    if not report.get("passed"):
        raise RuntimeError("continuity soak failed; inspect the measured report")


if __name__ == "__main__":
    main()
