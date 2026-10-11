"""Low priority process for YuE2 admission, decoding and transition measurement."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path


def dispatch(action: str, payload: dict) -> dict:
    if action == "audit_audio":
        import shutil

        from .audio_evidence import load_evidence
        from .generation import admit_render

        source, output = Path(payload["source"]), Path(payload["output"])
        if source.resolve() == output.resolve() or output.exists():
            raise ValueError("audit output must be a new copy, never the original")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        try:
            asset = admit_render(
                output,
                theme=payload["theme"],
                duration_s=payload["duration_s"],
                energy=0.5,
                music_plan=payload.get("music_plan"),
                generator=payload.get("generator", "yue2"),
                render_metadata={"audit_source": str(source)},
            )
            return {
                "manifest": asset.manifest.model_dump(mode="json"),
                "evidence": load_evidence(output),
            }
        except ValueError as exc:
            return {"error": str(exc), "evidence": load_evidence(output)}
    if action == "benchmark_writer":
        from .writer_eval import benchmark_wave

        return benchmark_wave(payload)
    if action == "plan_v2":
        from vexa_contracts import TrackPlan

        from .generation_v2 import write_plan

        brief = write_plan(
            TrackPlan.model_validate(payload["plan"]),
            payload["settings"],
            Path(payload["cache_path"]),
        )
        return {"brief": asdict(brief)}
    if action == "plan":
        from .ace_planner import plan_in_worker

        return plan_in_worker(
            payload["theme"],
            payload["count"],
            **({"generator": payload["generator"]} if payload.get("generator") else {}),
        )
    if action == "generate":
        from .generation import generate_theme

        asset = generate_theme(payload.pop("theme"), Path(payload.pop("library")), **payload)
        return {"manifest": asset.manifest.model_dump(mode="json"), "path": str(asset.path)}
    if action == "prepare":
        import numpy as np
        from vexa_audio.cues import CueMap
        from vexa_audio.engine import AudioEngine, EngineConfig
        from vexa_audio.equalizer import EqualizerSettings
        from vexa_audio.loader import PreloadedTrack
        from vexa_audio.preview import PreviewRenderer
        from vexa_contracts import AssetManifest

        from .depot import DepotAsset
        from .prepare import prepare_for_tempo
        from .prepared_audio import persist_pcm, track_metadata
        from .transitions import score_rendered_transition

        manifest = AssetManifest.model_validate(payload["manifest"])
        asset = DepotAsset(manifest, Path(payload["path"]), frozenset(), {})
        cue = CueMap.load(asset.path.with_suffix(".cues.json"))
        if not cue.valid_for(asset.path) or cue.entry_s is None or cue.exit_s is None:
            raise ValueError("candidate has no valid measured cues")
        engine = AudioEngine(EngineConfig(sample_rate=payload["sample_rate"]))
        track, cue = prepare_for_tempo(
            engine, asset, cue, payload["target_bpm"], Path(payload["cache_dir"])
        )
        outgoing = PreloadedTrack(
            samples=np.load(payload["outgoing_samples"], mmap_mode="r", allow_pickle=False),
            **payload["outgoing_track"],
        )
        outgoing_cue = CueMap(**payload["outgoing_cue"])
        renderer = PreviewRenderer(
            sample_rate=payload["sample_rate"], out_dir=payload["preview_dir"],
            blocksize=payload.get("blocksize", 2048),
            equalizer=EqualizerSettings(**payload.get("equalizer", {})),
            gains=tuple(payload.get("gains", (1.0, 1.0))),
        )
        report = score_rendered_transition(
            renderer, outgoing, track, outgoing_cue, cue, name=payload["name"]
        )
        samples_path = persist_pcm(track, Path(payload["cache_dir"]))
        metadata = track_metadata(track)
        return {
            "samples_path": str(samples_path),
            "track": metadata,
            "cue": asdict(cue),
            "report": {**asdict(report), "preview_path": str(report.preview_path)},
        }
    raise ValueError(f"unknown offline action: {action}")


def preflight_render(payload: dict):
    """Missing components must fail before waiting behind a long native GPU render."""
    from vexa_yue2.backends import BackendUnavailable

    from .generators import components, selection

    generator = payload.get("generator", "yue2")
    if generator == "yue2":
        for path in components("yue2", "q8").values():
            if not path.is_file():
                raise BackendUnavailable(f"YuE2 component missing: {path}")
    else:
        selection(generator, payload.get("preset"), require_available=True)


def main() -> None:
    # Reserve two logical CPUs for audio/UI; children inherit this affinity and priority.
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) > 2:
            os.sched_setaffinity(0, cpus[2:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    job, result = (Path(value) for value in sys.argv[1:])
    from vexa_yue2.backends import BackendUnavailable

    try:
        body = json.loads(job.read_text())
        os.nice(5 if body["action"] == "prepare" else 10)
        from contextlib import nullcontext

        from .background import gpu_job_lock

        if body["action"] == "generate":
            preflight_render(body["payload"])

        if body["action"] in ("plan", "plan_v2", "benchmark_writer"):
            from .generation import _gpu_index

            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_index()
            except RuntimeError as exc:
                raise BackendUnavailable(str(exc)) from exc
        lock = (
            gpu_job_lock()
            if body["action"] in ("plan", "plan_v2", "generate", "benchmark_writer")
            else nullcontext()
        )
        with lock:
            output = {"ok": True, "result": dispatch(body["action"], body["payload"])}
    except Exception as exc:
        logging.exception("offline audio work failed")
        output = {
            "ok": False,
            "error": str(exc),
            "unavailable": isinstance(exc, BackendUnavailable),
        }
    result.write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main()
