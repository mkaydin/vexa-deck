"""Low priority process for YuE2 admission, decoding and transition measurement."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path


def dispatch(action: str, payload: dict) -> dict:
    if action == "plan":
        from .ace_planner import plan_in_worker

        return plan_in_worker(payload["theme"], payload["count"])
    if action == "generate":
        from .generation import generate_theme

        asset = generate_theme(payload.pop("theme"), Path(payload.pop("library")), **payload)
        return {"manifest": asset.manifest.model_dump(mode="json"), "path": str(asset.path)}
    if action == "prepare":
        import numpy as np
        from vexa_audio.cues import CueMap
        from vexa_audio.engine import AudioEngine, EngineConfig
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
            sample_rate=payload["sample_rate"], out_dir=payload["preview_dir"]
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


def main() -> None:
    # Reserve two logical CPUs for audio/UI; children inherit this affinity and priority.
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) > 2:
            os.sched_setaffinity(0, cpus[2:])
    os.nice(10)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    job, result = (Path(value) for value in sys.argv[1:])
    from vexa_yue2.backends import BackendUnavailable

    try:
        body = json.loads(job.read_text())
        from contextlib import nullcontext

        from .background import gpu_job_lock

        if body["action"] == "plan":
            from .generation import _gpu_index

            os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_index()
        lock = gpu_job_lock() if body["action"] in ("plan", "generate") else nullcontext()
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
