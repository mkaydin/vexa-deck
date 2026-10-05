"""File-backed PCM handoff for measured transitions; no decode on the audio host."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
from vexa_audio.cues import CueMap
from vexa_audio.loader import PreloadedTrack

from .background import run_job
from .depot import DepotAsset
from .transitions import TransitionReport


def track_metadata(track: PreloadedTrack) -> dict:
    return {
        field.name: getattr(track, field.name) for field in fields(track) if field.name != "samples"
    }


def persist_pcm(track: PreloadedTrack, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{track.content_sha256}_{track.sample_rate}.npy"
    if not path.exists():
        temporary = path.with_suffix(".partial.npy")
        with temporary.open("wb") as output:
            np.save(output, track.samples, allow_pickle=False)
        temporary.replace(path)
    return path


def prepare_candidate(
    asset: DepotAsset,
    *,
    current: DepotAsset,
    current_track: PreloadedTrack,
    current_cue: CueMap,
    target_bpm: float,
    cache_dir: Path,
    preview_dir: Path,
    name: str,
    cancel,
) -> tuple[PreloadedTrack, CueMap, TransitionReport]:
    outgoing = cache_dir / f"{current_track.content_sha256}_{current_track.sample_rate}.npy"
    if not outgoing.exists():
        raise ValueError("playing track PCM was not prepared before playback")
    body = run_job(
        "prepare",
        {
            "manifest": asset.manifest.model_dump(mode="json"),
            "path": str(asset.path),
            "sample_rate": current_track.sample_rate,
            "target_bpm": target_bpm,
            "cache_dir": str(cache_dir),
            "preview_dir": str(preview_dir),
            "name": name,
            "outgoing_samples": str(outgoing),
            "outgoing_track": track_metadata(current_track),
            "outgoing_cue": {
                field.name: getattr(current_cue, field.name) for field in fields(current_cue)
            },
        },
        cancel=cancel,
    )
    samples = np.load(body["samples_path"], mmap_mode="r", allow_pickle=False)
    if samples.dtype != np.float32 or samples.ndim != 2:
        raise ValueError("prepared PCM must be stereo float32")
    track = PreloadedTrack(samples=samples, **body["track"])
    if track.channels != 2 or samples.shape[1] != 2:
        raise ValueError("prepared PCM must have two channels")
    # Touch pages in small slices on this control thread before the callback can see them.
    for offset in range(0, track.frames, 65536):
        np.sum(samples[offset : offset + 65536])
    cue = CueMap(
        **{
            **body["cue"],
            "beat_times_s": tuple(body["cue"]["beat_times_s"]),
            "downbeat_times_s": tuple(body["cue"]["downbeat_times_s"]),
        }
    )
    report = TransitionReport(
        **{**body["report"], "preview_path": Path(body["report"]["preview_path"])}
    )
    return track, cue, report
