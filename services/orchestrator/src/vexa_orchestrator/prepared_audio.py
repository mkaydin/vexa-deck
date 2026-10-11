"""File-backed PCM handoff for measured transitions; no decode on the audio host."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
from vexa_audio.cues import CueMap
from vexa_audio.equalizer import EqualizerSettings
from vexa_audio.loader import PreloadedTrack, file_sha256

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
    receipt = path.with_suffix(".pcm.json")
    valid = False
    if path.exists() and receipt.exists():
        try:
            saved = json.loads(receipt.read_text())
            valid = (
                saved.get("source_sha256") == track.content_sha256
                and saved.get("frames") == track.frames
                and saved.get("pcm_sha256") == file_sha256(path)
            )
        except (OSError, ValueError):
            pass
    if not valid:
        temporary = path.with_suffix(".partial.npy")
        with temporary.open("wb") as output:
            np.save(output, track.samples, allow_pickle=False)
        temporary.replace(path)
        temporary_receipt = receipt.with_suffix(".tmp")
        temporary_receipt.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_sha256": track.content_sha256,
                    "sample_rate": track.sample_rate,
                    "frames": track.frames,
                    "channels": track.channels,
                    "pcm_sha256": file_sha256(path),
                }
            )
        )
        temporary_receipt.replace(receipt)
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
    timeout_s: float = 300,
    blocksize: int = 2048,
    equalizer: EqualizerSettings | None = None,
    gains: tuple[float, float] = (1.0, 1.0),
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
            "blocksize": blocksize,
            "equalizer": (equalizer or EqualizerSettings()).to_dict(),
            "gains": gains,
            "outgoing_samples": str(outgoing),
            "outgoing_track": track_metadata(current_track),
            "outgoing_cue": {
                field.name: getattr(current_cue, field.name) for field in fields(current_cue)
            },
        },
        cancel=cancel,
        timeout_s=timeout_s,
    )
    samples = np.load(body["samples_path"], mmap_mode="r", allow_pickle=False)
    if samples.dtype != np.float32 or samples.ndim != 2:
        raise ValueError("prepared PCM must be stereo float32")
    track = PreloadedTrack(samples=samples, **body["track"])
    path = Path(body["samples_path"])
    saved = json.loads(path.with_suffix(".pcm.json").read_text())
    if (
        saved.get("pcm_sha256") != file_sha256(path)
        or saved.get("source_sha256") != track.content_sha256
        or saved.get("sample_rate") != track.sample_rate
        or saved.get("frames") != track.frames
    ):
        raise ValueError("prepared PCM integrity receipt does not match")
    if track.channels != 2 or samples.shape[1] != 2:
        raise ValueError("prepared PCM must have two channels")
    # Touch pages in small slices on this control thread before the callback can see them.
    for offset in range(0, track.frames, 65536):
        if not np.isfinite(samples[offset : offset + 65536]).all():
            raise ValueError("prepared PCM contains nonfinite samples")
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
