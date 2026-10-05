"""Pitch preserving tempo preparation outside the audio callback."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from vexa_audio.cues import CueMap
from vexa_audio.engine import AudioEngine
from vexa_audio.loader import PreloadedTrack, file_sha256

from .depot import DepotAsset


def prepare_for_tempo(engine: AudioEngine, asset: DepotAsset, cue: CueMap,
                      target_bpm: float, cache_dir: Path) -> tuple[PreloadedTrack, CueMap]:
    """Stretch a candidate before loading; retime every cue by the same measured ratio."""
    bpm = asset.manifest.beat_grid.bpm
    if file_sha256(asset.path) != asset.manifest.content_sha256:
        raise ValueError("asset changed after admission")
    if abs(bpm / target_bpm - 1.0) <= 0.005:
        return engine.preload(asset.path, bpm=bpm,
                              expected_hash=asset.manifest.content_sha256), cue
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared = cache_dir / f"{asset.manifest.content_sha256}_{target_bpm:.3f}.wav"
    if not prepared.exists():
        temporary = prepared.with_suffix(".partial.wav")
        process = subprocess.run(
            ["rubberband", "--fast", "--quiet", "--tempo", f"{bpm}:{target_bpm}",
             str(asset.path), str(temporary)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"tempo preparation failed: {process.stderr[-300:]}")
        temporary.replace(prepared)
    track = engine.preload(prepared, bpm=target_bpm)
    scale = bpm / target_bpm
    retimed = replace(cue, bpm=target_bpm,
        beat_times_s=tuple(round(value * scale, 4) for value in cue.beat_times_s),
        downbeat_times_s=tuple(round(value * scale, 4) for value in cue.downbeat_times_s),
        entry_s=cue.entry_s * scale if cue.entry_s is not None else None,
        exit_s=cue.exit_s * scale if cue.exit_s is not None else None)
    return track, retimed
