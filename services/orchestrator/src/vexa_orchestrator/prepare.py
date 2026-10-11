"""Pitch preserving tempo preparation outside the audio callback."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from vexa_audio.cues import CueMap
from vexa_audio.engine import AudioEngine
from vexa_audio.loader import PreloadedTrack, file_sha256

from .depot import DepotAsset


def prepare_for_tempo(
    engine: AudioEngine, asset: DepotAsset, cue: CueMap, target_bpm: float, cache_dir: Path
) -> tuple[PreloadedTrack, CueMap]:
    """Stretch a candidate before loading; retime every cue by the same measured ratio."""
    bpm = asset.manifest.beat_grid.bpm
    if file_sha256(asset.path) != asset.manifest.content_sha256:
        raise ValueError("asset changed after admission")
    if abs(bpm / target_bpm - 1.0) <= 0.005:
        return engine.preload(asset.path, bpm=bpm, expected_hash=asset.manifest.content_sha256), cue
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared = (
        cache_dir
        / f"stretch-v2_{asset.manifest.content_sha256}_{target_bpm:.6f}_{engine.sample_rate}.wav"
    )
    receipt = prepared.with_suffix(".receipt.json")
    valid = False
    if prepared.exists() and receipt.exists():
        try:
            saved = json.loads(receipt.read_text())
            valid = saved.get("source_sha256") == asset.manifest.content_sha256 and saved.get(
                "prepared_sha256"
            ) == file_sha256(prepared)
        except (OSError, ValueError):
            pass
    if not valid:
        temporary = prepared.with_suffix(".partial.wav")
        process = subprocess.run(
            [
                "rubberband",
                "--fast",
                "--quiet",
                "--tempo",
                f"{bpm}:{target_bpm}",
                str(asset.path),
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"tempo preparation failed: {process.stderr[-300:]}")
        temporary.replace(prepared)
        receipt_tmp = receipt.with_suffix(".tmp")
        receipt_tmp.write_text(
            json.dumps(
                {
                    "source_sha256": asset.manifest.content_sha256,
                    "prepared_sha256": file_sha256(prepared),
                    "target_bpm": target_bpm,
                    "algorithm": "rubberband-fast-v2",
                }
            )
        )
        receipt_tmp.replace(receipt)
    track = engine.preload(prepared, bpm=target_bpm)
    scale = bpm / target_bpm
    if abs(track.duration_s - asset.manifest.audio.duration_s * scale) > 0.5:
        raise ValueError("tempo prepared duration does not match the measured ratio")
    retimed = replace(
        cue,
        bpm=target_bpm,
        source_sha256=track.content_sha256,
        beat_times_s=tuple(round(value * scale, 4) for value in cue.beat_times_s),
        downbeat_times_s=tuple(round(value * scale, 4) for value in cue.downbeat_times_s),
        entry_s=cue.entry_s * scale if cue.entry_s is not None else None,
        exit_s=cue.exit_s * scale if cue.exit_s is not None else None,
    )
    return track, retimed
