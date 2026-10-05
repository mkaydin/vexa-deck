"""Measure the audio that the decks would actually overlap, before a rule commits it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from vexa_audio.cues import CueMap
from vexa_audio.loader import PreloadedTrack
from vexa_audio.preview import PreviewRenderer, PreviewSpec


@dataclass(frozen=True, slots=True)
class TransitionReport:
    score: float
    rms_jump_db: float
    spectral_distance: float
    peak_dbfs: float
    preview_path: Path


def score_rendered_transition(
    renderer: PreviewRenderer,
    outgoing: PreloadedTrack,
    incoming: PreloadedTrack,
    outgoing_cue: CueMap,
    incoming_cue: CueMap,
    *,
    name: str,
    lead_s: float = 3.0,
    fade_s: float = 4.0,
) -> TransitionReport:
    """Score level and spectrum changes in the rendered preview, not track-wide tags.

    The score is a ranking signal. It cannot override admission, tempo, key or cue rules.
    """
    if outgoing_cue.exit_s is None or incoming_cue.entry_s is None:
        raise ValueError("both tracks need measured transition cues")
    preview = renderer.render_transition(
        PreviewSpec(name, lead_s, fade_s, 1.5), outgoing, incoming,
        out_name=name,
        outgoing_start_s=max(0.0, outgoing_cue.exit_s - lead_s),
        incoming_entry_s=0.0,
    )
    audio, sr = sf.read(preview.path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    before = mono[: int(lead_s * sr)]
    after = mono[int((lead_s + fade_s) * sr):]
    if len(before) < 1024 or len(after) < 1024:
        raise ValueError("rendered preview too short to score")
    def rms(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block * block) + 1e-12))
    jump = abs(20.0 * np.log10(rms(after) / rms(before)))
    # A spectrum comparison at the actual overlap boundaries catches abrupt timbre changes.
    def spectrum(block: np.ndarray) -> np.ndarray:
        window = np.hanning(min(len(block), 8192)).astype(np.float32)
        spec = np.abs(np.fft.rfft(block[:len(window)] * window))
        return spec / (np.sum(spec) + 1e-9)
    distance = float(np.sum(np.abs(spectrum(before) - spectrum(after))) / 2.0)
    peak = float(np.max(np.abs(audio)))
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-9))
    score = max(0.0, 1.0 - min(1.0, jump / 12.0) * 0.55 - distance * 0.45)
    return TransitionReport(float(round(score, 4)), float(round(jump, 3)),
                            float(round(distance, 4)), float(round(peak_dbfs, 3)),
                            preview.path)
