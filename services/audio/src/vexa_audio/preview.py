"""Transition previews.

The missing piece between "we have a library" and "we have labels". A listener cannot answer
"which transition sounds better" in the abstract — they have to hear them.

Renders a short clip of each candidate transition so they can be compared back to back, with a
marker between them. Everything here is **offline file rendering**: no device is opened, no
realtime path is touched, and the output is a plain WAV that any player can open.

``IDEAS.md:11`` describes auditioning a preview *while the set continues on the public output*.
That needs a third output bus and is a later step. It is not needed to collect the first labels,
because during collection the listener is not also listening to a set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .loader import PreloadedTrack


@dataclass(frozen=True, slots=True)
class PreviewSpec:
    """One candidate transition, as it will be heard."""

    label: str
    #: How long before the change, in seconds.
    lead_in_s: float = 4.0
    #: Crossfade length in seconds.
    fade_s: float = 2.0
    #: Silence appended so consecutive previews do not butt together.
    tail_s: float = 1.5


@dataclass(frozen=True, slots=True)
class Preview:
    """A rendered preview, ready to play."""

    label: str
    path: Path
    duration_s: float


class PreviewRenderer:
    """Renders auditable comparisons from preloaded tracks.

    The point of rendering to a file rather than streaming is comparability: every listener hears
    the identical clip, so the answer is about the transition and not about playback conditions.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        channels: int = 2,
        out_dir: str | Path = "var/previews",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- rendering ----------------------------------------------------------

    def render_transition(
        self,
        spec: PreviewSpec,
        outgoing: PreloadedTrack,
        incoming: PreloadedTrack,
        *,
        out_name: str,
        entry_bar: int = 0,
        incoming_entry_bar: int = 0,
        outgoing_start_s: float | None = None,
        incoming_entry_s: float | None = None,
    ) -> Preview:
        """One clip: the tail of ``outgoing`` crossfading into the head of ``incoming``.

        Equal-power, because a linear fade loses ~3 dB in the middle and would make every
        transition sound worse than it is — a systematic bias in exactly the comparison we are
        asking a listener to make.
        """
        sr = self.sample_rate
        lead = int(spec.lead_in_s * sr)
        fade = int(spec.fade_s * sr)
        tail = int(spec.tail_s * sr)
        total = lead + fade + tail

        a = _slice_from(outgoing, entry_bar, lead + fade, self.channels,
                        start_s=outgoing_start_s)
        b = _slice_from(incoming, incoming_entry_bar, fade + tail, self.channels,
                        start_s=incoming_entry_s)
        if len(a) != lead + fade or len(b) != fade + tail:
            raise ValueError("preview cue extends beyond an audio asset")

        audio = np.zeros((total, self.channels), dtype=np.float32)
        audio[:lead] = a[:lead]
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[lead : lead + fade] = (
                a[lead:] * np.cos(ramp * np.pi / 2)[:, None]
                + b[:fade] * np.sin(ramp * np.pi / 2)[:, None]
            )
        audio[lead + fade :] = b[fade:]

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.891:  # -1 dBFS, same ceiling the master limiter uses
            audio *= 0.891 / peak

        path = self.out_dir / f"{out_name}.wav"
        _write(path, audio, sr)
        return Preview(label=spec.label, path=path, duration_s=total / sr)

    def render_hold(
        self, spec: PreviewSpec, track: PreloadedTrack, *, out_name: str, entry_bar: int = 0
    ) -> Preview:
        """Render the actual continue option without a restart or a crossfade."""
        frames = int((spec.lead_in_s + spec.fade_s + spec.tail_s) * self.sample_rate)
        audio = _slice_from(track, entry_bar, frames, self.channels)
        if len(audio) != frames:
            raise ValueError("hold preview extends beyond an audio asset")
        path = self.out_dir / f"{out_name}.wav"
        _write(path, audio, self.sample_rate)
        return Preview(spec.label, path, frames / self.sample_rate)

    def render_comparison(
        self,
        outgoing: PreloadedTrack,
        candidates: list[tuple[str, PreloadedTrack]],
        *,
        name: str,
        lead_in_s: float = 4.0,
        fade_s: float = 2.0,
    ) -> dict[str, Preview]:
        """Every candidate, each rendered identically so they can be compared fairly.

        Clips for this decision are cleared first. Without that, re-rendering a decision after its
        option set changed leaves the old clips behind under overlapping indices, so a listener
        replaying ``decision000_1_*`` hears two different things under one name.
        """
        self.clear(name)
        results: dict[str, Preview] = {}
        for index, (label, track) in enumerate(candidates):
            spec = PreviewSpec(label=label, lead_in_s=lead_in_s, fade_s=fade_s)
            out_name = f"{name}_{index}_{_slug(label)}"
            results[label] = (
                self.render_hold(spec, outgoing, out_name=out_name)
                if label == "continue_current"
                else self.render_transition(spec, outgoing, track, out_name=out_name)
            )
        return results

    def clear(self, name: str) -> int:
        """Remove a previous render of one decision. Returns how many files went."""
        removed = 0
        for path in self.out_dir.glob(f"{name}_*.wav"):
            path.unlink()
            removed += 1
        return removed

    def render_manifest(
        self, previews: dict[str, Preview], path: str | Path
    ) -> Path:
        """Write what was played, so a label can be traced back to its audio.

        ``README.md:22`` asks for reproducibility: the request, the assets, the action, the
        outcome. An annotation with no record of what it referred to is not reproducible.
        """
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            label: {"path": str(p.path), "duration_s": round(p.duration_s, 2)}
            for label, p in previews.items()
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


def _slice_from(
    track: PreloadedTrack, entry_bar: int, frames: int, channels: int,
    *, start_s: float | None = None,
) -> np.ndarray:
    """Source samples from a cue; the caller decides where they land in the preview."""
    start = (round(start_s * track.sample_rate) if start_s is not None else
             track.sample_at_bar(entry_bar) if track.can_seek_by_bar else entry_bar)
    start = max(0, min(start, track.frames))
    block = track.samples[start : start + frames]
    if block.shape[1] != channels:
        block = block[:, :channels] if block.shape[1] > channels else np.repeat(
            block, channels // block.shape[1] + 1, axis=1
        )[:, :channels]
    return block


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), audio, sr, subtype="PCM_16")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")[:40]
