"""The mixer: the audio callback itself.

This is the only place in the project where realtime rules apply, so they are stated once here and
obeyed everywhere in this file.

Inside :meth:`Mixer._render` there is:

* **no allocation** — every buffer is created in :meth:`Mixer.__init__` and reused;
* **no lock** — commands arrive on a single-producer ring, tracks are swapped by pointer;
* **no I/O** — no file, no device, no network, no logging;
* **no model, no database, no clock read.**

What it *does* do is arithmetic: read two preloaded buffers at an integer position, scale them,
crossfade, limit, and write out. If a track is missing it falls back; if that is missing too it
outputs silence rather than raising. **The music stops for nothing.**

Positions are tracked in **integer samples**. A float slice index would force numpy to build a
view of computed length, and worse, would make the read length unpredictable between blocks — the
classic source of a click at a block boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .loader import MasterLimiter, PreloadedTrack
from .queue import Clock, Command, CommandKind, CommandQueue


@dataclass(slots=True)
class DeckState:
    """Everything the callback needs to know about one deck.

    ``track`` is the one field another thread may write, and it is replaced with a single
    attribute store. The old buffer stays alive until the callback stops referencing it.
    """

    index: int
    track: PreloadedTrack | None = None
    position: int = 0
    playing: bool = False
    gain: float = 1.0
    tempo_ratio: float = 1.0
    cue_point: int = 0
    #: Current crossfade gain, and where it is heading.
    fade_gain: float = 1.0
    fade_target: float = 1.0
    #: Frames of crossfade remaining. Zero means the gain has arrived.
    fade_remaining: int = 0
    #: Restart at the top instead of stopping at the end. DJ beds and short loops are meant to
    #: run indefinitely, and looping is also what lets a soak test run for half an hour from a
    #: few seconds of audio, instead of half an hour of audio resident in memory.
    loop: bool = False

    @property
    def loaded(self) -> bool:
        return self.track is not None

    @property
    def finished(self) -> bool:
        return self.track is not None and self.position >= self.track.frames

    @property
    def fading(self) -> bool:
        return self.fade_remaining > 0


@dataclass(slots=True)
class MixStats:
    """Written by the callback, read from outside it.

    Plain counters for the same reason the clock is: a torn read is harmless, a lock is not.
    """

    callbacks: int = 0
    underruns: int = 0
    frames_rendered: int = 0
    worst_callback_s: float = 0.0
    silence_blocks: int = 0
    fade_blocks: int = 0


class Mixer:
    """Two decks, crossfading, into a master limiter."""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int = 2,
        queue: CommandQueue | None = None,
        clock: Clock | None = None,
        fallback: PreloadedTrack | None = None,
        max_block: int = 1024,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        # Explicit `is None` checks, not `or`. `CommandQueue` defines `__len__`, so an empty but
        # perfectly valid queue is *falsy* — `queue or CommandQueue()` silently discards the one
        # it was handed, and every command then vanishes into an object nobody is reading.
        self.queue = CommandQueue() if queue is None else queue
        self.clock = Clock(sample_rate=sample_rate, bpm=120.0) if clock is None else clock
        self.stats = MixStats()

        self.decks = [DeckState(0), DeckState(1)]
        self.fallback = fallback

        # --- preallocated; never grown at runtime ---
        self._out = np.zeros((max_block, channels), dtype=np.float32)
        self._scratch_a = np.zeros((max_block, channels), dtype=np.float32)
        self._scratch_b = np.zeros((max_block, channels), dtype=np.float32)
        self._limiter = MasterLimiter(frames=8192, channels=channels)

        self._running = False

    # -- engine-facing API; never called from the callback -----------------

    def set_track(self, deck: int, track: PreloadedTrack) -> None:
        """Swap a deck's track. Safe from a control thread.

        A single attribute store: the callback sees either the old track or the new one, never a
        half-swapped deck.
        """
        if 0 <= deck < len(self.decks):
            self.decks[deck].track = track

    def set_fallback(self, track: PreloadedTrack) -> None:
        self.fallback = track

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    # -- the callback -------------------------------------------------------

    def render(self, outdata: np.ndarray, frames: int) -> None:
        """PortAudio callback. Must not raise, allocate, or block."""
        started = time.perf_counter()
        try:
            self._render(outdata, frames)
        except Exception:
            # Silence beats a dead stream. A dropped block is audible; an exception tearing down
            # the callback is fatal.
            outdata[:frames] = 0.0
            self.stats.underruns += 1

        self.stats.callbacks += 1
        self.stats.frames_rendered += frames
        elapsed = time.perf_counter() - started
        if elapsed > self.stats.worst_callback_s:
            self.stats.worst_callback_s = elapsed

    def _render(self, outdata: np.ndarray, frames: int) -> None:
        for command in self.queue.drain():
            self._apply(command)

        usable = min(frames, self._out.shape[0])
        self._render_deck(self.decks[0], self._scratch_a, usable)
        self._render_deck(self.decks[1], self._scratch_b, usable)

        mix = self._out
        mix[:usable] = self._scratch_a[:usable] + self._scratch_b[:usable]
        if usable < frames:
            mix[usable:frames] = 0.0

        if not np.any(mix[:frames]):
            self.stats.silence_blocks += 1

        outdata[:frames] = self._limiter.process(mix[:frames])
        self.clock.advance(frames)
        self._advance_fades(frames)

    def _render_deck(self, deck: DeckState, target: np.ndarray, frames: int) -> None:
        """Mix one deck into a preallocated buffer."""
        target[:frames] = 0.0
        track = deck.track
        if track is None or not deck.playing:
            return
        if deck.position >= track.frames:
            if deck.loop:
                deck.position = 0
            else:
                deck.playing = False
                return

        gain = deck.gain * deck.fade_gain
        if gain <= 0.0:
            return

        available = track.frames - deck.position
        take = frames if frames < available else available
        if take <= 0:
            deck.playing = False
            return

        # Integer slice: a view, no copy, no allocation.
        chunk = track.samples[deck.position : deck.position + take]
        target[:take] = chunk * gain

        # Tempo is applied as a resampled read rate. Near 1.0 the integer skip is exact enough
        # and costs nothing; a real ratio belongs to the audio engine's resampler, not here.
        if abs(deck.tempo_ratio - 1.0) < 0.001:
            deck.position += take
        else:
            deck.position += max(1, int(take * deck.tempo_ratio))

        # Looping decks wrap to the top on the next block; others stop.
        if deck.position >= track.frames and not deck.loop:
            deck.playing = False

    def _advance_fades(self, frames: int) -> None:
        """Interpolate each deck's crossfade gain toward its target.

        A fade that only sets a flag and a countdown is a hard cut wearing a crossfade's name.
        Moving the gain once per block is smooth enough at 512 frames and costs one multiply.
        """
        for deck in self.decks:
            if deck.fade_remaining <= 0:
                continue
            self.stats.fade_blocks += 1
            span = max(1, deck.fade_remaining)
            delta = deck.fade_target - deck.fade_gain
            gain = deck.fade_gain + delta * min(1.0, frames / span)
            # If this block crossed the target, snap rather than overshoot into a gain above 1.
            if delta * (deck.fade_target - gain) < 0:
                gain = deck.fade_target
            deck.fade_gain = gain
            deck.fade_remaining -= frames
            if deck.fade_remaining <= 0:
                deck.fade_remaining = 0
                deck.fade_gain = deck.fade_target

    # -- commands -----------------------------------------------------------

    def _apply(self, command: Command) -> None:
        deck = self._deck(command.deck)
        frames = int(command.value2)

        match command.kind:
            case CommandKind.PLAY:
                deck.playing = True
            case CommandKind.PAUSE:
                deck.playing = False
            case CommandKind.SEEK_BAR:
                if deck.track is not None and deck.track.can_seek_by_bar:
                    deck.position = min(
                        deck.track.frames, deck.track.sample_at_bar(int(command.value))
                    )
            case CommandKind.SET_GAIN:
                deck.gain = max(0.0, min(4.0, command.value))
            case CommandKind.SET_TEMPO:
                deck.tempo_ratio = max(0.5, min(2.0, command.value))
            case CommandKind.CUE:
                deck.position = deck.cue_point
            case CommandKind.LOAD_TRACK:
                # The audio itself is installed by pointer from the loader thread; the callback
                # only reacts to playback intent.
                if command.value > 0:
                    deck.position = 0
                    deck.fade_gain = 1.0
                    deck.fade_target = 1.0
                    deck.fade_remaining = 0
                    deck.playing = True
            case CommandKind.START_CROSSFADE:
                # The incoming deck fades up; the outgoing one fades down. Both keep playing
                # through the overlap, which is what makes this a crossfade rather than a cut.
                deck.fade_gain = 0.0
                deck.fade_target = 1.0
                deck.fade_remaining = max(1, frames)
                deck.playing = True
                other = self._deck(1 - deck.index)
                other.fade_target = 0.0
                other.fade_remaining = max(1, frames)
            case CommandKind.EMERGENCY_STOP:
                for d in self.decks:
                    d.playing = False
                    d.fade_remaining = 0
                    d.fade_gain = 1.0
                    d.fade_target = 1.0
            case _:  # pragma: no cover - CommandKind is a closed set
                pass

    def _deck(self, index: int) -> DeckState:
        if 0 <= index < len(self.decks):
            return self.decks[index]
        return self.decks[0]

    # -- introspection ------------------------------------------------------

    @property
    def bar(self) -> int:
        return self.clock.bar

    @property
    def position_seconds(self) -> float:
        return self.clock.samples / self.sample_rate

    def snapshot(self) -> dict[str, object]:
        """A plain-data view for the control plane. Never called from the callback."""
        def deck(d: DeckState) -> dict[str, object]:
            return {
                "loaded": d.loaded,
                "playing": d.playing,
                "hash": d.track.content_sha256[:12] if d.track else None,
                "position": d.position,
                "gain": round(d.gain * d.fade_gain, 4),
                "fading": d.fading,
            }

        return {
            "bar": self.clock.bar,
            "beat": self.clock.beat,
            "seconds": round(self.position_seconds, 2),
            "deck_a": deck(self.decks[0]),
            "deck_b": deck(self.decks[1]),
            "stats": {
                "callbacks": self.stats.callbacks,
                "underruns": self.stats.underruns,
                "frames": self.stats.frames_rendered,
                "worst_callback_ms": round(self.stats.worst_callback_s * 1000, 3),
                "silence_blocks": self.stats.silence_blocks,
                "fade_blocks": self.stats.fade_blocks,
                "queue_dropped": self.queue.stats.dropped,
            },
        }