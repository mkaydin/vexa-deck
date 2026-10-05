"""The command queue between the control plane and the audio thread.

This is the boundary the whole architecture rests on. ``README.md:18``: audio continuity is the
highest priority, and a model timeout must not interrupt the stream. So the callback cannot ask
the orchestrator anything — not "is this asset ready", not "should we transition", not even "what
time is it". It pulls already-decided commands from here and does arithmetic.

Single-producer, single-consumer. One writer thread (the control plane) and one reader (the audio
callback). No locks, because a lock in the callback is a stall waiting to happen; correctness comes
from the SPSC discipline instead.

The queue is a fixed-size ring of preallocated slots. If the consumer falls behind, commands are
**dropped and counted** rather than blocking the producer or growing without bound. A full queue
is a control-plane problem to observe, not a reason to stop the music.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .loader import PreloadedTrack


class CommandKind(Enum):
    """What the callback can be told to do. Deliberately a closed set.

    Anything not in this enum cannot reach the audio thread, which is what "models are bounded"
    means in practice. There is no escape hatch for an arbitrary DSP command.
    """

    LOAD_TRACK = "load_track"        # put a preloaded track on a deck
    PLAY = "play"
    PAUSE = "pause"
    SEEK_BAR = "seek_bar"
    SET_GAIN = "set_gain"
    SET_TEMPO = "set_tempo"
    CUE = "cue"                      # return a deck to its cue point
    START_CROSSFADE = "start_crossfade"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True, slots=True)
class Command:
    """One instruction. Frozen: the consumer must not be able to mutate what it is reading."""

    kind: CommandKind
    deck: int = 0
    #: Content hash of a preloaded track, or an empty string.
    track_hash: str = ""
    value: float = 0.0
    value2: float = 0.0
    #: When set, the command applies only once the transport reaches this bar.
    at_bar: int | None = None
    #: Ready PCM reference; installed together with position/fade state at a block boundary.
    track: PreloadedTrack | None = None


@dataclass(slots=True)
class QueueStats:
    """Observed from outside the callback. Read by the engine's health reporting."""

    enqueued: int = 0
    delivered: int = 0
    #: Dropped because the consumer fell behind. A nonzero value is a design smell, not a crash.
    dropped: int = 0

    @property
    def lost(self) -> int:
        return self.enqueued - self.delivered


class CommandQueue:
    """A fixed-capacity SPSC ring of :class:`Command`.

    The indices are plain ints written by each side only. CPython's GIL makes each assignment
    atomic, and neither side spins on the other's index — the producer simply refuses to overwrite
    an unread slot.
    """

    __slots__ = ("_head", "_mask", "_slots", "_tail", "stats")

    def __init__(self, capacity: int = 256) -> None:
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power of two so the mask wrap is cheap")
        self._slots: list[Command | None] = [None] * capacity
        self._mask = capacity - 1
        self._head = 0  # written by producer
        self._tail = 0  # written by consumer
        self.stats = QueueStats()

    def __len__(self) -> int:
        return (self._head - self._tail) & self._mask

    @property
    def full(self) -> bool:
        return len(self) == self._mask

    def put(self, command: Command) -> bool:
        """Enqueue without blocking. Returns False if the queue was full and the command dropped.

        Never waits for the consumer: the producer is the control plane, and it must never be
        slowed down by audio.
        """
        head = self._head
        if ((head + 1) & self._mask) == self._tail:
            self.stats.dropped += 1
            return False
        self._slots[head & self._mask] = command
        # Publish the write before the index, or the consumer could read a half-built slot.
        self._head = (head + 1) & self._mask
        self.stats.enqueued += 1
        return True

    def put_many(self, commands: Sequence[Command]) -> int:
        return sum(1 for c in commands if self.put(c))

    def drain(self) -> list[Command]:
        """Consumer side. Returns commands in order and clears the slots.

        The list is a small allocation, but it is bounded by the queue capacity and happens once
        per callback block rather than per sample. Everything inside the mixer is preallocated.
        """
        tail = self._tail
        head = self._head
        if tail == head:
            return []

        out: list[Command] = []
        while tail != head:
            slot = self._slots[tail & self._mask]
            if slot is not None:
                out.append(slot)
                self._slots[tail & self._mask] = None
                self.stats.delivered += 1
            tail = (tail + 1) & self._mask
        self._tail = tail
        return out


@dataclass(slots=True)
class Clock:
    """Musical position shared with the control plane.

    The callback is the only writer, so this is a plain float that the control plane may read at
    any time. It is deliberately not locked: a torn read is at worst one slightly stale bar
    counter, and a lock here could stall the audio thread.
    """

    sample_rate: float
    bpm: float
    beats_per_bar: int = 4
    _samples: float = 0.0

    @property
    def samples_per_beat(self) -> float:
        return self.sample_rate * 60.0 / self.bpm

    @property
    def samples_per_bar(self) -> float:
        return self.samples_per_beat * self.beats_per_bar

    @property
    def samples(self) -> float:
        return self._samples

    @property
    def bar(self) -> int:
        return int(self._samples // self.samples_per_bar)

    @property
    def beat(self) -> int:
        return int((self._samples % self.samples_per_bar) // self.samples_per_beat)

    def advance(self, frames: int) -> None:
        self._samples += frames

    def reset(self) -> None:
        self._samples = 0.0


def preallocate_buffer(frames: int, channels: int) -> np.ndarray:
    """A zeroed stereo scratch buffer, allocated once at startup."""
    return np.zeros((frames, channels), dtype=np.float32)