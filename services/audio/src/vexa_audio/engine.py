"""The audio engine: owns the PortAudio stream and the loader thread.

Two rules decide the shape of this module.

**The engine must outlive every service.** ``ROADMAP.md:15`` requires that stopping the planning
process does not stop playback. So the stream, the mixer and the decoded tracks all live here,
behind one object, and nothing in the orchestrator owns them.

**The fallback bed is loaded before the set starts, not on demand.** ``ARCHITECTURE.md:85``: keep
at least one approved fallback bed and one compatible transition loop loaded. If a transition
fails, the music continues — the bed was already in memory, so nothing had to be fetched inside a
callback.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .loader import PreloadedTrack, TrackCache
from .mixer import Mixer
from .queue import Clock, Command, CommandKind, CommandQueue


@dataclass(slots=True)
class EngineConfig:
    """Device and buffer settings.

    These are the numbers ``PLAN.md`` Q2 asks to be *measured* rather than assumed. The defaults
    are a starting point on a PulseAudio desktop device; ``tools/measure_audio.py`` is how they
    get replaced with something known good for this machine.
    """

    sample_rate: int = 44100
    channels: int = 2
    #: Frames per callback. Larger absorbs more jitter at the cost of latency; 512 is 11.6 ms at
    #: 44.1 kHz, which is a reasonable desktop starting point.
    blocksize: int = 512
    device: int | str | None = None
    #: Pre-roll so the callback is never the first thing to touch the device.
    warmup_callbacks: int = 8


@dataclass(slots=True)
class EngineStatus:
    running: bool = False
    device_name: str = ""
    sample_rate: int = 0
    blocksize: int = 0
    latency_s: float = 0.0
    fallback_loaded: bool = False
    cached_tracks: int = 0
    underruns: int = 0
    queue_dropped: int = 0
    worst_callback_ms: float = 0.0


class AudioEngine:
    """The realtime side of the system."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.queue = CommandQueue(capacity=256)
        self.clock = Clock(sample_rate=self.config.sample_rate, bpm=120.0)
        self.mixer = Mixer(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            queue=self.queue,
            clock=self.clock,
        )
        self.cache = TrackCache(sample_rate=self.config.sample_rate)
        self._stream = None
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Open the device and start the callback thread.

        Raises:
            RuntimeError: if the device cannot be opened. Failing loudly here is correct — this is
                startup, not playback, and there is no set to protect yet.
        """
        import sounddevice as sd

        with self._lock:
            if self._stream is not None:
                return

            def callback(outdata, frames, _time, _status):
                # A status flag here would report an underrun. Counting it is enough; logging
                # from a callback is not allowed.
                self.mixer.render(outdata, frames)

            # `OutputStream`, not `RawOutputStream`: RawOutputStream hands the callback a raw
            # buffer object, and wrapping it in an array per block would allocate on the
            # realtime path. OutputStream supplies a numpy array directly, so the mixer can
            # work on the output in place.
            self._stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.blocksize,
                dtype="float32",
                device=self.config.device,
                callback=callback,
            )
            self._stream.start()
            self.mixer.start()
            for _ in range(self.config.warmup_callbacks):
                self._render_silence()

    def _render_silence(self) -> None:
        """Pre-roll so the first real block does not land on a cold device."""
        block = np.zeros((self.config.blocksize, self.config.channels), dtype=np.float32)
        self.mixer.render(block, self.config.blocksize)

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                finally:
                    self._stream = None
            self.mixer.stop()

    @property
    def running(self) -> bool:
        return self._stream is not None

    # -- loading (never on the audio thread) --------------------------------

    def preload(self, path: str | Path, *, bpm: float | None = None,
                expected_hash: str | None = None) -> PreloadedTrack:
        """Decode a file into memory. Safe to call from any thread."""
        return self.cache.preload(path, bpm=bpm, expected_hash=expected_hash)

    def preload_async(self, path: str | Path, *, bpm: float | None = None,
                      on_done: Callable[[PreloadedTrack | None], None] | None = None) -> None:
        """Preload on a background thread.

        The point is that a control thread can ask for a track without waiting for a decode, so a
        long file never delays a scheduler decision.
        """
        def worker() -> None:
            try:
                track = self.preload(path, bpm=bpm)
            except Exception:
                track = None
            if on_done:
                on_done(track)

        threading.Thread(target=worker, daemon=True).start()

    def install_fallback(self, path: str | Path, *, bpm: float | None = None) -> PreloadedTrack:
        """Load the bed that keeps the music going when a transition cannot land."""
        track = self.preload(path, bpm=bpm)
        self.mixer.set_fallback(track)
        return track

    # -- deck control (enqueues; never blocks) ------------------------------

    def play(self, deck: int = 0) -> None:
        self.queue.put(Command(CommandKind.PLAY, deck=deck))

    def pause(self, deck: int = 0) -> None:
        self.queue.put(Command(CommandKind.PAUSE, deck=deck))

    def seek_bar(self, bar: int, deck: int = 0) -> None:
        self.queue.put(Command(CommandKind.SEEK_BAR, deck=deck, value=float(bar)))

    def set_gain(self, gain: float, deck: int = 0) -> None:
        self.queue.put(Command(CommandKind.SET_GAIN, deck=deck, value=gain))

    def set_tempo(self, ratio: float, deck: int = 0) -> None:
        self.queue.put(Command(CommandKind.SET_TEMPO, deck=deck, value=ratio))

    def cue(self, deck: int = 0) -> None:
        self.queue.put(Command(CommandKind.CUE, deck=deck))

    def set_loop(self, enabled: bool, deck: int = 0) -> None:
        """Make a deck restart at the top instead of stopping at the end.

        Applied on the control thread as a direct flag write. The callback reads it as part of the
        deck state it already owns, so there is no command latency and nothing to allocate.
        """
        if 0 <= deck < len(self.mixer.decks):
            self.mixer.decks[deck].loop = enabled

    def emergency_stop(self) -> None:
        """Hard stop for both decks. Reachable at all times, even when everything else is busy."""
        self.queue.put(Command(CommandKind.EMERGENCY_STOP))

    def load_and_play(self, track: PreloadedTrack, deck: int = 0, *, fade_frames: int = 0) -> None:
        """Install a preloaded track and start it.

        The track swap happens here, on the control thread, as a single attribute store. The
        callback only receives the playback intent.
        """
        self.mixer.set_track(deck, track)
        self.queue.put(Command(CommandKind.LOAD_TRACK, deck=deck, value=1.0))
        if fade_frames > 0:
            self.queue.put(
                Command(CommandKind.START_CROSSFADE, deck=deck, value2=float(fade_frames))
            )

    def crossfade_to(self, track: PreloadedTrack, deck: int = 1, *, fade_frames: int) -> None:
        """Fade the incoming deck up and the other one down.

        Both decks keep playing through the overlap, which is what makes it a crossfade rather
        than a cut.
        """
        self.mixer.set_track(deck, track)
        self.queue.put(Command(CommandKind.START_CROSSFADE, deck=deck, value2=float(fade_frames)))

    # -- introspection ------------------------------------------------------

    def status(self) -> EngineStatus:
        import sounddevice as sd

        stats = self.mixer.snapshot()["stats"]
        name = ""
        latency = 0.0
        if self._stream is not None:
            try:
                info = sd.query_devices(self.config.device, "output")
                name = info["name"]
                # Prefer what PortAudio reports over blocksize/rate, which is only nominal.
                latency = float(self._stream.latency or 0.0) or float(
                    info["default_low_input_latency"] or 0.0
                )
            except Exception:  # status must never raise
                name = ""
        return EngineStatus(
            running=self.running,
            device_name=name,
            sample_rate=self.config.sample_rate,
            blocksize=self.config.blocksize,
            latency_s=latency or (self.config.blocksize / self.config.sample_rate),
            fallback_loaded=self.mixer.fallback is not None,
            cached_tracks=len(self.cache),
            underruns=int(stats["underruns"]),
            queue_dropped=int(stats["queue_dropped"]),
            worst_callback_ms=float(stats["worst_callback_ms"]),
        )

    def snapshot(self) -> dict[str, object]:
        return self.mixer.snapshot()