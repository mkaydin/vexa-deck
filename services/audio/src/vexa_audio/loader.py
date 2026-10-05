"""Decoding and preparing audio for the realtime path.

Everything expensive happens here, on a loader thread, never in the audio callback. A callback that
decodes a FLAC or resamples is a callback that will glitch.

Two rules:

* **A track is identified by its content hash**, not its path (``PLAN.md`` §6). A scheduler
  decision names a hash, so a file that changed underneath it cannot be played.
* **The device rate is the working rate.** Resampling once, here, means the callback only ever
  reads sequential samples.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PreloadedTrack:
    """Audio in memory, ready for the callback to read sequentially.

    ``samples`` is ``(n, channels)`` float32 at the device rate. Read-only from here on; the
    callback slices views into it and never mutates it.
    """

    content_sha256: str
    samples: np.ndarray
    sample_rate: int
    channels: int
    #: Sample index corresponding to musical bar 0, when the beat grid is known.
    bar0_sample: float = 0.0
    samples_per_bar: float = 0.0
    duration_s: float = 0.0
    #: True peak measured at load, so the mixer can predict clipping before it happens.
    peak: float = 0.0

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    def sample_at_bar(self, bar: int) -> int:
        """Sample index for a musical bar.

        Only meaningful when ``samples_per_bar > 0``; otherwise the caller must not seek by bar.
        """
        if self.samples_per_bar <= 0:
            return 0
        return int(self.bar0_sample + bar * self.samples_per_bar)

    @property
    def can_seek_by_bar(self) -> bool:
        return self.samples_per_bar > 0


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_track(
    path: str | Path,
    *,
    sample_rate: int,
    bpm: float | None = None,
    beats_per_bar: int = 4,
    expected_hash: str | None = None,
) -> PreloadedTrack:
    """Decode, downmix/upmix, and resample to the device rate.

    Raises:
        ValueError: if ``expected_hash`` is given and does not match. A hash mismatch means the
            file changed after a decision was made against it, and playing it would act on a
            different artifact than the one that was approved.
    """
    import soundfile as sf

    path = Path(path)
    actual = file_sha256(path)
    if expected_hash and actual != expected_hash:
        raise ValueError(
            f"content hash mismatch for {path.name}: expected {expected_hash[:12]}, "
            f"found {actual[:12]}"
        )

    data, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    channels = data.shape[1]

    if source_rate != sample_rate:
        data = _resample(data, source_rate, sample_rate)

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    duration = data.shape[0] / sample_rate

    samples_per_bar = 0.0
    bar0 = 0.0
    if bpm and bpm > 0:
        samples_per_bar = sample_rate * 60.0 / bpm * beats_per_bar

    return PreloadedTrack(
        content_sha256=actual,
        samples=np.ascontiguousarray(data, dtype=np.float32),
        sample_rate=sample_rate,
        channels=channels,
        bar0_sample=bar0,
        samples_per_bar=samples_per_bar,
        duration_s=duration,
        peak=peak,
    )


def _resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample off the realtime path.

    Linear interpolation rather than a windowed sinc: by this point the material has already been
    through a quality gate, and the accuracy that matters for a 44.1/48 kHz conversion between two
    studio rates is marginal for a crossfade. Revisit if the gates ever pass material it should
    not.
    """
    if source_rate == target_rate:
        return data
    frames_in = data.shape[0]
    frames_out = round(frames_in * target_rate / source_rate)
    if frames_out < 1:
        return np.zeros((1, data.shape[1]), dtype=np.float32)

    source_idx = np.arange(frames_in, dtype=np.float64)
    target_idx = np.linspace(0.0, frames_in - 1, frames_out)
    out = np.empty((frames_out, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        out[:, channel] = np.interp(target_idx, source_idx, data[:, channel]).astype(np.float32)
    return out


class TrackCache:
    """A thread-safe cache of decoded audio, keyed by content hash.

    Read by the loader thread and by the callback (read-only after load), so the lock is only
    ever taken on the loading side. The callback receives a finished
    :class:`PreloadedTrack` by pointer and never asks for one.
    """

    def __init__(self, *, sample_rate: int, capacity: int = 32) -> None:
        self.sample_rate = sample_rate
        self.capacity = capacity
        self._tracks: dict[str, PreloadedTrack] = {}
        self._lock = threading.Lock()

    def get(self, digest: str) -> PreloadedTrack | None:
        with self._lock:
            return self._tracks.get(digest)

    def put(self, track: PreloadedTrack) -> None:
        with self._lock:
            if len(self._tracks) >= self.capacity:
                # Simple eviction. A real DJ session revisits recent material far more than old
                # material, so dropping the oldest insertion is the right default.
                oldest = next(iter(self._tracks))
                del self._tracks[oldest]
            self._tracks[track.content_sha256] = track

    def preload(
        self,
        path: str | Path,
        *,
        bpm: float | None = None,
        expected_hash: str | None = None,
    ) -> PreloadedTrack:
        """Load if absent, return if present. Never blocks the caller for longer than one decode."""
        if expected_hash:
            cached = self.get(expected_hash)
            if cached is not None:
                return cached
        track = load_track(
            path, sample_rate=self.sample_rate, bpm=bpm, expected_hash=expected_hash
        )
        self.put(track)
        return track

    def __len__(self) -> int:
        with self._lock:
            return len(self._tracks)

    def warm(
        self,
        tracks: list[tuple[str | Path, float | None]],
        on_done: Callable[[str | Path], None] | None = None,
    ) -> None:
        """Preload several files. Used to warm the fallback bed before a set starts."""
        for path, bpm in tracks:
            try:
                self.preload(path, bpm=bpm)
            except Exception:
                pass
            finally:
                if on_done:
                    on_done(path)


class MasterLimiter:
    """Stereo linked sample peak limiter with a continuous 250 ms release.

    Instant attack protects the output ceiling. The gain envelope persists across callbacks;
    unlike a block peak limiter it cannot jump back to unity on each quiet block. This is
    sample peak protection, not an oversampled true peak or lookahead limiter.
    Working arrays are allocated once and the returned output is reused.
    """

    def __init__(self, *, frames: int, channels: int = 2, ceiling: float = 0.891,
                 sample_rate: int = 44100) -> None:
        if frames < 1 or channels < 1 or not 0 < ceiling <= 1 or sample_rate < 1:
            raise ValueError("invalid limiter configuration")
        self.ceiling = ceiling
        self._gain = 1.0
        self._out = np.empty((frames, channels), dtype=np.float32)
        self._absolute = np.empty_like(self._out)
        self._peak = np.empty(frames, dtype=np.float64)
        self._envelope = np.empty(frames, dtype=np.float64)
        # deficit[i] = max(required_deficit[i], release * deficit[i-1]).
        # Dividing by release**i turns the recurrence into a vectorized cumulative max.
        self._release = np.exp(-np.arange(1, frames + 1) / (sample_rate * 0.25))
        self._inverse_release = 1 / self._release

    def process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        if n > len(self._out):
            raise ValueError("audio block exceeds limiter capacity")
        if not n:
            return self._out[:0]
        peak, envelope = self._peak[:n], self._envelope[:n]
        np.abs(block, out=self._absolute[:n])
        np.maximum.reduce(self._absolute[:n], axis=1, out=peak)
        np.maximum(peak, self.ceiling, out=peak)
        np.divide(self.ceiling, peak, out=envelope)
        np.subtract(1.0, envelope, out=envelope)
        np.multiply(envelope, self._inverse_release[:n], out=envelope)
        np.maximum(envelope, 1.0 - self._gain, out=envelope)
        np.maximum.accumulate(envelope, out=envelope)
        np.multiply(envelope, self._release[:n], out=envelope)
        np.subtract(1.0, envelope, out=envelope)
        self._gain = float(envelope[-1])
        np.multiply(block, envelope[:, None], out=self._out[:n])
        return self._out[:n]

    def reset(self) -> None:
        self._gain = 1.0
