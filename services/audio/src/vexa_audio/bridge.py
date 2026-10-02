"""Bridge: scheduler decisions become engine commands.

This is the seam the project has been missing. Everything upstream produces a
:class:`~vexa_contracts.ScheduledAction`; everything downstream is a deck playing audio. Until
this module existed, nothing connected them.

The design rule is one-directional. **Audio never calls up.** The bridge reads the orchestrator's
committed action and pushes commands into the engine's queue; if the orchestrator is unreachable
the bridge simply does nothing and the current track keeps playing. Audio is never blocked by the
control plane — that is ``README.md:18`` stated as an implementation constraint.

Crossfades are computed from the scheduler's commit bar rather than guessed: bars to the commit
point at the current tempo gives the exact frame count, so the fade is however long the plan said
it should be.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vexa_contracts import AssetManifest, ScheduledAction, SessionState

from vexa_audio.engine import AudioEngine
from vexa_audio.loader import PreloadedTrack


@dataclass(slots=True)
class BridgeConfig:
    #: Which deck a transition lands on. Deck B is the incoming deck; deck A keeps playing under
    #: the crossfade and is faded out.
    incoming_deck: int = 1
    outgoing_deck: int = 0
    #: Minimum crossfade, in bars. Below this a transition reads as a cut rather than a blend.
    min_fade_bars: int = 4
    #: Ceiling, so a slow tempo cannot produce a 40-second fade.
    max_fade_bars: int = 32
    #: Bounce at the end of a track instead of falling silent.
    loop_decks: bool = True


@dataclass(slots=True)
class BridgeStats:
    applied: int = 0
    skipped_not_ready: int = 0
    skipped_no_asset: int = 0
    preload_failures: int = 0


class AudioBridge:
    """Applies orchestrator decisions to the audio engine."""

    def __init__(
        self,
        engine: AudioEngine,
        *,
        config: BridgeConfig | None = None,
        library: dict[str, Path] | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or BridgeConfig()
        #: content hash -> file on disk. Injected so the bridge never walks a filesystem.
        self.library: dict[str, Path] = library or {}
        self.stats = BridgeStats()

    def register(self, manifest: AssetManifest, path: Path) -> None:
        """Make an admitted asset reachable by the engine."""
        self.library[manifest.content_sha256] = path

    def apply(
        self,
        *,
        action: ScheduledAction,
        state: SessionState,
        current_asset: AssetManifest | None = None,
    ) -> bool:
        """Execute a committed action. Returns whether anything was actually played."""
        if action.asset_sha256 is None:
            self.stats.skipped_no_asset += 1
            return False

        path = self.library.get(action.asset_sha256)
        if path is None:
            self.stats.skipped_no_asset += 1
            return False

        if not Path(path).exists():
            self.stats.skipped_no_asset += 1
            return False

        try:
            track: PreloadedTrack = self.engine.preload(
                path, bpm=state.clock.bpm, expected_hash=action.asset_sha256
            )
        except Exception:  # a failed load must not stop the current track
            self.stats.preload_failures += 1
            return False

        if self.config.loop_decks:
            self.engine.set_loop(True, deck=self.config.incoming_deck)

        fade = self._fade_frames(state, action)
        self.engine.crossfade_to(
            track, deck=self.config.incoming_deck, fade_frames=fade
        )
        self.stats.applied += 1
        return True

    def _fade_frames(self, state: SessionState, action: ScheduledAction) -> int:
        """Frames of crossfade, derived from the musical plan rather than guessed."""
        bars = max(self.config.min_fade_bars, action.commit_bar - state.clock.bar)
        bars = min(bars, self.config.max_fade_bars)
        # seconds_per_bar is in SECONDS; the engine needs FRAMES. Dropping the sample-rate factor
        # produced a 16-frame "fade" — a sub-millisecond cut wearing a crossfade's name.
        return int(state.clock.seconds_per_bar * bars * self.engine.sample_rate)

    # -- manual control -----------------------------------------------------

    def start(self, manifest: AssetManifest) -> bool:
        """Begin playing an asset immediately, for a session's opening bed."""
        path = self.library.get(manifest.content_sha256)
        if path is None:
            return False
        try:
            track = self.engine.preload(
                path, bpm=manifest.beat_grid.bpm if manifest.beat_grid else None
            )
        except Exception:
            self.stats.preload_failures += 1
            return False
        if self.config.loop_decks:
            self.engine.set_loop(True, deck=self.config.outgoing_deck)
        self.engine.load_and_play(track, deck=self.config.outgoing_deck)
        return True

    def snapshot(self) -> dict[str, object]:
        return {
            "stats": {
                "applied": self.stats.applied,
                "skipped_not_ready": self.stats.skipped_not_ready,
                "skipped_no_asset": self.stats.skipped_no_asset,
                "preload_failures": self.stats.preload_failures,
            },
            "library_size": len(self.library),
        }
