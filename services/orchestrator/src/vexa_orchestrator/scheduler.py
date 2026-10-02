"""The scheduler.

``ARCHITECTURE.md:64`` and ``:68``. The scheduler commits transitions in **musical time** — beats,
bars, phrase boundaries — and only when the assets are loaded before the required deadline.

The rule that makes this trustworthy: **a late command is dropped, never executed late.** A
transition that fires 200 ms after its phrase boundary is worse than no transition at all, because
the listener hears a dropped beat rather than a DJ choice. When a deadline passes the scheduler
holds the current track and says so.

Nothing here blocks. The audio callback is a different process entirely; this schedules *future*
commands for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from vexa_contracts import (
    ActionType,
    AssetManifest,
    DeckId,
    FeasibleAction,
    ScheduledAction,
    SessionState,
)

#: Transitions land on multiples of this many bars.
PHRASE_BARS = 8


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    #: Wall-clock lead time by which a transition's asset must be loaded.
    load_lead_time_s: float = 5.0
    #: Beyond this many bars ahead, a plan is too fragile to commit.
    max_commit_horizon_bars: int = 64
    #: A committed action older than this is abandoned even if unexpired.
    max_plan_age_s: float = 30.0


@dataclass(frozen=True, slots=True)
class ScheduleOutcome:
    """What happened to a commit attempt. Explicit, so nothing fails silently."""

    action_id: str
    committed: bool
    reason: str
    action: ScheduledAction | None = None

    def __bool__(self) -> bool:
        return self.committed


@dataclass(slots=True)
class Scheduler:
    """Commits actions against a musical clock, with deadline enforcement."""

    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    _committed: ScheduledAction | None = None

    # -- musical helpers ----------------------------------------------------

    @staticmethod
    def next_boundary(state: SessionState, bars: int = PHRASE_BARS) -> int:
        """The next bar that is a multiple of ``bars``."""
        return ((state.clock.bar // bars) + 1) * bars

    @staticmethod
    def bars_until(state: SessionState, target_bar: int) -> int:
        return max(0, target_bar - state.clock.bar)

    @staticmethod
    def is_on_phrase(state: SessionState, bars: int = PHRASE_BARS) -> bool:
        return state.clock.bar % bars == 0 and state.clock.beat == 0

    # -- committing ---------------------------------------------------------

    def commit(
        self,
        *,
        state: SessionState,
        action: FeasibleAction,
        asset: AssetManifest | None,
        fallback_action_id: str | None = None,
        now: datetime | None = None,
    ) -> ScheduleOutcome:
        """Try to commit ``action``. Refuses rather than commits late."""
        now = now or datetime.now(UTC)
        commit_bar = action.start_at_session_bar
        horizon = self.bars_until(state, commit_bar)

        if horizon <= 0:
            return ScheduleOutcome(
                action.action_id, False, f"commit bar {commit_bar} is not in the future"
            )
        if horizon > self.config.max_commit_horizon_bars:
            return ScheduleOutcome(
                action.action_id,
                False,
                f"{horizon} bars ahead exceeds the "
                f"{self.config.max_commit_horizon_bars}-bar commit horizon",
            )
        if commit_bar % PHRASE_BARS != 0:
            return ScheduleOutcome(
                action.action_id,
                False,
                f"bar {commit_bar} is not a {PHRASE_BARS}-bar phrase boundary",
            )
        if action.fade_bars > horizon:
            return ScheduleOutcome(
                action.action_id,
                False,
                f"{action.fade_bars}-bar crossfade does not fit in {horizon} bars",
            )

        if action.is_safe:
            # continue_current / play_fallback_loop need no asset to be schedulable.
            pass
        elif asset is None:
            return ScheduleOutcome(
                action.action_id, False, f"action {action.action_id} needs an asset"
            )
        elif not asset.admissible():
            return ScheduleOutcome(
                action.action_id, False, f"asset {asset.asset_id} is not schedulable"
            )
        elif not asset.audio.has_headroom():
            return ScheduleOutcome(
                action.action_id, False, f"asset {asset.asset_id} has no peak headroom"
            )
        elif asset.asset_id != action.asset_id:
            return ScheduleOutcome(
                action.action_id,
                False,
                f"action names asset {action.asset_id} but {asset.asset_id} was supplied",
            )

        scheduled = ScheduledAction.from_action(
            action,
            state,
            asset_version=asset.version if asset else None,
            asset_sha256=asset.content_sha256 if asset else None,
            fallback_action_id=fallback_action_id,
            load_lead_time_s=self.config.load_lead_time_s,
        )
        self._committed = scheduled
        return ScheduleOutcome(
            action.action_id, True, f"committed at bar {commit_bar}", scheduled
        )

    # -- draining -----------------------------------------------------------

    def due(
        self, *, state: SessionState, now: datetime | None = None
    ) -> ScheduleOutcome | None:
        """Return the committed action if it has reached its commit bar.

        The staleness check runs here, not at commit time: a newer user request may have landed in
        between, and a result arriving after the listener changed their mind must not take effect
        (``ARCHITECTURE.md:81``).
        """
        if self._committed is None:
            return None
        now = now or datetime.now(UTC)
        pending = self._committed

        if pending.is_stale(state.generation):
            self._committed = None
            return ScheduleOutcome(
                pending.action_id,
                False,
                f"superseded: planned at generation {pending.generation}, "
                f"session is now {state.generation}",
            )
        if pending.deadline_expired(at=now):
            self._committed = None
            return ScheduleOutcome(
                pending.action_id, False, "asset was not loaded before the deadline"
            )
        if now - pending.committed_at > timedelta(seconds=self.config.max_plan_age_s):
            self._committed = None
            return ScheduleOutcome(
                pending.action_id, False, "plan aged out before its commit bar"
            )
        if state.clock.bar < pending.commit_bar:
            return None

        self._committed = None
        return ScheduleOutcome(
            pending.action_id, True, f"executed at bar {pending.commit_bar}", pending
        )

    @property
    def pending(self) -> ScheduledAction | None:
        return self._committed

    def clear(self) -> None:
        """Drop any committed action. Used when a new request invalidates the plan outright."""
        self._committed = None

    # -- applying to session state -----------------------------------------

    def apply(
        self,
        *,
        state: SessionState,
        action: FeasibleAction,
        asset: AssetManifest | None,
    ) -> DeckId:
        """Move a committed action into session state. Returns the deck that received the audio.

        Applying is a state transition on the control plane. The audio engine consumes these as
        future commands; nothing here opens a device.
        """
        if action.is_safe and action.action_type is ActionType.CONTINUE_CURRENT:
            target = DeckId.A
        else:
            target = DeckId.B

        incoming = state.deck(target)
        if asset is not None:
            incoming.asset_id = asset.asset_id
            incoming.position_bar = action.entry_bar
            incoming.playing = True
            incoming.tempo_ratio = action.tempo_ratio
        else:
            incoming.playing = False

        if action.action_type is not ActionType.CONTINUE_CURRENT:
            state.deck(DeckId.A if target is DeckId.B else DeckId.B).playing = False

        state.clock.bar = max(state.clock.bar, action.start_at_session_bar)
        state.clock.bpm = asset.beat_grid.bpm if asset and asset.beat_grid else state.clock.bpm
        state.committed_action_id = action.action_id
        state.energy = action.energy_fit
        return target


def assets_ready(assets: Iterable[AssetManifest]) -> set[str]:
    """IDs of assets currently allowed to be scheduled."""
    return {a.asset_id for a in assets if a.admissible()}