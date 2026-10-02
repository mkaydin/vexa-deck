"""Session and transport state.

Musical time is the primary clock. The scheduler reasons in bars and beats, never in wall-clock
seconds, because a transition is only correct if it lands on a phrase boundary
(``ARCHITECTURE.md:68``). Wall-clock fields exist for the UI and for deadline arithmetic, but no
musical decision reads them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .asset import Contract


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DeckId(StrEnum):
    A = "a"
    B = "b"


class DeckState(BaseModel):
    """One deck. ``asset_id is None`` means the deck is idle."""

    model_config = ConfigDict(extra="forbid")

    deck_id: DeckId
    asset_id: str | None = None
    #: Bars from the asset's own start.
    position_bar: int = Field(default=0, ge=0)
    playing: bool = False
    gain_db: float = Field(default=0.0, ge=-60.0, le=6.0)
    tempo_ratio: float = Field(default=1.0, gt=0.0, le=2.0)


class MusicalClock(BaseModel):
    """Position within the set, in musical time."""

    model_config = ConfigDict(extra="forbid")

    bar: int = Field(default=0, ge=0)
    beat: int = Field(default=0, ge=0, le=3)
    bpm: float = Field(gt=0.0, default=120.0)
    beats_per_bar: int = Field(default=4, ge=1, le=32)

    @model_validator(mode="after")
    def _beat_in_bar(self) -> MusicalClock:
        if self.beat >= self.beats_per_bar:
            raise ValueError(
                f"beat {self.beat} is outside a {self.beats_per_bar}-beat bar"
            )
        return self

    @property
    def absolute_beat(self) -> int:
        return self.bar * self.beats_per_bar + self.beat

    @property
    def beats_per_second(self) -> float:
        return self.bpm / 60.0

    @property
    def seconds_per_bar(self) -> float:
        """Seconds in one bar.

        ``60 * beats_per_bar / bpm``: at 120 BPM in 4/4 that is 2.0 s, which is right -- two beats
        per second, four to a bar.

        This was previously ``240 * beats_per_bar / bpm``, a 4x error that made every
        bars-to-seconds conversion in the system wrong by exactly one time signature.
        """
        return 60.0 * self.beats_per_bar / self.bpm

    def bars_until(self, target_bar: int) -> int:
        return max(0, target_bar - self.bar)


class SessionState(Contract):
    """A running set.

    ``generation`` increments on every accepted user request. A scheduled action carries the
    generation it was planned under, so a late-arriving generation result cannot reverse the
    listener's latest direction (``ARCHITECTURE.md:81``).
    """

    session_id: str = Field(min_length=1)
    theme: str = Field(min_length=1)
    generation: int = Field(default=0, ge=0)

    clock: MusicalClock = Field(default_factory=MusicalClock)
    decks: dict[DeckId, DeckState] = Field(default_factory=dict)

    #: Always loaded, per the continuity rule in ``ARCHITECTURE.md:85``.
    fallback_asset_id: str | None = None

    #: ID of the committed transition, if any.
    committed_action_id: str | None = None
    active_request_id: str | None = None

    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    started_at: datetime = Field(default_factory=_utcnow)

    def deck(self, deck_id: DeckId) -> DeckState:
        """Return the deck, creating an idle one on first use."""
        return self.decks.setdefault(deck_id, DeckState(deck_id=deck_id))

    @property
    def has_safe_fallback(self) -> bool:
        return self.fallback_asset_id is not None