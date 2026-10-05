"""The feasibility filter.

``ARCHITECTURE.md:64``: *"The feasibility filter validates readiness, time-to-load, beat alignment,
tempo-change bounds, harmonic policy, vocal collisions, and recent repetition. It always includes a
safe ``continue_current`` or ``play_fallback_loop`` action."*

This runs before any model sees anything. It is pure, deterministic, and does no I/O, so the same
state always produces the same menu — which is what makes a decision reproducible
(``README.md:22``).

Two properties matter more than the individual checks:

1. **It never returns an empty menu.** If no asset is playable, the safe actions are still there.
   An empty candidate list would leave the scheduler with nothing to do at a phrase boundary.
2. **It is the first of two gates.** A selector's answer is re-checked by :func:`revalidate`
   before it can be scheduled, because a fine-tuned model will occasionally return something this
   filter would never have produced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from vexa_contracts import (
    ActionType,
    AssetManifest,
    Estimate,
    FeasibleAction,
    ReadinessState,
    RequestConstraints,
    SessionState,
)

#: Tag values that count as "this asset has a prominent vocal".
VOCAL_TAGS = frozenset({"vocal", "vocals", "sung", "lead vocal", "choir"})


def _tags(asset: AssetManifest, name: str) -> list[Estimate]:
    return asset.tags.get(name, [])


def _has_vocals(asset: AssetManifest) -> bool:
    return any(tag.value.strip().lower() in VOCAL_TAGS for tag in _tags(asset, "vocal"))


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Tunable bounds. Each is a musical decision, not a constant of convenience."""

    #: Longest crossfade the engine will attempt, in bars.
    max_fade_bars: int = 32
    #: Crossfade used when nothing better is known, in bars.
    default_fade_bars: int = 8
    #: How far ahead a transition may be scheduled, in bars.
    max_lead_bars: int = 64
    #: Minimum bars between now and the commit point.
    min_lead_bars: int = 4
    #: A family played within this many bars counts as recent repetition.
    repetition_window_bars: int = 128
    #: Hard engine ceiling on tempo change. Beyond this it is not a transition, it is a mistake.
    max_tempo_ratio: float = 1.10
    #: Confidence floor for accepting a key estimate as meaningful.
    key_confidence_floor: float = 0.5
    #: Non-safe candidates kept in the menu. Deliberately small: Laya has an option-token budget
    #: and high option counts degrade its confidence (``LAYA_DATA.md:11``).
    max_candidates: int = 6


@dataclass(frozen=True, slots=True)
class Rejection:
    """Why an asset did not make the menu. Surfaced in the UI, never swallowed."""

    asset_id: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.asset_id}: {self.reason}"


@dataclass(slots=True)
class FilterResult:
    """The menu plus its rejections, so a thin or empty menu can be explained."""

    candidates: list[FeasibleAction] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def safe_actions(self) -> list[FeasibleAction]:
        return [a for a in self.candidates if a.is_safe]

    @property
    def transitions(self) -> list[FeasibleAction]:
        return [a for a in self.candidates if not a.is_safe]


class FeasibilityFilter:
    """Builds the bounded menu of actions that are actually playable right now."""

    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()

    # -- asset-level checks -------------------------------------------------

    def asset_is_playable(self, asset: AssetManifest) -> str | None:
        """Return a rejection reason, or ``None`` if the asset may be scheduled."""
        if asset.readiness is not ReadinessState.READY:
            return f"not ready ({asset.readiness.value})"
        if not asset.quality.passed:
            return f"quality gates failed: {', '.join(asset.quality.failures)}"
        if asset.beat_grid is None:
            return "no beat grid, cannot cue against it"
        if not asset.beat_grid.reliable:
            return f"beat grid unreliable (confidence {asset.beat_grid.confidence:.2f})"
        if not asset.sections and not asset.loops:
            return "no detected section or approved loop to enter at"
        if not asset.audio.has_headroom():
            return "no true-peak headroom for a crossfade and master limiter"
        return None

    def entry_point(self, asset: AssetManifest) -> tuple[int, str] | None:
        """The bar to enter at, preferring a beat-aligned loop over a raw section start."""
        for loop in asset.loops:
            if loop.beat_aligned:
                return loop.start_bar, f"beat-aligned loop at bar {loop.start_bar}"
        if asset.sections:
            first = asset.sections[0]
            return first.start_bar, f"{first.kind} section at bar {first.start_bar}"
        if asset.loops:
            loop = asset.loops[0]
            return loop.start_bar, f"unaligned loop at bar {loop.start_bar}"
        return None

    # -- request-level checks ----------------------------------------------

    def violates_constraints(self, asset: AssetManifest, c: RequestConstraints) -> str | None:
        if asset.beat_grid is None:
            return "no tempo to compare"
        energy = asset.energy()
        if c.min_energy is not None or c.max_energy is not None:
            if energy is None:
                # An unmeasured asset cannot satisfy a stated floor or ceiling. Admitting it
                # would mean playing something we cannot claim matches the request — and the
                # listener asked for a specific direction, not "whatever is available".
                return "energy unmeasured, so a stated energy bound cannot be honoured"
            if c.min_energy is not None and energy < c.min_energy:
                return f"energy {energy:.2f} below requested floor {c.min_energy:.2f}"
            if c.max_energy is not None and energy > c.max_energy:
                return f"energy {energy:.2f} above requested ceiling {c.max_energy:.2f}"
        if c.forbid_vocals and _has_vocals(asset):
            return "requested vocals off but asset has vocals"
        # Tempo bounds were declared on the contract and never checked here, so "keep it under
        # 100" or "between 118 and 126" filtered nothing. Relative tempo is handled by
        # ``max_tempo_ratio`` elsewhere; this is the absolute bound the listener asked for, and
        # it is a different thing.
        bpm = asset.beat_grid.bpm
        if c.min_bpm is not None and bpm < c.min_bpm:
            return f"tempo {bpm:.1f} below requested floor {c.min_bpm:.1f}"
        if c.max_bpm is not None and bpm > c.max_bpm:
            return f"tempo {bpm:.1f} above requested ceiling {c.max_bpm:.1f}"
        return None

    def vocal_collision(
        self, current: AssetManifest | None, candidate: AssetManifest
    ) -> str | None:
        """Two prominent vocals must not overlap unless explicitly requested
        (``ARCHITECTURE.md:87``)."""
        if current is None:
            return None
        current_vocals = {tag.value.strip().lower() for tag in _tags(current, "vocal")}
        candidate_vocals = {tag.value.strip().lower() for tag in _tags(candidate, "vocal")}
        return (
            "would overlap two prominent vocals"
            if current_vocals & candidate_vocals
            else None
        )

    def key_compatible(
        self, current: AssetManifest | None, candidate: AssetManifest
    ) -> str | None:
        """Harmonic policy.

        Unknown or low-confidence keys are permissive. Refusing everything on the strength of a
        shaky estimate would stall the set, and stalling the set violates a stronger rule.
        """
        if current is None or current.key is None or candidate.key is None:
            return None
        if current.key.confidence < self.config.key_confidence_floor:
            return None
        if candidate.key.confidence < self.config.key_confidence_floor:
            return None
        current_mode = _mode(current.key.value)
        candidate_mode = _mode(candidate.key.value)
        if current_mode != candidate_mode and not _relative_keys(
                current.key.value, candidate.key.value):
            return f"mode clash ({current.key.value} -> {candidate.key.value})"
        return None

    def tempo_ratio(
        self, state: SessionState, asset: AssetManifest, c: RequestConstraints
    ) -> float | None:
        """Playback-rate ratio to reach the asset's tempo, or ``None`` if out of bounds.

        The bound is the stricter of the engine limit and whatever this request asked for, so a
        listener who wants gentle changes gets them even when the engine would tolerate more.
        """
        if asset.beat_grid is None or state.clock.bpm <= 0:
            return None
        ratio = asset.beat_grid.bpm / state.clock.bpm
        bound = min(self.config.max_tempo_ratio, c.max_tempo_ratio)
        if abs(ratio - 1.0) > bound - 1.0:
            return None
        return round(ratio, 4)

    # -- the menu -----------------------------------------------------------

    def next_phrase_bar(self, state: SessionState) -> int:
        """The next 8-bar boundary, which is where transitions normally land."""
        return ((state.clock.bar // 8) + 1) * 8

    def build(
        self,
        *,
        state: SessionState,
        library: Iterable[AssetManifest],
        constraints: RequestConstraints | None = None,
        current_asset: AssetManifest | None = None,
        recent_family_ids: Sequence[str] = (),
        commit_bar: int | None = None,
    ) -> FilterResult:
        """Assemble the menu. Never returns an empty candidate list."""
        c = constraints or RequestConstraints()
        result = FilterResult()

        target_bar = commit_bar if commit_bar is not None else self.next_phrase_bar(state)
        target_bar = max(target_bar, state.clock.bar + self.config.min_lead_bars)
        target_bar = min(target_bar, state.clock.bar + self.config.max_lead_bars)

        scored: list[tuple[float, FeasibleAction]] = []
        for asset in library:
            action = self._evaluate(
                asset,
                state=state,
                constraints=c,
                current_asset=current_asset,
                recent_family_ids=recent_family_ids,
                target_bar=target_bar,
                result=result,
            )
            if action is not None:
                scored.append((self._score(asset, c), action))

        scored.sort(key=lambda pair: (-pair[0], pair[1].action_id))
        result.candidates.extend(action for _, action in scored[: self.config.max_candidates])
        result.candidates.extend(self.safe_actions(state, target_bar))
        return result

    def _evaluate(
        self,
        asset: AssetManifest,
        *,
        state: SessionState,
        constraints: RequestConstraints,
        current_asset: AssetManifest | None,
        recent_family_ids: Sequence[str],
        target_bar: int,
        result: FilterResult,
    ) -> FeasibleAction | None:
        """Return a transition action for ``asset``, or record why there isn't one."""
        reject = lambda reason: result.rejections.append(  # noqa: E731
            Rejection(asset.asset_id, reason)
        )

        # Transitioning to the asset already playing is a no-op at best. The filter is given the
        # current asset precisely so it can exclude it; without that check the menu offers "go to
        # the track you are on", and a selector is free to pick it.
        if current_asset is not None and asset.asset_id == current_asset.asset_id:
            reject("already playing")
            return None

        if reason := self.asset_is_playable(asset):
            reject(reason)
            return None
        entry = self.entry_point(asset)
        if entry is None:
            reject("no usable entry point")
            return None
        entry_bar, entry_reason = entry

        if asset.family_id in recent_family_ids:
            reject(f"family {asset.family_id} played recently")
            return None
        if reason := self.violates_constraints(asset, constraints):
            reject(reason)
            return None
        if reason := self.vocal_collision(current_asset, asset):
            reject(reason)
            return None
        if reason := self.key_compatible(current_asset, asset):
            reject(reason)
            return None

        ratio = self.tempo_ratio(state, asset, constraints)
        if ratio is None:
            bpm = asset.beat_grid.bpm if asset.beat_grid else 0.0
            reject(
                f"tempo change to {bpm:.0f} BPM exceeds the "
                f"{self.config.max_tempo_ratio:.2f}x bound"
            )
            return None

        lead = target_bar - state.clock.bar
        return FeasibleAction(
            action_id=f"transition_to_{asset.asset_id}",
            action_type=ActionType.TRANSITION,
            asset_id=asset.asset_id,
            entry_bar=entry_bar,
            start_at_session_bar=target_bar,
            fade_bars=max(0, min(self.config.default_fade_bars, lead, self.config.max_fade_bars)),
            tempo_ratio=ratio,
            rationale=f"{entry_reason}, compatible tempo",
            energy_fit=asset.energy() if asset.energy() is not None else 0.5,
        )

    def _score(self, asset: AssetManifest, c: RequestConstraints) -> float:
        """Rank by closeness to the requested energy. Deterministic; ties broken by action id."""
        energy = asset.energy()
        if energy is None:
            return 0.5
        target = c.max_energy if c.max_energy is not None else 0.5
        return 1.0 - abs(energy - target)

    def safe_actions(self, state: SessionState, target_bar: int) -> list[FeasibleAction]:
        """Always-present options. ``ARCHITECTURE.md:64``."""
        actions = [
            FeasibleAction(
                action_id="continue_current",
                action_type=ActionType.CONTINUE_CURRENT,
                start_at_session_bar=target_bar,
                rationale="keep the current track playing",
                energy_fit=state.energy,
            )
        ]
        if state.has_safe_fallback:
            actions.append(
                FeasibleAction(
                    action_id="play_fallback_loop",
                    action_type=ActionType.PLAY_FALLBACK_LOOP,
                    asset_id=state.fallback_asset_id,
                    start_at_session_bar=target_bar,
                    rationale="fallback bed is loaded and ready",
                    energy_fit=0.5,
                )
            )
        return actions


def _mode(key: str) -> str:
    """Extract the mode from a key string, defaulting to major for bare notes."""
    parts = key.strip().lower().split()
    if len(parts) < 2:
        return "major"
    tail = parts[-1]
    if tail in ("minor", "min", "m"):
        return "minor"
    if tail in ("major", "maj"):
        return "major"
    return tail


def _relative_keys(left: str, right: str) -> bool:
    """Relative major/minor share a key signature and can crossfade safely."""
    roots = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
             "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
             "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
    a = roots.get(left.split()[0].upper())
    b = roots.get(right.split()[0].upper())
    if a is None or b is None:
        return False
    return ((_mode(left) == "major" and _mode(right) == "minor" and (a - b) % 12 == 3)
            or (_mode(left) == "minor" and _mode(right) == "major" and (b - a) % 12 == 3))


def revalidate(
    action: FeasibleAction, offered: Sequence[FeasibleAction]
) -> FeasibleAction | None:
    """The second deterministic gate.

    Whatever a selector returns is checked against the menu that was actually offered. A selector
    that names an action nobody offered — or that mutates an offered action's parameters — is
    refused, and the caller falls back to the rules.
    """
    for candidate in offered:
        if candidate.action_id != action.action_id:
            continue
        unchanged = (
            candidate.action_type is action.action_type
            and candidate.asset_id == action.asset_id
            and candidate.start_at_session_bar == action.start_at_session_bar
            and candidate.entry_bar == action.entry_bar
            and candidate.fade_bars == action.fade_bars
            and abs(candidate.tempo_ratio - action.tempo_ratio) <= 1e-6
        )
        return candidate if unchanged else None
    return None
