"""The Laya adapter.

Wraps ``laya`` so the orchestrator can ask one bounded question and get back an action id. The
model never sees raw audio, never picks from an unbounded list, and never writes a DSP command.

Three things this module refuses to do, each for a reason the docs or Laya's own documentation
gives explicitly:

1. **Send a long menu.** Laya has an option-token budget, and high option counts degrade
   confidence selection. The orchestrator already shortlists; this is the hard backstop.
2. **Trust an uncalibrated confidence.** The shipped checkpoints are over-confident. A confidence
   threshold means nothing until temperatures are fitted, so :class:`CalibrationGate` is closed by
   default and the adapter reports raw scores until it is opened deliberately.
3. **Let a ``noul`` answer substitute for validation.** The yes/no question is the affirmative
   half of the second gate, not a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vexa_contracts import ActionType, DecisionRequest, DecisionResponse, FeasibleAction

#: Laya's documented ceiling before high-cardinality choice starts hurting confidence.
MAX_OPTIONS = 8

#: Action ids that always mean "hold position". Used to detect a menu with no safe option.
SAFE_IDS = frozenset({"continue_current", "play_fallback_loop"})


class LayaUnavailable(RuntimeError):
    """The checkpoint or its dependencies are not installed. Never fatal to playback."""


@dataclass(slots=True)
class CalibrationGate:
    """Holds a per-question-type temperature fit and decides whether confidence may gate.

    Two rules from Laya's fine-tuning documentation are encoded here rather than left to memory:

    * Fit on a slice held out **before** training. Fitting on data the run already saw measures
      the fit rather than the calibration.
    * An inherited ``temperature_by_options`` takes precedence at inference and silently masks a
      new fit. Loading a checkpoint therefore *removes* it rather than trusting it.
    """

    #: One temperature per Laya question type. ``None`` means uncalibrated.
    temperatures: dict[str, float] = field(default_factory=dict)
    #: Whether confidence is trusted enough to gate on. Closed until deliberately opened.
    calibrated: bool = False

    def apply(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return a checkpoint config with a fitted calibration and no inherited override."""
        if not self.calibrated:
            return config
        updated = dict(config)
        updated["temperature"] = dict(self.temperatures)
        # The documented trap: leftover per-option temperatures win at inference and hide the fit.
        updated.pop("temperature_by_options", None)
        return updated

    def open(self, temperatures: dict[str, float]) -> None:
        """Mark calibrated. Only call this after fitting on a held-out slice."""
        if not temperatures:
            raise ValueError("cannot open the calibration gate without fitted temperatures")
        self.temperatures = dict(temperatures)
        self.calibrated = True

    def adjust(self, question_type: str, logit_confidence: float) -> float:
        """Apply temperature scaling to a raw confidence.

        Temperature scaling moves confidence but not the argmax, so an uncalibrated gate refuses
        to gate rather than pretending a raw score means something.
        """
        if not self.calibrated:
            return logit_confidence
        temperature = self.temperatures.get(question_type, 1.0)
        return _scale(logit_confidence, temperature)


def _scale(confidence: float, temperature: float) -> float:
    """Temperature-scale a probability. ``temperature > 1`` softens an over-confident model."""
    if temperature <= 0:
        return confidence
    logit = _logit(confidence)
    return _sigmoid(logit / temperature)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    from math import log

    return log(p / (1 - p))


def _sigmoid(x: float) -> float:
    from math import exp

    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True, slots=True)
class LayaSettings:
    checkpoint: str = "convaiinnovations/laya-typed-decisions"
    device: str = "cuda:1"
    #: ``max_len`` must exceed the state text; the head budget bounds the option list.
    max_len: int = 1024
    head_max_len: int = 256
    language: str = "en"


class LayaAdapter:
    """Loads a checkpoint lazily and answers one bounded choice per decision."""

    def __init__(
        self,
        settings: LayaSettings | None = None,
        calibration: CalibrationGate | None = None,
    ) -> None:
        self.settings = settings or LayaSettings()
        self.calibration = calibration or CalibrationGate()
        self._agent: Any = None

    @property
    def name(self) -> str:
        """Satisfies ``DecisionPolicy`` -- ``ShadowPolicy`` reads it to label its observations."""
        return f"laya:{self.settings.checkpoint}"

    @property
    def loaded(self) -> bool:
        return self._agent is not None

    def load(self) -> None:
        """Load the checkpoint. Raises :class:`LayaUnavailable` rather than crashing the caller."""
        if self._agent is not None:
            return
        try:
            import laya
        except ImportError as exc:
            raise LayaUnavailable(
                "laya is not installed; install the 'gpu' extra to enable the selector"
            ) from exc
        try:
            self._agent = laya.load(self.settings.checkpoint)
        except Exception as exc:
            raise LayaUnavailable(f"could not load {self.settings.checkpoint}: {exc}") from exc

    def unload(self) -> None:
        self._agent = None

    # -- question construction ---------------------------------------------

    def build_question(self, request: DecisionRequest) -> dict[str, Any]:
        """A typed ``choice`` question over a shortlist of feasible actions.

        Raises:
            ValueError: if the menu is empty or exceeds what Laya can answer reliably.
        """
        candidates = self._shortlist(request.candidates)
        if len(candidates) < 2:
            raise ValueError("Laya needs at least two options to choose between")
        return {
            "choice": {
                "type": "choice",
                "instructions": (
                    "Which action best follows the listener's request at the next phrase "
                    "boundary?"
                ),
                "criteria": [self._label(a) for a in candidates],
            }
        }


    def _shortlist(self, candidates: list[FeasibleAction]) -> list[FeasibleAction]:
        """Bound the option count, keeping every safe action.

        Safe actions are never dropped: without them the model cannot express "do nothing", and
        it learns to over-transition (``LAYA_DATA.md:57``).
        """
        safe = [a for a in candidates if a.action_id in SAFE_IDS or a.is_safe]
        transitions = sorted(
            (a for a in candidates if a not in safe),
            key=lambda a: (-a.energy_fit, a.action_id),
        )
        room = max(0, MAX_OPTIONS - len(safe))
        chosen = safe + transitions[:room]
        if len(chosen) > MAX_OPTIONS:  # pragma: no cover - guarded by `room`
            chosen = chosen[:MAX_OPTIONS]
        return chosen

    @staticmethod
    def _label(action: FeasibleAction) -> str:
        """A short factual label. Descriptions come from measured features, never invention."""
        if action.action_type is ActionType.CONTINUE_CURRENT:
            return "keep the current track playing"
        if action.action_type is ActionType.PLAY_FALLBACK_LOOP:
            return "fallback loop bed"
        bits = [f"{action.action_type.value} to {action.asset_id}"]
        if action.tempo_ratio:
            bits.append(f"tempo ratio {action.tempo_ratio:.2f}")
        bits.append(f"energy fit {action.energy_fit:.2f}")
        return ", ".join(bits)

    # -- inference ----------------------------------------------------------

    def choose(self, request: DecisionRequest) -> DecisionResponse:
        """Ask Laya which action it would take.

        This never raises for a model problem. A failure returns a zero-confidence response with
        ``fell_back`` set, so the orchestrator's chain policy hands the decision to the rules and
        playback is unaffected (``README.md:18``).
        """
        try:
            self.load()
            question = self.build_question(request)
            state = self.state_text(request)
            result = self._agent.predict(
                state,
                question,
                max_len=self.settings.max_len,
                head_max_len=self.settings.head_max_len,
                lang=self.settings.language,
            )
        except LayaUnavailable as exc:
            # A missing or unloadable checkpoint is a configuration state, not a crash. The set
            # must keep playing, so this reports a fallback rather than propagating.
            return DecisionResponse(
                decision_id=request.decision_id,
                action_id=_safe_action_id(request),
                confidence=0.0,
                selector="laya",
                fell_back=True,
                reasoning=f"laya unavailable: {exc}",
            )
        except Exception:
            return DecisionResponse(
                decision_id=request.decision_id,
                action_id="continue_current",
                confidence=0.0,
                selector="laya",
                fell_back=True,
                reasoning="laya inference failed; rules decide",
            )

        answer = (result.get("answers") or {}).get("choice") or {}
        label = answer.get("choice", "")
        raw_confidence = float(answer.get("answer_confidence", 0.0))
        confidence = self.calibration.adjust("choice", raw_confidence)

        action = next((a for a in request.candidates if self._label(a) == label), None)
        if action is None:
            return DecisionResponse(
                decision_id=request.decision_id,
                action_id="continue_current",
                confidence=0.0,
                selector="laya",
                fell_back=True,
                reasoning=f"laya returned an unrecognised option {label!r}",
            )

        return DecisionResponse(
            decision_id=request.decision_id,
            action_id=action.action_id,
            confidence=confidence,
            selector="laya",
            reasoning=f"laya chose {action.action_id}",
        )

    @staticmethod
    def state_text(request: DecisionRequest) -> str:
        """Render the session for the model. Structured metadata only — never the waveform
        (``README.md:39``)."""
        state = request.state
        request_text = state.theme or "continue the current set"
        history = state.committed_action_id or "nothing committed yet"
        return (
            f"Theme: {request_text}. "
            f"Playing {state.clock.bpm:.0f} BPM at bar {state.clock.bar}, "
            f"energy {state.energy:.2f}. "
            f"Last action: {history}. "
            f"{len(request.candidates)} feasible actions."
        )


def _safe_action_id(request: DecisionRequest) -> str:
    """The hold-position option from the menu, so a fallback always names something real."""
    safe = next((a for a in request.candidates if a.is_safe), None)
    return safe.action_id if safe else request.candidates[0].action_id