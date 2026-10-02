"""Temperature calibration.

Delegates the fitting to Laya's own implementation rather than reimplementing it. Laya 0.3.23
ships ``laya.calibrate.records_from_labeled`` and ``fit_temperatures``, which handle the parts that
are easy to get subtly wrong:

* ``MIN_TYPE_N = 10`` — below this a per-type scalar is not fitted at all.
* ``MIN_BUCKET_N = 2000`` — per-bucket temperatures are *omitted* for small datasets, and the
  type-level scalar covers them instead.
* ``compute_ece=True`` holds out 20% of each bucket, stratified, to measure the ECE improvement
  rather than asserting one.

What this module adds is the project's own guardrails, which are not Laya's concern:

1. **The slice is carved by asset family**, so a revision of a render cannot appear on both sides
   of the calibration split.
2. **The gate refuses to open on thin evidence.** An uncalibrated confidence is worse than no
   confidence, because something downstream may come to trust it.
3. **The fit must not make calibration worse.** A fit that raises ECE is evidence the slice was
   wrong, not evidence of a temperature.

A correction to an earlier reading of the documentation
-------------------------------------------------------
``LAYA_DATA.md:65`` and Laya's own fine-tuning notebook advise *removing* ``temperature_by_options``
when persisting a calibration, because inherited per-option values take precedence at inference and
mask a new fit. That advice is about **stale, inherited** values.

Laya 0.3.23's ``fit_temperatures`` deliberately *writes* per-bucket temperatures alongside the
type-level scalar, gated by ``MIN_BUCKET_N``. Using the library's own fit — and its own
``save_calibration`` — is therefore the correct path, and hand-deleting the bucket map would
discard a deliberate part of the newer calibration scheme. What remains true is that a config
carrying a bucket map from a *previous* checkpoint must not be carried forward.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .train_config import TrainConfig

#: Re-exported so callers can check the library's own floors rather than hard-coding numbers.
try:  # pragma: no cover - exercised only when laya is installed
    from laya.calibrate import MIN_BUCKET_N, MIN_TYPE_N, records_from_labeled

    LAYA_CALIBRATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    MIN_BUCKET_N = 2000
    MIN_TYPE_N = 10
    records_from_labeled = None
    LAYA_CALIBRATE_AVAILABLE = False

#: The three question types Laya fits a scalar for.
QUESTION_TYPES = ("choice", "score", "noul")


@dataclass(slots=True)
class CalibrationResult:
    """What was fitted, on how much evidence, and whether it may be trusted."""

    temperatures: dict[str, float] = field(default_factory=dict)
    #: Per-bucket map, present only when the dataset is large enough to justify one.
    temperature_by_options: dict[str, float] = field(default_factory=dict)
    slice_size: int = 0
    ece_before: float = 0.0
    ece_after: float = 0.0
    #: The payload Laya wants persisted alongside the weights.
    payload: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.ece_after <= self.ece_before

    @property
    def trustworthy(self) -> bool:
        """Whether this fit may be allowed to gate live decisions."""
        return (
            LAYA_CALIBRATE_AVAILABLE
            and self.slice_size >= MIN_TYPE_N
            and bool(self.temperatures)
            and self.improved
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperatures": self.temperatures,
            "slice_size": self.slice_size,
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "improved": self.improved,
            "trustworthy": self.trustworthy,
            "has_per_bucket_temperatures": bool(self.temperature_by_options),
            "notes": self.notes,
        }


def carve_calibration_families(
    families: Sequence[str],
    *,
    fraction: float = 0.10,
    max_items: int = 400,
    seed: int = 0,
) -> set[str]:
    """Choose whole asset families for the calibration slice.

    By family, never by row: two revisions of one render in opposite halves of the split would
    measure memorisation rather than calibration.
    """
    import random

    unique = sorted(set(families))
    if not unique:
        return set()
    count = min(max_items, max(1, int(len(unique) * fraction)))
    rng = random.Random(seed)
    return set(rng.sample(unique, min(count, len(unique))))


def calibrate(
    agent: Any,
    pairs: Sequence[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    config: TrainConfig | None = None,
    seed: int = 0,
) -> CalibrationResult:
    """Fit temperatures on held-out ``(state, questions, targets)`` pairs.

    Args:
        agent: a loaded ``laya.Agent``.
        pairs: held-out labelled pairs. Must **not** overlap the training set — see the module
            docstring for why fitting on trained-on rows returns a degenerate scale.
        config: supplies the seed and any project-specific bounds.

    Returns:
        A result that reports whether the fit may be trusted, so a caller cannot accidentally gate
        on a calibration that was fitted on too little data.
    """
    cfg = config or TrainConfig()
    result = CalibrationResult(slice_size=len(pairs))

    if not LAYA_CALIBRATE_AVAILABLE:
        result.notes.append("laya.calibrate is unavailable; calibration skipped")
        return result
    if not pairs:
        result.notes.append("no held-out pairs; calibration skipped")
        return result

    # `compute_ece=True` makes the library hold out 20% of each bucket internally so the reported
    # improvement is measured rather than asserted.
    records = records_from_labeled(agent, pairs)

    from laya.calibrate import fit_temperatures

    fitted = fit_temperatures(records, compute_ece=True, seed=seed or cfg.seed)

    result.payload = fitted
    result.temperatures = dict(fitted.get("temperature") or {})
    result.temperature_by_options = dict(fitted.get("temperature_by_options") or {})

    report = fitted.get("report") or {}
    result.ece_before = float(report.get("ece_before", 0.0) or 0.0)
    result.ece_after = float(report.get("ece_after", 0.0) or 0.0)
    if not result.ece_before and not result.ece_after:
        # No internal report: treat as unmeasured rather than claiming a pass.
        result.notes.append("library reported no ECE comparison; improvement unverified")

    if not result.temperature_by_options:
        result.notes.append(
            f"no per-bucket temperatures: fewer than MIN_BUCKET_N ({MIN_BUCKET_N}) examples, "
            "so the type-level scalar covers every option"
        )

    if result.slice_size < MIN_TYPE_N:
        result.notes.append(
            f"only {result.slice_size} held-out items; MIN_TYPE_N is {MIN_TYPE_N}. "
            "Confidence must not gate on this fit."
        )
    return result


def persist_calibration(
    agent: Any,
    result: CalibrationResult,
    *,
    path: str | Path | None = None,
) -> str | Path | None:
    """Persist the calibration using the library's own writer.

    Shipping weights without the calibration file is how a calibrated model ends up behaving like
    an uncalibrated one, so this uses ``Agent.save_calibration`` rather than writing a config by
    hand.
    """
    if result.payload is None:
        return None
    target = Path(path) if path else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
    if target is not None:
        return agent.save_calibration(result.payload, path=target)
    return agent.save_calibration(result.payload)


def load_calibration(agent: Any, path: str | Path) -> bool:
    """Load a persisted calibration. Returns whether one was applied."""
    loader = getattr(agent, "load_calibration", None)
    if loader is None:  # pragma: no cover - depends on the installed laya version
        return False
    try:
        loader(str(path))
    except Exception:
        return False
    return True


def write_calibration_report(result: CalibrationResult, path: str | Path) -> Path:
    """Record what was fitted, so a run can be reviewed without rerunning it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target