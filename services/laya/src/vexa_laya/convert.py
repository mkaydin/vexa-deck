"""Conversion to Laya's documented formats.

Two outputs, both from the same canonical record:

* :func:`to_eval_jsonl` — the evaluation harness format documented at ``laya/docs/evals.md``:
  ``state``, ``questions``, ``expected``, optional ``tags``/``language``. Verifiable right now with
  ``laya-evals validate`` and no checkpoint needed.
* :func:`to_training_items` — the training question shape, one row per decision.

**Labels are the currency.** Laya's ``criteria`` are label strings, and ``expected`` and gold
distributions are keyed by those same strings. Keying them by internal option ids instead produces
a file where no label ever matches and every option silently falls through to the unannotated
weight — a dataset that looks well-formed and teaches nothing.

The mapping onto Laya's three typed questions:

============  ==========================================================
``choice``    which option to take — the menu, with labels
``noul``      whether a named transition is acceptable at all
``score``     how strongly it fits, when the annotator gave a graded answer
============  ==========================================================

``LAYA_DATA.md:9`` calls ``noul`` "a yes/no confidence gate", which is exactly right: it is the
affirmative half of the deterministic second validator, not a replacement for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import Annotation, CandidateType

#: What the annotator's source implies for the question type.
QUESTION_FOR_SOURCE = {
    "human_pairwise_review": "choice",
    "rule_policy": "choice",
    "teacher_model": "choice",
}

LANGUAGE = "en"


def describe(candidate: CandidateType) -> str:
    """One short, factual description of an option.

    Built from measured features only. ``IDEAS.md:19`` requires these reasons to be derived from
    recorded data rather than invented after the fact.
    """
    if candidate.type in ("continue", "continue_current"):
        return "keep the current track playing"
    if candidate.type in ("fallback", "play_fallback_loop"):
        return "fallback loop bed"
    parts = [candidate.type]
    if candidate.bpm is not None:
        parts.append(f"{candidate.bpm:.0f} BPM")
    if candidate.energy is not None:
        parts.append(f"energy {candidate.energy:.2f}")
    parts.append("vocals" if candidate.vocals else "instrumental")
    return ", ".join(parts)


def criteria(annotation: Annotation) -> dict[str, str] | list[str]:
    """Option criteria in the shape Laya's ``render_options`` expects for this question type.

    ``choice`` takes a **dict** mapping an option label to its description; ``score`` takes a
    list. Passing a list for a choice question fails at render time with
    ``'list' object has no attribute 'items'``. Since VEXA's labels are already self-describing,
    the description slot is empty and the label renders unchanged.
    """
    first = annotation.state.candidates[0]
    if first.type == "score":  # pragma: no cover - not produced by this project yet
        return ["poor fit", "partial fit", "good fit", "excellent fit"]
    return {describe(candidate): "" for candidate in annotation.state.candidates}


def label_of(annotation: Annotation, option_id: str) -> str:
    """The criteria label for an option id.

    Raises:
        KeyError: if the option was not offered at this decision point.
    """
    for candidate in annotation.state.candidates:
        if candidate.id == option_id:
            return describe(candidate)
    raise KeyError(f"option {option_id!r} is not among this decision's candidates")


def state_text(annotation: Annotation) -> str:
    """Render the decision context as the text Laya actually reads.

    Deterministic and stable on purpose: the evaluation harness fingerprints question text, so
    rewording this function silently invalidates every saved baseline. ``laya/docs/evals.md`` is
    explicit that a reworded instruction is a different experiment.
    """
    ctx = annotation.state
    current = ctx.current
    request = ctx.request or "continue the set as it is going"
    history = (
        f"recently played: {', '.join(current.recent_families)}"
        if current.recent_families
        else "no recent plays recorded"
    )
    return (
        f"Listener request: {request}. "
        f"Currently playing: {current.bpm:.0f} BPM"
        + (f" in {current.key}" if current.key else "")
        + (f", in the {current.section} section" if current.section else "")
        + f", {current.bars_to_boundary} bars to the next phrase boundary. "
        f"Current energy {current.energy:.2f}"
        + (", with vocals" if current.vocals else ", instrumental")
        + f". {history}. "
        f"Feasible next actions: {len(ctx.candidates)}."
    )


def question_id(annotation: Annotation) -> str:
    """Stable per-decision id, so the harness can slice by question."""
    return f"act:{annotation.family_id}"


def _has_spread(annotation: Annotation) -> bool:
    return bool(annotation.also_acceptable or annotation.rejected_action_ids)


def _fit_level(annotation: Annotation) -> float:
    """A graded fit from the accept/reject structure.

    Coarse on purpose. The annotation is the only ground truth available, and inventing a finer
    scale from it would manufacture precision the label never contained.
    """
    total = len(annotation.state.candidates)
    if total == 0:
        return 0.5
    accepted = 1 + len(annotation.also_acceptable)
    return round(min(1.0, accepted / total), 3)


def to_eval_record(annotation: Annotation) -> dict[str, Any]:
    """One row in Laya's evaluation JSONL format."""
    qid = question_id(annotation)
    question_type = QUESTION_FOR_SOURCE.get(annotation.label_source.kind, "choice")

    row: dict[str, Any] = {
        "state": state_text(annotation),
        "questions": {
            qid: {
                "type": question_type,
                "instructions": annotation.question,
                "criteria": criteria(annotation),
            }
        },
        # Keyed by the label that appears in `criteria`, which is what Laya matches against.
        "expected": {qid: label_of(annotation, annotation.preferred_action_id)},
        "tags": [
            annotation.family_id,
            annotation.label_source.kind,
            f"energy_{annotation.state.current.energy:.1f}",
        ],
        "language": LANGUAGE,
    }

    # A graded answer becomes a second, typed question rather than being flattened into the
    # choice. Mixing the two would ask the model to rank and to rate in the same breath.
    if question_type == "choice" and _has_spread(annotation):
        rating_id = f"{qid}:fit"
        row["questions"][rating_id] = {
            "type": "score",
            "instructions": "How well does this action fit the listener's request?",
            "criteria": ["poor fit", "partial fit", "good fit", "excellent fit"],
        }
        row["expected"][rating_id] = _fit_level(annotation)
    return row


def to_eval_jsonl(annotations: list[Annotation], path: str | Path) -> Path:
    """Write the evaluation set. Validate it with ``laya-evals validate <path>``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            handle.write(
                json.dumps(to_eval_record(annotation), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return target


def _raw_weight(annotation: Annotation, candidate: CandidateType) -> float:
    """Unnormalised target mass for one candidate.

    The trainer consumes teacher distributions rather than hard labels. A single chosen option
    carries the most mass; options the listener also accepted keep a share, because forcing one
    label asserts something the annotation never claimed.
    """
    if candidate.id == annotation.preferred_action_id:
        return 0.7
    if candidate.id in annotation.also_acceptable:
        return 0.2
    if candidate.id in annotation.rejected_action_ids:
        return 0.0
    # Not annotated either way: share the remainder without asserting a preference.
    return 0.1


def gold_distribution(annotation: Annotation) -> dict[str, float]:
    """The target distribution over option labels, summing to 1.

    The raw weights are *relative*, not absolute: a menu where the listener rejected two of three
    options sums to 0.9, not 1.0. Laya trains against distributions, so they are normalised here
    rather than left summing to whatever the annotation happened to produce.
    """
    raw = {
        describe(candidate): _raw_weight(annotation, candidate)
        for candidate in annotation.state.candidates
    }
    total = sum(raw.values())
    if total <= 0:  # pragma: no cover - the preferred option always carries mass
        uniform = 1.0 / max(len(raw), 1)
        return {label: uniform for label in raw}
    return {label: round(mass / total, 6) for label, mass in raw.items()}


def to_training_items(annotations: list[Annotation]) -> list[dict[str, Any]]:
    """Rows in the shape the fine-tuning notebook's ``load_dataset`` calls expect.

    Each case carries ``state``, ``questions`` and ``gold``. Kept separate from the eval format
    because the two are allowed to drift — the notebook's schema is not a published contract the
    way the eval JSONL is.
    """
    items: list[dict[str, Any]] = []
    for annotation in annotations:
        qid = question_id(annotation)
        items.append(
            {
                "state": state_text(annotation),
                "questions": {
                    qid: {
                        "type": "choice",
                        "instructions": annotation.question,
                        "criteria": criteria(annotation),
                    }
                },
                "gold": {qid: {"distribution": gold_distribution(annotation)}},
                "family_id": annotation.family_id,
                "session_id": annotation.session_id,
                "label_source": annotation.label_source.kind,
            }
        )
    return items


def calibration_slice(
    annotations: list[Annotation], *, max_items: int = 400
) -> list[Annotation]:
    """Carve out the slice used to fit temperatures.

    Must be taken **before** training and must not overlap it. Laya's own documentation calls this
    "the step most likely to be dropped when copying the loop, and it is load-bearing the moment
    anyone gates on confidence" — fitting on data the run already trained on measures the fit, not
    the calibration, and returns a degenerate scale.
    """
    ordered = sorted(annotations, key=lambda a: (a.family_id, a.session_id))
    cap = min(max_items, max(1, len(ordered) // 10))
    return ordered[:cap]