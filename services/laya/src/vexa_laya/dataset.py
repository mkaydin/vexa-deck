"""The canonical annotation record.

``LAYA_DATA.md:49`` is explicit that this is *this project's* record format, not a claim about
what Laya's trainer consumes. Three formats sit between the annotation and the model:

    annotation record  ->  Laya training question  ->  Laya evaluation JSONL
    (here)                    (typed questions)          (state/questions/expected)

The raw record is kept so trainer versions can change without losing labels. That is the whole
reason this module exists separately from the converter.

Two fields carry most of the design weight:

* ``family_id`` — splits are made on this, never on random rows (``LAYA_DATA.md:69``). Variants of
  one render must stay together or the model memorises a family instead of learning a decision.
* ``preferred_action_id`` is one of several acceptable answers, not a single right one. Subjective
  choices are recorded as such rather than averaged into a fake consensus.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Candidate types that mean "hold position". Ids are arbitrary; types are the contract.
SAFE_CANDIDATE_TYPES = frozenset({"continue", "continue_current", "fallback", "play_fallback_loop"})


class CandidateType(BaseModel):
    """One option in the menu, as the annotator saw it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    asset: str | None = None
    bpm: float | None = None
    energy: float | None = None
    vocals: bool | None = None
    #: Measured facts the annotator could see. Kept separate from the option's identity.
    features: dict[str, float] = Field(default_factory=dict)


class StateSnapshot(BaseModel):
    """What was playing when the decision was made."""

    model_config = ConfigDict(extra="forbid")

    bpm: float
    key: str | None = None
    section: str | None = None
    bars_to_boundary: int = Field(ge=0)
    energy: float = Field(ge=0.0, le=1.0)
    vocals: bool = False
    last_action: str | None = None
    recent_families: list[str] = Field(default_factory=list)


class DecisionContext(BaseModel):
    """The full decision point."""

    model_config = ConfigDict(extra="forbid")

    request: str = ""
    current: StateSnapshot
    candidates: list[CandidateType] = Field(min_length=2)


class LabelSource(BaseModel):
    """Where a label came from and how much to trust it.

    Provenance per row is not bookkeeping — ``LAYA_DATA.md:85`` requires it, and a model trained on
    synthetic labels can only ever reproduce the rules that generated them.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    #: e.g. ``human_pairwise_review``, ``rule_policy``, ``teacher_model``.
    name: str = ""
    annotators: list[str] = Field(default_factory=list)
    #: For subjective choices, the spread across annotators rather than a single forced label.
    agreement: float | None = Field(default=None, ge=0.0, le=1.0)


class Annotation(BaseModel):
    """One labelled decision point — the canonical record."""

    model_config = ConfigDict(extra="forbid")

    state: DecisionContext
    question: str = "Which feasible action best follows the request at the next phrase boundary?"
    #: The chosen option. Several may be acceptable; this is the primary one.
    preferred_action_id: str = Field(min_length=1)
    #: Other options a listener also accepted. Presence prevents over-transitioning.
    also_acceptable: list[str] = Field(default_factory=list)
    #: Options a listener explicitly rejected. Stronger signal than silence.
    rejected_action_ids: list[str] = Field(default_factory=list)
    #: The asset family this decision was made on. Splits key on this.
    family_id: str = Field(min_length=1)
    #: The session this decision came from. Splits key on this too.
    session_id: str = Field(min_length=1)
    label_source: LabelSource = Field(default_factory=LabelSource)
    #: Annotator confidence in the label itself, not in the model.
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    #: The family of the *asset that was actually played*, when different from the decision family.
    played_asset_family: str | None = None
    #: Free-form notes kept verbatim. Losing these loses the reasoning.
    notes: str = ""

    @model_validator(mode="after")
    def _preferred_must_be_offered(self) -> Annotation:
        offered = {c.id for c in self.state.candidates}
        if self.preferred_action_id not in offered:
            raise ValueError(
                f"preferred_action_id {self.preferred_action_id!r} is not among the offered "
                f"options {sorted(offered)}"
            )
        for group, name in (
            (self.also_acceptable, "also_acceptable"),
            (self.rejected_action_ids, "rejected_action_ids"),
        ):
            unknown = set(group) - offered
            if unknown:
                raise ValueError(f"{name} names options that were not offered: {sorted(unknown)}")
        if self.preferred_action_id in self.rejected_action_ids:
            raise ValueError("an action cannot be both preferred and rejected")
        return self

    @property
    def is_safe_choice(self) -> bool:
        """Whether the label teaches holding position rather than always transitioning.

        Keyed on the chosen candidate's *type*, not its id. Ids are arbitrary shortlist slots;
        only the type says whether this was a hold-position decision.

        ``LAYA_DATA.md:57``: without ``continue_current`` and a safe fallback in the dataset, the
        model learns to over-transition.
        """
        chosen = next(
            (c for c in self.state.candidates if c.id == self.preferred_action_id), None
        )
        return chosen is not None and chosen.type in SAFE_CANDIDATE_TYPES


def write_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> Path:
    """Write records as JSONL. Blank lines and ``#`` comments are allowed on read."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
    return target


def read_jsonl(path: str | Path) -> Iterator[Annotation]:
    """Read a JSONL file of :class:`Annotation` records, skipping blanks and comments."""
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                yield Annotation.model_validate_json(stripped)
            except Exception as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc




@dataclass(frozen=True, slots=True)
class Split:
    """A family-level split. Every variant of one render lands in exactly one partition."""

    train: list[Annotation] = field(default_factory=list)
    validation: list[Annotation] = field(default_factory=list)
    test: list[Annotation] = field(default_factory=list)

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def all_families(self) -> set[str]:
        return {a.family_id for a in (*self.train, *self.validation, *self.test)}


def split_by_family(
    annotations: Sequence[Annotation],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 0,
) -> Split:
    """Partition by asset family and session, never by row.

    ``LAYA_DATA.md:69``: *"Variants of one YuE2 render must stay in the same split."* A random row
    split would put two revisions of the same track on both sides of the boundary, and the model
    would score well by recognising the render rather than by choosing well.
    """
    import random

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    groups: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        # Family is the split key, but session is folded in so a whole session stays whole too.
        key = f"{annotation.family_id}"
        groups.setdefault(key, []).append(annotation)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = len(keys)
    train_end = max(1, int(total * train_ratio)) if total else 0
    validation_end = train_end + max(1, int(total * validation_ratio)) if total else train_end

    split = Split()
    for index, key in enumerate(keys):
        bucket = split.train if index < train_end else (
            split.validation if index < validation_end else split.test
        )
        bucket.extend(groups[key])
    return split