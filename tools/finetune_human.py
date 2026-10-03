"""Stage 3: fine-tune on human labels, with a guard that refuses to train on anything else.

``FINETUNE-DECISION.md`` is explicit that the only labels that train a shipped model are human
A/B judgements. Everything else — rule-derived, teacher-distilled, synthetic — exists to prove the
machinery works and must never reach a promotion.

That rule is enforced here, in one place, as a refusal rather than a convention. The failure mode
this prevents is specific and unpleasant: a checkpoint trained on rule-derived labels scores well
on our own benchmark, looks like a successful fine-tune, and adds nothing at runtime. Nothing in a
training log would reveal it.

Order matters and is enforced rather than documented:

1. Load labels and **refuse** if any row is not human-reviewed.
2. Split by family, then **verify** no family appears in two splits.
3. Carve the calibration slice **before** training, so it cannot overlap (``LAYA_DATA.md:65``).
4. Train on what is left.
5. Report the held-out numbers. No promotion decision is made here — that is ``tools/judge_shadow``.

Usage::

    uv run --no-sync python tools/finetune_human.py --labels data/sessions/labels.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services/laya/src"))
sys.path.insert(0, str(ROOT / "packages/contracts/src"))

from vexa_laya.convert import calibration_slice, to_training_items  # noqa: E402
from vexa_laya.dataset import Annotation, split_by_family  # noqa: E402

#: Label sources allowed to train a shipped model. Everything else is a rehearsal.
HUMAN_SOURCES: frozenset[str] = frozenset({"human_pairwise_review"})

#: Below this, a split cannot mean anything and a reviewer should collect more first.
MIN_ROWS = 50

#: Laya's own guidance on how much to hold back for temperature fitting.
MAX_CALIBRATION_ITEMS = 400


def load(labels: Path) -> list[Annotation]:
    annotations = [
        Annotation.model_validate_json(line)
        for line in labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not annotations:
        raise SystemExit(f"no annotations in {labels}")
    return annotations


def refuse_unless_human(annotations: list[Annotation]) -> None:
    """Hard stop on any label that did not come from a person listening.

    This is the check most worth having in the codebase. ``IDEAS.md:57`` warns that training on
    synthetic rules reproduces the rules; the resulting model scores well on our own benchmark and
    changes nothing at runtime, which is worse than no model because it looks like success.
    """
    kinds = Counter(a.label_source.kind for a in annotations)
    offenders = sorted(set(kinds) - HUMAN_SOURCES)
    if offenders:
        print("REFUSING TO TRAIN.", file=sys.stderr)
        print(f"  found {dict(kinds)}", file=sys.stderr)
        print(f"  only {sorted(HUMAN_SOURCES)} may train a shipped model", file=sys.stderr)
        print(
            "  rule-derived labels prove the pipeline; they cannot teach a model what sounds "
            "good.\n  Collect A/B judgements with tools/collect_labels.py.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def verify_no_family_leaks(split) -> None:
    """No family may appear in more than one split.

    A leak here would inflate every number in the run: the model scores well by recognising the
    render it saw in training rather than by choosing well.
    """
    train = {a.family_id for a in split.train}
    validation = {a.family_id for a in split.validation}
    test = {a.family_id for a in split.test}
    overlaps = {
        "train/validation": train & validation,
        "train/test": train & test,
        "validation/test": validation & test,
    }
    bad = {name: sorted(fams) for name, fams in overlaps.items() if fams}
    if bad:
        print(f"ABORT: family leakage across splits: {bad}", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="JSONL of human labels")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "finetune-human")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--min-annotations", type=int, default=MIN_ROWS)
    args = parser.parse_args()

    annotations = load(args.labels)
    print(f"loaded {len(annotations)} annotations")
    print(f"  sources: {dict(Counter(a.label_source.kind for a in annotations))}")
    print(f"  annotators: {sorted({n for a in annotations for n in a.label_source.annotators})}")

    refuse_unless_human(annotations)

    if len(annotations) < args.min_annotations:
        print(
            f"\nSTOP: {len(annotations)} annotations is below the {args.min_annotations} needed "
            "for a\n      split to mean anything. Keep listening and re-run.",
            file=sys.stderr,
        )
        return 1

    split = split_by_family(annotations)
    verify_no_family_leaks(split)
    print(
        f"\nsplit by family: train={len(split.train)} validation={len(split.validation)} "
        f"test={len(split.test)}  (no family spans two splits)"
    )
    # Carved from the **training split only**. ``calibration_slice`` sorts the rows it is given by
    # family and takes the first N, so calling it on the whole dataset hands back rows that
    # straddle train and validation -- it is not split-aware. Carving from ``split.train`` makes
    # the disjointness structural instead of something to check for afterwards.
    calibration = calibration_slice(split.train, max_items=MAX_CALIBRATION_ITEMS)
    calibration_families = {a.family_id for a in calibration}
    train_items = [a for a in split.train if a.family_id not in calibration_families]

    leaked = calibration_families & (
        {a.family_id for a in split.validation} | {a.family_id for a in split.test}
    )
    if leaked:
        print(f"ABORT: calibration slice overlaps held-out data: {sorted(leaked)}", file=sys.stderr)
        return 2

    items = to_training_items(train_items)
    print(
        f"calibration slice: {len(calibration)} rows held out before training "
        f"({len(calibration_families)} families)"
    )
    print(f"training rows: {len(items)}   epochs: {args.epochs}")

    if not items:
        print("no training rows after holding out the calibration slice", file=sys.stderr)
        return 1

    print(
        "\nThe training run itself is tools/verify_finetune_pipeline.py --from-annotations, which\n"
        "already runs the published RLCD loop. This command exists to gate it: it refuses\n"
        "non-human labels and verifies the splits before any GPU time is spent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())