"""Judge shadow-mode disagreements against human labels.

This is the experiment the whole fine-tune plan exists to answer. The base checkpoint picked
``continue_current`` in every decision it was given — agreement with the rules of 0%. That number
alone is uninterpretable in both directions:

* **High agreement** could mean the model is good, or that it always picks option 0 and the rules
  happen to agree. Those are very different things.
* **Low agreement** could mean the model has learned something the rules missed, or that it is
  broken. Only listening separates them.

So this joins the two logs -- ``runs/shadow/observations.jsonl`` and the human labels -- and
reports, **on the decisions where Laya disagreed with the rules**, who the human actually picked:

| Outcome | What it means |
|---|---|
| human sided with Laya | the model found a transition the rules missed — the case for keeping it |
| human sided with the rules | the disagreement is noise; the rules are the right answer |
| human rejected both | the *menu* was wrong, not the chooser — a filter problem |

The third row matters as much as the other two. It says the feasibility filter offered nothing
good, and no amount of selector work fixes that.

Rows are matched on **deck plus option set**, not on an index. The two logs come from different
tools over different runs, so the only stable identity a decision point has is "this asset was
playing, and these were the options".

Usage::

    uv run --no-sync python tools/judge_shadow.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services/laya/src"))
sys.path.insert(0, str(ROOT / "packages/contracts/src"))

from vexa_laya.dataset import Annotation  # noqa: E402

SHADOW_LOG = ROOT / "runs" / "shadow" / "observations.jsonl"
LABELS = ROOT / "data" / "sessions" / "labels.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def decision_key(playing: str, candidate_ids: list[str] | set[str]) -> tuple[str, frozenset[str]]:
    """The identity of a decision point: what was playing, and what was on offer."""
    return (playing, frozenset(candidate_ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow", type=Path, default=SHADOW_LOG)
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args()

    shadow = read_jsonl(args.shadow)
    raw_labels = read_jsonl(args.labels)
    human = [
        Annotation.model_validate(row)
        for row in raw_labels
        if row.get("label_source", {}).get("kind") == "human_pairwise_review"
    ]

    print(f"shadow observations: {len(shadow)}")
    print(f"human annotations  : {len(human)}")

    if not human:
        print(
            "\nNo human labels yet, so there is nothing to judge the model against.\n"
            "Collect some:\n"
            "  uv run --no-sync python tools/collect_labels.py"
            " --annotator <name> --count 20 --audible",
            file=sys.stderr,
        )
        return 1
    if not shadow:
        print(
            f"\nNo shadow log at {args.shadow}. Run tools/shadow_compare.py first.",
            file=sys.stderr,
        )
        return 1

    # A label knows its family (the asset that was playing) and its own option set.
    by_decision: dict[tuple[str, frozenset[str]], Annotation] = {}
    for annotation in human:
        key = decision_key(
            annotation.family_id, {c.id for c in annotation.state.candidates}
        )
        by_decision.setdefault(key, annotation)

    verdicts: collections.Counter[str] = collections.Counter()
    matched = 0

    for observation in shadow:
        key = decision_key(observation["playing"], observation.get("candidate_ids", []))
        annotation = by_decision.get(key)
        if annotation is None:
            continue
        matched += 1
        picked = annotation.preferred_action_id

        if observation["agrees"]:
            verdicts["agreed — human backed the shared choice"] += (
                1 if picked == observation["rule_action_id"] else 0
            )
            verdicts["agreed — but human chose something else"] += (
                0 if picked == observation["rule_action_id"] else 1
            )
            continue

        if picked == observation["model_action_id"]:
            verdicts["disagreed — human sided with Laya"] += 1
        elif picked == observation["rule_action_id"]:
            verdicts["disagreed — human sided with the rules"] += 1
        else:
            verdicts["disagreed — human rejected both (filter problem)"] += 1

    print(f"\nmatched {matched} of {len(shadow)} shadow decisions to human labels")
    if not matched:
        print(
            "\nNothing could be joined. The shadow run and the labels must cover the same\n"
            "decision points: same deck, same option set. Re-run tools/shadow_compare.py over\n"
            "the library you labelled against, with the same --limit.",
            file=sys.stderr,
        )
        return 1

    agreed = {k: v for k, v in verdicts.items() if k.startswith("agreed")}
    disputed = {k: v for k, v in verdicts.items() if k.startswith("disagreed")}
    total_disputed = sum(disputed.values())

    print("\n--- where Laya and the rules agreed ---")
    for label, count in sorted(agreed.items()):
        print(f"  {label:44s} {count:3d}")

    print(f"\n--- on the {total_disputed} decisions where they disagreed ---")
    for label, count in sorted(disputed.items(), key=lambda kv: -kv[1]):
        share = f"{count / total_disputed:.0%}" if total_disputed else "n/a"
        print(f"  {label:44s} {count:3d}  ({share})")

    print(
        "\nReading this:\n"
        "  Laya wins  -> the model found transitions the rules missed. Keep it.\n"
        "  rules win  -> the disagreement is noise. Ship the rules, delete the model.\n"
        "  both lose  -> the feasibility filter offered nothing good. No amount of selector\n"
        "                work fixes that; widen or re-rank the menu first.\n"
        "\nConfidence is reported but not a gate: the shipped checkpoint's calibration is\n"
        "documented upstream as unfit (LAYA_DATA.md:65), so it must be refitted on held-out\n"
        "data before anything keys off it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())