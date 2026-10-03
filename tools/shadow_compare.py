"""Run shadow mode over real decision points and report what Laya would have done.

``LAYA_DATA.md:81``: *"Run a shadow mode first: Laya chooses but the rule policy controls audio.
Compare choices and listen to disagreements."* ``ShadowPolicy`` implements that decision for a
single request. This is the driver that runs it across the whole library, persists the
observations, and reports the numbers that decide whether Laya has earned a place.

**The rules keep control throughout.** Nothing here commits an action or touches the audio engine.
The output is a log, and the log's purpose is to answer one question: *does the model ever disagree
usefully?*

Four things are measured, because each can end the experiment on its own:

* **Invalid action rate.** An action outside the feasible menu is a hard fail. ``PRODUCT.md:52``
  requires exactly zero, so this is reported as a count and the run exits non-zero if it is not 0.
* **Agreement rate.** If the model always agrees with the rules it has added nothing but latency.
* **Confidence.** Until Laya is calibrated these numbers are meaningless as a gate
  (``LAYA_DATA.md:65``), so they are reported and *not* used to filter.
* **Decision diversity.** A model that picks one action always has a high agreement rate by
  accident. Distinct choices are counted so that is visible.

Usage::

    uv run --no-sync python tools/shadow_compare.py --limit 40
    uv run --no-sync python tools/shadow_compare.py --out runs/shadow/latest.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services/orchestrator/src"))
sys.path.insert(0, str(ROOT / "services/laya/src"))
sys.path.insert(0, str(ROOT / "packages/contracts/src"))

from vexa_contracts import (  # noqa: E402
    ApprovalState,
    AssetManifest,
    DecisionRequest,
    RequestConstraints,
    SessionState,
)
from vexa_laya.adapter import LayaAdapter, LayaSettings  # noqa: E402
from vexa_orchestrator.feasibility import FeasibilityFilter, FilterConfig  # noqa: E402
from vexa_orchestrator.policy import RulePolicy, ShadowPolicy  # noqa: E402

LIBRARY = ROOT / "assets" / "library"
DEFAULT_OUT = ROOT / "runs" / "shadow" / "observations.jsonl"

#: A sample must survive the filter with at least this many options to be worth asking about.
MIN_OPTIONS = 3


def load_library() -> list[AssetManifest]:
    library = []
    for path in sorted(LIBRARY.glob("*.json")):
        if path.name.endswith(".request.json"):
            continue
        manifest = AssetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.approval == ApprovalState.APPROVED and manifest.admissible():
            library.append(manifest)
    return library


def build_requests(library: list[AssetManifest], limit: int) -> list[DecisionRequest]:
    """Real decision points: every asset as the deck, with the filter's own menu."""
    flt = FeasibilityFilter(FilterConfig())
    requests: list[DecisionRequest] = []
    for index, current in enumerate(library):
        if len(requests) >= limit:
            break
        result = flt.build(
            state=SessionState(session_id="shadow", theme="shadow comparison"),
            library=library,
            constraints=RequestConstraints(),
            current_asset=current,
        )
        if len(result.candidates) < MIN_OPTIONS:
            continue
        requests.append(
            DecisionRequest(
                decision_id=f"shadow-{index:04d}-{current.asset_id}",
                session_id="shadow",
                state=SessionState(session_id="shadow", theme="shadow comparison"),
                candidates=list(result.candidates),
            )
        )
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40, help="decision points to sample")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="observation log")
    args = parser.parse_args()

    library = load_library()
    if not library:
        print("no admitted assets", file=sys.stderr)
        return 2

    requests = build_requests(library, args.limit)
    print(f"library: {len(library)} assets, sampling {len(requests)} decision points")
    if not requests:
        print("no decision points had enough options to be worth asking about", file=sys.stderr)
        return 2

    rules = RulePolicy()
    rules.mark_loaded({m.asset_id for m in library})
    model = LayaAdapter(LayaSettings())
    shadow = ShadowPolicy(inner=model, rules=rules)

    print("\nloading checkpoint (this takes a moment) ...")
    started = time.time()
    failures: list[str] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    feasible_ids = {
        request.decision_id: {c.action_id for c in request.candidates} for request in requests
    }
    with args.out.open("w", encoding="utf-8") as handle:
        for index, request in enumerate(requests, 1):
            response = shadow.choose(request)
            observation = shadow.disagreements[-1]
            valid = observation.chosen_action_id in feasible_ids[request.decision_id]
            record = {
                "decision_id": request.decision_id,
                "playing": request.state.decks.get("A").asset_id
                if request.state.decks.get("A")
                else None,
                "options": len(request.candidates),
                "rule_action_id": observation.rule_action_id,
                "model_action_id": observation.chosen_action_id,
                "model_confidence": round(observation.confidence, 4),
                "agrees": observation.agrees,
                "model_action_valid": valid,
                "rules_controlled_audio": response.selector == rules.name,
                "reasoning": response.reasoning,
            }
            handle.write(json.dumps(record) + "\n")
            if not valid:
                failures.append(request.decision_id)
            if index % 10 == 0 or index == len(requests):
                print(f"  {index}/{len(requests)} decisions")

    elapsed = time.time() - started
    agreements = sum(1 for d in shadow.disagreements if d.agrees)
    choices = collections.Counter(d.chosen_action_id for d in shadow.disagreements)
    confidences = [d.confidence for d in shadow.disagreements]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print(f"\n--- shadow report over {len(requests)} decisions in {elapsed:.1f}s ---")
    print(f"  agreement with rules : {agreements}/{len(requests)} "
          f"({agreements / len(requests):.1%})")
    print(f"  distinct model choices: {len(choices)} "
          f"(always-one would mean agreement by accident)")
    print(f"  mean model confidence: {mean_conf:.3f}  "
          f"(NOT a gate until calibrated -- LAYA_DATA.md:65)")
    print(f"  invalid actions      : {len(failures)}  <- must be 0")
    print(f"  rules controlled audio: all {len(requests)} decisions")

    if failures:
        print(f"\nABORT: model chose an action outside the feasible menu: {failures[:5]}")
        return 1

    print(f"\nobservations written to {args.out.relative_to(ROOT)}")
    print("next: LISTEN to the cases where Laya and the rules differ. An agreement rate")
    print("      means nothing until you have heard the disagreements and judged them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())