"""Generate rule-derived labels over the real depot.

**These labels must never train a shipped model.** ``IDEAS.md:57``: *"Training Laya solely from
synthetic rules may reproduce the rules without improving listening quality."* Every row is stamped
``rule_derived``, and :func:`promotable` is what a promotion step has to call -- so the exclusion is
a property of the data, not a convention somebody has to remember.

Why generate them at all, if they cannot ship? Because this is the **pipeline stress test** for
Stage 1 of ``FINETUNE-DECISION.md``. ``tools/verify_finetune_pipeline.py`` proves the loop runs on
twelve families of hand-written rows. That does not prove the loop survives what it will actually
see: real asset ids, real menu sizes, real bpm and key strings, family structure derived from the
assets themselves, and a few thousand rows rather than a few dozen.

The rule is stated, simple, and deliberately the sort of thing a DJ would already do: **move
toward the arc target, and hold position when the deck is already close enough.** If a model
trained on this learns the rule, that is the expected and only outcome.

Usage::

    uv run --no-sync python tools/rule_derived_labels.py --out data/labels/rule_derived.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services/laya/src"))
sys.path.insert(0, str(ROOT / "services/orchestrator/src"))
sys.path.insert(0, str(ROOT / "packages/contracts/src"))

from vexa_contracts import (  # noqa: E402
    ApprovalState,
    AssetManifest,
    RequestConstraints,
    SessionState,
)
from vexa_laya.dataset import Annotation, DecisionContext, LabelSource, StateSnapshot  # noqa: E402
from vexa_orchestrator.feasibility import (  # noqa: E402
    FeasibilityFilter,
    FilterConfig,
)

LIBRARY = ROOT / "assets" / "library"

#: Energy targets the synthetic arc sweeps through. Deliberately coarse: a rule-derived label has
#: no business pretending to more resolution than the rule it came from.
ARC_TARGETS: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8)

#: Within this distance the deck is already on target, so the correct action is to hold.
HOLD_TOLERANCE = 0.12


def load_library() -> dict[str, AssetManifest]:
    """Every committed manifest. The manifest is the source of truth, not a fresh analysis."""
    library: dict[str, AssetManifest] = {}
    for path in sorted(LIBRARY.glob("*.json")):
        if path.name.endswith(".request.json"):
            continue
        manifest = AssetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.approval == ApprovalState.APPROVED and manifest.admissible():
            library[manifest.asset_id] = manifest
    return library


def _energy(manifest: AssetManifest) -> float:
    value = manifest.energy()
    return 0.5 if value is None else value


def build_rows(library: dict[str, AssetManifest], *, per_asset: int = 4) -> list[Annotation]:
    """One decision per (playing asset, arc target), using the **real** filter menu.

    The menu comes from ``FeasibilityFilter`` rather than being written by hand, so option counts,
    asset ids and rejection reasons are the shapes the model will meet at runtime. The menu is
    capped at ``FilterConfig.max_candidates``, and a deck that already sits on target keeps its
    options -- ``LAYA_DATA.md:57`` warns that a dataset without ``continue_current`` teaches the
    model to over-transition.
    """
    flt = FeasibilityFilter(FilterConfig())
    rows: list[Annotation] = []
    assets = sorted(library.values(), key=lambda m: m.asset_id)

    for current in assets:
        current_bpm = current.beat_grid.bpm if current.beat_grid else 0.0
        if current_bpm <= 0:
            continue
        for target in (ARC_TARGETS[:per_asset] or ARC_TARGETS):
            session = SessionState(session_id=f"rule-{current.asset_id}", theme="rule-derived arc")
            result = flt.build(
                state=session,
                library=assets,
                constraints=RequestConstraints(),
                current_asset=current,
            )
            if len(result.candidates) < 2:
                continue

            current_energy = _energy(current)
            hold = abs(current_energy - target) <= HOLD_TOLERANCE

            candidates = []
            for action in result.candidates:
                target_asset = library.get(action.asset_id) if action.asset_id else None
                candidates.append(
                    {
                        "id": action.action_id,
                        "type": action.action_type.value,
                        "asset": action.asset_id,
                        # ``continue_current`` carries no asset, so it inherits the deck's own
                        # tempo and energy. That is what it *is*: staying exactly where we are.
                        "bpm": (
                            round(target_asset.beat_grid.bpm, 2)
                            if target_asset is not None and target_asset.beat_grid
                            else round(current_bpm, 2)
                        ),
                        "energy": (
                            round(_energy(target_asset), 2)
                            if target_asset is not None
                            else round(current_energy, 2)
                        ),
                        "vocals": False,
                    }
                )

            # The stated rule, as a soft preference rather than a forced label.
            if hold:
                weights = [2.0 if c["id"] == CONTINUE else 0.6 for c in candidates]
            else:
                weights = [2.0 if abs(c["energy"] - target) < 0.12 else 0.6 for c in candidates]
            total = sum(weights)
            probabilities = [w / total for w in weights]
            preferred = max(range(len(candidates)), key=lambda i: probabilities[i])

            rows.append(
                Annotation(
                    state=DecisionContext(
                        request=(
                            f"Move toward energy {target:.2f}. "
                            f"Currently playing {current_bpm:.0f} BPM "
                            f"at energy {current_energy:.2f}. "
                            + ("The deck is already close; hold unless something is clearly better."
                               if hold else
                               "Move toward the target using the closest available option.")
                        ),
                        current=StateSnapshot(
                            bpm=round(current_bpm, 2),
                            key=current.key.value if current.key else None,
                            bars_to_boundary=8,
                            energy=round(current_energy, 3),
                            vocals=False,
                            last_action=None,
                            recent_families=[],
                        ),
                        candidates=candidates,
                    ),
                    preferred_action_id=candidates[preferred]["id"],
                    also_acceptable=[
                        c["id"]
                        for i, c in enumerate(candidates)
                        if i != preferred and probabilities[i] >= 0.4
                    ],
                    # family_id is the *playing* asset's family: variants of one decision must
                    # never straddle a split (``LAYA_DATA.md:69``).
                    family_id=current.family_id or f"family_{current.asset_id}",
                    session_id=f"rule-{current.asset_id}",
                    label_source=LabelSource(
                        kind="rule_derived",
                        name="vexa-deck tempo/energy rule policy",
                        annotators=[],
                    ),
                    quality=1.0,
                )
            )
    return rows


#: The hold action id the filter emits. Kept as a constant because it is a contract-level string,
#: not a display label.
CONTINUE = "continue_current"


def promotable(rows: list[Annotation]) -> list[Annotation]:
    """Rows eligible to train a shipped model. Rule-derived rows are excluded here and nowhere else.

    A promotion step must call this rather than filtering inline, so there is exactly one place
    that knows which sources are allowed to ship.
    """
    return [row for row in rows if row.label_source.kind != "rule_derived"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data" / "labels" / "rule_derived.jsonl",
    )
    parser.add_argument("--per-asset", type=int, default=4, help="decisions per playing asset")
    args = parser.parse_args()

    library = load_library()
    if not library:
        print("no admitted assets found", file=sys.stderr)
        return 2

    rows = build_rows(library, per_asset=args.per_asset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json")) + "\n")

    families = len({row.family_id for row in rows})
    print(f"assets={len(library)} rows={len(rows)} families={families}")
    print(f"promotable={len(promotable(rows))}  <- must be 0; rule-derived labels never ship")
    print(f"written to {args.out}")
    return 0 if rows and not promotable(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())