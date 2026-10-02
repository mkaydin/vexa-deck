"""Verify the fine-tuning pipeline end to end on a real GPU.

**What this is:** a machinery test. It proves the RLCD loop runs, the loss decreases, the memory
fits, and the artifacts can be written.

**What this is not:** a usable model. The labels here are generated from a stated rule, and
``IDEAS.md:57`` is explicit that training solely on synthetic rules "may reproduce the rules
without improving listening quality". The checkpoint this writes is stamped
``SYNTHETIC-SMOKE-TEST`` and must never be promoted.

Real labels come from the A/B transition-preview harness in ``docs/FINETUNE.md``.

Usage::

    uv run --no-sync python tools/verify_finetune_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vexa_laya.finetune import (  # noqa: E402
    LossWeights,
    TrainingItem,
    build_training_item,
    fine_tune,
    load_base,
    save_checkpoint,
    write_run_report,
)
from vexa_laya.train_config import Metrics, TrainConfig  # noqa: E402

#: The device is chosen by NAME because torch and nvidia-smi disagree on index order.
DEVICE_NAME = "RTX 5060 Ti"
CHECKPOINT = "convaiinnovations/laya-typed-decisions"


def resolve_device(name: str) -> str:
    """Resolve a GPU by name to a torch ordinal.

    ``nvidia-smi`` and torch enumerate the two cards in opposite order, so an index-based config
    silently puts fine-tuning on the wrong card.
    """
    import torch

    for index in range(torch.cuda.device_count()):
        if name.lower() in torch.cuda.get_device_name(index).lower():
            return f"cuda:{index}"
    raise SystemExit(f"no GPU matching {name!r}; refusing to guess an index")


def synthetic_rows(count: int) -> list[dict]:
    """Structurally realistic rows whose labels come from a *stated* rule.

    The rule: prefer the option whose energy is closest to the request's target, and prefer
    holding position when the request is already satisfied. Deliberately simple, so that if the
    fine-tune learns it we know it learned the rule and not something subtle.
    """
    rows = []
    for i in range(count):
        target_energy = 0.3 + 0.05 * (i % 8)
        options = [
            {"label": "keep the current track playing", "energy": round(target_energy, 2)},
            {"label": f"transition, 124 BPM, energy {target_energy:.2f}, instrumental",
             "energy": round(target_energy, 2)},
            {"label": f"transition, 121 BPM, energy {target_energy - 0.3:.2f}, vocals",
             "energy": round(target_energy - 0.3, 2)},
        ]
        # The stated rule, expressed as a gold distribution.
        distances = [abs(o["energy"] - target_energy) for o in options]
        raw = [1.0 / (0.1 + d) for d in distances]
        total = sum(raw)
        gold = {o["label"]: r / total for o, r in zip(options, raw, strict=True)}

        rows.append(
            {
                "state": (
                    f"Listener request: move toward energy {target_energy:.2f}. "
                    f"Currently playing: 122 BPM in A minor, 8 bars to the next phrase boundary. "
                    f"Current energy {target_energy:.2f}, instrumental. "
                    f"no recent plays recorded. Feasible next actions: 3."
                ),
                "questions": {
                    "act": {
                        "type": "choice",
                        "instructions": "Which action best follows the listener's request "
                                        "at the next phrase boundary?",
                        # A choice question's criteria are a dict of label -> description. The
                        # labels are self-describing, so the description slot is empty.
                        "criteria": {o["label"]: "" for o in options},
                    }
                },
                "gold": {"act": gold},
                "family_id": f"family_{i % 12}",
            }
        )
    return rows


def main() -> int:
    import torch
    device = resolve_device(DEVICE_NAME)
    print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    cfg = TrainConfig(epochs=2, micro_batch_size=4, gradient_accumulation_steps=2, seed=0)

    print("\nloading base checkpoint via laya.load ...")
    _agent, model, tokenizer, load_warnings = load_base(CHECKPOINT)
    print(f"  {CHECKPOINT}")
    for message in load_warnings:
        if "temperature" in message.lower():
            print(f"  WARNING: {message}")

    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    largest = max(float(p.detach().float().abs().max()) for p in model.parameters())
    print(f"  parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  weights finite: {finite}  max|w|: {largest:.3f}")
    if not finite or largest > 1e4:
        print("  ABORT: weights look randomly initialised", file=sys.stderr)
        return 1

    rows = synthetic_rows(48)
    items: list[TrainingItem] = []
    skipped = 0
    for row in rows:
        qid = next(iter(row["questions"]))
        item = build_training_item(
            tokenizer, row["state"], row["questions"][qid], row["gold"][qid], cfg
        )
        if item is None:
            skipped += 1
            continue
        items.append(item)
    print(f"\ntokenised {len(items)} items, skipped {skipped} (marker/option mismatch)")
    if not items:
        print("no usable training items", file=sys.stderr)
        return 1

    print("\ntraining (RLCD, published recipe) ---")
    history, usage = fine_tune(
        model=model, tokenizer=tokenizer, items=items, cfg=cfg,
        device=device, weights=LossWeights(),
    )

    first, last = history[0], history[-1]
    print(f"\nloss {first.loss:.4f} -> {last.loss:.4f}  (delta {last.loss - first.loss:+.4f})")
    print(f"peak VRAM: {usage['peak_vram_gib']:.2f} GiB of "
          f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GiB")

    out = ROOT / "runs" / "pipeline-verification"
    target = save_checkpoint(model, tokenizer, out / "checkpoint")
    (target / "SYNTHETIC-SMOKE-TEST").write_text(
        "Trained on rule-generated labels. Verifies the pipeline only.\n"
        "MUST NOT be promoted: IDEAS.md:57 warns that synthetic labels can reproduce the rules\n"
        "without improving listening quality. Real labels come from A/B human review.\n",
        encoding="utf-8",
    )
    print(f"\nwrote {target.relative_to(ROOT)} (stamped SYNTHETIC-SMOKE-TEST)")

    write_run_report(
        out / "report.json",
        config=cfg,
        history=history,
        metrics=Metrics(),
        extra={
            "purpose": "pipeline verification only",
            "base_checkpoint": CHECKPOINT,
            "device": device,
            "labels": "rule-generated (synthetic)",
            "items": len(items),
            "skipped": skipped,
            **usage,
        },
    )
    print(f"wrote {(out / 'report.json').relative_to(ROOT)}")

    print("\n" + "=" * 64)
    ok = last.loss < first.loss
    print(f"loss decreased across epochs: {ok}")
    print("NOTE: this verifies machinery, not quality. Real labels are required")
    print("      before any fine-tuned checkpoint may control audio.")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())