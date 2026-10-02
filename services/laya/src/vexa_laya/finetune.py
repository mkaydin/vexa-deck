"""RLCD fine-tuning for VEXA//DECK's decision head.

This is Laya's fine-tuning loop, adapted from the published notebook
(``notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb``) to a **single** GPU, because this
machine allocates the fine-tune card exclusively (``PLAN.md`` §3.6: YuE2's torch backend and a
fine-tune cannot share it).

The loss is RLCD, and the structure is not decorative:

1. Sample ``G`` noisy logit distributions by adding zero-mean Gaussian noise to the logits.
2. Score each with proper scoring rules (``proper_reward``: spherical 0.75, ranked 1.0) against
   the gold distribution, and normalise the resulting advantage.
3. A policy-gradient term pushes up the noisy samples that scored well.
4. A full-weight soft cross-entropy term against the same gold distribution.

Both halves read the **target distribution**, not a hard label. That is why
``dataset.gold_distribution`` normalises: a target that does not sum to one is not a target.

Exploration noise anneals 0.4 -> 0.1 across epochs.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .train_config import Metrics, TrainConfig


@dataclass(slots=True)
class TrainingItem:
    """One tokenised decision, ready to collate."""

    ids: list[int]
    markers: list[int]
    qtype: int
    target: list[float]
    label: int


def build_training_item(tokenizer: Any, state: str, question: dict[str, Any],
                        gold: dict[str, float], cfg: TrainConfig) -> TrainingItem | None:
    """Tokenise one labelled question into a training item.

    Returns ``None`` when the marker count disagrees with the option count. That check is in the
    published recipe and is not optional: a mismatch means the head is sliced at the wrong width
    and the target vector is aligned against the wrong options, which trains silently and wrongly.
    """
    from laya.common import QTYPES, build_sequence, render_options

    qtype = question["type"]
    criteria = question.get("criteria", [])

    # `choice` criteria are a dict of label -> description; `score` criteria are a list. Either
    # way the gold distribution is keyed by the *rendered* option string, so the target vector
    # must be built from the rendered labels rather than the raw criteria.
    if qtype == "choice":
        labels = list(render_options({"t": qtype, "crit": criteria}))
        target = [gold.get(label, 0.0) for label in labels]
    elif qtype == "noul":
        target = [gold.get("false", 0.5), gold.get("true", 0.5)]
    elif qtype == "score":
        target = [gold.get(str(i), 0.0) for i in range(len(criteria) or 4)]
    else:
        raise ValueError(f"unknown question type {qtype!r}")

    total = sum(target)
    target = [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)

    options = render_options({"t": qtype, "crit": criteria})
    sequence, markers = build_sequence(
        tokenizer,
        state,
        {"t": qtype, "ins": question.get("instructions", ""), "crit": criteria},
        cfg.max_len,
        cfg.head_max_len,
    )
    # Marker count must equal option count, or the head is sliced at the wrong width and the
    # target is aligned against the wrong options. Fails loudly rather than training wrongly.
    if len(markers) != len(options):
        return None

    return TrainingItem(
        ids=sequence,
        markers=markers,
        qtype=QTYPES[qtype],
        target=target,
        label=target.index(max(target)),
    )


def collate_train_batch(items: Sequence[TrainingItem], pad_id: int) -> dict[str, Any]:
    """Pad a batch into the tensors the model expects."""
    import torch

    n = len(items)
    length = max(len(it.ids) for it in items)
    kmax = max(len(it.markers) for it in items)

    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, kmax), dtype=torch.long)
    marker_mask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)

    for i, item in enumerate(items):
        ids[i, : len(item.ids)] = torch.tensor(item.ids, dtype=torch.long)
        attention[i, : len(item.ids)] = 1
        k = len(item.markers)
        marker_pos[i, :k] = torch.tensor(item.markers, dtype=torch.long)
        marker_mask[i, :k] = True
        target[i, : len(item.target)] = torch.tensor(item.target, dtype=torch.float32)

    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([it.qtype for it in items], dtype=torch.long),
        "label": torch.tensor([it.label for it in items], dtype=torch.long),
    }


@dataclass(slots=True)
class LossWeights:
    """The published reward mixture."""

    #: Spherical score weight, which rewards matching the target distribution's shape.
    spherical: float = 0.75
    #: Ranked probability score weight.
    ranked: float = 1.0
    #: Soft cross-entropy weight. 1.0 in the published recipe: both halves matter.
    soft_ce: float = 1.0


@dataclass(slots=True)
class EpochLog:
    epoch: int
    loss: float
    loss_rl: float
    loss_ce: float
    seconds: float
    items: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "loss": round(self.loss, 6),
            "loss_rl": round(self.loss_rl, 6),
            "loss_ce": round(self.loss_ce, 6),
            "seconds": round(self.seconds, 2),
            "items": self.items,
        }


def rlcd_loss(
    logits: Any,
    target: Any,
    qtype: Any,
    marker_mask: Any,
    *,
    sigma: float,
    weights: LossWeights,
    group_size: int = 4,
) -> tuple[Any, Any, Any]:
    """The RLCD objective. Returns ``(total, policy_gradient, soft_ce)``.

    ``sigma`` is the exploration noise, annealed across epochs by the caller.
    """
    import torch
    from laya.common import proper_reward

    mask = marker_mask
    k = mask.sum(-1, keepdim=True).float()

    # 1. Zero-mean noise projected onto the option axis, so the perturbation stays inside the
    #    simplex rather than pushing mass at masked positions.
    noise = torch.randn((group_size, *logits.shape), device=logits.device) * sigma * mask
    noise = (noise - noise.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + noise
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    # 2. Proper scoring reward against the gold distribution, then a normalised advantage.
    with torch.no_grad():
        reward = proper_reward(
            q, target.unsqueeze(0), qtype, mask,
            w_sph=weights.spherical, w_rps=weights.ranked,
        )
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)

    # 3. Policy gradient over the noisy samples, plus full-weight soft cross-entropy.
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(advantage * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return (loss_rl + weights.soft_ce * loss_ce), loss_rl, loss_ce


def sigma_at(epoch: int, epochs: int, start: float = 0.4, end: float = 0.1) -> float:
    """Linear anneal from ``start`` to ``end`` across the run."""
    progress = epoch / max(1, epochs - 1)
    return start + (end - start) * progress


def fine_tune(
    *,
    model: Any,
    tokenizer: Any,
    items: Sequence[TrainingItem],
    cfg: TrainConfig,
    device: str,
    weights: LossWeights | None = None,
    log: Any = print,
) -> tuple[list[EpochLog], dict[str, float]]:
    """Run the documented recipe on one GPU.

    Returns the per-epoch log and peak VRAM, which goes into the run report so the numbers are
    measured rather than guessed.
    """
    import torch

    weights = weights or LossWeights()
    model = model.to(device)
    model.train()

    encoder_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    if not encoder_params:
        encoder_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.encoder_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = max(1, len(items) // cfg.micro_batch_size)
    total_updates = steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16_autocast and device.startswith("cuda"))

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history: list[EpochLog] = []
    for epoch in range(cfg.epochs):
        started = time.time()
        sigma = sigma_at(epoch, cfg.epochs)
        order = list(items)
        random.shuffle(order)

        epoch_loss = epoch_rl = epoch_ce = 0.0
        batches = 0
        optimizer.zero_grad(set_to_none=True)

        for start in range(0, len(order), cfg.micro_batch_size):
            chunk = order[start : start + cfg.micro_batch_size]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tokenizer.pad_token_id)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.fp16_autocast):
                logits, _activation = model(
                    batch["input_ids"], batch["attention_mask"],
                    batch["marker_pos"], batch["marker_mask"], batch["qtype"],
                )

            total, loss_rl, loss_ce = rlcd_loss(
                logits.float(), batch["target"], batch["qtype"], batch["marker_mask"],
                sigma=sigma, weights=weights,
            )
            # The published recipe divides by accumulation so the gradient magnitude is
            # independent of how many micro-batches make up an update.
            # Scale *before* backward. Omitting `scaler.scale` while still calling `unscale_`
            # lets fp16 gradients underflow, which is worse than not using autocast at all.
            scaler.scale(total / cfg.gradient_accumulation_steps).backward()

            epoch_loss += float(total.detach())
            epoch_rl += float(loss_rl.detach())
            epoch_ce += float(loss_ce.detach())
            batches += 1

            final_partial = start + cfg.micro_batch_size >= len(order)
            if batches % cfg.gradient_accumulation_steps == 0 or final_partial:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        entry = EpochLog(
            epoch=epoch,
            loss=epoch_loss / max(1, batches),
            loss_rl=epoch_rl / max(1, batches),
            loss_ce=epoch_ce / max(1, batches),
            seconds=time.time() - started,
            items=len(order),
        )
        history.append(entry)
        log(
            f"epoch {epoch + 1}/{cfg.epochs}  sigma={sigma:.3f}  "
            f"loss={entry.loss:.4f} (rl={entry.loss_rl:.4f} ce={entry.loss_ce:.4f})  "
            f"{entry.seconds:.1f}s"
        )

    peak = 0.0
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    return history, {"peak_vram_gib": round(peak, 3)}


def load_base(checkpoint: str = "convaiinnovations/laya") -> tuple[Any, Any, Any, list[str]]:
    """Load the base checkpoint through Laya's supported entry point.

    ``laya.load`` rather than a hand-assembled :func:`laya.common.build_model`. The hand-assembled
    path silently produced a **randomly initialised** model on this machine — weights reaching
    ``2.8e37`` with NaN entries — and trained to a NaN loss without raising. ``laya.load``
    loads the same 421.3M parameters correctly.

    Returns ``(agent, model, tokenizer, warnings)``. Warnings are returned rather than swallowed:
    the shipped ``laya-typed-decisions`` checkpoint carries a per-bucket temperature of 0.1006,
    below Laya's valid ``[0.5, 5]`` range, which the loader clamps and flags as uncalibrated.
    """
    import warnings as _warnings

    import laya

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        agent = laya.load(checkpoint)
    messages = [str(w.message) for w in caught]
    return agent, agent.model, getattr(agent, "tok", None), messages


def inspect_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    """Report whether a checkpoint's shipped temperatures are usable.

    A value outside Laya's valid range is clamped at load time and the affected entry is marked
    uncalibrated, so shipping one is worse than shipping none — it looks configured.
    """
    buckets = cfg.get("temperature_by_options") or {}
    values = list(buckets.values()) if isinstance(buckets, dict) else list(buckets)
    out_of_range = [float(v) for v in values if not (0.5 <= float(v) <= 5.0)]
    return {
        "per_bucket_count": len(values),
        "out_of_range": [round(v, 6) for v in out_of_range],
        "usable": not out_of_range,
        "type_level": cfg.get("temperature"),
    }


def save_checkpoint(model: Any, tokenizer: Any, path: str | Path) -> Path:
    """Write weights and tokenizer. The calibration config is written separately and must ship
    alongside — see ``calibration.persist_calibration``."""

    from safetensors.torch import save_file

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(target / "model.safetensors"))
    if getattr(tokenizer, "save_pretrained", None):
        tokenizer.save_pretrained(str(target / "tokenizer"))
    return target


def write_run_report(
    path: str | Path,
    *,
    config: TrainConfig,
    history: Sequence[EpochLog],
    metrics: Metrics,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist everything needed to review the run without rerunning it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "epochs": config.epochs,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch": config.effective_batch,
            "encoder_lr": config.encoder_lr,
            "head_lr": config.head_lr,
            "max_len": config.max_len,
            "head_max_len": config.head_max_len,
            "seed": config.seed,
        },
        "epochs": [entry.as_dict() for entry in history],
        "metrics": metrics.as_dict(),
        **(extra or {}),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def temperature_bounds() -> tuple[float, float]:
    """The library's own clamp range, so we never fit outside what it accepts."""
    try:
        from laya.calibrate import TEMP_MAX, TEMP_MIN

        return float(TEMP_MIN), float(TEMP_MAX)
    except ImportError:  # pragma: no cover
        return 0.1, 10.0


def cosine_lr(step: int, total: int, base: float, eta_min: float = 1e-6) -> float:
    """Cosine schedule, exposed so a dry run can check the shape without a GPU."""
    if total <= 0:
        return base
    return eta_min + 0.5 * (base - eta_min) * (1 + math.cos(math.pi * step / total))