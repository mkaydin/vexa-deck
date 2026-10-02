"""Training configuration.

Every value here comes from Laya's published fine-tuning recipe rather than from taste. The
numbers are in the fine-tuning notebook and in ``docs/finetune.md``; they are recorded here with
their provenance so a future change is a deliberate edit rather than a silent default.

The recipe targets a 16 GB card. Both of this machine's GPUs are smaller than that except the
RTX 5060 Ti, which matches it exactly — so fine-tuning belongs on **GPU 0**, which is also the
card YuE2's torch backend wants. See ``PLAN.md`` §3.6: the two are mutually exclusive, and a
fine-tune is never started while a generation job is queued.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Provenance for every knob, so the numbers are auditable rather than folklore.
RECIPE_SOURCE = (
    "Laya docs/finetune.md and notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb"
)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """The documented recipe, parameterised by device."""

    #: Four epochs is what the notebook runs for its demo and its real data alike.
    epochs: int = 4

    #: 8 sequences per micro-batch, accumulated to an effective batch. On a single card the
    #: published 2-GPU x 4-accumulation figure is reproduced with 4 micro-batches of 16.
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8

    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    max_grad_norm: float = 1.0
    fp16_autocast: bool = True
    gradient_checkpointing: bool = True

    #: Sequence budget. `max_len` bounds the state text, `head_max_len` bounds the option list.
    max_len: int = 1024
    head_max_len: int = 256
    max_tokens_per_batch: int = 4096

    #: Calibration slice: up to 400 items or 10%, taken *before* sharding.
    calibration_max_items: int = 400
    calibration_fraction: float = 0.10
    #: Temperatures are clamped to this range by the published fit.
    temperature_bounds: tuple[float, float] = (0.1, 10.0)
    #: Used when a slice has too few items to fit anything meaningful.
    temperature_default: float = 1.2

    seed: int = 0

    @property
    def effective_batch(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def with_device(self, *, micro_batch: int | None = None) -> TrainConfig:
        """A copy adjusted for a card with less memory than the recipe assumed."""
        return replace(self, micro_batch_size=micro_batch or self.micro_batch_size)


@dataclass(frozen=True, slots=True)
class Paths:
    """Where a run reads and writes. Nothing outside these directories is touched."""

    root: str
    dataset: str = "data/sessions/labels.jsonl"
    output: str = "runs/laya"

    @property
    def train(self) -> str:
        return f"{self.output}/train.jsonl"
@dataclass(slots=True)
class Metrics:
    """What a run reports. Calibration is reported next to accuracy, always.

    Every field is ``None`` until measured. An unmeasured metric must never read as a passing
    zero: a report full of ``0.0`` looks like a clean run and would let an unevaluated checkpoint
    through a promotion gate. ``promotion_gates`` therefore fails closed.
    """

    choice_accuracy: float | None = None
    soft_accuracy: float | None = None
    brier: float | None = None
    ece: float | None = None
    mean_confidence: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    #: Held-out pairwise win rate against the rule baseline. The number that decides promotion.
    win_rate_vs_rules: float | None = None
    #: Fraction of chosen actions that deterministic validation would have refused.
    invalid_action_rate: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """Whether every gated metric has a real value."""
        return all(
            value is not None
            for value in (
                self.choice_accuracy, self.ece, self.invalid_action_rate, self.win_rate_vs_rules,
            )
        )

    def as_dict(self) -> dict[str, object]:
        def fmt(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "measured": self.measured,
            "choice_accuracy": fmt(self.choice_accuracy),
            "soft_accuracy": fmt(self.soft_accuracy),
            "brier": fmt(self.brier),
            "ece": fmt(self.ece),
            "mean_confidence": fmt(self.mean_confidence),
            "latency_p50_ms": fmt(self.latency_p50_ms),
            "latency_p95_ms": fmt(self.latency_p95_ms),
            "win_rate_vs_rules": fmt(self.win_rate_vs_rules),
            "invalid_action_rate": fmt(self.invalid_action_rate),
            **self.extra,
        }

    def promotion_gates(
        self, *, min_accuracy: float, max_ece: float, min_win_rate: float = 0.5
    ) -> list[tuple[str, bool]]:
        """The conditions a fine-tune must meet before it may control audio.

        Fails closed: an unmeasured metric is a failing gate, not a passing one. Invalid actions
        are absolute — Laya may only pick from the menu the filter produced, so a nonzero rate is
        a bug rather than a quality score.
        """
        def at_least(value: float | None, floor: float) -> bool:
            return value is not None and value >= floor

        def at_most(value: float | None, ceiling: float) -> bool:
            return value is not None and value <= ceiling

        return [
            ("metrics were measured", self.measured),
            (f"choice_accuracy >= {min_accuracy}", at_least(self.choice_accuracy, min_accuracy)),
            (f"ece <= {max_ece}", at_most(self.ece, max_ece)),
            ("invalid_action_rate == 0", self.invalid_action_rate == 0.0),
            (f"win_rate_vs_rules >= {min_win_rate}",
             at_least(self.win_rate_vs_rules, min_win_rate)),
        ]

    def promotable(self, *, min_accuracy: float, max_ece: float) -> bool:
        return all(passed for _, passed in self.promotion_gates(
            min_accuracy=min_accuracy, max_ece=max_ece))
