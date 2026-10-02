# Fine-tuning Laya

How the VEXA//DECK decision head is trained, calibrated, evaluated, and — only then — allowed to
control audio.

The governing constraint is the same one as everywhere else in this project: a model improves the
set, and it may never endanger it.

---

## 1. What is actually being trained

Laya is a **non-autoregressive decision model** over structured text — a ModernBERT-large encoder
(measured: **421.3M parameters**) with a small typed-decision head. It answers bounded questions
about a menu the orchestrator already validated:

| Type | Question | Used for |
|---|---|---|
| `choice` | which of these options | which action to take |
| `score` | how strongly | graded fit |
| `noul` | is this acceptable at all | affirmative half of the second validator |

It never sees audio, never invents an action, and never writes a DSP command. The full contract is
in [`LAYA_DATA.md`](LAYA_DATA.md).

---

## 2. Hard-won facts about the runtime

These were established by measurement on this machine and are not in the upstream docs.

### 2.1 Load the model through `laya.load`, never by hand

```python
# WRONG — silently produces a randomly initialised model.
from laya.common import build_model
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=True)
# -> max |weight| = 2.8e37, NaN parameters, trains to a NaN loss, raises nothing

# RIGHT
import laya
agent = laya.load("convaiinnovations/laya-typed-decisions")
model  = agent.model      # 421.3M params, max |weight| = 4.2, all finite
tokenizer = agent.tok     # note: `.tok`, not `.tokenizer`
```

The hand-assembled path is the one in the upstream notebook. On this machine it loads nothing. The
pipeline verification script **asserts weight sanity and aborts** rather than training garbage:
`tools/verify_finetune_pipeline.py`.

### 2.2 `choice` criteria are a dict, not a list

```python
"criteria": {"keep the current track playing": "",
             "transition, 124 BPM, energy 0.68, instrumental": ""}
```

`render_options` calls `crit.items()` for a choice question. A list fails with
`'list' object has no attribute 'items'`. `score` questions are the opposite — a plain list.

The **label is the key**, and both `expected` and the gold distribution are keyed by it. Keying by
internal option ids instead produces a well-formed dataset where nothing ever matches.

### 2.3 The shipped checkpoint ships an invalid temperature

Loading `laya-typed-decisions` emits:

```
laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected entries as uncalibrated.
```

Measured: `temperature_by_options` has 6 entries, min 0.1006, max 1.9834 — one is **below the
valid floor** and gets clamped. The type-level `temperature` is a list of three values near 1.04.

**This is the concrete reason confidence must not gate until we refit.** Shipping an out-of-range
value is worse than shipping none, because it looks configured. `inspect_calibration()` reports
this; `CalibrationResult.trustworthy` gates on it.

### 2.4 Devices are selected by name

`nvidia-smi` and torch enumerate the two cards in **opposite order**:

```
nvidia-smi index 0 = RTX 5060 Ti     torch cuda:0 = RTX 4060
nvidia-smi index 1 = RTX 4060        torch cuda:1 = RTX 5060 Ti
```

Any config saying `cuda:0` meaning "the big card" is wrong. `resolve_device()` matches on name.

Also measured: the 4060's `sm_89` is **absent** from this torch build's arch list and runs via PTX
JIT. First-launch JIT cost is large — the 5060 Ti's first fp16 matmul measured 349 ms against
2.9 ms warm. Never benchmark without warmup.

---

## 3. Getting labels

**Playback logs are not enough.** They record which action fired, never how the unplayed
alternatives would have sounded, so they cannot rank anything (`LAYA_DATA.md:61`).

The collection loop:

1. The feasibility filter offers **3–8 genuinely playable** options at a decision point.
2. Render a **10–20 second audio preview** of each (`IDEAS.md:11`).
3. A trained listener picks the best and flags unacceptable ones.
4. **Always include `continue_current` and a safe fallback.** Without them the model learns to
   over-transition (`LAYA_DATA.md:57`).
5. **Keep disagreements.** Annotator spread is signal about subjectivity, not noise to average
   away (`LAYA_DATA.md:59`).

Sample hard cases deliberately: beat ambiguity, tempo doubling, weak key estimates, sparse-to-dense
changes, a new request, and generation arriving late.

### Dataset sources

Ranked by trustworthiness in [`PLAN.md`](PLAN.md) §7.1. The two that matter are ours:
**A/B transition previews** (real preference, no legal ambiguity) and **decisions on our own
approved YuE2 library**. The DJ-mix transitions dataset stays blocked: its card states no license.

---

## 4. Splits

**By asset family and complete session, never by random row.** Variants of one YuE2 render must
stay in the same split or the model memorises a render instead of learning a decision.

```python
from vexa_laya.dataset import split_by_family
split = split_by_family(annotations, train_ratio=0.7, validation_ratio=0.15, seed=0)
```

`tests/test_laya.py::test_a_family_never_spans_two_splits` asserts no family leaks across buckets.

---

## 5. Training

`services/laya/src/vexa_laya/finetune.py`, following the published recipe.

| Knob | Value | Source |
|---|---|---|
| epochs | 4 | notebook |
| effective batch | 64 | 8 × 2 GPUs × 4 accumulation |
| encoder LR | 2.5e-5 | notebook |
| head LR | 1e-4 | notebook |
| optimiser | AdamW, wd 0.01, cosine → 1e-6 | notebook |
| grad clip | 1.0 | notebook |
| `max_len` / `head_max_len` | 1024 / 256 | `laya-typed-decisions` config |
| exploration σ | 0.4 → 0.1, annealed | notebook |
| reward | spherical 0.75, ranked 1.0 | notebook |

The loss is RLCD: sample noisy logit projections, score them with proper scoring rules against the
gold distribution, normalise the advantage, take a policy-gradient step, and add a full-weight soft
cross-entropy term. Both halves read the **distribution**, not a hard label.

### Two bugs this codebase already fixed

Both were caught only by running, and both would have produced a plausible-looking run:

- **`scaler.scale(loss).backward()` is required.** Calling `scaler.unscale_()` without scaling
  first raises; and omitting the scale while un-scaling lets fp16 gradients underflow.
- **`build_training_item` returns `None` on a marker/option count mismatch.** The head is sliced at
  the marker width, so a mismatch aligns the target vector against the wrong options and trains
  *silently and wrongly*.

### Verifying the pipeline

```bash
uv run --no-sync python tools/verify_finetune_pipeline.py
```

**Measured on this machine** (RTX 5060 Ti, 48 items, 2 epochs):

```
epoch 1/2  sigma=0.400  loss=0.9210 (rl=-0.1969 ce=1.1180)  9.5s
epoch 2/2  sigma=0.100  loss=0.5409 (rl=-0.4555 ce=0.9963)  1.2s
peak VRAM: 7.81 GiB of 15.5 GiB
```

This proves the machinery. It proves **nothing about quality** — the labels are rule-generated,
and `IDEAS.md:57` warns that synthetic labels can reproduce the rules without improving listening
quality. The checkpoint this writes is stamped `SYNTHETIC-SMOKE-TEST` and must never be promoted.

---

## 6. Calibration

`services/laya/src/vexa_laya/calibration.py` delegates to Laya's own
`laya.calibrate.fit_temperatures` rather than reimplementing it, because the library handles the
easy-to-get-wrong parts: `MIN_TYPE_N = 10` for a per-type scalar and `MIN_BUCKET_N = 2000` for
per-bucket temperatures, which are therefore *omitted* for a dataset our size.

Project guardrails on top:

- the slice is carved **by asset family**, before training;
- the gate **refuses to open** on fewer than `MIN_TYPE_N` held-out items;
- a fit that **raises** ECE is rejected — that is evidence the slice was wrong, not a temperature;
- `compute_ece=True` makes the library hold out 20% internally, so the improvement is *measured*.

```python
result = calibrate(agent, held_out_pairs, config=cfg)
if result.trustworthy:
    persist_calibration(agent, result)
```

**A correction to an earlier reading.** `LAYA_DATA.md:65` and the upstream notebook say to *remove*
`temperature_by_options` when persisting a calibration. That advice is about **stale, inherited**
values. Laya 0.3.23's `fit_temperatures` deliberately *writes* per-bucket temperatures gated by
`MIN_BUCKET_N`, so hand-deleting them would discard a deliberate part of the current scheme. Use
the library's own fit and its own `save_calibration`.

---

## 7. Evaluation

Always report **calibration next to accuracy** — the training signal is a distribution, so
accuracy alone will not tell you whether confidence is usable.

| Metric | Why |
|---|---|
| choice accuracy / soft accuracy | agreement with the chosen label / the gold distribution |
| Brier | whether probabilities are meaningful |
| ECE (15 bins) | whether confidence may gate |
| pairwise win rate vs rules | **the number that decides promotion** |
| invalid action rate | **must be 0** — a bug, not a quality score |
| latency p50/p95 | must land before the scheduling deadline |

Timing is informational: Laya's harness deliberately excludes `*_ms` from baseline comparison,
because timing noise is not a quality regression.

---

## 8. Shadow mode, then promotion

```python
shadow = ShadowPolicy(inner=LayaAdapter(), rules=RulePolicy())
```

Shadow mode lets Laya choose while the **rules keep control of audio**, logging every disagreement.
Compare, listen to the disagreements, and only then consider promotion.

### Promotion gates

| Gate | Condition |
|---|---|
| Held-out blind preference | beats the rule policy |
| Invalid action rate | **exactly 0** |
| ECE | at or below target after calibration |
| Latency | answers before the scheduling deadline under load |
| Low-confidence fallback | works |

Until all five hold, Laya runs in shadow. The rules are the permanent fallback, so the set never
depends on it.

---

## 9. Hardware

| Phase | Device | Why |
|---|---|---|
| Inference | RTX 4060 (GPU 1) | 421M params ≈ 0.85 GB fp16; leaves room for a GGUF generation job |
| **Fine-tuning** | **RTX 5060 Ti (GPU 0)** | the recipe targets a 16 GB card; `sm_120` is natively compiled |

Measured peak for fine-tuning: **7.81 GiB of 15.5 GiB**, so the 5060 Ti fits with room to spare.

Fine-tuning and YuE2's **torch** backend both want GPU 0 and **cannot run concurrently** — see
`PLAN.md` §3.6. The GGUF backends run on GPU 1 alongside Laya inference.