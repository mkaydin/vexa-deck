# Datasets for fine-tuning Laya

Every entry below was **verified against its source on 2026-10-02**, not carried over from
[`LAYA_DATA.md`](LAYA_DATA.md). Where the two disagree, this file is correct and `LAYA_DATA.md`
should be updated.

The governing constraint comes from `IDEAS.md:57`: *"Training Laya solely from synthetic rules may
reproduce the rules without improving listening quality."* No dataset on this page labels **DJ
decisions**. That is the whole problem, and it is stated plainly in §2.

---

## 1. The short version

| Dataset | Use it for | License | Reality |
|---|---|---|---|
| **Your own A/B previews** | **training** | yours | The only source of real DJ preference. Does not exist yet |
| `LocalLLaMA/typed-decisions` | format + RLCD targets | Apache-2.0 | Right schema, wrong domain, synthetic |
| `disco-eth/edm-cue` | cue timing priors | **MIT** | Metadata only, no audio |
| `mir-aidj` djmix (`hf.yi-lab.net/djmix`) | transition distributions | **unspecified** | **Blocked** — no license |
| `Raveform` (mir-aidj) | phrase/section structure | check terms | Structure annotations for EDM in mixes |
| `M-DJCUE` | small timing eval set | GPL-3.0 (code) | 134 tracks, audio not distributed |
| `MTG-Jamendo` | retrieval/tagging | Apache-2.0 code, **CC per-track audio** | 55k tracks, not DJ decisions |
| `UnmixDB` | DSP behaviour tests | check | Synthetic transitions |

**Nothing here can train the model to DJ well.** They supply priors, evaluation sets, and a
format. The labels have to come from listening.

---

## 2. The gap, stated honestly

Laya's supervised signal is *preferences between concrete options at a decision point*. For a DJ
that means: given a session state and 3–8 feasible next actions, which did a listener actually
prefer?

No public dataset measures that. The closest things measure **timing** (where the cue points are)
or **tagging** (what genre/mood a track is). Neither is a judgement about which of two available
transitions sounds better right now.

So the critical path is unchanged and is not a download:

```
starter library  ->  A/B preview harness  ->  human labels  ->  fine-tune
```

Everything in §3 exists to make that loop faster or better founded, not to replace it.

---

## 3. Format reference — `LocalLLaMA/typed-decisions`

**This is the dataset Laya's own published fine-tune uses.** It is worth studying closely even
though it is off-domain.

- <https://huggingface.co/datasets/LocalLLaMA/typed-decisions>
- License **Apache-2.0**. 1,200 train / 400 test cases. Five typed questions per case → 6,000
  train decisions, 2,000 test decisions.
- Configs: `agent_trace_observability`, `customer_service`, `invoice_processing`,
  `security_incidents`.
- Row shape: `state` and `questions` are JSON strings; `gold` holds **full gold distributions**.

```python
from datasets import load_dataset
import json

ds = load_dataset("LocalLLaMA/typed-decisions", "customer_service", split="test")
row = ds[0]
state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
print(gold["urgency"]["probabilities"])
```

### What it teaches us, and what it does not

**Teaches (worth copying):**

- Gold is **soft**. Each case is labelled three times by a ~4B teacher at temperature 0.7, and the
  gold is the mean distribution. That is precisely the RLCD target shape `services/laya/convert.py`
  emits, and it is why forcing a single label throws information away.
- Every option carries a written description that is **part of the input**, not decoration.
- Per-question ceilings vary wildly (0.560 for `agent_trace/urgency` to 0.937 for
  `customer_service/category`). Read scores per question, never only the average.

**Does not teach:** anything about music, DJs, or transitions. It is tagged `synthetic`.

### Reference points to beat

| Reference | Accuracy | What it is |
|---|---|---|
| Random | 0.318 | floor |
| Majority class | 0.461 | ignores the input |
| Prior | 0.470 | label frequencies only |
| Teacher self-agreement | 0.735 | ceiling — above this is learning teacher quirks |
| Laya typed-decisions | 0.766 | fine-tuned on this benchmark |
| Base `laya` (no fine-tune) | 0.362 | what we start from |

**Caveat carried from the benchmark itself:** *"Gold is the mean of three samples from a teacher, so
a score measures agreement with that teacher, not correctness."*

---

## 4. Prior and evaluation datasets

### `disco-eth/edm-cue` — MIT, the safest one

- <https://huggingface.co/datasets/disco-eth/edm-cue>
- **License MIT.** ~4,710 tracks, ~21k expert-annotated cue points, beat grid and key metadata.
- Tabular/parquet, 1K–10K rows. Paper `arXiv:2407.06823`, code at `ETH-DISCO/cue-detr`.

**Use:** timing priors, and a sanity check on our own beat-grid detector.
**Not for:** preferences. Cue points say *where a human tapped in*, not *what they would have
played next*.
**Note:** metadata only. No audio is distributed, so it cannot become library material.

### Raveform — structure annotations for EDM in real mixes

- <https://mir-aidj.github.io/raveform>, dataset via `mir-aidj/djmix-dataset`
- Paper: *Raveform: A Dataset of Metrical and Functional Structure Annotations for EDM Tracks in DJ
  Mixes*, TISMIR 2026.

**Use:** phrase/section boundaries and where transitions actually fall inside real mixes — the
closest public analogue to our scheduler's problem.

**Directly relevant finding from the paper**, which we should adopt:

> *"For tracks with substantial tempo changes or more irregular rhythms, the estimated beat grid can
> become unstable."*

That is the exact condition under which our `min_beat_confidence = 0.6` gate refuses to schedule.
Independent corroboration that the gate is right.

**Check before deriving weights:** the audio is third-party and track links are not redistribution
rights.

### The DJ Mix Dataset — **blocked**

- Repo `mir-aidj/djmix-dataset`; the data itself is served from **<https://hf.yi-lab.net/djmix>**,
  not from `huggingface.co/datasets/djmix/...` as `LAYA_DATA.md` currently states. Correct that.
- Splits: `transitions` (64.7k rows, 19 columns), `beats-tracks` (63k rows), `beats-mixes`
  (5.04k rows).

**Use if it were licensed:** studying real transition distributions — what overlap lengths and
transition types actually occur.

**Blocked.** The dataset card states no explicit license. Using it for training or for deriving
weights without verified permission is exactly the failure `LAYA_DATA.md:85` warns about. Keep it
blocked until terms are established in writing.

### `MZehren/M-DJCUE` — small but human

- <https://github.com/MZehren/M-DJCUE>, 134 EDM tracks, multi-annotator IN/OUT cue labels.
- **Code GPL-3.0.** Source audio is copyrighted and not distributed.

**Use:** a small external set to check our beat tracker and cue logic against humans.
**Not for:** training; 134 tracks cannot teach a decision policy.

### `MTG-Jamendo` — tagging and retrieval

- <https://github.com/MTG/mtg-jamendo-dataset>, 55,000+ full tracks, 195 tags.
- **Repo code Apache-2.0; audio is Creative Commons, per track.** Check
  `audio_licenses.txt` for each track you actually use.

**Use:** pretraining or evaluating the *retrieval* half of the system — mood, genre, instrument
tags. Its terms suit a non-commercial project.

**Not for:** DJ actions. It has no notion of a transition or a decision point.

### `UnmixDB` — synthetic transitions

**Use:** testing DSP behaviour against transitions whose parameters are known exactly.
**Not for:** preference — "programmed transitions are not human DJ preference labels".

### Adjacent work worth reading

- **DJTransGAN** — *Automatic DJ Transitions with Differentiable Audio Effects and Generative
  Adversarial Networks*, `ChenPaulYu/DJTransGAN`. Relevant to how transitions are *rendered* for
  preview, which is what the A/B harness needs.
- **`mir-aidj/djmix-analysis`** — mix-to-track subsequence alignment (ISMIR 2020). The technique
  behind recovering where a transition happened inside a recorded mix.

---

## 5. Building the labels that actually matter

From `LAYA_DATA.md:61` and `IDEAS.md:11`. The loop:

1. The feasibility filter offers **3–8 genuinely playable** options at a decision point.
2. Render a **10–20 second audio preview** of each.
3. A trained listener picks the best and flags the unacceptable ones.
4. **Always include `continue_current` and a safe fallback.** Without them the model learns to
   over-transition (`LAYA_DATA.md:57`).
5. **Keep disagreements.** Annotator spread is signal about subjectivity (`LAYA_DATA.md:59`).
6. Sample hard cases deliberately: beat ambiguity, tempo doubling, weak key estimates,
   sparse-to-dense changes, generation arriving late.

### Sampled deliberately, not scraped

- ≥5 annotators on the first 100 decisions, so the disagreement structure is measured rather than
  assumed.
- Hold out whole **asset families**, never random rows. Variants of one render must stay together
  or the model memorises the render instead of learning the decision.
- Keep the raw canonical record (`services/laya/dataset.py`) so trainer versions can change without
  losing labels.

---

## 6. A correction to the shipped checkpoint's calibration

Verified from the `convaiinnovations/laya-typed-decisions` model card, and independently measured
on this machine:

> *"Its `temperature_by_options` was inherited from the base checkpoint and overrides the per-type
> temperatures fitted for this model — refit on your own held-out data before relying on the
> probabilities."*
>
> *"The fine-tuning run fitted `[1.0148, 1.0374, 1.0575]` on a slice of the same items it had just
> trained on... Until this checkpoint is refit, treat its confidence as uncalibrated."*

Loading it here emits:

```
laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected entries as uncalibrated.
```

And its measured per-type temperatures are `[1.0148024559020996, 1.0374259948730469,
1.0575125217437744]` — identical to the documented values.

**So: the base checkpoint's confidence must not gate anything until we refit on our own held-out
data.** `services/laya/calibration.py` enforces this — the gate refuses to open without
`MIN_TYPE_N` held-out items and refuses a fit that raises ECE. Its measured ECE is 0.213 against
Jev's 0.144.

One more documented limit worth respecting, and we already do:

> *"Keep `choice` questions under ~20 options. Options share a fixed 256-token head budget, so a
> large label space leaves few tokens per label and accuracy falls off sharply."*

`vexa_laya.adapter.MAX_OPTIONS = 8` is comfortably inside that.

---

## 7. Order of work

1. **Get more labels.** Everything else is downstream. A/B preview harness → annotators.
2. **Refit calibration** on held-out data from our own families. Until then, confidence is
   decorative and shadow mode stays on.
3. **Use the public sets for what they are good at** — beat-grid validation (EDM-CUE, M-DJCUE),
   retrieval pretraining (MTG-Jamendo), transition-distribution study (djmix, if licensed),
   section structure (Raveform).
4. **Use `typed-decisions` for the format**, and for a pipeline check that the training code runs
   end to end on a known-good benchmark before it ever sees our labels.
5. **Never** fine-tune on a dataset whose labels came from the rule policy we are trying to beat.
