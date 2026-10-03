# Fine-tuning decision

**Decided: 2026-10-02.** What we fine-tune Laya on, in what order, and what we refuse to do.

This supersedes the dataset discussion in [`DATASETS.md`](DATASETS.md), which remains
authoritative for *individual dataset licences and provenance*. This file is authoritative for
*the plan*.

---

## 1. The decision

**Train on human A/B labels, collected on a library of roughly 100 tracks.**

Everything else is rehearsal:

| Source | Role | Ships? |
|---|---|---|
| `LocalLLaMA/typed-decisions` | proves the RLCD loop consumes real soft labels | no |
| Rule-derived labels over our own depot | pipeline stress test at realistic scale | **never** |
| **Human A/B labels, ~300–400** | **the decision head** | **yes** |
| EDM-CUE, M-DJCUE, Raveform, MTG-Jamendo | validating our analyzer and retrieval | no |

### What we refuse to do

**We will not train on labels derived from the rule policy we are trying to beat.**

`IDEAS.md:57`: *"Training Laya solely from synthetic rules may reproduce the rules without
improving listening quality."* It would also produce a model that scores well on our own benchmark
and adds nothing at runtime — the worst available outcome, because it looks like success.

Rule-derived rows are still generated, because a pipeline untested at scale is untested. They are
tagged `rule_derived` and **structurally excluded from promotion**, not excluded by convention.

### The honest ceiling

This plan's quality ceiling is a person listening. There is no dataset shortcut. **If annotation
capacity is not available, the correct outcome is to ship the deterministic rule policy** — it is
explainable, has no calibration problem, and costs nothing. Laya earns its place only by
measurably beating it.

---

## 2. Why the library must grow first

**10 tracks cannot produce a meaningful label set.**

Pairwise transitions among 10 tracks give roughly 90 ordered pairs, of which perhaps 30–40 survive
the tempo/key/repetition filter as genuinely distinct decision situations. Collecting 400 labels
there yields 400 opinions about ~40 decisions: a model that memorises the library, and a
family-level split with nothing sensible to separate.

| Library | Distinct decision situations | Usable for a fine-tune |
|---|---|---|
| 10 (current) | ~30–40 | no — overfits |
| **~100** | **~300+** | **yes** |

100 tracks costs **~22 minutes** of GPU at the measured 13 s per 45 s of audio. It is the cheapest
thing standing between us and a meaningful label round, and it needs no listening.

---

## 3. Stages

### Stage 1 — depot growth and pipeline proof *(no listening required)*

1. Generate ~100 YuE2 tracks clustered across tempo and mood, mastered, gated, admitted.
2. Derive rule-based labels over every valid pair → `rule_derived`, thousands of rows.
3. Run the full RLCD loop on them. This proves: tokenisation, family-level splits, the calibration
   slice held out before training, loss behaviour at scale, and that the trained adapter loads.

**Exit criterion:** loss falls, splits hold no family overlap, an adapter round-trips through
`laya.load`, and every produced row is stamped non-promotable.

**This stage can ship nothing, and that is correct.**

### Stage 2 — human labels *(the real work)*

1. `tools/collect_labels.py` over the 100-track depot.
2. ~300–400 decisions, **five annotators on the first ~100** so disagreement is measured rather
   than assumed (`LAYA_DATA.md:59`).
3. `continue_current` always offered — without it the model learns to over-transition
   (`LAYA_DATA.md:57`).

At ~30 s per decision that is roughly 2.5–3 hours per annotator.

**Exit criterion:** a held-out family split the model has never seen.

### Stage 3 — fine-tune

Human labels only. 4 epochs, encoder LR 2.5e-5, head LR 1e-4, calibration slice carved out
*before* training (`LAYA_DATA.md:65`, and Laya's own documentation calls this the step most likely
to be dropped and the one that matters most the moment anything gates on confidence).

### Stage 4 — shadow mode

Laya proposes, rules play, disagreements logged. Promotion requires all of:

| Gate | Condition |
|---|---|
| Held-out blind preference | beats the rule policy |
| Invalid action rate | **exactly 0** |
| ECE | at or below target after refitting on held-out data |
| Latency | answers before the scheduling deadline |

**If Laya never disagrees usefully, we ship the rules and delete the model.** That is a legitimate
outcome, not a failure.

---

## 4. Algorithmic generation as depot filler

Searched 2026-10-02. Two families exist:

- **Algorithmic composition**: `isobar` (mature, Python), `subsequence`, `midigen` (GPL-3.0),
  `pymusik`
- **ML composition**: Magenta (`melody_rnn`, `improv_rnn`, `music_vae`)

### What it buys, and why it matters

Verified by writing a 4-track file directly: **tempo is exact** (122.0001 BPM as written),
**and the parts are genuinely separate** — kick, bass and chords are distinct tracks, not
estimates.

That solves two problems YuE2 cannot:

1. **Exact metadata.** YuE2 was asked for a slow jazz bed and produced 144 BPM. Every filter
   input — tempo, key, energy — is an *estimate* on generative audio and *ground truth* on
   algorithmic audio.
2. **True stems.** We could not get kick/drum/bass out of YuE2 by any route: not natively, not by
   editing its score (tested — removing the instrumental voice produced audio just as loud as the
   full mix), and not by separation (HTDemucs-6s and BS-RoFormer both produce a ~141 % residual,
   i.e. overlapping estimates rather than a partition). Algorithmic generation *specifies* the
   parts, so separation is never needed.

### The domain shift, stated plainly

A selector trained on algorithmic transitions and deployed on YuE2 transitions is a **domain
shift**. The `DecisionRequest` state is bpm, key, section, bars_to_boundary, energy, vocals,
recent_families — no timbre — so the *task* is the same. But the mapping from features to "sounds
good" may differ between synthesised loops and dense generative audio.

This is a real risk and shadow mode is precisely the mechanism for catching it. It does not
invalidate the approach; it bounds the claim.

### Strudel — the strongest candidate

**Strudel** (<https://strudel.cc>) is a JavaScript port of the TidalCycles pattern language,
browser-first. It renders through WebAudio, so a headless renderer needs browser automation —
but several exist, and the mature one works.

Tested `dehenne/wirbel` (**AGPL-3.0-only**, npm, has CI):

```
setcpm(124/60/4)
stack(
  s("bd*4").gain(1.0),
  s("~ cp").gain(.55),
  s("~ ~ oh ~").gain(.3),
  note("<C2 C2 F2 G2>*8").s("sawtooth").lpf(400).gain(.6)
)
```

```
$ wirbel house.strudel --format wav --duration 12 --json
{"ok":true,"output":"/tmp/strudelout/house.wav","duration":12,"cps":0.00861}
```

**12 seconds of stereo audio in 0.52 s.** Two orders of magnitude faster than YuE2 (13 s per
45 s), and it produced a genuine four-on-the-floor pattern with a filtered saw bass.

Worth noting: `jebin2/strudel-render` advertises *"Pure renderer; loopability is config"* —
directly relevant, since a DJ library wants loopable beds.

I first measured **130.8 BPM against a requested 124, confidence 0.00**, and recorded beat
detection as a blocker. **That was my error, not a defect.** The pattern used `setcpm`, which is
cycles per *minute*; wirbel divided by 60 again, so the 12 s render contained **0.103 cycles —
about 0.4 beats**. One detected onset was the correct answer to a near-silent clip. `setcps` is
the right call.

Re-tested across the range, with the corrected pattern:

| Requested | Measured | Error | Confidence | Stable grid |
|---|---|---|---|---|
| 90 | 90.7 | 0.81 % | 1.00 | yes |
| 105 | 104.2 | 0.79 % | 1.00 | yes |
| 124 | 125.0 | 0.81 % | 1.00 | yes |
| 140 | 140.6 | 0.45 % | 1.00 | yes |
| 174 | 175.8 | 1.02 % | 1.00 | yes |

**Beat detection works on synthetic material.** YuE2 library re-checked at the same time — no
regression, all six tracks still at confidence 1.00.

The real finding is narrower and more useful: **measured tempo carries ~1 % error**, from frame
quantisation at `REFERENCE_RATE` 48 kHz with a 512-hop. So when the generator *specified* the
tempo, we store the specified value and treat measurement as **verification, not metadata**.
That distinction matters more for the depot than the bug did.

One genuine issue remains: the render measured **-27.65 LUFS**, below our -24 floor. Our existing
mastering stage fixes it.

### Decision

**Use Strudel to fill the depot and the Stage 1 pipeline test. Keep YuE2 as the production
aesthetic.**

Rationale: 0.5 s per render makes a 100-track depot nearly free, tempo is specified rather than
estimated, and patterns are per-instrument so stems are genuine. YuE2 stays for the audible
library because it sounds like music rather than a drum machine.

Licence is the deciding factor against `isobar`: Strudel examples carry **CC BY-NC-SA 4.0** and
wirbel is **AGPL-3.0-only**. Both are fine for this non-commercial project but neither is
permissive, so patterns and rendered audio must be kept as separate artefacts, never vendored
into the source tree.

**Tempo handling:** store the generator's specified tempo; use the measured value as a gate
(reject beyond ~2 % disagreement), never as the stored metadata.

---

## 5. What is already true

- YuE2 backend built and running on the RTX 5060 Ti; 10 tracks generated, all admitted.
- Loudness verified against the BS.1770-4 calibration tone; mastering gives 0.30 LU library spread.
- Audio engine: 0 underruns, worst callback 0.61 ms of an 11.6 ms budget.
- Laya RLCD loop runs end to end (loss 0.9210 → 0.5409, 7.81/15.5 GiB) — on rule-generated labels,
  so it proves machinery only.
- A/B harness renders equal-power previews and records canonical annotations.
- 253 tests, lint clean.

## 6. What is not true yet

- **No human labels exist.**
- **The loop has never run as a continuous session with a model choosing.** Every component works
  in isolation; none have run together with real Laya decisions driving transitions.
- The shipped Laya checkpoint's calibration is documented upstream as unfit — its per-type
  temperatures were fitted on training data and an inherited `temperature_by_options` overrides
  them. Confidence must not gate until refitted on held-out data.

---

## 7. Immediate next action

**Stage 1.** Grow the depot to ~100 tracks and run the rule-derived pipeline test.

It needs no listening, it cannot ship, and it answers the question that matters next: *does the
fine-tune machinery hold up at realistic scale?*

Only after that does asking a person to listen for three hours make sense.
