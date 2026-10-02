# Laya fine-tuning and data plan

## What Laya should learn

Laya is a fast, non-generative decision model over text or structured state. In VEXA//DECK it should answer bounded questions such as:

1. **When should a transition begin?** Continue, next 4/8/16-bar boundary, or a known section marker.
2. **Which feasible action should happen next?** Stay, add/remove a prepared layer, swap bass, start a specific ready asset, or use a fallback bed.
3. **Is the proposed transition acceptable?** A yes/no confidence gate alongside deterministic checks.

Do not train it to infer beat positions from raw audio or to produce music. The analysis pipeline provides measured features; the orchestration layer creates feasible options. Keep option descriptions concise: Laya has an option-token budget, so shortlist candidates before ranking. [Official Laya repository](https://github.com/NandhaKishorM/laya).

## Dataset research and intended use

| Source | What it contributes | How to use it | Limitation |
|---|---|---|---|
| [EDM-CUE](https://huggingface.co/datasets/disco-eth/edm-cue) | 4,710 tracks and about 21k expert cue points, with beat-grid and key metadata | Pretrain/evaluate candidate cue timing | Cue points do not label the best next song or transition effect; metadata is MIT, audio is not included |
| [M-DJCUE](https://github.com/MZehren/M-DJCUE) | Human IN/OUT cue annotations for 134 EDM tracks, multiple annotators | Small external timing evaluation set | Copyrighted source audio is not distributed; cue choices are subjective |
| [DJ Mix Dataset transitions](https://hf.yi-lab.net/djmix) | 64,748 rows with track pairs, overlap, cue timing, and alignment fields | Study real transition distributions and derive candidate examples after validation | Served from `hf.yi-lab.net/djmix`, **not** `huggingface.co/datasets/djmix/...`. Card states no explicit license; verify permissions and alignment quality before training or redistribution |
| [Raveform](https://mir-aidj.github.io/raveform/) | DJ set order, beats, estimated transitions, and human structure labels on 1,423 tracks | Learn phrase/section context and transition placement | Track links are not redistribution rights; exact annotation/data terms should be checked before publishing derived weights |
| [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | Genre, instruments, mood/theme tags for over 55k tracks | Train/evaluate catalog tagging and request-to-asset retrieval | It does not label DJ actions; dataset terms restrict it to non-commercial research/academic use and audio has per-track licenses |
| [UnmixDB](https://zenodo.org/records/1422385) | Synthetic, beat-aligned transitions with known parameters | Test transition analysis and DSP behavior | Programmed transitions are not human DJ preference labels |

**Primary domain dataset:** collect decisions from VEXA//DECK's own YuE2-generated and approved library. Public data provides useful timing and structure priors; the system-specific data teaches which of *our actual actions* sounds good.

## Canonical example

Create a JSONL record for each decision point, with a stable state representation and a question over concrete candidate actions:

```json
{
  "state": {
    "request": "darker, more energetic, no vocals",
    "current": {"bpm": 122, "key": "A minor", "section": "outro", "bars_to_boundary": 8, "energy": 0.46, "vocals": false},
    "history": {"last_action": "continue", "recent_families": ["family_8", "family_3"]},
    "candidates": [
      {"id": "a", "type": "continue", "energy": 0.46},
      {"id": "b", "type": "transition", "asset": "asset_42", "bpm": 124, "energy": 0.68, "vocals": false},
      {"id": "c", "type": "transition", "asset": "asset_53", "bpm": 121, "energy": 0.72, "vocals": true}
    ]
  },
  "question": "Which feasible action best follows the request at the next phrase boundary?",
  "preferred_action_id": "b",
  "label_source": "human_pairwise_review",
  "quality": 0.9
}
```

This is the project's **canonical annotation record**, not a claim that the official trainer consumes this exact JSON. Convert it into the current Laya training question format, then into the documented evaluation JSONL shape (`state`, `questions`, `expected`, optional tags). Keep the raw record so trainer versions can change without losing labels. [Laya evaluation format](https://github.com/NandhaKishorM/laya/blob/main/docs/evals.md).

## Label collection

1. Generate and verify a varied starter library with YuE2. Record model, decoder, prompt, seed, and source file hashes.
2. Run a deterministic feasibility filter to offer 3–8 genuinely playable options at each decision point.
3. Render short **audio A/B transition previews**, not just textual descriptions.
4. Ask trained listeners to choose the best option, identify unacceptable options, and optionally score continuity, energy fit, vocal clash, and request fit.
5. Include `continue_current` and safe fallback choices; otherwise the model will learn to over-transition.
6. Sample hard cases deliberately: beat ambiguity, tempo doubling/halving, weak key estimates, sparse-to-dense changes, new user requests, and generation arriving late.
7. Keep disagreements between annotators. They reveal subjective choices and should not be hidden by one forced label.

Historical playback logs alone are insufficient: they show which action occurred, but not how unplayed alternatives would have sounded. Human comparisons or controlled offline renders are needed for reliable ranking labels.

## Training sequence

**Stage A — baseline:** implement rule selection and measure it. **Stage B — timing:** use cue/structure datasets to evaluate or specialize a cue decision. **Stage C — system action:** fine-tune Laya on VEXA-specific candidate decisions. **Stage D — calibration:** fit confidence on a separate calibration set and measure whether low-confidence cases should fall back to rules. Laya's official repository provides a fine-tuning notebook and warns that inherited option-specific temperatures can override a newly fitted calibration if exported incorrectly. [Laya fine-tuning notes](https://github.com/NandhaKishorM/laya).

## Splits and evaluation

Split by **asset family and complete session**, not random decision rows. Variants of one YuE2 render must stay in the same split. Freeze a held-out test set before tuning. Report results per genre, BPM range, vocal condition, request type, and asset availability.

| Metric | Why it matters |
|---|---|
| Choice accuracy / top-k accuracy | Agreement with expert choices |
| Pairwise preference win rate | Whether selected transitions sound better than baseline |
| Expected calibration error / Brier score | Whether confidence supports fallback decisions |
| Invalid-action rate | Should be zero after deterministic validation |
| Timing error in bars | Whether transitions land on intended phrase boundaries |
| Session-level listening rating | Detects repetition and poor long-term energy flow |
| Inference latency under load | Ensures decisions arrive before scheduling deadlines |

Run a **shadow mode** first: Laya chooses but the rule policy controls audio. Compare choices and listen to disagreements. Enable Laya control only after it improves blind listening results while meeting continuity and invalid-action gates.

## Rights and provenance

Open-source code, public metadata, dataset audio, YuE2 weights, and trained Laya weights are distinct artifacts. Keep a source-and-license manifest for each training row. Do not package third-party audio with the app by default. Before publishing fine-tuned weights, verify permissions for each data source; a public download page or a repository code license alone does not settle training-data and redistribution terms. YuE2 model weights are published under CC BY-NC 4.0, while its code has a separate license. [YuE2 model card](https://huggingface.co/m-a-p/YuE2-3B).
