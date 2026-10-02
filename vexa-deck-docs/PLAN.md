# VEXA//DECK — Merged Plan

**Status:** authoritative. Supersedes the planning content of `ROADMAP.md`, `PRODUCT.md`,
`ARCHITECTURE.md`, `LAYA_DATA.md`, and `IDEAS.md`. Those documents remain the *specification*
(what the product is, what the contracts mean, why). This document is the *build plan*
(what we build, in what order, on what hardware, in what runtime).

Everything below is grounded in the repository docs plus runtime verification of YuE2-3B and
Laya 0.3.23. Where a claim comes from measurement rather than the original docs, the evidence
is cited.

---

## 1. Locked decisions

These come from the project owner and are treated as settled.

| # | Decision | Consequence |
|---|---|---|
| D1 | `character-sheets/vexa-pixel-portrait.png` is the repository README icon and the application icon | Canonical brand asset; derived sizes generated from it. Character turnaround sheets are **reference art only**, not shipped sprites |
| D2 | Character sheets become the animated on-program hero; animation is a later pass | The GUI reserves a character stage; no sprite sheet is produced until the animation design pass |
| D3 | The LLM integration is **OpenAI-compatible**, not vendor-pinned | Any `/v1/chat/completions` endpoint works. The planner can be pointed at OpenAI, a local llama.cpp/vLLM server, LM Studio, or any compatible gateway with a config change. See §5.3 |
| D4 | Model inference runs **locally on this host**, never in Docker | Preserves GPU passthrough, avoids CUDA-in-container friction, keeps weights on local disk. See §4 |
| D5 | Services run in **Docker containers** | Orchestrator, planner, library/index, analyzer, and job queue are containerized for reproducible dependencies and clean restarts |
| D6 | Two GPUs: **RTX 5060 Ti (16 GB) is primary**, RTX 4060 (8 GB) secondary | Primary takes generation, fine-tuning and the bf16 reference pipeline; the secondary holds Laya inference and can take a Q8_0 GGUF render concurrently. Resolved by name, never index — §3.1 |
| D7 | YuE2 runs through a **pluggable generation backend**, defaulting to a quantized GGUF build | Both GPUs become generation-capable and generation can run concurrently. Backends: `yue2.cpp` GGUF (3.8–5.8 GB), `torch` BF16 (11.18 GiB, reference), `audio.cpp` GGUF (7755–8867 MiB). Chosen by Phase B0 measurement, not assumption. See §3.4 |

---

## 2. What the backend actually is

The GUI is one half. The backend is three cooperating concerns, and keeping them distinct is
what stops the system from violating its own central invariant.

> **Audio continuity is the highest priority.** A model timeout or failed generation must not
> interrupt the stream. (`README.md:18`)

That single rule determines the process topology. Anything that can be slow, hang, OOM, or
restart is *outside* the audio process. Anything on the audio path is deterministic and
allocation-free in its callback.

```
┌─ DOCKER ────────────────────────────────────────────────────────────┐
│                                                                     │
│  orchestrator ──► library/index ──► feasibility filter ──► scheduler │
│       │                                                               │
│       ├──► planner (OpenAI-compatible LLM, config-pinned)           │
│       └──► analyzer (decode, loudness, beat/key/structure, gates)    │
│                                                                     │
│  job queue (Redis)  ◄── enqueue ──► claim ──► result                  │
└─────────────────────────────────────────────────────────────────────┘
            │                                    │              │
            │ versioned HTTP/JSON contract      │              │
            ▼                                    ▼              ▼
┌─ HOST (local, no Docker) ─────────────────────────────┐┌──────────────┐
│  laya service ──► GPU 1 inference (resident)          ││ audio engine │
│    decision + shadow mode + calibration gate          ││ deterministic│
│                                                        ││ no model in  │
│  yue2 worker ─┬─► GPU 0  torch BF16   (reference)     ││ the callback │
│   one          └─► GPU 1  yue2.cpp Q8_0 (default)     │└──────────────┘
│   GenerationBackend protocol — same job contract       │
│                                                        │
│  GUI (later)                                          │
└────────────────────────────────────────────────────────┘
```

The orchestrator is the only component that both containers and host processes talk to. Every
message across that line is a versioned contract (§6). The GUI never talks to a model directly.

---

## 3. Hardware reality

### 3.1 What was measured

```
GPU 0: NVIDIA GeForce RTX 5060 Ti   16311 MiB   compute_cap 12.0 (Blackwell, sm_120)
GPU 1: NVIDIA GeForce RTX 4060       8188 MiB   compute_cap  8.9 (Ada, sm_89)
Host RAM: 14 GiB total, ~10 GiB available
Driver: 615.71.09
```

**Enumeration order differs between `nvidia-smi` and torch, and this is a trap.** Measured:

```
nvidia-smi index 0  = RTX 5060 Ti     torch cuda:0 = RTX 4060
nvidia-smi index 1  = RTX 4060        torch cuda:1 = RTX 5060 Ti
```

Any configuration that says `cuda:0` meaning "the big card" is wrong. Devices are therefore
selected by name, never by index.

**Measured torch capability** (torch 2.14.1+cu130, CUDA 13.0):

| Device | Compute | In torch's compiled arch list | Warmed fp16 matmul |
|---|---|---|---|
| RTX 5060 Ti | `sm_120` | **yes** | 47.2 TFLOP/s |
| RTX 4060 | `sm_89` | **no** | 22.4 TFLOP/s |

The 4060's `sm_89` is absent from the build's arch list (`sm_75, sm_80, sm_86, sm_90, sm_100,
sm_120`) and runs via PTX JIT. Ordinary ops work, but kernels with no PTX fallback may not.
**Both GPUs support bf16.** First-launch JIT cost is real and large — the 5060 Ti's first fp16
matmul measured 349 ms against 2.9 ms warm, so benchmark numbers taken without warmup are
meaningless.

This reinforces the split in §3.3: fine-tuning belongs on the 5060 Ti, whose architecture is
natively compiled.

### 3.2 Model requirements, verified

| Model | Requirement | Evidence |
|---|---|---|
| **YuE2-3B** | Peak VRAM **11.18 GiB** (full CoT); **14.08 GiB** at max context. "Use a 24GB GPU and 24GB available host RAM." One song at a time. BF16 AR/NAR, FP32 VAE. | YuE2 model card speed/resource table |
| **YuE2-3B speed** | 71.04 s generation / 214.85 s audio on RTX 4090 (≈3.0× realtime) | same |
| **Laya** | 421M params (ModernBERT-large). ~0.85 GB in fp16. 33 ms/question on T4, 7.2 ms batched | Laya README |
| **Laya fine-tune** | Recipe tuned for a **16 GB** card | Laya `docs/finetune.md` |
| **YuE2-3B GGUF** (`yue2.cpp`, Q8_0) | **3.8 GB** peak at `--max-seq 8192`; **5.8 GB** at full 24576 context. Weights 3.81 GB. Only one module resident in VRAM at a time; the KV cache persists. | `Serveurperso/YuE2-GGUF` model card |
| **YuE2-3B GGUF** (`audio.cpp`, Q8_0 + F16 VAE) | **8867 MiB** peak, longform, RTX 5090 | `audio-cpp/Yue2-3B-GGUF` model card |
| **YuE2-3B GGUF** (`audio.cpp`, Q4_0 + F16 VAE) | **7755 MiB** peak, longform, RTX 5090 | same |
| **YuE2 VAE** | **Never quantize.** 530 MB F32. "Its weights are the audio." | `Serveurperso/YuE2-GGUF` model card |
| **Quantization floor** | **No Q4 for the backbone.** "An audio code LM breaks below Q5." Q8_0 is the sane default. | same |

### 3.3 Pinned GPU split

**There are two viable generation backends, and they fit different GPUs.** This is the plan's
central hardware conclusion, and it changed once the quantized GGUF weights were checked.

The reference PyTorch pipeline loads the whole model resident and peaks at 11.18 GiB, which
**cannot** fit the 4060. The GGUF backends trade residency for memory — `yue2.cpp` keeps only
one module in VRAM at a time and lets the halves trade places around a persistent KV cache —
and land between **3.8 GB and 5.8 GB**, which **does** fit it.

```
GPU 0 — RTX 5060 Ti, 15.9 GiB usable  (Blackwell, sm_120)
  ├── yue2 worker (torch, BF16)   11.18 GiB peak   → FITS, ~4.7 GiB headroom
  │                              (14.08 max-ctx)  → FITS, ~1.8 GiB headroom (tight)
  └── laya fine-tuning            recipe wants 16  → FITS COMFORTABLY

GPU 1 — RTX 4060, 8.0 GiB usable  (Ada, sm_89)
  ├── laya inference (resident)   ~0.85 GiB        → comfortable
  └── yue2 worker (yue2.cpp Q8_0) 3.8–5.8 GiB     → FITS
                                 (audio.cpp Q4_0)  7755 MiB → does NOT fit
                                 (torch BF16)      11.18 GiB → does NOT fit
```

**Consequences:**

1. **Both GPUs can generate, and they can generate at the same time.** This is the whole reason
   the GGUF route matters. It is not a degraded fallback — it is a second worker, on a second
   card, with a different memory profile. Phase B4's job queue and priority model exist for
   exactly this, and it also realizes the local/remote worker contract from `IDEAS.md:41`
   without needing a remote machine at all.
2. **The 5060 Ti is freed for Laya fine-tuning**, which needs a 16 GB card and could not use the
   4060 at all. Fine-tuning and torch-generation still cannot share GPU 0 (§3.6).
3. **Q8_0 is the default backbone, not Q4.** Q8_0 is described as near-lossless; the quantization
   floor for an audio code LM is Q5, and Q4 is explicitly rejected. Beat-grid stability is a
   top-ranked risk in this project (`ROADMAP.md:52`) and there is no reason to spend quality on
   a memory problem we do not have.
4. **The VAE is never quantized**, in any backend.
5. **Host RAM is 14 GiB against YuE2's documented 24 GiB recommendation.** This risk now applies
   to the torch backend specifically. The GGUF/C++ backends are far lighter on host RAM, which is
   a further reason to prefer them as the default. Still unverified on this box.
6. **Blackwell (sm_120) needs a CUDA 12.8+ PyTorch build.** `torch==2.14.1` (current, and the
   version Laya requires) satisfies this. FlashAttention on sm_120 is a **separate, unverified**
   risk on the torch path; the GGUF/CUDA path does not depend on it.

### 3.4 Choosing the default backend

All three are implemented behind one `GenerationBackend` protocol. The default is **decided by
measurement in Phase B0, not by argument here**, because the deciding factor is audio quality
under quantization — and that is exactly the kind of thing that cannot be read off a spec.

| Backend | Weights | Peak VRAM | Card | For |
|---|---|---|---|---|
| `yue2.cpp` GGUF Q8_0 | 3.81 GB | 3.8–5.8 GB | 4060 or 5060 Ti | **default candidate** — concurrent generation on both GPUs |
| `torch` BF16 | 7.26 GB | 11.18 GiB | 5060 Ti only | reference-quality renders, editable-ABC plan→edit→generate, ground truth for comparing the others |
| `audio.cpp` GGUF | 2.67–4.26 GB | 7755–8867 MiB | 5060 Ti | secondary alternative; `audio.cpp` is the more established of the two GGUF runtimes |

The torch backend is kept even if GGUF wins, for two reasons: it is the reference implementation
whose output the quantized backends are judged against, and it is the only one of the three that
supports **Plan-only → edit the ABC → regenerate**, which the agentic-editing workflow in
`IDEAS.md` depends on. (`yue2.cpp` exposes `yue-plan`, `audio.cpp` does not.)

**The honest caveat: every VRAM figure above is from someone else's GPU.** `yue2.cpp`'s
3.8/5.8 GB is self-reported without a named device; `audio.cpp`'s 7755/8867 MiB is measured on an
RTX 5090. None of it was measured on Ada (sm_89) or Blackwell (sm_120), and none of it was
measured at our target song lengths. Phase B0 exists to replace these with our own numbers.

### 3.5 Expected generation throughput

The RTX 5060 Ti is compute-weak relative to a 4090 (roughly half the BF16 throughput) though its
memory bandwidth is higher, so the torch path is **predicted at ~2–3 minutes per song instead of
71 seconds**. The GGUF paths trade some precision for a large memory win and may be comparable or
faster. All of these are projections. Phase B0 produces the real numbers.

None of them change the design: the docs only ever needed generation to be *background*, never
gating first audio. Even a five-minute render is a perfectly good background job. What the
measurement decides is queue behaviour and how many workers we can usefully run at once.

### 3.6 GPU scheduling rule

One owner per GPU at a time. A simple lease queue in the orchestrator:

```
5060 Ti (primary)  : [ yue2 torch job ]  OR  [ yue2 gguf job ]  OR  [ laya finetune ]
4060  (secondary)  : [ yue2 gguf job ]   OR  [ laya inference, resident ~0.85 GB ]
```

`workers/yue2/backends.py` owns the device table (`PRIMARY_GPU_NAME`, `primary_gpu()`,
`secondary_gpu()`), so "primary" means one thing everywhere rather than a default repeated across
call sites.

Laya *inference* stays resident on GPU 1 — it is a few hundred MB and needs to answer inside a
scheduling deadline. A GGUF generation job takes the rest of that card. Fine-tuning is a batch
activity that takes the whole of GPU 0, and is simply not started while a torch generation job is
queued or running. Since a DJ session is not fine-tune-time, this is a non-conflict in practice,
but the rule is explicit rather than assumed.


---

### 3.7 The audio engine, measured

**Decision: Python + `sounddevice` (PortAudio).** Chosen to get the underrun criterion *measured*
before committing to a heavier stack, per this document's own Q2. Revisit if the numbers below
stop being comfortable.

Two decks, a crossfader, a lookahead master limiter, and a lock-free command queue. All decoding
and resampling happens on a loader thread; the callback only reads preloaded buffers and does
arithmetic. See `services/audio/`.

**Measured on this machine** (PulseAudio default output, 44.1 kHz, 512-frame blocks):

| Metric | Measured | Budget |
|---|---|---|
| Worst callback time | **0.61–1.99 ms** | 11.6 ms per block (5–17% used) |
| Underruns over 90 s | **0** in 7787 callbacks | 0 |
| Queue drops | **0** | 0 |
| Crossfade | linear ramp over 2 s, no dip in output level | — |

**Configuration that works:** 44100 Hz, 512 frames (11.6 ms), 2 channels. Larger blocks absorb
more jitter at the cost of latency; 11.6 ms is imperceptible for a background set.

#### Bugs this found, that no design review would have

Three defects, all of which had to be *heard* rather than reasoned about:

1. **`queue or CommandQueue()` silently discarded the engine's queue.** `CommandQueue` defines
   `__len__`, so an empty-but-valid queue is **falsy**; the `or` default built a fresh one, and
   every command vanished into an object nobody was reading. The engine appeared to start and
   played perfect silence. Fixed with explicit `is None` checks, and pinned by
   `test_the_queue_is_never_falsy_just_because_it_is_empty`.
2. **`RawOutputStream` hands the callback a buffer, not an array.** The callback caught the
   `TypeError` and correctly output silence — the safety net working, the engine useless.
   `OutputStream` supplies a numpy array directly.
3. **The crossfade set a flag and a countdown but never interpolated the gain.** It was a hard
   cut wearing a crossfade's name. Now a per-block ramp, verified to hold output level through the
   overlap.

#### Memory: a self-inflicted OOM

The first soak attempt computed `track_seconds = args.seconds * 3`. For a 30-minute run that is
5400 seconds of stereo audio per deck — about **3.8 GB resident**, on a 14 GB machine that was
already running the container stack. It got OOM-killed.

Fixed by giving decks a `loop` flag, which is what a DJ bed does anyway: a 60-second bed loops
indefinitely, so a half-hour soak costs **141 MB**. The lesson generalises — *a soak test that
scales its fixtures with its duration is a memory bomb, not a test.*

---

## 4. Runtime: what is containerized and what is not

The line is **not** "cloud vs local". The line is *does this component need to be predictable and
low-latency, and does it need the GPU or the audio device*.

| Component | Runtime | Why |
|---|---|---|
| orchestrator | **Docker** | Stateless control plane; restart must not affect audio |
| library/index | **Docker** | Pure data + search |
| analyzer | **Docker** | CPU-heavy, isolated, restartable |
| planner (LLM) | **Docker** | Network-egressing, slowest, most failure-prone |
| job queue (Redis) | **Docker** | Standard; gives us priority + retry semantics for free |
| laya service | **Host** | Needs GPU 1; model stays on local disk (D4) |
| yue2 worker (torch) | **Host** | Needs GPU 0 + FlashAttention + large host RAM (D4) |
| yue2 worker (GGUF) | **Host** | Needs a GPU and a CUDA/Vulkan build; spawned as a **subprocess per song**, so containerizing it would only add process nesting |
| audio engine | **Host** | Needs realtime audio. Containerized ALSA/Pulse adds latency jitter and underrun risk for zero benefit. Built in `services/audio/` — see §3.7 |
| GUI | **Host** (later) | Electron/Tauri on the desktop |

The audio engine is host-native for the same reason the models are: a container boundary on the
realtime path is a latency risk that buys nothing.

The GGUF backends are worth one extra note: they are C++ binaries, not Python packages, and the
worker talks to them over a **subprocess boundary**, not an in-process API. That is a feature, not
a compromise — it means cancelling a running generation is a process kill, the GPU is released
when the process exits, and a crashed render cannot take the worker down with it.

---

## 5. Backend services

### 5.1 `services/orchestrator` — the control plane

The deterministic heart. In order of precedence:

1. **Session state** — decks, musical clock, current energy, committed action, fallback bed.
2. **Feasibility filter** — deterministic, never model-driven. Rejects candidates on: asset
   not `ready`, time-to-load exceeds the deadline, beat alignment unavailable, tempo ratio outside
   bounds, harmonic policy violation, vocal collision, recent repetition. **Always emits a safe
   `continue_current` and `play_fallback_loop` action.** (`ARCHITECTURE.md:64`)
3. **Decision policy** — pluggable. Two implementations:
   - `RulePolicy` — deterministic. The baseline, and the permanent fallback. (`ROADMAP.md:37`)
   - `LayaPolicy` — delegates to the Laya service. Starts in **shadow mode**: it chooses, the
     rule policy still controls audio, and disagreements are logged. (`LAYA_DATA.md:81`)
4. **Scheduler** — commits in musical time (beats, bars, phrase boundaries). A transition commits
   only when its assets are loaded before the required deadline. A late command is dropped, never
   executed mid-phrase by accident. (`ARCHITECTURE.md:64`)
5. **Request state machine** — `received → interpreted → matched_ready → scheduled → applied`, with
   the generation branch. Each request carries an ID and a **generation number**; a newer request
   invalidates an older uncommitted transition. A late-arriving generation result cannot reverse
   the listener's latest direction. (`ARCHITECTURE.md:81`)

### 5.2 GPU leases and job queue

Redis-backed priority queue. Jobs are the `GenerationJob` contract (§6. Retries happen inside a
budget; a failure marks the job failed and leaves playback untouched.

### 5.3 `services/planner` — OpenAI-compatible LLM

Per **D3**, this speaks plain `/v1/chat/completions`. The base URL and model ID are configuration:

```env
PLANNER_BASE_URL=https://api.openai.com/v1
PLANNER_MODEL=<any model id>
PLANNER_API_KEY=<key>
```

Point it at a local llama.cpp or vLLM server instead and nothing else changes. This is precisely why
the vendor is not hardcoded — model choice is expected to change, and the seam is already drawn.

Two hard rules from the docs are enforced here:
- The planner receives **structured music metadata**, never raw audio. It interprets free-form
  user intent into a structured brief and constraints. (`README.md:39`)
- **A local rule-based fallback adapter must exist and be usable.** If the LLM is unreachable,
  timeouts, or is simply not configured, the orchestrator keeps working with the rule-based
  parser. Basic operation never requires a remote service. (`README.md:7`, `ARCHITECTURE.md:24`)

### 5.4 `services/laya` — decision model (GPU 1)

- Wraps `laya==0.3.23`, checkpoint `convaiinnovations/laya-typed-decisions` (421M, 1024 ctx) for
  English decision rows.
- Exposes typed questions: `choice` (which action), `noul` (is this transition acceptable — the
  yes/no confidence gate from `LAYA_DATA.md:9`), `score` (how strongly).
- **Option-token budget is real.** Laya has a head budget; the orchestrator shortlists to 3–8
  genuinely playable options *before* asking. (`LAYA_DATA.md:11`, and confirmed as a documented
  Laya limitation: high option counts degrade confidence selection.)
- **Calibration gate.** Laya ships over-confident. Confidence thresholds are meaningless until
  temperatures are fitted. Any export must (a) fit one temperature per question type on a slice
  held out **before** training, and (b) **delete any inherited `temperature_by_options`**, which
  takes precedence at inference and silently masks a new fit. This is the exact trap
  `LAYA_DATA.md:65` warns about, and Laya's own docs call it "the part most likely to be dropped."
- **Deterministic re-validation.** Whatever Laya returns is re-checked by the second
  deterministic validator before it can be scheduled. Laya never writes DSP commands.

### 5.5 `workers/yue2` — generation (GPU 0 **and** GPU 1)

- Versioned job contract, **identical across every backend and identical for a local and a remote
  worker**, so a later handoff needs no protocol change. (`IDEAS.md:41`)
- One `GenerationBackend` protocol, three implementations, selected per job by capacity:

  | Backend | Runtime | Exposes plan→edit→generate? |
  |---|---|---|
  | `yue2.cpp` Q8_0 | C++/GGML, subprocess per song | yes — `yue-plan` emits the ABC score |
  | `torch` BF16 | Python, resident model | yes — `pipe.plan()` then regenerate with an edited `abc` |
  | `audio.cpp` | C++/GGML, subprocess per song | **no** — plan-only is not exposed |

- Because each C++ backend is a **subprocess per song**, job cancellation is a real process-tree
  kill rather than a cooperative flag. The GPU is released when the process exits.
- Output goes to the analyzer and quality gates. It becomes `ready` only after passing. A failed
  or cancelled job has **zero effect on playback**. (`ROADMAP.md:33`)
- Provenance recorded on every render: backend, model revision, quant level, decoder, prompt,
  seed, source hashes. A quantized render is a *different artifact* from a bf16 one and must be
  distinguishable downstream.
- **Quantization is recorded as a quality caveat, not silently accepted.** Beat-grid stability
  under Q8_0 is measured in Phase B0 and re-checked in the gates; if quantized renders prove
  unreliable for cueing, they stay in the library as review-only material and the torch backend
  takes over for anything the scheduler must beat-align.

### 5.6 Analyzer and quality gates

An asset becomes `ready` only after all of these pass (`ARCHITECTURE.md:46`):

| Gate | Checks |
|---|---|
| decode | readable, correct channel count and sample rate |
| duration | within the requested band |
| loudness | integrated LUFS in range; true-peak headroom |
| beat grid | tempo detected and grid stable enough to cue against |
| loop boundary | approved loop points are beat-aligned and do not click |
| quality | not silent, not clipped, no gross artifacts |

Loudness and peak are computed deterministically in-process (ITU-R BS.1770 K-weighting +
oversampled true peak). Beat/key/structure use established analysis libraries. **Analysis runs
at ingestion, never in the audio callback.** Stems, when produced, come from separate
source separation during ingestion — YuE2 is not a stem separator (`README.md:39`).

---

## 6. Contracts (v1)

Versioned from the start, before any service code. `ARCHITECTURE.md:106`

| Contract | Purpose | Key fields |
|---|---|---|
| `AssetManifest` | one playable material | `asset_id`, `family_id`, `source_type`, `parent_id`, content hash, codec/rate/channels, duration, bpm, beat-grid version, time signature, key **estimate + confidence**, section markers, energy/mood/instrument/vocal tags, integrated loudness, true peak, loop points, quality flags, approval state, model/decoder versions, prompt/seed, license/provenance |
| `SessionState` | live set state | decks A/B, musical clock, current asset + bar, energy, committed action, fallback, active request, job refs |
| `DecisionRequest` | what the selector is asked | state snapshot, feasible action IDs, deadline, policy version |
| `FeasibleAction` | a concrete option | `action_id`, `asset_id`, `entry_bar`, `start_at_session_bar`, `fade_bars`, `tempo_ratio`, `transition_preset`, rationale |
| `ScheduledAction` | committed transition | action ID, commit beat, **asset version/hash**, fallback ID |
| `GenerationJob` | YuE2 work | brief, seed, priority, request ID + generation number, state, target family |
| `RequestStatus` | honest UI state | `applied now` / `scheduled` / `generating` / `ready` / `unable to fulfil` (`README.md:21`) |

Two invariants worth stating because they are enforced in code, not comments:

- **Model-produced tags are estimates with confidence, never facts.** `LAYA_DATA.md:44`
- **A path is not an identity.** Assets are identified by content hash and version, never by
  filename, so a stale scheduler decision cannot act on a file that has since changed underneath
  it. `ScheduledAction` carries the hash for exactly this reason.

---

## 7. Laya dataset plan

The fine-tune target is `laya-typed-decisions`. On its own benchmark, the base checkpoint scores
**0.362** near chance against a 0.318 baseline; the fine-tuned checkpoint reaches **0.766**.
Fine-tuning is where essentially all the value is, so the dataset is a first-class deliverable,
not an afterthought.

### 7.1 Label sources, ranked by trustworthiness

| Rank | Source | What it gives | Status |
|---|---|---|---|
| **1** | **Own A/B transition previews** | True DJ preference on *our actual actions* | Must be built. Highest value, no legal ambiguity |
| **2** | Own approved YuE2 library decisions | Real action distributions on our assets | Must be built. Blocked on Phase B1 |
| 3 | `laya-typed-decisions` teacher | Soft label bootstrap, cold start only | Use to seed, never as final ground truth |
| 4 | MTG-Jamendo tags | 55k+ genre/mood/instrument labels for retrieval | Non-commercial research use — **fits this project**, per-track audio licenses |
| 5 | EDM-CUE | 4,710 tracks, ~21k expert cue points, beat grid + key | Metadata MIT, audio excluded. Cue timing priors |
| 6 | Raveform | 1,423 tracks with structure + transition placement | Check terms before deriving weights |
| 7 | M-DJCUE | 134 tracks, multi-annotator IN/OUT cues | Small external timing eval set |
| 8 | UnmixDB | Synthetic beat-aligned transitions with known parameters | DSP behaviour testing only — not preference |
| — | DJ Mix Dataset transitions | 64,748 rows, real transition distributions | **BLOCKED** — dataset card states no explicit license |

### 7.2 Why #1 has to be real

Playback logs alone are insufficient. They record which action fired, but never how the
**unplayed** alternatives would have sounded — so they cannot rank anything. The data collection
loop is therefore:

1. Feasibility filter offers 3–8 genuinely playable options at a decision point.
2. Render short **audio previews** of each (10–20 s, per `IDEAS.md:11`).
3. A trained listener picks the best and flags unacceptable ones.
4. Always include `continue_current` and a safe fallback — **otherwise the model learns to
   over-transition.** (`LAYA_DATA.md:57`)
5. Keep disagreements between annotators. They are signal about subjectivity, not noise to be
   averaged away. (`LAYA_DATA.md:59`)

### 7.3 Splitting

Split by **asset family and complete session**, never by random decision row. Variants of one
YuE2 render must stay in the same split, or the model will memorise a family instead of learning
the decision. (`LAYA_DATA.md:69`) The held-out test set is frozen before tuning.

### 7.4 Known sharp edges

Carried straight from Laya's own documentation, because each will otherwise surface as a mystery:

- High option counts degrade confidence selection → keep the shortlist short.
- Forced-choice negation can follow the *question* over the *state* → do not phrase questions as
  negations.
- The loop is only as good as its targets: RLCD imitates a teacher's distribution, and that
  teacher's quality is the ceiling.

---

## 8. Build phases

Each phase has an exit criterion that can be *measured*, not asserted.

### Phase A — Contracts and skeleton
Versioned schemas for all seven contracts, JSON Schema export, repo layout, CI, test harness.
**Exit:** a clean checkout validates a fixture manifest against the schema and round-trips every
contract through serialize → parse.

### Phase B0 — Hardware verification *(do this first, it is cheap)*

Measure before building on any of it. Five unknowns, in priority order:

1. **Backend bake-off.** Render the *same* brief and seed on every backend that runs here, then
   compare: `yue2.cpp` Q8_0 on the 4060, `yue2.cpp` Q8_0 on the 5060 Ti, `torch` BF16 on the
   5060 Ti. Record peak VRAM, wall time, and — critically — **tempo and beat-grid stability
   under quantization**, which is what actually decides the default.
2. **Host RAM headroom** against YuE2's documented 24 GB recommendation on a 14 GB box. Applies
   to the torch backend; the C++ backends are far lighter.
3. **sm_120 + FlashAttention** for the torch path. Confirm or lose that backend.
4. **CUDA build viability** for `yue2.cpp` / `audio.cpp` against driver 615.71.09 on both sm_89
   and sm_120.
5. **Laya on the 4060** — loads, answers, and meets the latency budget under a 1024-token head.

**Exit:** a written numbers table replacing every figure in §3.2 that is not ours. If no backend
fits, or quantized renders cannot be beat-gridded reliably, the generation strategy changes
here — before any orchestrator work depends on it.

### Phase B1 — Audio core *(no models)*
Two decks, fallback bed, preloading, transport, gain/EQ/crossfade, master limiter, beat-aware
scheduling with hard limits. Model-free.
**Exit:** a representative 30-minute unattended set with **zero underruns and zero dead air**;
killing the planning process does not stop playback. (`ROADMAP.md:15`)

### Phase B2 — Library, ingestion, gates
Immutable asset IDs, family/version links, local index, metadata extraction, quality gates,
preview playback, manual approval, sections and approved loops. Optional stem separation here.
**Exit:** a corrupt or unanalyzed asset can never become `ready`; every played asset has
provenance.

### Phase B3 — Requests and rule-based DJ
Local parser/rules adapter, candidate retrieval, deterministic feasibility filter, rule selector,
session history, all three request state machines with fallback.
**Exit:** a request for an absent asset leaves audio uninterrupted and shows a visible queued
status; newer requests supersede older uncommitted plans; manual override always wins.

### Phase B4 — YuE2 worker
GPU lease, offline/background generation, job priority, versioned outputs, analysis, review,
cache admission, full provenance.
**Exit:** failed or cancelled jobs do not affect playback; stale outputs never override a newer
request; measured generation times recorded.

### Phase B5 — Planner service
OpenAI-compatible client, local rule-based fallback adapter, retry/timeout/failover behaviour.
**Exit:** killing the LLM mid-session changes nothing audible; the fallback parser carries the set.

### Phase B6 — Laya service, shadow mode
Adapter, annotation tooling, A/B collection harness, calibration, dataset build. Runs in shadow:
Laya chooses, rules control audio, disagreements logged.
**Exit:** inference reliably completes before the scheduling deadline under load; disagreements
are captured and analysable; no invalid action ever reaches the scheduler.

### Phase B7 — Laya fine-tune and promotion
Train on the frozen dataset, calibrate temperatures, evaluate against the rule policy on held-out
families, compare blind listening.
**Exit:** held-out human listening preference improves over the rule policy; invalid-action rate
stays at zero; low-confidence fallback works. **Only then** may Laya control audio.

### Phase C — GUI
Character stage (D1/D2), decks, library, transition timeline, request panel, generation queue,
keyboard access, high contrast, reduced motion, packaging.
**Exit:** a new user imports or generates approved music, starts a set, steers it, inspects
pending work, and stops it — without a terminal.

---

## 9. Risks, with the ones that can kill a phase marked

| Risk | Severity | Early test | Mitigation |
|---|---|---|---|
| No generation backend fits both GPUs | **Phase-killing** | Phase B0, day one | `--max-seq` trades context for memory; `cot="off"`/`melody` peak slightly lower; CPU/Vulkan fallback exists; or a remote worker via the identical job contract |
| Quantized renders lose tempo/beat-grid fidelity | **High — decides the default backend** | Phase B0 bake-off, measuring beat stability not just VRAM | Keep torch BF16 as reference; gate quantized output more strictly; if unusable for cueing, quantized renders stay review-only |
| GGUF CUDA builds fail on sm_89/sm_120 with driver 615.71.09 | High | Phase B0, day one | Vulkan or CPU backend; `audio.cpp` as an alternative runtime; torch on the 5060 Ti |
| 14 GB host RAM too small for the torch backend | Medium (GGUF backends unaffected) | Phase B0, day one | Prefer the C++ backends by default; reduce max context |
| GPU 0 contention between torch generation and fine-tuning | Medium | Lease enforcement from the first commit | Explicit exclusive lease (§3.6) |
| Generated tracks vary in tempo or phrasing | High | Beat-grid + cue-point analysis in B4 | Robust beat maps; reject ambiguous automatic transitions |
| YuE2 outputs loop or separate poorly | High | Score loop/stem quality on diverse short candidates | Prefer full tracks and verified sections; quarantine poor stems |
| Laya learns shortcuts from synthetic labels | High | Family-level split, unseen-family comparison | Human A/B labels; never train solely from rules (`IDEAS.md:57`) |
| User requests outrun generation | Medium | Simulate conflicting live requests | Versioned intent + commit checks (`ARCHITECTURE.md:81`) |
| Non-commercial assets bundled accidentally | Medium | Audit package contents against the manifest | Source, weights, audio, datasets stay separate downloads |
| LLM provider changes or disappears | **Low** | — | Already neutralized by D3; the OpenAI-compatible seam is the mitigation |

---

## 10. Deliberate limits

Not oversights. From `PRODUCT.md:56-61` and `IDEAS.md:53-57`.

- Prepared tracks and short verified loops first. Unrestricted stem recombination is deferred.
- No claim that a YuE2 revision preserves an unchanged waveform outside an edited section.
- No promise that an absent style is heard instantly.
- No automatic publication or redistribution of third-party dataset audio or model weights.
- No model inference in the audio callback. Ever.
- No "always generate the next segment" — generate toward known library gaps.

---

## 11. Open decisions

| # | Question | Recommendation | Blocks |
|---|---|---|---|
| Q1 | Repository license | **AGPL-3.0-or-later.** Non-commercial open project; keeps derived work open. YuE2 weights are CC BY-NC 4.0 and stay separate regardless. `pyproject.toml` currently states this provisionally | Packaging (Phase C) |
| Q2 | Audio engine + GUI toolkit | **Answered for the engine: Python + `sounddevice` (PortAudio).** See §3.7. GUI toolkit still open | ~~Phase B1~~ |
| Q3 | Starter library content | Needs a small, redistributable, self-generated set covering ≥3 moods and an energy range | Phase B1 exit |
| Q4 | Whether YuE2 ever runs remote | Decide after Phase B0 measurement, not before | Phase B4 |

---

## 12. Document map

| Document | Role |
|---|---|
| **PLAN.md (this file)** | Build order, hardware, runtime topology. Authoritative for *how* |
| `PRODUCT.md` | User journeys and acceptance criteria. Authoritative for *what the product is* |
| `ARCHITECTURE.md` | Service semantics, asset model, scheduling. Authoritative for *contract meaning* |
| `LAYA_DATA.md` | Dataset sources, labeling, evaluation, provenance. Authoritative for *data policy* |
| `DATASETS.md` | **Verified** dataset status, licences and what each one can and cannot teach. Supersedes `LAYA_DATA.md` where they disagree |
| `FINETUNE.md` | The fine-tune runbook: measured facts, bugs found, promotion gates |
| `VISUAL_DIRECTION.md` | Pixel-art GUI and Vexa character brief. Authoritative for *look* |
| `IDEAS.md` | Optional experiments. Authoritative for *what comes after the core* |

---

## 13. Verified sources

All accessed 2026-10-02.

- YuE2-3B model card — VRAM table, speed, hardware requirements:
  `https://huggingface.co/m-a-p/YuE2-3B`
- YuE2 generation workflow:
  `https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/generation-and-covers.md`
- Laya repository — architecture, checkpoints, params, license:
  `https://github.com/NandhaKishorM/laya`
- Laya fine-tuning — recipe, calibration, known sharp edges:
  `https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md`
- Laya evaluation harness — dataset format:
  `https://github.com/NandhaKishorM/laya/blob/main/docs/evals.md`
- `yue2.cpp` GGUF backend — Q8_0 VRAM figures, module-eviction design, quantization floor,
  plan/synth/transcribe tools:
  `https://huggingface.co/Serveurperso/YuE2-GGUF`
- `audio.cpp` GGUF backend — measured peak VRAM per quant level on RTX 5090:
  `https://huggingface.co/audio-cpp/Yue2-3B-GGUF`
- Practical GGUF integration notes — runtime boundary, WAV→FLAC handling, per-song subprocess,
  and which capabilities each backend does *not* expose:
  `https://github.com/vrgamegirl19/Yue2_Studio/blob/main/docs/gguf.md`