# Technical architecture

## Runtime separation

```text
User request ──> Intent/planner ──> Session state ──> Candidate retrieval
                                                │              │
                                                │         Feasibility filter
                                                │              │
Audio output <── Real-time audio engine <── Scheduler <── Laya decision
                    ▲                           │
                    │                           └── Fallback rule policy
              Ready asset cache
                    ▲
             Analysis + approval <── YuE2 worker <── Generation queue
```

The audio callback must never wait for Laya, Sol, YuE2, the database, network, or disk reads. Those systems produce **future scheduled commands**. The engine consumes already-loaded audio and bounded commands through a thread-safe, nonblocking boundary. If a deadline is missed, it continues the current track or a prepared loop.

## Suggested services

| Service | Inputs | Outputs | Failure behavior |
|---|---|---|---|
| Intent/planner | User text, session summary | Structured target and constraints | Retain current target; ask the user only if essential |
| Library/index | Asset manifests, feature vectors | Shortlist of ready assets | Return an empty shortlist and invoke safe fallback |
| Laya selector | State plus small feasible action list | Action ID, score/confidence | Rule policy chooses `continue` or safe transition |
| Scheduler | Action, beat clock, asset readiness | Timed engine command | Drop late command; never execute midphrase by accident |
| YuE2 worker | Generation brief and seed | Audio, score, model/version metadata | Retry within budget or mark failed; playback unaffected |
| Analyzer | Audio and provenance | Beat/key/structure/loudness/quality metadata | Quarantine asset until fixed or reviewed |

## Asset model

Represent a **family** of related materials rather than a flat pile of files:

```text
AssetFamily
  ├─ original YuE2 stereo render
  ├─ revised renders (each a separate version)
  ├─ detected sections and approved loops
  ├─ optional separated stems (estimated, not ground truth)
  └─ analysis and quality reports
```

Each asset manifest should contain `asset_id`, `family_id`, `source_type`, `parent_id`, file hash, codec/sample rate/channels, duration, BPM/beat-grid version, time signature, key estimate, section markers, energy/mood/instrument/vocal tags, integrated loudness and peak, loop points, quality flags, approval state, model/decoder versions, source prompt/seed, and license/provenance. Treat model-produced tags as estimates with confidence, not facts.

An asset becomes `ready` only after decode, duration, loudness, beat-grid, loop-boundary, and audio quality checks pass. The runtime may preload and resample ready assets to a common output format. Avoid placing database or decoder work in the real-time thread.

## Decision contract

The orchestrator creates a small list of feasible actions before invoking Laya. An action is concrete:

```json
{
  "action_id": "transition_to_asset_42",
  "asset_id": "asset_42",
  "entry_bar": 17,
  "start_at_session_bar": 129,
  "fade_bars": 16,
  "tempo_ratio": 1.016,
  "transition_preset": "bass_swap"
}
```

The feasibility filter validates readiness, time-to-load, beat alignment, tempo-change bounds, harmonic policy, vocal collisions, and recent repetition. It always includes a safe `continue_current` or `play_fallback_loop` action. Laya selects an action ID; it never writes arbitrary DSP commands to the audio thread. A second deterministic validator checks the chosen action before scheduling.

## Scheduling and latency

Use musical time for decisions: beats, bars, phrase boundaries, and known cue points. The scheduler must commit a transition only when its assets are loaded before the required deadline. An audio buffer of 256 frames at 48 kHz spans about 5.3 ms; this illustrates why model inference belongs outside the audio callback. Actual output latency depends on the device, buffering, processing, and operating system and must be measured.

Laya's published single-question T4 timing is on the order of tens of milliseconds, which is suitable for ahead-of-time selection. It does not make the full application audio path real-time. YuE2's published full-song generation benchmark is on the order of tens of seconds; queueing and validation add more time. [Laya benchmark](https://github.com/NandhaKishorM/laya), [YuE2 benchmark](https://huggingface.co/m-a-p/YuE2-3B).

## Live-request state machine

```text
received -> interpreted -> matched-ready -> scheduled -> applied
                       └-> missing -> queued -> generating -> validating
                                                    ├-> ready -> scheduled
                                                    └-> failed/review-needed
```

Every request gets an ID and a generation number. A newer request can invalidate an older uncommitted transition; the worker may finish and store its asset, but the scheduler rechecks the current request before use. This prevents a delayed generation result from reversing the listener's latest direction.

## Sound continuity and safety

- Keep at least one approved fallback bed and one compatible transition loop loaded.
- Apply bounded fades, headroom, and a master limiter; analyze clipping after every offline render.
- Do not mix two prominent vocals by default unless explicitly requested.
- Apply tempo/pitch changes with a real-time-capable DSP library and account for its algorithmic delay. [Rubber Band documentation](https://www.breakfastquay.com/rubberband/code-doc/).
- Separate stems during ingestion, not in the audio callback; quality-check leakage and artifacts. [Demucs](https://github.com/hadeskers/demucs-music).
- Log underruns and decisions without blocking the audio callback.

## Deployment modes

**Local core:** audio engine, library, retrieval, Laya, and GUI run on the user's machine. **Optional remote planner:** GPT-6 Sol can turn free-form requests into structured briefs and help plan an energy arc; a local adapter supplies basic operation without it. **YuE2 worker:** local GPU where available, or a user-operated remote worker. Remote jobs must not be required for continuous playback.

## API sketches

- `POST /sessions`: create a set from a theme and initial constraints.
- `POST /sessions/{id}/requests`: submit a live change; return request ID and status.
- `GET /sessions/{id}/state`: current audio state, next transition, generation jobs.
- `POST /jobs/generate`: enqueue a YuE2 brief.
- `POST /assets/{id}/approve`: promote a verified candidate to the ready library.
- Internal `DecisionRequest`: state snapshot, feasible action IDs, deadline, policy version.
- Internal `ScheduledAction`: action ID, commit beat, asset version/hash, fallback ID.

Version the contracts from the start. Exact transport (in-process messages, IPC, or HTTP) is an implementation choice for the prototype.
