# Implementation roadmap

The phases below are ordered by dependency. Time estimates should be made after measuring the target hardware and choosing the desktop/audio stack.

## Phase 0 — Decisions and evidence

**Deliverables:** target operating systems, audio device assumptions, repository license, model/weight download policy, asset manifest schema, and a small licensed or self-generated starter library.

**Exit criteria:** a clean checkout can read a sample manifest, verify asset hashes, and identify which files may be redistributed. Confirm the YuE2 and Laya runtime versions to be tested.

## Phase 1 — Continuous audio core

Build two decks, a prepared fallback loop, preloading, transport controls, gain/EQ/crossfade, and a master output path. Add beat-aware scheduling and hard limits on transitions. This phase can use hand-picked audio; it should not depend on any model.

**Exit criteria:** a representative 30-minute unattended set has no audio underrun or dead air on target hardware; stopping a planning process does not stop playback. Measure device-to-output latency and transition alignment.

## Phase 2 — Library and ingestion

Implement immutable asset IDs, family/version links, a local index, metadata extraction, quality gates, preview playback, and manual approval. Support sections and approved loops. Optional stem separation runs during ingestion.

**Exit criteria:** corrupt or unanalyzed assets never become `ready`; a user can find assets by theme, BPM, key, energy, vocals, and readiness. Every played asset has provenance.

## Phase 3 — Requests and rule-based DJ

Translate user prompts into structured constraints using a local parser/rules adapter first. Build candidate retrieval, a deterministic feasibility filter, a rule selector, and a session history. Implement `play now`, `prepare`, and live request state machines with fallback behavior.

**Exit criteria:** a request for an absent asset leaves audio uninterrupted and produces a visible queued status. Newer requests cancel or supersede older uncommitted plans. A manual override always wins.

## Phase 4 — YuE2 worker

Add offline and background generation, job priority, versioned outputs, analysis, review, and cache admission. Keep generation on a separate process or worker so GPU memory spikes do not block audio. Add optional remote-worker support only if needed.

**Exit criteria:** failed or canceled jobs do not affect playback; a validated output can enter the library and be scheduled later; stale outputs do not override a newer user request. Record measured generation times and acceptance rates.

## Phase 5 — Laya specialization

Create the annotation format and A/B review tool. Build cue/action datasets, train a baseline Laya checkpoint, calibrate confidence, and compare against the Phase 3 rule policy. Start with shadow mode.

**Exit criteria:** held-out human listening preference improves, invalid actions remain blocked, low-confidence fallback works, and inference completes before scheduling deadlines under expected load.

## Phase 6 — Pixel GUI and release packaging

Build the Vexa stage, deck controls, library, upcoming-action timeline, request panel, and generation queue. Add keyboard access, high-contrast text, reduced-motion behavior, local settings, and model setup instructions. Package without bundling assets or weights whose distribution terms are unclear.

**Exit criteria:** a new user can import or generate approved music, start a set, steer it, inspect pending work, and stop it without using a terminal.

## Risks to test early

| Risk | Early experiment | Mitigation |
|---|---|---|
| YuE2 outputs are difficult to loop or separate cleanly | Generate diverse short candidates and score loop/stem quality | Prefer full tracks and verified sections; quarantine poor stems |
| Generated tracks vary in tempo or phrasing | Analyze beat grids and inspect cue points | Use robust beat maps and reject ambiguous automatic transitions |
| Generation competes with audio for GPU/CPU | Run worker stress tests during playback | Process separation, resource limits, optional remote worker |
| Laya learns shortcuts from synthetic labels | Blindly compare against rule policy on unseen asset families | Human A/B labels, family-level split, hard-case evaluation |
| User requests change faster than generation completes | Simulate conflicting live requests | Versioned intent and commit checks before playback |
| Non-commercial assets are accidentally bundled | Audit package contents and manifest | Keep source, weights, audio, and datasets as separate downloads |

## Initial backlog

1. Choose the audio engine prototype and measure safe buffer sizes on target machines.
2. Define the versioned `AssetManifest`, `SessionState`, `DecisionRequest`, and `ScheduledAction` schemas.
3. Create a small, reviewed starter library covering at least three moods and a range of energies.
4. Build continuous playback and transition scheduling without AI.
5. Build ingestion, quality gates, and a searchable catalog.
6. Implement the three user journeys using a rule policy.
7. Integrate YuE2 generation and the optional Sol planner.
8. Collect human decision labels and fine-tune Laya.
9. Complete the pixel-art GUI and release packaging.
