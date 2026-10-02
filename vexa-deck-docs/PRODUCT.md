# Product specification

## Product promise

VEXA//DECK should feel as though a DJ has already started the set while the system works out the next moves. The user describes a musical direction and hears a prepared, compatible asset promptly. The app then maintains a coherent progression using beat-aware transitions, user feedback, and newly generated material.

**Primary audience:** people who want an adaptive background set or an interactive, AI-assisted DJ experience. **Initial platform:** desktop, with local audio output and a local asset library. Network streaming and multi-user rooms are later extensions.

## User journeys

### 1. Start immediately

1. The user enters a prompt, optionally specifying intensity, vocals, genre, language, and duration.
2. The intent parser returns a structured brief. The library retrieves playable candidates.
3. The app starts a verified opening bed/track. If no exact match exists, it clearly identifies the closest prepared option while generating better matches.
4. The planner sets a near-term energy arc. Laya chooses between feasible next actions at phrase boundaries.
5. The UI shows what is playing, the next planned transition, and any background generation.

**Acceptance criterion:** generation is never required before audio can start, provided the installed library contains at least one valid starter asset. The measured start target should be set after a local hardware benchmark; it is not a model guarantee.

### 2. Prepare a theme

1. The user enters a theme, target length, and optional constraints.
2. The planner creates an asset brief: starter, low/medium/high-energy sections, transition-friendly loops, and optional vocal variants.
3. YuE2 jobs produce song candidates. The pipeline analyzes each output, optionally separates stems, creates safe loop regions, and records provenance.
4. Automated checks reject corrupt files, clipping, unusable beat grids, silence, or incompatible length. The user can listen and approve a set before playing.
5. Only approved assets enter the ready library.

### 3. Change a live set

Requests have three classes:

| Class | Example | Expected behavior |
|---|---|---|
| Immediate control | "Lower the volume," "mute vocals" when stems exist | Apply a bounded parameter change or schedule it at the next safe boundary |
| Ready-library change | "Make it darker and faster" when compatible assets exist | Re-plan and schedule a transition from ready assets |
| Generation-dependent change | "Add a new Turkish vocal synthwave section" when absent | Keep playing, enqueue YuE2 work, report progress, introduce only after validation |

When requests conflict, the most recent active user direction supersedes older **uncommitted** plans. Already playing audio is changed through a safe transition. A completed generation job that no longer matches the current direction may remain in the library but should not automatically enter the set.

### 4. Manual control

The user can skip, lock a track, set a cue, change the transition type, disable automatic mixing, and stop generation. User commands outrank automated decisions. The UI should make it obvious which decisions are automatic and which are user-directed.

## Functional requirements

- Two playable decks or equivalent layered playback, plus a looping emergency bed.
- Cue/phrase scheduling aligned to a verified beat grid.
- Tempo adjustment, pitch preservation where practical, gain staging, EQ, and a master limiter.
- Asset search by mood, genre, instrument, vocals, BPM, key, energy, and readiness.
- Background job states: queued, running, analyzing, review-needed, ready, failed, canceled.
- Nonblocking user-request handling and visible time-to-availability category, without inventing an exact ETA.
- Session history containing requests, decisions, asset IDs, timestamps, and overrides.
- Optional GPT-6 Sol integration through a planner interface, with a usable local fallback.

## Deliberate first-release limits

- Focus on prepared tracks and short, verified loops; arbitrary real-time recombination of unrelated stems comes later.
- No claim that a YuE2 revision preserves an unchanged waveform outside an edited section.
- No promise that a new, absent style can be heard instantly.
- No automatic publication or redistribution of downloaded dataset audio or model weights.

## Success measures

Measure these on representative desktop hardware and a held-out asset library:

- Playback start latency and uninterrupted-playback rate.
- Underruns, clipped samples, and large perceived loudness jumps.
- Beat/phrase alignment error at transitions.
- Percentage of live requests correctly classified as immediate, scheduled, or generation-dependent.
- Human blind preference for transitions versus a deterministic baseline.
- Diversity across a 30- to 60-minute set, including repeat rate and energy-arc adherence.
- Generation acceptance rate, average queue time, and percentage of generated assets actually used.

The GUI must report measured behavior, not substitute an animation for a successful audio action.
