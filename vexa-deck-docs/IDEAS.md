# Ideas backlog

These ideas are optional experiments. The continuous-playback core and the three primary user journeys should work before they expand the product.

## Near-term ideas

### A visible musical horizon

Show the next three likely sections as a small timeline: **playing**, **committed**, and **possible**. A user can steer the possible future without an abrupt change to what is already playing. Generation jobs feed this horizon but cannot silently replace committed audio.

### A transition audition button

Allow the listener to hear a 10–20-second private preview of the next proposed transition while the public/master output continues. This creates a practical way to collect human preference data for Laya.

### Energy arc editor

Let the user draw a rough curve—calm opening, gradual rise, short peak, soft finish. The planner converts it into section targets; Laya chooses local actions that move toward the curve without violating compatibility rules.

### Explain the next move

Expose short, factual reasons such as “instrumental intro, compatible tempo, starts after the current 8-bar phrase.” These should be derived from recorded features and constraints, not invented after the fact.

### Prepared theme packs

Create a self-contained pack containing approved audio, feature manifests, artwork, and a set brief. A pack can be imported locally and reviewed before playback. Distribution must include the applicable license and provenance for every asset.

## Longer-term experiments

### Stem-aware performance

Perform bass swaps, vocal cutouts, and percussion-layer transitions when stem quality is sufficient. The scheduler should know which stems derive from the same source family and prefer proven combinations.

### Audience steering

In a shared room, let listeners request an energy or mood direction without directly controlling transport. Aggregate requests at a slower cadence to avoid rapid oscillation. The host remains able to override.

### Adaptive generation budget

Generate only what the current library is missing: an instrumental bridge, a lower-energy opener, or a compatible ending. Estimate usefulness from gaps in the planned energy arc and asset inventory, then schedule work with a GPU budget.

### Seamless local-to-remote worker handoff

Use the same versioned job contract for a local GPU and a user-operated remote YuE2 worker. The live audio engine continues locally. A disconnected worker merely leaves a generation request pending.

### DJ style profiles

Allow separate profiles for gentle ambient blending, club-style phrase transitions, or experimental stem layering. Treat each profile as a policy and evaluation setting, not merely a color theme. Collect separate listening judgments when styles differ.

### Share a reproducible set plan

Export the decisions, requests, and asset references for a set. Another user with the same permitted assets can replay or edit the plan. Do not bundle model weights or third-party audio automatically.

## Experiments to reject quickly if they fail

- **Unrestricted stem recombination:** likely to expose key, phrase, and separation artifacts unless candidate compatibility is tightly constrained.
- **Model inference inside the audio callback:** conflicts with deterministic timing and continuity goals.
- **Always generating the next segment:** wastes GPU time and creates a hard dependency on generation speed; generate toward known library gaps instead.
- **Training Laya solely from synthetic rules:** may reproduce the rules without improving listening quality. Compare against human preference data on unseen assets.
