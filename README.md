<img src="assets/icons/vexa-icon-256.png" width="128" align="right" alt="VEXA//DECK">

# VEXA//DECK

**An open-source, AI-assisted DJ that starts playing immediately and composes its future in the background.**

VEXA//DECK is a desktop music system with a pixel-art cyberpunk interface and a character named
Vexa behind the decks. You describe an atmosphere — *"a quiet rain-soaked jazz bar"*, *"warm house
that becomes more energetic"* — and it builds a continuous set from prepared musical assets.

## The one rule everything is built around

> Audio continuity is the highest priority. A model timeout or failed generation must not
> interrupt the stream.

Every architectural decision follows from this. Models make **bounded** decisions: the playback
engine, beat alignment, gain limits and transition deadlines stay deterministic. A selector picks
an action from a menu that has already been validated; it can never write a DSP command. If any
model is slow, dead, or wrong, the set keeps playing.

## Current state

**Backend foundations.** Versioned contracts, the orchestrator's deterministic decision path, and
pluggable generation backends. No GUI yet.

| Component | Status |
|---|---|
| `packages/contracts` | 24 versioned schemas, JSON Schema export, invariants enforced structurally |
| `services/orchestrator` | Feasibility filter, rule policy, beat-clock scheduler, request state machine, HTTP API |
| `services/planner` | OpenAI-compatible client + local rule-based fallback |
| `services/laya` | Dataset schema, family-level splits, eval/training converters, bounded-choice adapter, RLCD fine-tune loop |
| `workers/yue2` | BS.1770-4 loudness, true peak, tempo/key/structure analysis, readiness gates |
| `services/audio` | **Two decks, crossfade, master limiter.** Runs on the host; nothing plays without it |

**The audio engine plays.** 7787 callbacks over 90 s with **zero underruns**, worst callback
0.61 ms of an 11.6 ms budget. Run `uv run --no-sync python tools/audio_smoke.py --seconds 1800`
for the full 30-minute soak.

**The Laya fine-tune pipeline is verified running on the RTX 5060 Ti.** Loss 0.9210 → 0.5409,
peak VRAM 7.81 GiB of 15.5 GiB. Those numbers came from rule-generated labels and prove the
machinery, not the quality — real labels come from A/B human review.

Phase gates, hardware allocation, and the reasoning behind each choice are in
**[`vexa-deck-docs/PLAN.md`](vexa-deck-docs/PLAN.md)**. Fine-tuning is in
**[`vexa-deck-docs/FINETUNE.md`](vexa-deck-docs/FINETUNE.md)**.

## Quick start

```bash
uv sync --extra core          # Python 3.12, uv-managed
uv run pytest                 # contract invariants
uv run ruff check .
```

### Running the stack

```bash
cp .env.example .env      # every value is optional
docker compose up -d
curl localhost:8080/health
curl localhost:8081/health
```

Only the orchestrator, planner, analyzer and job queue run in containers. The Laya service, the
YuE2 worker, the audio engine and the GUI run **on the host**, where the GPUs and the audio device
are — model weights never enter an image, and a container boundary on the realtime audio path
would add latency jitter for nothing. See [`vexa-deck-docs/PLAN.md`](vexa-deck-docs/PLAN.md) §4.

Model weights are never bundled. They are separate downloads under their own terms — YuE2 weights
are CC BY-NC 4.0, Laya is Apache-2.0.

## Documentation

| Document | What it is |
|---|---|
| [PLAN.md](vexa-deck-docs/PLAN.md) | **Build order, hardware, runtime topology.** Start here |
| [PRODUCT.md](vexa-deck-docs/PRODUCT.md) | User journeys and acceptance criteria |
| [ARCHITECTURE.md](vexa-deck-docs/ARCHITECTURE.md) | Service semantics, asset model, scheduling |
| [LAYA_DATA.md](vexa-deck-docs/LAYA_DATA.md) | Dataset sources, labeling, evaluation, provenance |
| [VISUAL_DIRECTION.md](vexa-deck-docs/VISUAL_DIRECTION.md) | Pixel-art GUI and the Vexa character brief |
| [IDEAS.md](vexa-deck-docs/IDEAS.md) | Experiments beyond the core |

## Licence

Source is AGPL-3.0-or-later. Model weights, generated audio, and third-party datasets remain
separate artifacts governed by their own terms — see `PLAN.md` §11.

---

*Vexa is DJing behind the decks in [`character-sheets/`](character-sheets/). Those sheets are
reference art; the animated on-program hero is a later design pass.*