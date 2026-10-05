# VEXA//DECK

**An open-source, AI-assisted DJ that starts playing immediately and composes its future in the background.**

VEXA//DECK is a desktop-first music system with a dark ASCII cyberpunk interface and a local looping video panel with Normal, Pixel and ASCII modes. A listener describes an atmosphere—"a quiet rain-soaked jazz bar," "warm house that becomes more energetic," or "keep the rhythm but remove the vocals"—and the system builds a continuous set from prepared musical assets. It can also prepare a themed library before playback. While a set is running, new requests can change the direction of the mix; assets that do not exist yet are generated asynchronously and introduced when ready.

This is a **non-commercial project**. The application source can be published openly, while model weights, generated audio, and third-party datasets remain separate artifacts governed by their own terms. The application should work without a proprietary planning service by providing a local/rule-based planning adapter; GPT-6 Sol is an optional planner, not a requirement for basic playback.

## The experience

1. **Play now.** Enter a mood or theme. VEXA selects a compatible prepared bed or track, starts audio, then builds the next few transitions.
2. **Prepare a set.** Enter a theme and target duration. YuE2 generates candidates in advance; analysis and review turn them into a searchable library.
3. **Steer a live set.** Request a change while music plays. Available assets are scheduled at a musically appropriate boundary. Missing assets enter a generation queue while playback continues.
4. **Take control.** The user can approve, skip, lock, or manually override the next transition at any time.

## Design principles

- **Audio continuity is the highest priority.** A model timeout or failed generation must not interrupt the stream.
- **Models make bounded decisions.** The playback engine, beat alignment, gain limits, and transition deadlines are deterministic.
- **The system knows what is ready.** Only analyzed and verified assets can be selected for immediate playback.
- **A request has an honest status.** The UI distinguishes `applied now`, `scheduled`, `generating`, `ready`, and `unable to fulfill`.
- **Every choice is reproducible.** Store the user request, selected assets, model versions, action, timing, and outcome.
- **The user owns the set.** Manual DJ controls and local library management remain available.

## Major components

| Component | Responsibility |
|---|---|
| Audio engine | Continuous deck playback, looping, beat sync, pitch/tempo, EQ, gain, crossfades, output protection |
| Asset library | Audio files, loops, sections, optional separated stems, metadata, provenance, readiness |
| Analysis pipeline | BPM, beat grid, key, structure, loudness, vocal presence, quality checks |
| Laya decision service | Fast selection among a small set of feasible next actions |
| Planner adapter | Interprets user intent and shapes the set; GPT-6 Sol is one optional implementation |
| YuE2 worker | Generates new songs/arrangements in the background or during preparation |
| GUI | Local video styles/settings, decks, library, upcoming transitions, request and generation state |

## Important model boundaries

YuE2 publicly exposes song-level stereo generation and symbolic planning. It is **not** a native, guaranteed-isolated drum/bass/vocal stem generator. Stems in this project are derived by a separate source-separation step and must be quality-checked. YuE2's published RTX 4090 benchmark reports roughly 71 seconds to generate about 215 seconds of audio; that is useful for background preparation, but does not establish subsecond first-audio latency. Laya is a text/state decision model, not an audio renderer. GPT-6 Sol does not accept or emit audio in its published model specification, so it receives structured music metadata rather than the live waveform. [YuE2 model card](https://huggingface.co/m-a-p/YuE2-3B), [YuE2 generation workflow](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/generation-and-covers.md), [Laya repository](https://github.com/NandhaKishorM/laya), [GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol).

## Documentation map

- [Product specification](PRODUCT.md): user journeys, behavior, interface, and feature priorities.
- [Technical architecture](ARCHITECTURE.md): runtime services, data model, scheduling, interfaces, and failure handling.
- [Laya and data plan](LAYA_DATA.md): dataset research, labeling, fine-tuning, evaluation, and provenance.
- [Roadmap](ROADMAP.md): implementation phases, acceptance criteria, and risks.
- [Visual direction](VISUAL_DIRECTION.md): dark ASCII cyberpunk GUI and local video modes.
- [Ideas backlog](IDEAS.md): distinctive features and experiments beyond the initial release.

## Proposed repository layout

```text
vexa-deck/
  README.md
  docs/
  apps/desktop/             # GUI and desktop packaging
  crates/audio-engine/      # or an equivalent native real-time module
  services/orchestrator/
  services/laya/
  workers/yue2/
  packages/contracts/       # versioned events and schemas
  data/examples/            # small redistributable fixtures, no model weights
  tests/
```

The runtime may be local-only for the first release. The folder structure expresses boundaries, not a mandatory programming language choice.
