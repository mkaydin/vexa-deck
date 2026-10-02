# Visual and interaction direction

## Identity

**Display name:** VEXA//DECK  
**Repository slug:** `vexa-deck`  
**Character:** DJ Vexa, a high-density pixel-art woman performing behind a pair of decks.

The project should look like a playable nighttime music console, not a generic analytics dashboard. The character supplies personality and feedback, while the actual controls and audio state remain easy to read.

## Screen concept

```text
┌──────────────────────────────────────────────────────────────┐
│ VEXA//DECK        SET: RAIN-SOAKED JAZZ       ● LIVE  01:24 │
├──────────────┬───────────────────────┬───────────────────────┤
│ THEME / CHAT │  DJ VEXA PIXEL STAGE  │ READY LIBRARY         │
│ new request  │  reactive lights      │ tracks / loops / stems│
│ energy arc   │  decks in foreground  │ filters + previews    │
├──────────────┴───────────────────────┴───────────────────────┤
│ DECK A  waveform    TRANSITION TIMELINE    waveform  DECK B  │
│ cue · EQ · tempo    next phrase / action    cue · EQ · tempo  │
├──────────────────────────────────────────────────────────────┤
│ GENERATION: 2 queued · 1 validating  |  MASTER  |  OVERRIDE  │
└──────────────────────────────────────────────────────────────┘
```

On a smaller window, prioritize the two decks, request entry, and next transition; move the library and generation queue into collapsible panels. A narrow character animation must never conceal transport controls.

## Character behavior

- **Idle:** breathing and subtle head movement.
- **Playing:** hands move in time with the beat; movement intensity follows energy rather than every audio transient.
- **Transition pending:** Vexa reaches toward the relevant deck; the timeline clearly shows the scheduled bar.
- **Generation running:** a small console or holographic cassette displays progress; Vexa continues DJing with existing music.
- **User override:** the character visibly yields control without slowing the manual action.
- **Error:** calm, specific status text replaces celebratory animation.

These are product animation ideas, not claims that the current portrait is a finished sprite sheet. The existing portrait can guide face, hairstyle, and palette; a later animation pass needs consistent front/side poses, hands, equipment geometry, and frame dimensions.

## Art direction

Use authentic pixel clusters with a stable virtual resolution and integer scaling where possible. Palette: near-black violet, deep plum, magenta, electric blue, and small cyan accents. Reserve the brightest colors for active audio states and actionable controls. The stage can contain cables, equalizer lights, rainy city windows, and signage, but text and waveforms should remain legible. Avoid continuous flashing; respect reduced-motion settings.

The reference-inspired portrait establishes Vexa's long dark hair, tilted expression, jewelry, and violet lighting. Build the production sprite as an original consistent asset rather than repeatedly downsampling a photograph. Store portrait/key art separately from runtime sprites.

## Interaction details

- Show `READY`, `SCHEDULED AT BAR 129`, or `GENERATING` beside each request.
- On the transition timeline, make the current beat, next phrase, committed action, and fallback visible.
- Preview an asset without loading it into the live deck.
- Let users pin an asset, forbid vocals, or set a maximum energy and see those constraints reflected in candidate choices.
- Keep transport, volume, and emergency stop accessible even while model services are busy.

## Accessibility

Provide keyboard controls, text labels in addition to color, readable type at common desktop sizes, a low-motion option, and a non-animated status view. Pixel aesthetics should never make controls ambiguous.
