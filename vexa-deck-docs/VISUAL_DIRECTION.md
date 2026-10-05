# Visual and interaction direction

## Identity

**Display name:** VEXA//DECK  
**Repository slug:** `vexa-deck`  
**Center:** a flexible local video player with Normal, Pixel and ASCII modes. Old character assets are removed.

Use a dark ASCII cyberpunk music console with clear controls and audio state.

## Screen concept

```text
V E X A // D E C K                 :: AFTER_HOURS / RADIO ::
+------------------+ +--------------------+ +------------------+
| [01] // SESSION  | |                    | | [02] // NEXT     |
| theme input_     | |                    | | ready record     |
| [START THE SET]  | |                    | | event history    |
| generation      | |                    | | [LIKE] [DISLIKE] |
| [MUSIC DEPOT]    | |                    | | [STOP PLAYBACK]  |
+------------------+ +--------------------+ +------------------+
                     +--------------------+
                     | FFT::32 .###..##.. |
                     +--------------------+
                     +---------+ +--------+
                     | DECK A  | | DECK B |
                     | [===..] | | [==..] |
                     +---------+ +--------+
```

On a smaller window, prioritize the two decks, request entry and transport controls;
move secondary information into collapsible panels.

## Center panel

Keep the center blank until a user uploads local video. Above it, provide Upload, the three
style choices and Settings. Videos loop silently, independently of the DJ music transport.
The settings dialog manages the repeating video list and live pixel/ASCII controls.
Fill mode uses the allocated area without gutters and crops the video; Fit preserves the
full source frame. Keep both choices available. Save preferences and local file paths.
Use local assets and renderer code; no remote pages, analytics or advertisements.
Pause video for reduced motion and while hidden/minimized. Cap effect processing FPS.
The window can resize down to 720 × 700 for narrow desktop tiles. Below 1000 pixels wide,
hide the header subtitle and place transport status and action buttons on separate rows.
Place both deck readouts below it. The 160-pixel spectrum strip spans the full console
beneath the main panels and above transport controls, with nine segments per band.
In short windows, hide quick-select presets so the theme input stays readable.

## Art direction

Palette: near-black blue, dark slate, muted teal borders, cyan, violet and sage accents.
Use monospace typography, `+`, `-` and `|` panel borders, and bracketed buttons.
The live spectrum uses `#` and `.`; deck progress uses `=` and `.` inside brackets.
Use installed DejaVu Sans Mono, Noto Sans Mono or Liberation Mono, with the system
fixed-width font as a fallback. Theme input remains a native Unicode text control.
Reserve bright colors for audio states and actionable controls. Respect reduced motion.

## Interaction details

- Use a dark slate title bar with the app icon and monospace window title. Minimize and
  maximize/restore icons are cyan; close is violet, with restrained hover and keyboard-focus
  outlines. Share this chrome across the main window, depot and diagnostics.
- Delegate title-bar dragging and six-pixel edge/corner resizing to the compositor using
  [Qt system move/resize](https://doc.qt.io/qt-6/qwindow.html#startSystemMove).
  Double-click the title bar to maximize or restore.
- Show `READY`, `SCHEDULED AT BAR 129`, or `GENERATING` beside each request.
- On the transition timeline, make the current beat, next phrase, committed action, and fallback visible.
- Preview an asset without loading it into the live deck.
- Let users pin an asset, forbid vocals, or set a maximum energy and see those constraints reflected in candidate choices.
- Keep transport, volume, and emergency stop accessible even while model services are busy.

## Accessibility

Provide keyboard controls, text labels in addition to color, readable type at common desktop sizes, a low-motion option, and a non-animated status view. ASCII styling should never make controls ambiguous.
