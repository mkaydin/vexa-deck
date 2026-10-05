# Video art repository analysis for VEXA

Reviewed: 2026-10-04. Scope: source inspection of both upstream repositories and
VEXA's current desktop video slot. No browser performance benchmark or rendered
video comparison was performed. Performance conclusions below are architectural
expectations, not measured frame rates. The application was not changed.

## Reviewed revisions

| Repository | Revision | Latest checked commit date |
| --- | --- | --- |
| [video-to-pixel-art](https://github.com/collidingScopes/video-to-pixel-art) | `773cbdea04cae8a3e87d2f273b1c2e5a41851085` | 2024-12-15 |
| [ascii](https://github.com/collidingScopes/ascii) | `af6b92f7c48e2b46f7f29d18ba8409a0f93900ff` | 2025-05-24 |

Read-only review clones are under `/tmp/vexa-review-video-to-pixel-art` and
`/tmp/vexa-review-ascii`; neither repository was admitted as a project dependency.

## Summary

Both are standalone browser applications that stylize existing video or webcam
frames. Neither generates a character, rigs a model, changes clothing, removes
people, or creates new movements. A filmed DJ retains her actual anatomy and
movements after conversion, which can address the distorted limb placements from
the previous procedural character approach. Clip selection, loop seams, cropping,
and the relationship between gestures and VEXA's actual deck states still need
their own implementation.

For the current dark ASCII interface, the ASCII treatment is the closer visual
match. The pixel shader is the stronger starting point for a live adjustable
filter. For initial integration, prepare both treatments offline and play muted
clips with Qt, reducing runtime processing while YuE2 generates music.

## 1. video-to-pixel-art

### Rendering pipeline

`HTMLVideoElement → WebGL texture → fragment shader → canvas`.

The shader rounds texture coordinates to a pixel grid, reads the video color,
adds a 4×4 Bayer threshold, chooses the nearest color from one of 11 palettes,
then optionally mixes in a Sobel edge highlight. Controls include pixel size
(1–32), dithering strength, palette, edge threshold, intensity and color. Webcam
and uploaded video sources are supported; uploaded output width is capped at
1200 pixels. PNG capture and video export are included.

Source: [pixelShader.js](https://github.com/collidingScopes/video-to-pixel-art/blob/773cbdea04cae8a3e87d2f273b1c2e5a41851085/pixelShader.js).

### Strengths for VEXA

- Processing runs in the fragment shader rather than JavaScript pixel loops.
- Existing filmed motion remains recognizable.
- Palette reduction, block size and edges give useful art direction controls.
- The core filter can be extracted from the page and adapted independently.

### Findings to fix before reuse

1. **Source changes can accumulate animation loops.** `render()` schedules itself
   unconditionally at lines 630–633. Webcam and upload handlers start another
   loop without cancelling the old one. `cleanupVideoSource()` can set
   `currentVideo = null` while `drawScene()` dereferences `currentVideo.paused`.
   Use one owned loop, cancellation and a source-generation token.
2. **Video readiness uses a timer.** Upload playback starts after a fixed one
   second delay, rather than waiting for usable frame data and checking the
   `play()` promise. Metadata handlers also accumulate across uploads.
3. **Startup contains an incorrect timer call.** The last line calls
   `startDefaultVideo()` immediately and passes its result to `setInterval`.
   The state variable `playAnimationToggle` also differs from the defined
   `animationPlayToggle` in the default-start function.
4. **The raster is not tied to a stable art resolution.** Pixel size depends on
   canvas dimensions. Bayer dithering uses output screen pixels rather than
   logical pixel cells, potentially introducing fine texture inside a block.
   Edge color blending can also produce colors outside the selected palette.
5. **Runtime recovery is incomplete.** No WebGL context-loss recovery or
   frame-driven rendering callback was found. The filter runs on animation
   callbacks even when the source video has not produced a new frame.
6. **Export needs hardening.** The exporter uses a separate 30 FPS interval,
   fixed AVC configuration, synthetic frame-number timestamps and an in-memory
   muxer. It lacks encoder capability checks and queue backpressure. Rounded
   encoder dimensions are not explicitly applied to a matching export canvas.
   Encoder instances are flushed but not explicitly closed.

Export source: [canvasVideoExport.js](https://github.com/collidingScopes/video-to-pixel-art/blob/773cbdea04cae8a3e87d2f273b1c2e5a41851085/canvasVideoExport.js).

## 2. ascii

### Rendering pipeline

`HTMLVideoElement → 2D canvas → getImageData → per-cell RGB averages → luminance
→ character selection → fillText on output canvas`.

Controls include resolution, font size, threshold, inversion, two text colors,
background hue/saturation and gradient, randomness, partial effect width, and
two text modes. “Random Text” primarily maps brightness to a fixed glyph ramp;
“User Text” repeats an entered string and varies glyph size with brightness.
There are also randomized column reveals and glyph changes.

Source: [ASCII.js](https://github.com/collidingScopes/ascii/blob/af6b92f7c48e2b46f7f29d18ba8409a0f93900ff/ASCII.js).

### Strengths for VEXA

- Direct visual match for the current terminal-style UI and icon.
- Two-color text supports cyan/violet highlights on the dark background.
- Brightness thresholds can suppress background detail.
- Repeated custom text could incorporate `VEXA`, although making the whole
   portrait from that word needs visual evaluation for face readability.

### Findings to fix before reuse

1. **Substantial work happens on the browser main thread.** Each frame reads the
   full processing canvas, scans pixels inside each cell, allocates temporary
   arrays, and draws text individually. Canvas dimensions are reset each frame.
   This is an avoidable source of CPU load and allocation churn. Downsample to
   the character grid first and use cached glyphs or a shader atlas for live use.
2. **Export duration is tied to rendering callback count.** `videofps = 12`,
   but `loop()` encodes once per animation callback. Timestamps advance by
   `frameNumber / 12`, regardless of elapsed source time. If rendering actually
   runs at 60 callbacks/second, one second of playback produces approximately
   five seconds of encoded timestamps. This is a conditional calculation, not
   an observed benchmark. Use source timestamps or deterministic frame sampling.
3. **Fallback recording shares the wrong encode path.** The MediaRecorder
   fallback sets recording active, while `loop()` still calls the WebCodecs
   encoder function. On a fresh fallback recording, `videoEncoder` has not been
   initialized. Separate capture modes explicitly.
4. **Uploads rely on a fixed two second delay.** There is no empty-file guard,
   prior upload URL revocation or removal of old metadata listeners. Webcam
   capture should also be stopped when switching directly to an upload.
5. **Randomness can obscure faces and shimmer.** Use zero randomness as the
   initial portrait preset, then adjust after seeing real clips.
6. **Not a text-stream exporter.** Output is a raster canvas/video. Actual
   character-grid export is still listed as a TODO. Image upload is also a TODO
   despite the file picker accepting images.
7. **The page contains external services.** `index.html` loads Umami analytics,
   Google advertising, Google Fonts and CDN icons. Video processing itself is
   local; the entire page is not an offline shell. An embedded version should
   package only the needed renderer and local assets.

Page source: [index.html](https://github.com/collidingScopes/ascii/blob/af6b92f7c48e2b46f7f29d18ba8409a0f93900ff/index.html).

## 3. Packaging and licenses

Both root licenses are MIT, copyright 2024 Alan Ang. The checked vendored
`dat.gui` license is Apache-2.0; `mp4-muxer` is MIT. Preserve applicable license
and attribution files if their code is copied. These software notices do not
establish the provenance or reuse permissions of a chosen reference video.

Sources: [pixel license](https://github.com/collidingScopes/video-to-pixel-art/blob/773cbdea04cae8a3e87d2f273b1c2e5a41851085/LICENSE.txt),
[ASCII license](https://github.com/collidingScopes/ascii/blob/af6b92f7c48e2b46f7f29d18ba8409a0f93900ff/LICENSE.txt),
[dat.gui license](https://github.com/collidingScopes/ascii/blob/af6b92f7c48e2b46f7f29d18ba8409a0f93900ff/dat.gui-master/LICENSE),
[muxer license](https://github.com/collidingScopes/ascii/blob/af6b92f7c48e2b46f7f29d18ba8409a0f93900ff/mp4-muxer-main/LICENSE).

Both have vendored dependencies and globally coupled page code rather than a
renderer library API. No top-level application test suite or renderer CI was
found in either checked tree; vendored dependencies do have their own tooling.

## 4. VEXA integration options

Current `apps/desktop/pixel_stage.py` contains an intentionally blank
`VideoViewport` that fills its allocated area. QtMultimedia, QtOpenGLWidgets and
QtWebEngineWidgets are importable in the system Python. Availability alone does
not verify decode acceleration, codecs or WebGL performance on Wayland.

| Option | Benefit | Cost / constraint |
| --- | --- | --- |
| Prepared video loops + QtMultimedia | Stable appearance, low filter cost during generation | Effects are baked; video decoding still consumes resources |
| Local renderer in QtWebEngine | Fastest reuse of the JavaScript controls | Browser process overhead and GPU contention must be measured |
| Native Qt video + shader | More control over composition and resource use | Requires porting the filter and handling frame formats and graphics lifecycle |

### Recommended sequence

1. Select short source clips with suitable idle movement, deck actions and
   expressive moments. Inspect crop and loop seams before conversion.
2. Produce pixel and ASCII previews of exactly the same footage. Use the
   current palette: background `#080c10`, cyan `#63e6d2`, violet `#b194df`.
   Preserve readable facial midtones instead of reducing all colors to neon.
3. Correct timestamps and export dimensions, then make deterministic muted
   loops. Start around 24–30 FPS for source motion; adjust only after visual
   review. A smaller art grid should be enlarged with nearest-neighbor scaling
   for pixel output.
4. Add local video playback to the center panel with an explicit fit/fill
   choice. Filling removes gutters but crops footage when aspect ratios differ;
   use a selected focal point so the face and deck remain visible.
5. Connect an external clip state controller to actual deck loading, active
   deck and transition progress. A generic filmed gesture cannot guarantee
   exact A/B hand contact or exact fader position. Use suitable clips and safe
   cut points, with return to idle when an action clip is unavailable.
6. Keep clip audio muted so VEXA's music engine remains the audio source.
   Measure GUI frame pacing, process CPU/GPU load and audio underruns both with
   generation idle and during YuE2 generation on the RTX 5060 Ti. Pause visual
   updates when minimized and support reduced motion.

The preferred first result is a small, reviewed set of prepared ASCII/pixel
video loops. Live effect controls can follow once concurrent music-generation
and playback measurements demonstrate sufficient headroom.
