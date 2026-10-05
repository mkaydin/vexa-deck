# Video renderer attribution

The pixel fragment shader and eleven upstream palettes are adapted from
[collidingScopes/video-to-pixel-art](https://github.com/collidingScopes/video-to-pixel-art),
revision `773cbdea04cae8a3e87d2f273b1c2e5a41851085`, MIT, copyright 2024 Alan Ang.
Changes: palette uniforms, VEXA palette, block-centered sampling, and Bayer
thresholds aligned to logical pixel cells.

The ASCII mode follows the luminance/character mapping and user-text controls in
[collidingScopes/ascii](https://github.com/collidingScopes/ascii), revision
`af6b92f7c48e2b46f7f29d18ba8409a0f93900ff`, under the same MIT license. Its brightness
ramp is retained. VEXA's implementation downsamples directly to the character
grid, caches colors, uses source-frame callbacks and caps processing FPS.

The MIT notice is preserved in LICENSE.txt. Upstream page UI, analytics, ads,
dat.gui, mp4-muxer and sample footage are not included. Controls are native Qt;
only local video display and effects run inside an embedded QtWebEngine view.
