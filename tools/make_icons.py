"""Generate the application icon ladder from the canonical portrait.

``character-sheets/vexa-pixel-portrait.png`` is the single source of truth (PLAN.md D1). Every
icon size is derived from it, so re-running this after a portrait revision is the whole workflow —
no size is ever hand-edited.

The portrait is centre-cropped to a square before resizing. Resizing a non-square image to a
square distorts the face; cropping keeps the composition the artist signed off on.

Usage::

    uv run --with pillow python tools/make_icons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "character-sheets" / "vexa-pixel-portrait.png"
OUT = ROOT / "assets" / "icons"

#: Sizes an application icon is actually consumed at.
SIZES = (16, 32, 48, 64, 128, 256, 512, 1024)


def square_crop(image: Image.Image) -> Image.Image:
    """Centre-crop to a square so nothing is stretched."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def main() -> int:
    if not SOURCE.exists():
        print(f"error: canonical portrait missing at {SOURCE}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE) as raw:
        print(f"source: {SOURCE.name} {raw.size} {raw.mode}")
        base = square_crop(raw.convert("RGB"))

    base.save(OUT / "vexa-icon.png", optimize=True)
    for size in SIZES:
        target = OUT / f"vexa-icon-{size}.png"
        base.resize((size, size), Image.LANCZOS).save(target, optimize=True)
        print(f"  wrote {target.relative_to(ROOT)}")

    print(f"\n{len(SIZES) + 1} icons written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())