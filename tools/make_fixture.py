"""Emit the sample asset manifest used by the Phase 0 exit check.

The fixture is committed JSON, not generated at test time, so the exit criterion is exactly what
it says: *a clean checkout* can read a sample manifest. This script exists so the fixture can be
regenerated when the contract changes, rather than hand-edited into an invalid state.

Usage::

    uv run python tools/make_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

from vexa_contracts import (
    ApprovalState,
    AssetManifest,
    AudioProperties,
    BackendKind,
    BeatGrid,
    KeyEstimate,
    LoopPoints,
    Provenance,
    QualityReport,
    ReadinessState,
    SectionMarker,
    SourceType,
)

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "fixtures" / "asset_manifest.sample.json"


def build() -> AssetManifest:
    """A plausible YuE2 render that has passed every gate and been approved."""
    return AssetManifest(
        asset_id="asset_42",
        family_id="family_8",
        source_type=SourceType.YUE2_RENDER,
        content_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        audio=AudioProperties(
            codec="flac",
            sample_rate_hz=48000,
            channels=2,
            duration_s=214.85,
            integrated_lufs=-9.2,
            true_peak_dbtp=-1.4,
        ),
        beat_grid=BeatGrid(bpm=122.0, grid_version=1, confidence=0.91),
        key=KeyEstimate(value="A minor", confidence=0.82),
        sections=[
            SectionMarker(section_id="intro", kind="intro", start_bar=0, end_bar=16),
            SectionMarker(section_id="build", kind="build", start_bar=16, end_bar=48),
            SectionMarker(section_id="drop", kind="drop", start_bar=48, end_bar=88),
        ],
        loops=[LoopPoints(start_bar=48, end_bar=64, beat_aligned=True)],
        tags={
            "mood": [{"value": "rain-soaked", "confidence": 0.88}],
            "instrument": [{"value": "rhodes", "confidence": 0.71}],
            "energy": [{"value": "0.46", "confidence": 0.9}],
        },
        quality=QualityReport(
            decode_ok=True,
            duration_ok=True,
            loudness_ok=True,
            beat_grid_ok=True,
            loop_boundary_ok=True,
            audio_quality_ok=True,
            flags=[],
            notes={"loudness": "-9.2 LUFS integrated, -1.4 dBTP"},
        ),
        approval=ApprovalState.APPROVED,
        readiness=ReadinessState.READY,
        provenance=Provenance(
            backend=BackendKind.YUE2_CPP,
            model_id="m-a-p/YuE2-3B",
            model_revision="1a96eca688d6ae5d7f0feb88573fec89920fcd19",
            quantization="Q8_0",
            decoder_id="m-a-p/YuE2-Vae",
            source_prompt="jazz bar, brushed drums, upright bass, no guitar",
            seed=831001,
            license="CC-BY-NC-4.0",
            source_hashes=[],
        ),
    )


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    manifest = build()
    TARGET.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {TARGET.relative_to(ROOT)} ({manifest.asset_id}, {manifest.audio.duration_s}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())