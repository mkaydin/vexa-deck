"""Audit measured vibe sidecars before they are used to select tracks.

Usage: uv run --no-sync python tools/audit_analysis.py [--library PATH]

Reports coverage, provenance, missing values, and descriptor distributions. Exits nonzero when a
sidecar is stale or malformed. A plausible-looking default must not enter a DJ decision menu.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from extract_vibe import EXTRACTOR_VERSION, LIBRARY, _sha256


def audit(library: Path) -> tuple[dict[str, object], list[str]]:
    wavs = sorted(library.glob("*.wav"))
    errors: list[str] = []
    rhythms: Counter[str] = Counter()
    alignments: list[float] = []
    tuning: list[float] = []
    missing_tuning = 0

    for wav in wavs:
        manifest_path = wav.with_suffix(".json")
        sidecar_path = wav.with_suffix(".vibe.json")
        if not manifest_path.exists() or not sidecar_path.exists():
            errors.append(f"{wav.name}: missing manifest or vibe sidecar")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{wav.name}: cannot read metadata: {exc}")
            continue

        content_hash = _sha256(wav)
        if manifest.get("content_sha256") != content_hash:
            errors.append(f"{wav.name}: manifest hash does not match audio")
        if sidecar.get("source_sha256") != content_hash:
            errors.append(f"{wav.name}: vibe hash does not match audio")
        if sidecar.get("extractor_version") != EXTRACTOR_VERSION:
            errors.append(f"{wav.name}: outdated vibe extractor version")

        alignment = sidecar.get("onset_grid_alignment")
        if alignment is not None:
            if (
                not isinstance(alignment, (int, float))
                or not math.isfinite(alignment)
                or not 0 <= alignment <= 1
            ):
                errors.append(f"{wav.name}: invalid onset alignment")
            else:
                alignments.append(float(alignment))
        elif sidecar.get("rhythm") != "unknown":
            errors.append(f"{wav.name}: missing alignment without unknown rhythm")

        hz = sidecar.get("tuning_hz")
        if hz is None:
            missing_tuning += 1
        elif not isinstance(hz, (int, float)) or not math.isfinite(hz) or not 400 <= hz <= 480:
            errors.append(f"{wav.name}: invalid tuning frequency")
        else:
            tuning.append(float(hz))

        for name in ("sub_bass_fraction", "presence_fraction"):
            value = sidecar.get(name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                errors.append(f"{wav.name}: invalid {name}")
        rhythms[str(sidecar.get("rhythm", "missing"))] += 1

    report: dict[str, object] = {
        "audio_files": len(wavs),
        "valid": not errors and bool(wavs),
        "errors": len(errors),
        "rhythm_counts": dict(sorted(rhythms.items())),
        "measured_alignment": len(alignments),
        "alignment_median": round(statistics.median(alignments), 3) if alignments else None,
        "alignment_range": (
            [round(min(alignments), 3), round(max(alignments), 3)] if alignments else None
        ),
        "measured_tuning": len(tuning),
        "missing_tuning": missing_tuning,
        "tuning_range_hz": [round(min(tuning), 2), round(max(tuning), 2)] if tuning else None,
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=LIBRARY)
    args = parser.parse_args()
    report, errors = audit(args.library)
    print(json.dumps(report, indent=2, sort_keys=True))
    for error in errors[:20]:
        print(error)
    if len(errors) > 20:
        print(f"... and {len(errors) - 20} more errors")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
