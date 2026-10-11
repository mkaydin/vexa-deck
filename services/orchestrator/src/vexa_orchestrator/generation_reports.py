"""Read stored track evidence without running models or decoding audio in the host."""

from __future__ import annotations

import json
from pathlib import Path

from vexa_contracts import AssetManifest

from .audio_evidence import load_evidence, sha256
from .generation_v2 import build_plan


def track_report(jobs, batch_id: str, slot: int) -> dict:
    batch = jobs.batch(batch_id)
    row = next((row for row in jobs.tracks(batch_id) if row["slot"] == slot), None)
    if row is None:
        raise ValueError("unknown track slot")
    snapshot = batch["snapshot"]
    brief = row["brief"] or {}
    plan = (brief.get("music_plan") or {}).get("track_plan") or build_plan(
        batch["theme"], slot, snapshot["generator"], controls=snapshot.get("controls")
    ).model_dump()
    manifest = None
    cues = None
    warnings = []
    evidence = None
    if row["asset_path"]:
        path = Path(row["asset_path"])
        if path.resolve().parent != Path(snapshot["library"]).resolve():
            raise ValueError("track report path is outside the batch depot")
        try:
            manifest = AssetManifest.model_validate_json(path.with_suffix(".json").read_text())
            cues = json.loads(path.with_suffix(".cues.json").read_text())
        except (OSError, ValueError):
            warnings.append("Published measurement report is unavailable; values remain unknown.")
        evidence = load_evidence(path, source_sha256=manifest.content_sha256 if manifest else None)
        if path.with_suffix(".evidence.json").exists() and not evidence:
            warnings.append(
                "Audio evidence is invalid or does not match the published content hash."
            )
    requested = {
        "genre": plan["intent"]["genre"],
        "vocal_mode": plan["intent"]["vocal_mode"],
        "language": plan["intent"]["language"],
        "bpm": plan["intent"]["bpm"],
        "duration_s": plan["duration_s"],
        "energy": plan["energy"],
    }
    measured = {
        "duration_s": manifest.audio.duration_s if manifest else None,
        "bpm": manifest.beat_grid.bpm if manifest and manifest.beat_grid else None,
        "beat_confidence": manifest.beat_grid.confidence
        if manifest and manifest.beat_grid
        else None,
        "integrated_lufs": manifest.audio.integrated_lufs if manifest else None,
        "true_peak_dbtp": manifest.audio.true_peak_dbtp if manifest else None,
        "genre": None,
        "vocal_mode": None,
        "language": None,
        "energy": None,
    }
    if evidence:
        final = evidence.get("final") or evidence.get("raw") or {}
        measured["stage"] = "final" if evidence.get("final") else "raw"
        for key in ("duration_s", "integrated_lufs", "true_peak_dbtp"):
            if key in final:
                measured[key] = final[key]
        measured.update(
            rms=final.get("rms"),
            spectral_balance=final.get("spectral_balance"),
            level_curve=final.get("level_curve"),
        )
        if evidence.get("timing"):
            measured.update(
                {key: evidence["timing"].get(key) for key in ("bpm", "beat_confidence")}
            )
    attempts = jobs.attempts(batch_id, slot)
    # History remains usable after retry/replace clears the current asset path.
    history = []
    for attempt in attempts:
        reference = attempt.get("evidence")
        if not reference:
            continue
        report_path = Path(reference["path"])
        if report_path.resolve().parent != Path(snapshot["library"]).resolve():
            warnings.append("Attempt evidence path is outside the depot; ignored.")
            continue
        try:
            if sha256(report_path) != reference["sha256"]:
                raise ValueError("evidence hash changed")
            stored = json.loads(report_path.read_text())
            history.append(
                {
                    "at": attempt["at"],
                    "outcome": attempt["outcome"],
                    "asset_id": stored["asset_id"],
                    "status": stored["status"],
                    "raw_sha256": (stored.get("raw") or {}).get("source_sha256"),
                    "final_sha256": (stored.get("final") or {}).get("source_sha256"),
                    "admission": stored.get("admission"),
                    "corrections": stored.get("corrections"),
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append("Historical attempt evidence is unavailable or changed.")
    return {
        "batch_id": batch_id,
        "slot": slot + 1,
        "state": row["state"],
        "title": brief.get("title"),
        "asset_id": row["asset_id"],
        "error": row["error"],
        "requested": requested,
        "measured": measured,
        "quality": evidence.get("quality")
        if evidence
        else manifest.quality.model_dump()
        if manifest
        else None,
        "cues": cues,
        "plan": plan,
        "brief": brief,
        "writer": snapshot["writer"],
        "controls": snapshot.get("controls"),
        "attempts": attempts,
        "audio_evidence": evidence,
        "evidence_history": history,
        "admission": evidence.get("admission") if evidence else None,
        "warnings": warnings,
        "semantic_evidence": "observation only; uncalibrated similarity is not "
        "a genre/vocal certification",
    }
