"""Run and recover independent track jobs without owning or resetting deck playback."""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from vexa_audio.cues import CueMap
from vexa_contracts import AssetManifest, GenerationSettings
from vexa_yue2.backends import BackendUnavailable

from .audio_evidence import load_evidence, sha256
from .depot import DepotAsset
from .generation import TrackBrief
from .generation_jobs import GenerationJobs
from .track_titles import unique_title


def cue_retry_brief(brief: TrackBrief) -> TrackBrief:
    """Correct a failed rhythmic outro without changing intent, clocks or lyrics."""
    metadata = copy.deepcopy(brief.music_plan or {})
    if metadata.get("cue_retry_correction"):
        return brief
    direction = (
        "Keep the established rhythmic foundation clearly audible and rhythmically "
        "steady throughout the final DJ outro, until the final bar. Preserve the requested "
        "genre, swing, tempo, instrumental palette and instrument exclusions. "
        "Thin melodic layers for mixing, "
        "but retain the rhythmic foundation; finish with a short final-bar release."
    )
    caption = metadata.get("minimax_caption")
    if caption and "\nArrangement\n" in caption:
        prefix, timeline = caption.split("\nArrangement\n", 1)
        sections = list(re.finditer(r"(?m)^\d{1,2}:[0-5]\d-\d{1,2}:[0-5]\d [^\n:]+: ", timeline))
        if sections:
            last = sections[-1]
            metadata["minimax_caption"] = (
                prefix + "\nArrangement\n" + timeline[: last.end()] + direction
            )
    form = metadata.get("sections") or []
    descriptions = (metadata.get("creative_draft") or {}).get("section_descriptions")
    if form and descriptions is not None and form[-1].get("section_id"):
        descriptions[form[-1]["section_id"]] = direction
    metadata["cue_retry_correction"] = {
        "version": 1,
        "reason": "transition_cues_unreliable",
        "direction": direction,
    }
    return replace(
        brief, style=brief.style + " DJ outro correction: " + direction, music_plan=metadata
    )


def recovered_asset(path: Path) -> DepotAsset | None:
    """Recover only a published, admitted render matching its measured cue hash."""
    try:
        manifest = AssetManifest.model_validate_json(path.with_suffix(".json").read_text())
        cue = CueMap.load(path.with_suffix(".cues.json"))
        if not manifest.admissible() or not cue.valid_for(path):
            return None
        if (
            manifest.content_sha256 != cue.source_sha256
            or cue.entry_s is None
            or cue.exit_s is None
        ):
            return None
        if path.with_suffix(".evidence.json").exists():
            evidence = load_evidence(path, source_sha256=manifest.content_sha256)
            if (
                not evidence
                or evidence["status"] != "admitted"
                or not evidence["admission"]["mixable"]
            ):
                return None
        return DepotAsset(manifest, path, frozenset(), {})
    except (OSError, ValueError, KeyError, TypeError):
        return None


class GenerationRunner:
    def __init__(
        self,
        jobs: GenerationJobs,
        batch_id: str,
        library: Path,
        *,
        planner,
        renderer,
        admitted,
        event,
        cancel,
    ):
        self.jobs, self.batch_id, self.library = jobs, batch_id, library
        self.planner, self.renderer = planner, renderer
        self.admitted, self.event, self.cancel = admitted, event, cancel

    def emit(self, kind: str, slot: int | None = None, **details):
        self.event(
            {
                "event": kind,
                "batch_id": self.batch_id,
                "track": slot + 1 if slot is not None else 0,
                "jobs": self.jobs.tracks(self.batch_id),
                "failed_attempts": self.jobs.failure_count(self.batch_id),
                **details,
            }
        )

    def interrupted(self):
        self.jobs.mark_batch(self.batch_id, "interrupted")
        self.emit("generation_interrupted")

    @staticmethod
    def evidence_reference(path: Path):
        report = path.with_suffix(".evidence.json")
        return {"path": str(report), "sha256": sha256(report)} if report.is_file() else None

    @staticmethod
    def quarantine(path: Path):
        manifest = path.with_suffix(".json")
        if manifest.exists():
            manifest.replace(path.with_suffix(".stale.json"))

    def run(self):
        batch = self.jobs.batch(self.batch_id)
        snapshot = batch["snapshot"]
        controls = GenerationSettings.model_validate(snapshot.get("controls", {}))
        self.jobs.mark_batch(self.batch_id, "running")
        for row in self.jobs.tracks(self.batch_id):
            slot = row["slot"]
            if self.cancel.is_set():
                self.interrupted()
                return
            if row["state"] in {"ready", "rendering"} and row["asset_path"]:
                asset = recovered_asset(Path(row["asset_path"]))
                if asset is not None:
                    self.admitted(asset, slot)
                    self.jobs.update(self.batch_id, slot, state="ready", error=None)
                    self.emit("generation_recovered", slot, asset_id=asset.manifest.asset_id)
                    continue
                reference = self.evidence_reference(Path(row["asset_path"]))
                if reference:
                    self.jobs.attempt(
                        self.batch_id,
                        slot,
                        "recovery",
                        "failed",
                        "previous render could not be recovered from published audio "
                        "and cue hashes",
                        evidence=reference,
                    )
                if row["state"] == "ready":
                    self.jobs.update(
                        self.batch_id,
                        slot,
                        state="rejected",
                        error="published asset is missing or changed; explicit retry required",
                    )
                    self.emit("generation_track_rejected", slot)
                    continue
                self.jobs.update(self.batch_id, slot, state="plan_ready")
            if row["state"] in {"rejected", "unavailable"}:
                continue
            brief = TrackBrief(**row["brief"]) if row["brief"] else None
            while brief is None and row["plan_attempts"] < controls.plan_attempts:
                if self.cancel.is_set():
                    self.interrupted()
                    return
                row["plan_attempts"] += 1
                self.jobs.update(
                    self.batch_id,
                    slot,
                    state="planning",
                    plan_attempts=row["plan_attempts"],
                    error=None,
                )
                self.emit("generation_planning", slot)
                try:
                    brief = self.planner(
                        batch["theme"],
                        slot,
                        snapshot["generator"],
                        settings={
                            **snapshot["writer"],
                            "seed": (controls.seed + 100003 * row.get("planning_revision", 0))
                            % 2147483647,
                        }
                        if "controls" in snapshot
                        else snapshot["writer"],
                        cache_path=self.jobs.path.parent
                        / "drafts"
                        / self.batch_id
                        / f"{slot}.json",
                        cancel=self.cancel,
                        **({"controls": snapshot["controls"]} if "controls" in snapshot else {}),
                    )
                    if self.cancel.is_set():
                        self.interrupted()
                        return
                    seen = {
                        (other["brief"] or {}).get("title", "").casefold()
                        for other in self.jobs.tracks(self.batch_id)
                        if other["slot"] != slot
                    }
                    brief = replace(brief, title=unique_title(brief.title, seen))
                    self.jobs.update(
                        self.batch_id, slot, brief=asdict(brief), state="plan_ready", error=None
                    )
                    self.jobs.attempt(self.batch_id, slot, "planning", "accepted")
                    self.emit("generation_planned", slot, origin=brief.origin)
                except Exception as exc:
                    if self.cancel.is_set():
                        self.interrupted()
                        return
                    self.jobs.update(self.batch_id, slot, error=str(exc))
                    self.jobs.attempt(self.batch_id, slot, "planning", "failed", str(exc))
                    self.emit("generation_plan_retry", slot, reason=str(exc))
                    if isinstance(exc, BackendUnavailable):
                        self.jobs.update(self.batch_id, slot, state="unavailable")
                        self.jobs.mark_batch(self.batch_id, "unavailable")
                        self.emit("generation_batch_failed", reason=str(exc))
                        return
            if brief is None:
                self.jobs.update(self.batch_id, slot, state="rejected")
                self.emit("generation_track_rejected", slot)
                continue
            # Explicit retry/resume may load a plan saved by an older app. Keep its
            # failure feedback even when the UI resets that slot's attempt budget.
            for previous in reversed(self.jobs.attempts(self.batch_id, slot)):
                if previous["stage"] == "planning" and previous["outcome"] == "accepted":
                    break
                if previous["outcome"] == "failed" and "transition_cues_unreliable" in (
                    previous["error"] or ""
                ):
                    corrected = cue_retry_brief(brief)
                    if corrected is not brief:
                        brief = corrected
                        self.jobs.update(self.batch_id, slot, brief=asdict(brief))
                        self.emit("generation_retry_adjusted", slot, reason="previous cue failure")
                    break
            success = False
            while row["render_attempts"] < controls.render_attempts:
                if self.cancel.is_set():
                    self.interrupted()
                    return
                row["render_attempts"] += 1
                render_id = (
                    "live-"
                    + uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{self.batch_id}:{slot}:{row['render_attempts']}"
                        + (f":revision:{row['revision']}" if row.get("revision") else ""),
                    ).hex[:12]
                )
                self.jobs.update(
                    self.batch_id,
                    slot,
                    state="rendering",
                    render_attempts=row["render_attempts"],
                    asset_id=render_id,
                    asset_path=str(self.library / f"{render_id}.wav"),
                    error=None,
                )
                self.emit(
                    "generation_started",
                    slot,
                    title=brief.title,
                    attempt=row["render_attempts"],
                    origin=brief.origin,
                )
                asset = None
                try:
                    asset = self.renderer(
                        batch["theme"],
                        self.library,
                        duration_s=brief.duration_s,
                        energy=brief.energy,
                        style=brief.style,
                        prompt_origin=brief.origin,
                        lyrics=brief.lyrics,
                        title=brief.title,
                        cancel=self.cancel,
                        generator=snapshot["generator"],
                        preset=snapshot["preset"],
                        render_id=render_id,
                        **(
                            {
                                "seed": (
                                    controls.seed
                                    + slot
                                    + 1009 * (row["render_attempts"] - 1)
                                    + 100003 * row.get("revision", 0)
                                )
                                % 2147483647,
                                "render_timeout_s": controls.rendering_timeout_s,
                            }
                            if "controls" in snapshot
                            else {}
                        ),
                        **({"music_plan": brief.music_plan} if brief.music_plan else {}),
                        **(
                            {"runtime_components": snapshot["runtime_components"]}
                            if "runtime_components" in snapshot
                            else {}
                        ),
                    )
                    if self.cancel.is_set():
                        self.quarantine(asset.path)
                        self.interrupted()
                        return
                    self.admitted(asset, slot)
                    self.jobs.update(
                        self.batch_id,
                        slot,
                        state="ready",
                        error=None,
                        asset_id=asset.manifest.asset_id,
                        asset_path=str(asset.path),
                    )
                    self.jobs.attempt(
                        self.batch_id,
                        slot,
                        "rendering",
                        "accepted",
                        evidence=self.evidence_reference(asset.path),
                    )
                    self.emit("generated", slot, asset_id=asset.manifest.asset_id)
                    success = True
                    break
                except Exception as exc:
                    if self.cancel.is_set():
                        self.quarantine(asset.path if asset else self.library / f"{render_id}.wav")
                        self.interrupted()
                        return
                    self.jobs.update(self.batch_id, slot, error=str(exc))
                    self.jobs.attempt(
                        self.batch_id,
                        slot,
                        "rendering",
                        "failed",
                        str(exc),
                        evidence=self.evidence_reference(self.library / f"{render_id}.wav"),
                    )
                    self.emit(
                        "generation_failed", slot, reason=str(exc), attempt=row["render_attempts"]
                    )
                    if "transition_cues_unreliable" in str(exc):
                        corrected = cue_retry_brief(brief)
                        if corrected is not brief:
                            brief = corrected
                            self.jobs.update(self.batch_id, slot, brief=asdict(brief))
                            self.emit(
                                "generation_retry_adjusted",
                                slot,
                                reason="retain measured rhythmic support through the DJ outro",
                                next_attempt=row["render_attempts"] + 1
                                if row["render_attempts"] < controls.render_attempts
                                else None,
                            )
                    if isinstance(exc, BackendUnavailable):
                        self.jobs.update(self.batch_id, slot, state="unavailable")
                        self.jobs.mark_batch(self.batch_id, "unavailable")
                        self.emit("generation_batch_failed", reason=str(exc))
                        return
            if not success:
                self.jobs.update(self.batch_id, slot, state="rejected")
                self.emit("generation_track_rejected", slot)
        rows = self.jobs.tracks(self.batch_id)
        complete = all(row["state"] == "ready" for row in rows)
        self.jobs.mark_batch(self.batch_id, "completed" if complete else "partial")
        errors = [row["error"] for row in rows if row["error"]]
        self.emit(
            "generation_batch_complete" if complete else "generation_batch_failed",
            reason=errors[-1] if errors else None,
        )
