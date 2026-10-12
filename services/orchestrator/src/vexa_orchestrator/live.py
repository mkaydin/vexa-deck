"""Host-side continuous DJ runner. All decoding and decisions stay off the audio callback."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vexa_audio.cues import CueMap
from vexa_audio.engine import AudioEngine
from vexa_audio.equalizer import EqualizerSettings
from vexa_audio.loader import PreloadedTrack
from vexa_audio.preview import PreviewRenderer, dsp_identity
from vexa_audio.queue import Command, CommandKind
from vexa_audio.telemetry import spectrum_levels
from vexa_contracts import DeckId, GenerationSettings, MusicalClock, SessionState

from .audio_priority import publish as publish_audio_priority
from .continuity import LatencyHistory, forecast, latency_key
from .depot import Depot, DepotAsset
from .feasibility import FeasibilityFilter, FilterConfig
from .generation import isolated_generate_theme as generate_theme
from .generation_jobs import GenerationJobs
from .generation_runner import GenerationRunner
from .generation_settings import capabilities, conditioning_theme
from .generation_v2 import build_plan, plan_track, writer_settings
from .generators import selection
from .prepared_audio import persist_pcm, prepare_candidate
from .track_titles import title_for_asset
from .transitions import TransitionReport
from .writers import WriterProfile, installed_local, pin_local

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Prepared:
    asset: DepotAsset
    track: PreloadedTrack
    cue: CueMap
    report: TransitionReport


class LiveSet:
    """One set owns the device and alternates A/B at measured exit cues."""

    def __init__(
        self,
        depot: Depot,
        engine: AudioEngine | None = None,
        *,
        preview_dir: Path = Path("var/previews"),
        min_track_duration_s: float = 0.0,
    ) -> None:
        self.depot = depot
        self.min_track_duration_s = min_track_duration_s
        self.engine = engine or AudioEngine()
        self.renderer = PreviewRenderer(sample_rate=self.engine.sample_rate, out_dir=preview_dir)
        self.filter = FeasibilityFilter(FilterConfig(max_tempo_ratio=1.05))
        self.state: SessionState | None = None
        self.current: DepotAsset | None = None
        self.current_track: PreloadedTrack | None = None
        self.cue: CueMap | None = None
        self.prepared: Prepared | None = None
        self._prepared_loaded = False
        self.active_deck = 0
        self.history: list[dict[str, object]] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prepare_thread: threading.Thread | None = None
        self._prepare_cancel = threading.Event()
        self._prepare_deadline: float | None = None
        self._preparation_misses = 0
        self._missed_for = None
        self._skip_pending = None
        self._skip_serial = 0
        self._skip_waiting = False
        self._next_skip_check = 0.0
        self._waiting = False
        self._playback_stopped = False
        self._played_hashes: set[str] = set()
        self._played_families: set[str] = set()
        self._inventory: list[tuple[str, str, float]] = []
        self._latencies = LatencyHistory(preview_dir.parent / "generation/latency.sqlite3")
        self._slot_started: dict[int, float] = {}
        self._attempt_started: float | None = None
        self._generation_thread: threading.Thread | None = None
        self._render_cancel = threading.Event()
        self._generation_theme: str | None = None
        self._generation_done_themes: set[tuple[str, str, str]] = set()
        self._generation_revision = 0
        self.generator, self.generator_preset = selection()
        self._generation_completed = 0
        self._startup_verified_count = 0
        self._startup_buffer_s = 0.0
        self._startup_required = 1
        self._next_inventory_check = 0.0
        self._last_replenished = 0.0
        self._generation_failed = 0
        self._generation_current = 0
        self._generation_artifact: Path | None = None
        self._generation_prompt_origin: str | None = None
        self._preparing = False
        self._prepared_for: tuple[str, int] | None = None
        self.generation_enabled = True
        self._generating = False
        self._pending_start: tuple[str, str] | None = None
        self.error: str | None = None
        self._last_audio_underruns = 0
        self._priority_updated = 0.0
        self._unhealthy_until = 0.0
        self._last_deadline_misses = 0
        self._equalizer_path = self.renderer.out_dir.parent / "config/equalizer.json"
        self._generator_path = self.renderer.out_dir.parent / "config/generation.json"
        self._writer_path = self.renderer.out_dir.parent / "config/writer.json"
        self._controls_path = self.renderer.out_dir.parent / "config/generation-controls.json"
        self._controls = GenerationSettings()
        if self._controls_path.is_file():
            try:
                self._controls = GenerationSettings.model_validate_json(
                    self._controls_path.read_text()
                )
            except (OSError, ValueError) as exc:
                LOGGER.warning("Ignoring invalid generation settings: %s", exc)
        self._session_controls = self._controls
        self._queued_snapshot = None
        self._generation_target = self._controls.track_count
        self._writer_profile = None
        if self._writer_path.is_file():
            try:
                self._writer_profile = WriterProfile.model_validate_json(
                    self._writer_path.read_text()
                ).model_dump()
            except (OSError, ValueError) as exc:
                LOGGER.warning("Ignoring invalid writer settings: %s", exc)
        self._jobs = GenerationJobs(self.renderer.out_dir.parent / "generation/jobs.sqlite3")
        self._generation_jobs: list[dict] = []
        self._generation_batch_id: str | None = None
        self._resume_batch_id: str | None = None
        self._retry_failed_batch = False
        self._force_new_batch = False
        self._generation_stage = "idle"
        self._generation_warning = None
        if self._generator_path.is_file():
            try:
                saved = json.loads(self._generator_path.read_text())
                self.generator, self.generator_preset = selection(
                    saved["generator"], saved["preset"]
                )
            except (ValueError, KeyError, TypeError, OSError) as exc:
                LOGGER.warning("Ignoring invalid generator settings: %s", exc)
        if self._equalizer_path.is_file():
            try:
                saved = json.loads(self._equalizer_path.read_text())
                self.engine.mixer.equalizer.configure(EqualizerSettings(**saved))
            except (ValueError, TypeError, OSError) as exc:
                LOGGER.warning("Ignoring invalid saved equalizer settings: %s", exc)

    def configure_generator(self, generator: str | None, preset: str | None = None) -> None:
        if generator is None and preset is None:
            return
        generator, preset = selection(generator or self.generator, preset, require_available=True)
        with self._lock:
            if (generator, preset) == (self.generator, self.generator_preset):
                return
            self._generator_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._generator_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"generator": generator, "preset": preset}) + "\n")
            temporary.replace(self._generator_path)
            self.generator, self.generator_preset = generator, preset
            if self._generating:
                self._queued_snapshot = self._new_snapshot()
            self._generation_revision += 1
            self._render_cancel.set()
            self._event({"event": "generator_changed", "generator": generator, "preset": preset})

    def configure_equalizer(self, settings: EqualizerSettings) -> dict[str, object]:
        with self._lock:
            path = self._equalizer_path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(settings.to_dict(), indent=2) + "\n")
            temporary.replace(path)
            self.engine.mixer.equalizer.configure(settings)
            self._invalidate_preparation()
            LOGGER.info("MASTER EQ %s", settings.to_dict())
            return self.engine.mixer.equalizer.snapshot()

    def writer_profile(self) -> dict:
        return self._writer_profile or writer_settings()

    def generation_settings(self) -> dict:
        return {
            "selected": self._controls.model_dump(),
            "capabilities": capabilities(self.generator),
        }

    def configure_generation_settings(self, controls: GenerationSettings) -> dict:
        with self._lock:
            self._controls_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._controls_path.with_suffix(".tmp")
            temp.write_text(controls.model_dump_json() + "\n")
            temp.replace(self._controls_path)
            self._controls = controls
            if self.state is None and not self._generating:
                self._session_controls = controls
            self._generation_done_themes.clear()
            self._event({"event": "generation_settings_saved", "applies_to": "new_batches"})
            return self.generation_settings()

    def preview_generation(
        self,
        theme: str,
        controls: GenerationSettings | None = None,
        *,
        generator: str | None = None,
    ) -> dict:
        controls = controls or self._controls
        generator, _preset = selection(generator or self.generator)
        plan = build_plan(theme, 0, generator, controls=controls.model_dump())
        warnings = []
        if controls.vocal_mode == "instrumental" and (
            controls.lyric_topic or controls.lyric_direction
        ):
            warnings.append("Lyric topic/direction are unused in instrumental mode.")
        if (
            self.writer_profile()["mode"] == "template"
            and plan.intent.vocal_mode != "instrumental"
            and not controls.custom_lyrics
        ):
            raise ValueError("template writer needs instrumental mode or custom original lyrics")
        return {
            "plan": plan.model_dump(),
            "controls": controls.model_dump(),
            "warnings": warnings,
            "capabilities": capabilities(generator),
            "overrides": [
                key
                for key, value in controls.model_dump().items()
                if value != GenerationSettings().model_dump()[key]
            ],
            "semantic_evidence": "targets and prompt hints; audio traits remain unverified",
        }

    def cancel_generation(self) -> dict:
        with self._lock:
            if self._generating:
                self._render_cancel.set()
                self._generation_stage = "cancelling"
                self._event({"event": "generation_cancel_requested"})
            return self.status()

    def track_report(self, batch_id: str, slot: int) -> dict:
        from .generation_reports import track_report

        self.generation_batch(batch_id)
        return track_report(self._jobs, batch_id, slot - 1)

    def retry_slot(self, batch_id: str, slot: int, *, replace=False) -> dict:
        with self._lock:
            if self._generating:
                raise ValueError("cancel or finish the active batch before changing a slot")
            self.generation_batch(batch_id)
            self._jobs.reset_slot(batch_id, slot - 1, replace=replace)
            if replace:
                cache = self._jobs.path.parent / "drafts" / batch_id / f"{slot - 1}.json"
                cache.unlink(missing_ok=True)
                cache.with_suffix(".feedback.json").unlink(missing_ok=True)
            return self.resume_generation(batch_id)

    def configure_writer(self, profile: WriterProfile) -> dict:
        settings = pin_local(profile.model_dump())
        if settings["mode"] == "local" and not settings["base"]:
            installed = installed_local(settings["model"])
            if not installed:
                from vexa_yue2.backends import BackendUnavailable

                raise BackendUnavailable("writer is not installed in the private model depot")
            if settings["digest"] and settings["digest"] != installed["digest"]:
                raise ValueError("writer digest does not match the installed model")
        with self._lock:
            self._writer_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._writer_path.with_suffix(".tmp")
            temp.write_text(json.dumps(settings) + "\n")
            temp.replace(self._writer_path)
            self._writer_profile = settings
            self._generation_done_themes.clear()
            self._event(
                {
                    "event": "writer_configured",
                    "model": settings["model"],
                    "mode": settings["mode"],
                    "applies_to": "new_batches",
                }
            )
            return settings

    def _event(self, event: dict[str, object]) -> None:
        record = {"at": datetime.now(UTC).isoformat(), **event}
        asset_id = event.get("asset_id")
        if asset_id:
            asset = getattr(self.depot, "assets", {}).get(asset_id)
            if asset is not None:
                record["title"] = title_for_asset(asset)
        self.history.append(record)
        LOGGER.info("LIVE %s", json.dumps(record, sort_keys=True, default=str))

    def feedback(self, rating: str, note: str = "") -> dict[str, object]:
        """Record listener evidence for later Laya training, without changing playback rules."""
        if rating not in {"like", "dislike"}:
            raise ValueError("rating must be like or dislike")
        with self._lock:
            if self.state is None or self.current is None:
                raise RuntimeError("no track is playing")
            record = {
                "at": datetime.now(UTC).isoformat(),
                "session_id": self.state.session_id,
                "theme": self.state.theme,
                "asset_id": self.current.manifest.asset_id,
                "title": title_for_asset(self.current),
                "rating": rating,
                "note": note[:500],
            }
            path = self.renderer.out_dir.parent / "feedback.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._event({"event": "feedback", "rating": rating, "asset_id": record["asset_id"]})
            return record

    def _cue(self, asset: DepotAsset) -> CueMap:
        cue = CueMap.load(asset.path.with_suffix(".cues.json"))
        if not cue.valid_for(asset.path) or cue.entry_s is None or cue.exit_s is None:
            raise ValueError(f"no valid measured cues for {asset.manifest.asset_id}")
        if not 0 <= cue.entry_s < cue.exit_s < asset.manifest.audio.duration_s:
            raise ValueError(f"invalid cue ordering for {asset.manifest.asset_id}")
        return cue

    def _load(self, asset: DepotAsset) -> tuple[PreloadedTrack, CueMap]:
        cue = self._cue(asset)
        track = self.engine.preload(
            asset.path,
            bpm=asset.manifest.beat_grid.bpm,
            expected_hash=asset.manifest.content_sha256,
        )
        persist_pcm(track, self.renderer.out_dir.parent / "prepared")
        return track, cue

    def _duration_eligible(self, asset: DepotAsset) -> bool:
        """Keep short legacy depot renders out of full-length live sets."""
        return asset.manifest.audio.duration_s >= self.min_track_duration_s

    def _invalidate_preparation(self):
        self._prepare_cancel.set()
        self.prepared = None
        self._prepared_loaded = False
        self._prepared_for = None
        self._inventory = []

    def _dsp_id(self):
        decks = self.engine.mixer.decks
        return dsp_identity(
            self.engine.sample_rate,
            self.engine.config.blocksize,
            self.engine.mixer.equalizer.settings,
            (decks[self.active_deck].gain, decks[1 - self.active_deck].gain),
        )

    def start(
        self,
        theme: str,
        *,
        session_id: str,
        controls: GenerationSettings | None = None,
        prepared_ids: list[str] | None = None,
        generation_enabled: bool = True,
    ) -> dict[str, object]:
        with self._lock:
            if self.state is not None:
                raise RuntimeError("a live set is already running")
            self._stop.clear()
            self.generation_enabled = generation_enabled
            self._session_controls = controls or self._controls
            theme = conditioning_theme(theme, self._session_controls)
            matches = (
                [(1.0, self.depot.assets[key]) for key in prepared_ids if key in self.depot.assets]
                if prepared_ids is not None
                else self.depot.search(
                    theme,
                    limit=80,
                    min_duration_s=self.min_track_duration_s,
                    **(
                        {"genre_override": self._session_controls.genre}
                        if self._session_controls.genre
                        else {}
                    ),
                )
                if self._session_controls.start_policy == "depot" or not self.generation_enabled
                else []
            )
            for similarity, asset in matches:
                if not self._duration_eligible(asset):
                    continue
                try:
                    track, cue = self._load(asset)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.error = str(exc)
                    continue
                try:
                    self.engine.clock.bpm = asset.manifest.beat_grid.bpm
                    self.engine.start()
                    self._playback_stopped = False
                    self.engine.set_loop(False, 0)
                    self.engine.load_and_play(track, 0, entry_frame=0)
                    self.current, self.cue, self.current_track = asset, cue, track
                    self._played_hashes.add(track.content_sha256)
                    self._played_families.add(asset.manifest.family_id)
                    self._waiting = False
                    self.error = None
                    self.state = SessionState(
                        session_id=session_id,
                        theme=theme,
                        clock=MusicalClock(bpm=asset.manifest.beat_grid.bpm),
                    )
                    self.state.deck(DeckId.A).asset_id = asset.manifest.asset_id
                    self.state.deck(DeckId.A).playing = True
                    self._event(
                        {
                            "event": "start",
                            "asset_id": asset.manifest.asset_id,
                            "theme_score": round(similarity, 3),
                            "play_start_s": 0.0,
                            "mix_entry_s": cue.entry_s,
                            "exit_s": cue.exit_s,
                            "file_duration_s": asset.manifest.audio.duration_s,
                        }
                    )
                    self._stop.clear()
                    self._thread = threading.Thread(target=self._run, daemon=True)
                    self._thread.start()
                    self._queue_generation(theme, session_id)
                    return self.status()
                except Exception as exc:
                    self.error = f"audio device could not start: {exc}"
                    return {"running": False, "generating": False, "reason": self.error}
            queued = self._queue_generation(theme, session_id)
            if not queued and not self.generation_enabled:
                self.error = (
                    "No compatible ready depot track. "
                    "Choose another vibe or enable music generation."
                )
                return {"running": False, "generating": False, "reason": self.error}
            return {
                "running": False,
                "generating": queued,
                "reason": "no full-length cueable depot match; music generation queued"
                if queued
                else ("no full-length cueable depot match; generation already attempted"),
            }

    def _queue_generation(self, theme: str, session_id: str, *, replenish: bool = False) -> bool:
        if not self.generation_enabled:
            self._generation_stage = "depot_only"
            return False
        try:
            snapshot = self._new_snapshot()
        except ValueError as exc:
            self.error = f"generation queue failed: {exc}"
            self._generation_stage = "queue_failed"
            self._event({"event": "generation_batch_failed", "reason": str(exc)})
            return False
        key = (theme, self.generator, self.generator_preset)
        if key in self._generation_done_themes and not replenish:
            return False
        if self._generating:
            if theme != self._generation_theme:
                self._generation_theme = theme
                self._queued_snapshot = snapshot
                self._generation_revision += 1
                self._render_cancel.set()
            return True
        self._generation_theme = theme
        self._queued_snapshot = snapshot
        self._generation_target = snapshot.get("controls", {}).get("track_count", 10)
        if replenish:
            self._last_replenished = time.monotonic()
            self._force_new_batch = True
            self._generation_done_themes.discard(key)
            self._event({"event": "generation_replenish", "theme": theme})
        self._generating = True
        self.error = None
        self._generation_warning = None
        self._generation_completed = 0
        self._startup_verified_count = 0
        self._startup_buffer_s = 0.0
        self._startup_required = 1
        self._generation_failed = 0
        self._generation_current = 0
        self._generation_artifact = None
        self._generation_prompt_origin = None
        self._generation_stage = "queued"
        self._generation_jobs = []
        self._generation_batch_id = None
        if self.state is None:
            self._pending_start = (theme, session_id)
        self._render_cancel = threading.Event()
        self._generation_thread = threading.Thread(target=self._generate, daemon=True)
        self._generation_thread.start()
        return True

    def _new_snapshot(self) -> dict:
        if self._resume_batch_id:
            return self._jobs.batch(self._resume_batch_id)["snapshot"]
        from .generators import components

        return {
            "pipeline_version": 2,
            "generator": self.generator,
            "preset": self.generator_preset,
            "runtime_components": {
                k: str(v) for k, v in components(self.generator, self.generator_preset).items()
            },
            "library": str(self.depot.library.resolve()),
            "writer": self.writer_profile(),
            "controls": self._session_controls.model_dump(),
        }

    def _start_prepared_batch(
        self, theme: str, snapshot: dict, rows: list[dict], verified_ids: set[str]
    ):
        pending = self._pending_start
        controls = GenerationSettings.model_validate(snapshot.get("controls", {}))
        required = (
            controls.track_count
            if controls.start_policy == "complete"
            else min(controls.buffer_tracks, controls.track_count)
            if controls.start_policy == "buffer"
            else 1
        )
        # Persisted READY rows must pass recovery before they count towards startup.
        ready = [
            row["asset_id"]
            for row in rows
            if row["state"] == "ready" and row["asset_id"] in verified_ids
        ]
        self._startup_verified_count = len(ready)
        supply = self._latencies.estimate("supply", latency_key(snapshot))
        preparation = self._latencies.estimate("prepare", "playback-v1")
        buffer_s = 0.0
        for index, asset_id in enumerate(ready):
            asset = getattr(self.depot, "assets", {}).get(asset_id)
            if asset is None:
                continue
            try:
                cue = CueMap.load(asset.path.with_suffix(".cues.json"))
                if cue.exit_s is None or cue.entry_s is None:
                    continue
                buffer_s += max(
                    0,
                    min(asset.manifest.audio.duration_s, cue.exit_s + 4)
                    - (cue.entry_s + 4 if index else 0),
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self._startup_buffer_s = buffer_s
        threshold = None
        if controls.start_policy == "buffer" and supply["samples"] >= 3:
            threshold = supply["p90_s"] + (preparation["p90_s"] or 0) + 15
            average = (
                buffer_s / len(ready) if buffer_s and ready else (controls.duration_s or 150) - 4
            )
            required = min(
                controls.track_count, max(required, math.ceil(threshold / max(1, average)))
            )
        self._startup_required = required
        if (
            pending
            and pending[0] == theme
            and len(ready) >= required
            and (threshold is None or buffer_s >= threshold or len(ready) == controls.track_count)
            and not self._stop.is_set()
            and not self._playback_stopped
            and not self._render_cancel.is_set()
        ):
            self._pending_start = None
            self.start(theme, session_id=pending[1], controls=controls, prepared_ids=ready)

    def _generate(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                theme, revision = self._generation_theme, self._generation_revision
                generator, preset = self.generator, self.generator_preset
                cancel = self._render_cancel
                resume_id, retry_failed = self._resume_batch_id, self._retry_failed_batch
                force_new = self._force_new_batch
                self._resume_batch_id = None
                self._retry_failed_batch = False
                self._force_new_batch = False
                snapshot = self._queued_snapshot or self._new_snapshot()
                self._generation_target = snapshot.get("controls", {}).get("track_count", 10)
                self._startup_verified_count = 0
                self._startup_buffer_s = 0.0
                self._slot_started.clear()
                self._attempt_started = None
            if theme is None:
                break
            try:
                with self._jobs.exclusive(cancel):
                    batch_id = resume_id or self._jobs.open_batch(
                        theme, snapshot, self._generation_target, force_new=force_new
                    )
                    if retry_failed:
                        self._jobs.reset_failed(batch_id)
                    self._generation_batch_id = batch_id
                    self._generation_target = len(self._jobs.tracks(batch_id))
                    verified_ids: set[str] = set()

                    def event(
                        record,
                        revision=revision,
                        theme=theme,
                        snapshot=snapshot,
                        verified_ids=verified_ids,
                    ):
                        with self._lock:
                            if revision != self._generation_revision:
                                return
                            rows = record.pop("jobs")
                            self._generation_jobs = [
                                {
                                    "slot": row["slot"] + 1,
                                    "state": row["state"],
                                    "title": (row["brief"] or {}).get("title"),
                                    "plan_attempts": row["plan_attempts"],
                                    "render_attempts": row["render_attempts"],
                                    "error": row["error"],
                                }
                                for row in rows
                            ]
                            self._generation_completed = sum(
                                row["state"] == "ready" for row in rows
                            )
                            self._generation_failed = record.pop("failed_attempts")
                            self._generation_current = record.get("track", 0)
                            now = time.monotonic()
                            slot = int(record.get("track", 0))
                            kind = record["event"]
                            if kind in {"generation_planning", "generation_started"}:
                                self._slot_started.setdefault(slot, now)
                                self._attempt_started = now
                            if (
                                kind in {"generation_failed", "generation_plan_retry"}
                                and self._attempt_started is not None
                            ):
                                self._latencies.record(
                                    "attempt",
                                    latency_key(snapshot),
                                    now - self._attempt_started,
                                    success=False,
                                )
                            if kind == "generated" and slot in self._slot_started:
                                self._latencies.record(
                                    "supply",
                                    latency_key(snapshot),
                                    now - self._slot_started.pop(slot),
                                    success=True,
                                )
                                self._attempt_started = None
                            if kind == "generation_track_rejected" and slot in self._slot_started:
                                self._latencies.record(
                                    "supply",
                                    latency_key(snapshot),
                                    now - self._slot_started.pop(slot),
                                    success=False,
                                )
                            if record["event"] == "generation_started":
                                self._generation_artifact = Path(
                                    rows[self._generation_current - 1]["asset_path"]
                                )
                            self._generation_stage = record["event"].removeprefix("generation_")
                            if record.get("origin"):
                                self._generation_prompt_origin = record["origin"]
                            if record["event"] == "generation_batch_failed":
                                self._generation_warning = (
                                    record.get("reason") or "some generation slots were rejected"
                                )
                                self.error = self._generation_warning
                            elif record["event"] == "generation_batch_complete":
                                if self.error == self._generation_warning:
                                    self.error = None
                                self._generation_warning = None
                            self._event(record)
                            self._start_prepared_batch(theme, snapshot, rows, verified_ids)

                    def admitted(
                        asset, slot, cancel=cancel, revision=revision, verified_ids=verified_ids
                    ):
                        with self._lock:
                            if cancel.is_set() or revision != self._generation_revision:
                                raise RuntimeError("stale generation admission")
                            self.depot.add(asset.manifest, asset.path)
                            verified_ids.add(asset.manifest.asset_id)
                            self._prepared_for = None

                    runner = GenerationRunner(
                        self._jobs,
                        batch_id,
                        self.depot.library,
                        planner=plan_track,
                        renderer=generate_theme,
                        admitted=admitted,
                        event=event,
                        cancel=cancel,
                    )
                    runner.run()
                    if self._jobs.batch(batch_id)["state"] == "completed":
                        self._generation_done_themes.add((theme, generator, preset))
            except Exception as exc:
                if not cancel.is_set():
                    with self._lock:
                        self.error = f"generation queue failed: {exc}"
                        self._generation_stage = "queue_failed"
                        self._event({"event": "generation_batch_failed", "reason": str(exc)})
                    LOGGER.exception("generation queue failed")
            with self._lock:
                if revision != self._generation_revision and not self._stop.is_set():
                    self._render_cancel = threading.Event()
                    continue
                self._generating = False
                if cancel.is_set():
                    self._generation_stage = "interrupted"
                self._generation_current = 0
                break

    def resume_generation(self, batch_id: str, *, retry_failed=False) -> dict:
        with self._lock:
            if self._generating:
                raise ValueError("generation is already active")
            batch = self._jobs.batch(batch_id)
            snapshot = batch["snapshot"]
            if batch["state"] == "completed":
                raise ValueError("batch is already complete")
            if snapshot["library"] != str(self.depot.library.resolve()):
                raise ValueError("batch belongs to a different depot")
            self.configure_generator(snapshot["generator"], snapshot["preset"])
            self._session_controls = GenerationSettings.model_validate(snapshot.get("controls", {}))
            self._retry_failed_batch = retry_failed
            self._resume_batch_id = batch_id
            self._generation_done_themes.discard(
                (batch["theme"], snapshot["generator"], snapshot["preset"])
            )
            self._stop.clear()
            self._queue_generation(
                batch["theme"], self.state.session_id if self.state else f"resume-{batch_id[:12]}"
            )
            return self.status()

    def generation_batch(self, batch_id: str) -> dict:
        batch = self._jobs.batch(batch_id)
        if batch["snapshot"]["library"] != str(self.depot.library.resolve()):
            raise ValueError("batch belongs to a different depot")
        rows = self._jobs.tracks(batch_id)
        return {
            "id": batch_id,
            "theme": batch["theme"],
            "state": batch["state"],
            "created": batch["created"],
            "updated": batch["updated"],
            "generator": batch["snapshot"]["generator"],
            "writer": batch["snapshot"]["writer"]["model"],
            "controls": batch["snapshot"].get("controls"),
            "completed": sum(row["state"] == "ready" for row in rows),
            "target": len(rows),
            "tracks": [
                {
                    "slot": row["slot"] + 1,
                    "state": row["state"],
                    "title": (row["brief"] or {}).get("title"),
                    "plan_attempts": row["plan_attempts"],
                    "render_attempts": row["render_attempts"],
                    "error": row["error"],
                }
                for row in rows
            ],
        }

    def generation_batches(self) -> list[dict]:
        return [
            self.generation_batch(batch["id"])
            for batch in self._jobs.batches()
            if batch["snapshot"]["library"] == str(self.depot.library.resolve())
        ]

    def steer(
        self,
        theme: str,
        *,
        controls: GenerationSettings | None = None,
        generation_enabled: bool | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if generation_enabled is not None:
                self.generation_enabled = generation_enabled
            requested = controls or self._controls
            changed_controls = requested != self._session_controls
            self._session_controls = requested
            theme = conditioning_theme(theme, requested)
            if changed_controls and self._generating:
                self._queued_snapshot = self._new_snapshot()
                self._generation_revision += 1
                self._render_cancel.set()
            if changed_controls:
                self._generation_done_themes.discard((theme, self.generator, self.generator_preset))
            if self.state is None:
                if self._pending_start is None:
                    raise RuntimeError("no live set")
                self._pending_start = (theme, self._pending_start[1])
                self._event({"event": "steer_pending", "theme": theme})
                self._queue_generation(theme, self._pending_start[1])
                return self.status()
            if self.state.theme == theme:
                self._queue_generation(theme, self.state.session_id)
                return self.status()
            self.state.theme = theme
            self.state.generation += 1
            # A late steer need not throw away compatible, already measured PCM.
            # Recheck the new scope before preserving the pending handoff.
            self._prepare_cancel.set()
            matches = self.depot.search(
                theme,
                limit=80,
                min_duration_s=self.min_track_duration_s,
                **({"genre_override": requested.genre} if requested.genre else {}),
            )
            matched = {asset.manifest.asset_id for _, asset in matches}
            permitted = self.filter.build(
                state=self.state,
                library=[self.prepared.asset.manifest] if self.prepared else [],
                current_asset=self.current.manifest if self.current else None,
                recent_family_ids=tuple(self._played_families),
            ).transitions
            if (
                self.prepared
                and self.prepared.asset.manifest.asset_id in matched
                and permitted
                and self.prepared.report.dsp_id == self._dsp_id()
            ):
                self._inventory = []
                self._prepared_for = None
                self._event(
                    {
                        "event": "prepared_retained",
                        "theme": theme,
                        "asset_id": self.prepared.asset.manifest.asset_id,
                    }
                )
            else:
                self._invalidate_preparation()
            self._event({"event": "steer", "theme": theme, "generation": self.state.generation})
            self._queue_generation(theme, self.state.session_id)
            return self.status()

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            try:
                underruns = self.engine.mixer.stats.underruns
                misses = self.engine.mixer.stats.deadline_misses
                now = time.monotonic()
                if underruns > self._last_audio_underruns or misses > self._last_deadline_misses:
                    self._unhealthy_until = now + 60
                self._last_deadline_misses = misses
                if now - self._priority_updated >= 2:
                    publish_audio_priority(
                        None
                        if self._playback_stopped
                        else "audio callback health recovering"
                        if now < self._unhealthy_until
                        else "imminent deck preparation"
                        if self._preparing
                        else None
                    )
                    self._priority_updated = now
                if underruns > self._last_audio_underruns:
                    self._last_audio_underruns = underruns
                    self._event(
                        {
                            "event": "audio_underrun",
                            "count": underruns,
                            "generating": self._generating,
                            "audio": self.engine.snapshot(),
                        }
                    )
                self.tick()
            except Exception as exc:
                self.error = str(exc)
                LOGGER.exception("live scheduling tick failed")

    def tick(self) -> None:
        with self._lock:
            if self._playback_stopped:
                return
            if self._skip_pending is not None:
                self._finish_skip()
                return
            if self._skip_waiting:
                if time.monotonic() < self._next_skip_check:
                    return
                self._next_skip_check = time.monotonic() + 2
                candidate = self._load_skip_candidate(self._skip_candidates())
                if candidate:
                    asset, track, cue = candidate
                    self._play_after_gap(asset, track, cue, manual=True, fade_in=True)
                    self._skip_waiting = False
                return
            if self.state is None or self.current is None or self.cue is None:
                return
            deck = self.engine.mixer.decks[self.active_deck]
            for index, deck_id in enumerate((DeckId.A, DeckId.B)):
                self.state.deck(deck_id).playing = self.engine.mixer.decks[index].playing
            self.state.clock.bar = self.engine.mixer.bar
            self.state.clock.beat = self.engine.mixer.clock.beat
            position_s = deck.position / self.engine.sample_rate
            exit_s = self.cue.exit_s
            if exit_s is None:
                return
            key = (self.current.manifest.asset_id, self.state.generation)
            if not deck.playing and deck.finished and not self._waiting:
                self._waiting = True
                self._prepared_for = None
                self._event({"event": "waiting_for_music", "reuse_policy": "never"})
            if (
                position_s >= exit_s
                and not self._waiting
                and self.prepared is None
                and self._missed_for != key
            ):
                self._missed_for = key
                self._preparation_misses += 1
                self._event(
                    {
                        "event": "preparation_deadline_missed",
                        "asset_id": key[0],
                        "policy": "finish_current_then_wait",
                    }
                )
            if self.prepared is not None and self.prepared.report.dsp_id != self._dsp_id():
                self._invalidate_preparation()
            if (
                self.prepared is None
                and not self._preparing
                and self._prepared_for != key
                and (position_s < exit_s - 2 or self._waiting)
            ):
                self._preparing = True
                self._prepared_for = key
                self._prepare_cancel = threading.Event()
                self._prepare_deadline = time.monotonic() + (
                    120.0 if self._waiting else max(0.1, exit_s - position_s - 2.0)
                )
                self._prepare_thread = threading.Thread(
                    target=self._prepare, args=(self.state.generation,), daemon=True
                )
                self._prepare_thread.start()
            incoming = 1 - self.active_deck
            standby = self.engine.mixer.decks[incoming]
            if (
                self.prepared is not None
                and not self._prepared_loaded
                and standby.fade_remaining == 0
                and (not standby.playing or standby.fade_gain == 0.0)
            ):
                self.engine.prepare_deck(
                    self.prepared.track,
                    incoming,
                    entry_frame=round(
                        self.prepared.report.incoming_entry_s * self.engine.sample_rate
                    ),
                )
                self.state.deck(
                    DeckId.B if incoming else DeckId.A
                ).asset_id = self.prepared.asset.manifest.asset_id
                self._prepared_loaded = True
                self._event(
                    {
                        "event": "deck_preloaded",
                        "deck": incoming,
                        "asset_id": self.prepared.asset.manifest.asset_id,
                    }
                )
            if self.prepared is not None and self._prepared_loaded:
                if self._waiting:
                    self._resume_playback(self.prepared)
                elif position_s >= exit_s:
                    tolerance = 2 * self.engine.config.blocksize / self.engine.sample_rate + 0.05
                    if position_s - exit_s <= tolerance:
                        self._commit(self.prepared)
                    else:
                        self._event(
                            {
                                "event": "late_transition_rejected",
                                "lateness_s": round(position_s - exit_s, 3),
                            }
                        )
                        self._invalidate_preparation()
            now = time.monotonic()
            if now >= self._next_inventory_check:
                self._next_inventory_check = now + 5
                self._replenish_if_needed()

    def _replenish_if_needed(self):
        if (
            self.state is None
            or self._generating
            or self._preparing
            or self.error
            or time.monotonic() - self._last_replenished < 60
        ):
            return
        continuity = self.status()["continuity"]
        threshold = continuity["recommended_buffer_s"]
        if threshold is not None and continuity["conditional_inventory_s"] < threshold:
            self._event(
                {
                    "event": "supply_shortfall",
                    "inventory_s": continuity["conditional_inventory_s"],
                    "required_s": threshold,
                }
            )
            self._queue_generation(self.state.theme, self.state.session_id, replenish=True)

    def _prepare(self, generation: int) -> None:
        cancel = self._prepare_cancel
        try:
            with self._lock:
                if self.state is None or self.current is None or self.cue is None:
                    return
                theme, current, cue = self.state.theme, self.current, self.cue
                current_track = self.current_track
                target_bpm = self.state.clock.bpm
                deadline = self._prepare_deadline or time.monotonic() + 120
                eq = self.engine.mixer.equalizer.settings
                gains = (
                    self.engine.mixer.decks[self.active_deck].gain,
                    self.engine.mixer.decks[1 - self.active_deck].gain,
                )
                identity = self._dsp_id()
                matches = self.depot.search(
                    theme,
                    limit=80,
                    min_duration_s=self.min_track_duration_s,
                    **(
                        {"genre_override": self._session_controls.genre}
                        if self._session_controls.genre
                        else {}
                    ),
                )
                shortlist = [
                    asset
                    for _, asset in matches
                    if asset.manifest.content_sha256 not in self._played_hashes
                    and asset.manifest.family_id not in self._played_families
                    and self._duration_eligible(asset)
                ]
                filtered = self.filter.build(
                    state=self.state,
                    library=[asset.manifest for asset in shortlist],
                    current_asset=current.manifest,
                    recent_family_ids=tuple(self._played_families),
                )
                allowed = {action.asset_id for action in filtered.transitions}
                self._event(
                    {
                        "event": "candidate_filter",
                        "allowed": len(allowed),
                        "rejected": len(filtered.rejections),
                        "reasons": [str(row) for row in filtered.rejections[:20]],
                        "reuse_policy": "never",
                    }
                )
            if current_track is None:
                return
            inventory = []
            for asset in shortlist:
                if asset.manifest.asset_id not in allowed or cancel.is_set():
                    continue
                try:
                    next_cue = self._cue(asset)
                    scale = asset.manifest.beat_grid.bpm / target_bpm
                    span = (
                        min(asset.manifest.audio.duration_s, next_cue.exit_s + 4) - next_cue.entry_s
                    )
                    inventory.append(
                        (
                            asset.manifest.content_sha256,
                            asset.manifest.family_id,
                            max(0, span * scale),
                        )
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            with self._lock:
                if not cancel.is_set() and self.current is current:
                    self._inventory = inventory
            # Publish the first measured candidate immediately. Refine only while it is
            # still unclaimed, with a bounded menu and an absolute preparation deadline.
            examined = 0
            for asset in shortlist:
                if asset.manifest.asset_id not in allowed:
                    continue
                remaining = deadline - time.monotonic()
                if cancel.is_set() or remaining <= 0 or examined >= 3:
                    break
                with self._lock:
                    if self._prepared_loaded:
                        break
                examined += 1
                started = time.monotonic()
                success = False
                try:
                    track, next_cue, report = prepare_candidate(
                        asset,
                        current=current,
                        current_track=current_track,
                        current_cue=cue,
                        target_bpm=target_bpm,
                        cache_dir=self.renderer.out_dir.parent / "prepared",
                        preview_dir=self.renderer.out_dir,
                        cancel=cancel,
                        timeout_s=min(remaining, 120),
                        blocksize=self.engine.config.blocksize,
                        equalizer=eq,
                        gains=gains,
                        name=f"live_{generation}_{current.manifest.asset_id}_{asset.manifest.asset_id}_{identity[:12]}",
                    )
                    if report.dsp_id != identity:
                        raise ValueError("prepared preview uses different playback DSP")
                    with self._lock:
                        if (
                            cancel.is_set()
                            or self._stop.is_set()
                            or self.current is not current
                            or self.state is None
                            or self.state.generation != generation
                            or self._dsp_id() != identity
                            or time.monotonic() > deadline
                        ):
                            return
                        success = True
                        if not self._prepared_loaded and (
                            self.prepared is None or report.score > self.prepared.report.score
                        ):
                            self.prepared = Prepared(asset, track, next_cue, report)
                            self._event(
                                {
                                    "event": "prepared",
                                    "asset_id": asset.manifest.asset_id,
                                    "score": report.score,
                                    "rms_jump_db": report.rms_jump_db,
                                    "spectral_distance": report.spectral_distance,
                                    "preparation_s": round(time.monotonic() - started, 3),
                                }
                            )
                except (OSError, ValueError, RuntimeError) as exc:
                    if not cancel.is_set():
                        self._event(
                            {
                                "event": "candidate_failed",
                                "asset_id": asset.manifest.asset_id,
                                "reason": str(exc),
                            }
                        )
                finally:
                    if not cancel.is_set():
                        self._latencies.record(
                            "prepare", "playback-v1", time.monotonic() - started, success=success
                        )
            with self._lock:
                if (
                    not cancel.is_set()
                    and self.current is current
                    and self.state is not None
                    and self.prepared is None
                    and not self._generating
                    and not self.error
                    and time.monotonic() - self._last_replenished >= 60
                ):
                    self._queue_generation(theme, self.state.session_id, replenish=True)
        except Exception as exc:
            if not cancel.is_set():
                self.error = f"transition preparation failed: {exc}"
                LOGGER.exception("transition preparation failed")
        finally:
            with self._lock:
                self._preparing = False
                self._prepare_deadline = None

    def skip_track(self):
        with self._lock:
            if self._skip_pending is not None:
                return self.status()
            if self.state is None or self._playback_stopped or self._waiting:
                raise RuntimeError("No playing track to skip.")
            if any(d.fade_remaining > 0 for d in self.engine.mixer.decks):
                raise RuntimeError("Wait for the current fade to finish before skipping.")
            candidates = self._skip_candidates()
            self._skip_serial += 1
            incoming = 1 - self.active_deck
            candidate = self._load_skip_candidate(candidates)
            track = candidate[1] if candidate else None
            command = Command(
                CommandKind.SKIP_TRACK,
                deck=incoming,
                value=float(self._skip_serial),
                value2=round(0.25 * self.engine.sample_rate),
                track=track,
            )
            if not self.engine.queue.put(command):
                raise RuntimeError("Playback command queue is full; try again.")
            self._invalidate_preparation()
            self._skip_pending = (self._skip_serial, incoming, candidate)
            self._event({"event": "skip_requested", "asset_id": self.current.manifest.asset_id})
            return self.status()

    def _skip_candidates(self):
        if self.state is None:
            return []
        matches = self.depot.search(
            self.state.theme, limit=80, min_duration_s=self.min_track_duration_s
        )
        candidates = [asset for _, asset in matches]
        # Prepared candidates already passed theme selection; revalidate their source on load.
        if self.prepared is not None:
            candidates.insert(0, self.prepared.asset)
        return candidates

    def _load_skip_candidate(self, candidates):
        for asset in candidates:
            if (
                not asset.manifest.admissible()
                or not self._duration_eligible(asset)
                or asset.manifest.content_sha256 in self._played_hashes
                or asset.manifest.family_id in self._played_families
            ):
                continue
            try:
                track, cue = self._load(asset)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            return asset, track, cue
        return None

    def _finish_skip(self):
        token, incoming, candidate = self._skip_pending
        if self.engine.mixer.skip_completed != token:
            return
        self._skip_pending = None
        if candidate:
            self.current, self.current_track, self.cue = candidate
            self.active_deck = incoming
            self._played_hashes.add(self.current.manifest.content_sha256)
            self._played_families.add(self.current.manifest.family_id)
            self.engine.clock.bpm = self.state.clock.bpm = self.current.manifest.beat_grid.bpm
            self.state.deck(
                DeckId.B if incoming else DeckId.A
            ).asset_id = self.current.manifest.asset_id
            self.state.deck(DeckId.B if incoming else DeckId.A).playing = True
            self.state.deck(DeckId.A if incoming else DeckId.B).playing = False
            self._waiting = self._skip_waiting = False
            self._event(
                {
                    "event": "skip_completed",
                    "asset_id": self.current.manifest.asset_id,
                    "play_start_s": 0.0,
                }
            )
        else:
            self._waiting = self._skip_waiting = True
            for deck_id in (DeckId.A, DeckId.B):
                self.state.deck(deck_id).playing = False
            self._event({"event": "skip_waiting"})

    def start_playback(self) -> dict[str, object]:
        """Restart explicitly stopped playback, or leave a natural silent gap."""
        with self._lock:
            if self._playback_stopped:
                if self.state is not None and self.current_track is not None:
                    # STOP resets transport; START replays the current track from its beginning.
                    # The stream is closed and both decks are silent before warming the device.
                    self.engine.start()
                    self.engine.set_loop(False, self.active_deck)
                    self.engine.set_gain(1.0, self.active_deck)
                    self.engine.load_and_play(self.current_track, self.active_deck, entry_frame=0)
                    self.state.deck(DeckId.B if self.active_deck else DeckId.A).playing = True
                    self._playback_stopped = False
                    self._waiting = False
                    self._event({"event": "playback_restart", "play_start_s": 0.0})
                    return self.status()
                if self._pending_start is not None:
                    rows = (
                        self._jobs.tracks(self._generation_batch_id)
                        if self._generation_batch_id
                        else []
                    )
                    ready = []
                    for row in rows:
                        asset = self.depot.assets.get(row["asset_id"])
                        if (
                            row["state"] != "ready"
                            or asset is None
                            or not asset.manifest.admissible()
                            or not self._duration_eligible(asset)
                        ):
                            continue
                        try:
                            self._load(asset)
                        except (OSError, ValueError, KeyError, TypeError):
                            continue
                        ready.append(row["asset_id"])
                    if not ready:
                        raise ValueError(
                            "No verified generated track is ready. Wait, then press START."
                        )
                    theme, session_id = self._pending_start
                    result = self.start(
                        theme,
                        session_id=session_id,
                        controls=self._session_controls,
                        prepared_ids=ready,
                    )
                    if not result.get("running"):
                        raise RuntimeError(str(result.get("reason", "Audio could not start.")))
                    self._pending_start = None
                    return self.status()
            if self.state is None or not self._waiting:
                raise RuntimeError("Playback is not waiting. Start a set first.")
            if any(deck.playing for deck in self.engine.mixer.decks):
                raise RuntimeError("Music is already playing.")
            rows = self._jobs.tracks(self._generation_batch_id) if self._generation_batch_id else []
            for row in rows:
                asset = self.depot.assets.get(row["asset_id"])
                if (
                    row["state"] != "ready"
                    or asset is None
                    or not asset.manifest.admissible()
                    or not self._duration_eligible(asset)
                    or asset.manifest.content_sha256 in self._played_hashes
                    or asset.manifest.family_id in self._played_families
                ):
                    continue
                try:
                    track, cue = self._load(asset)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    LOGGER.warning("Cannot restart with %s: %s", row["asset_id"], exc)
                    continue
                # No overlap remains after a gap, so the outgoing key/tempo is irrelevant.
                self.engine.clock.bpm = asset.manifest.beat_grid.bpm
                self.state.clock.bpm = asset.manifest.beat_grid.bpm
                self._play_after_gap(asset, track, cue, manual=True)
                return self.status()
            raise ValueError(
                "No unplayed generated track is ready. Wait for generation, then press START."
            )

    def _resume_playback(self, next_track: Prepared):
        self._play_after_gap(next_track.asset, next_track.track, next_track.cue)

    def _play_after_gap(
        self,
        asset: DepotAsset,
        track: PreloadedTrack,
        cue: CueMap,
        *,
        manual: bool = False,
        fade_in: bool = False,
    ):
        """After a gap this is a fresh start, not a stale transition preview."""
        assert self.state is not None
        incoming = 1 - self.active_deck
        self.engine.set_loop(False, self.active_deck)
        self.engine.set_loop(False, incoming)
        self.engine.set_gain(1.0, incoming)
        self.engine.load_and_play(
            track,
            incoming,
            entry_frame=0,
            fade_in_frames=round(0.25 * self.engine.sample_rate) if fade_in else 0,
        )
        self.state.deck(DeckId.B if incoming else DeckId.A).asset_id = asset.manifest.asset_id
        self.state.deck(DeckId.B if incoming else DeckId.A).playing = True
        self.state.deck(DeckId.A if incoming else DeckId.B).playing = False
        self.active_deck = incoming
        self.current, self.cue, self.current_track = asset, cue, track
        self._played_hashes.add(asset.manifest.content_sha256)
        self._played_families.add(asset.manifest.family_id)
        self._waiting = False
        self._event(
            {
                "event": "resume",
                "asset_id": asset.manifest.asset_id,
                "play_start_s": 0.0,
                "incoming_deck": incoming,
                "manual": manual,
            }
        )
        self._invalidate_preparation()

    def _commit(self, next_track: Prepared) -> None:
        assert self.state is not None
        permitted = self.filter.build(
            state=self.state,
            library=[next_track.asset.manifest],
            current_asset=self.current.manifest if self.current else None,
            recent_family_ids=tuple(self._played_families),
        ).transitions
        if (
            next_track.report.dsp_id != self._dsp_id()
            or not permitted
            or not self.current_track
            or next_track.report.outgoing_hash != self.current_track.content_sha256
            or next_track.report.incoming_hash != next_track.track.content_sha256
            or self.engine.mixer.decks[self.active_deck].tempo_ratio != 1.0
        ):
            self._invalidate_preparation()
            return
        incoming = 1 - self.active_deck
        self.engine.set_loop(False, incoming)
        self.engine.crossfade_to(
            next_track.track,
            incoming,
            fade_frames=round(next_track.report.fade_s * self.engine.sample_rate),
            entry_frame=round(next_track.report.incoming_entry_s * self.engine.sample_rate),
        )
        self.engine.set_loop(False, self.active_deck)
        outgoing_position = (
            self.engine.mixer.decks[self.active_deck].position / self.engine.sample_rate
        )
        outgoing_id = self.current.manifest.asset_id if self.current else None
        self._played_hashes.add(next_track.asset.manifest.content_sha256)
        self._played_families.add(next_track.asset.manifest.family_id)
        self.active_deck = incoming
        self.current, self.cue, self.current_track = (
            next_track.asset,
            next_track.cue,
            next_track.track,
        )
        self.state.deck(DeckId.A).asset_id = (
            next_track.asset.manifest.asset_id if incoming == 0 else outgoing_id
        )
        self.state.deck(DeckId.B).asset_id = (
            next_track.asset.manifest.asset_id if incoming == 1 else outgoing_id
        )
        self._event(
            {
                "event": "transition",
                "from": outgoing_id,
                "asset_id": next_track.asset.manifest.asset_id,
                "incoming_deck": incoming,
                "play_start_s": next_track.report.incoming_entry_s,
                "outgoing_position_s": outgoing_position,
                "dsp_id": next_track.report.dsp_id,
                "score": next_track.report.score,
                "preview": str(next_track.report.preview_path),
            }
        )
        self._invalidate_preparation()

    def status(self) -> dict[str, object]:
        active = self.engine.mixer.decks[self.active_deck]
        assets = getattr(self.depot, "assets", {})
        # Only send names shown by the decks/history, not the entire depot on every poll.
        visible_ids = {event.get("asset_id") for event in self.history[-12:]}
        if self.state:
            visible_ids.update(self.state.deck(deck).asset_id for deck in (DeckId.A, DeckId.B))
        titles = {key: title_for_asset(assets[key]) for key in visible_ids if key in assets}
        controls = GenerationSettings.model_validate(
            (self._queued_snapshot or {}).get("controls", {})
        )
        stage = self._generation_stage
        if (
            self._generating
            and not self._render_cancel.is_set()
            and stage == "started"
            and self._generation_artifact
        ):
            try:
                progress = json.loads(
                    self._generation_artifact.with_suffix(".progress.json").read_text()
                )
                if not isinstance(progress, dict):
                    raise ValueError("invalid audio progress metadata")
                if progress.get("asset_id") == self._generation_artifact.stem and progress.get(
                    "stage"
                ) in {"raw_check", "correcting", "analyzing", "rejected", "analysis_failed"}:
                    stage = progress["stage"]
            except (OSError, ValueError, TypeError):
                pass
        needed = (
            self._generation_target
            if controls.start_policy == "complete"
            else min(controls.buffer_tracks, self._generation_target)
            if controls.start_policy == "buffer"
            else 1
        )
        needed = max(needed, self._startup_required) if self._pending_start else needed
        supply = self._latencies.estimate(
            "supply",
            latency_key(
                self._queued_snapshot
                or {
                    "generator": self.generator,
                    "preset": self.generator_preset,
                    "controls": self._session_controls.model_dump(),
                    "writer": self._writer_profile or {},
                }
            ),
        )
        preparation = self._latencies.estimate("prepare", "playback-v1")
        remaining = (
            max(
                0,
                min(self.current_track.duration_s, self.cue.exit_s + 4)
                - active.position / self.engine.sample_rate,
            )
            if (self.current_track and self.cue and self.cue.exit_s is not None and active.playing)
            else 0
        )
        prepared_span = (
            max(
                0,
                min(self.prepared.track.duration_s, self.prepared.cue.exit_s + 4)
                - self.prepared.report.incoming_entry_s,
            )
            if (isinstance(self.prepared, Prepared) and self.prepared.cue.exit_s is not None)
            else 0
        )
        inventory = [
            row
            for row in self._inventory
            if row[0] not in self._played_hashes
            and row[1] not in self._played_families
            and (self.prepared is None or row[0] != self.prepared.asset.manifest.content_sha256)
        ]
        earliest = min(self._slot_started.values()) if self._slot_started else None
        continuity = forecast(
            current_remaining_s=remaining,
            prepared_span_s=prepared_span,
            candidate_spans=inventory,
            supply=supply,
            preparation=preparation,
            required=needed if self._pending_start else 0,
            ready=self._startup_verified_count,
            elapsed_s=time.monotonic() - earliest if earliest is not None else 0,
        )
        continuity.update(
            {
                "waiting": self._waiting,
                "stopped": self._playback_stopped,
                "skipping": self._skip_pending is not None,
                "skip_waiting": self._skip_waiting,
                "reuse_policy": "never",
                "preparing": self._preparing,
                "preparation_misses": self._preparation_misses,
                "preparation_deadline_s": max(0, self._prepare_deadline - time.monotonic())
                if self._prepare_deadline
                else None,
            }
        )
        return {
            "continuity": continuity,
            "asset_titles": titles,
            "current_title": title_for_asset(self.current),
            "prepared_title": title_for_asset(self.prepared.asset) if self.prepared else None,
            "running": self.engine.running,
            "session_id": self.state.session_id
            if self.state
            else self._pending_start[1]
            if self._pending_start
            else None,
            "theme": self.state.theme
            if self.state
            else self._pending_start[0]
            if self._pending_start
            else None,
            "current_asset_id": self.current.manifest.asset_id if self.current else None,
            "prepared_asset_id": (self.prepared.asset.manifest.asset_id if self.prepared else None),
            "active_deck": self.active_deck,
            "generating": self._generating,
            "prepared_loaded": self._prepared_loaded,
            "startup": {
                "policy": controls.start_policy,
                "required": max(needed, self._startup_required),
                "ready": self._startup_verified_count,
                "buffer_s": round(self._startup_buffer_s, 2),
            }
            if self._pending_start
            else None,
            "generation": {
                "target": self._generation_target,
                "completed": self._generation_completed,
                "failed": self._generation_failed,
                "current": self._generation_current,
                "theme": self._generation_theme,
                "prompt_origin": self._generation_prompt_origin,
                "generator": self.generator,
                "preset": self.generator_preset,
                "batch_id": self._generation_batch_id,
                "stage": stage,
                "warning": self._generation_warning,
                "rejected": sum(row["state"] == "rejected" for row in self._generation_jobs),
                "tracks": self._generation_jobs,
            },
            "bpm": self.state.clock.bpm if self.state else None,
            "deck_a_asset_id": self.state.deck(DeckId.A).asset_id if self.state else None,
            "deck_b_asset_id": self.state.deck(DeckId.B).asset_id if self.state else None,
            "current_position_s": round(active.position / self.engine.sample_rate, 2),
            "current_duration_s": self.current_track.duration_s if self.current_track else None,
            "prepared_duration_s": (self.prepared.track.duration_s if self.prepared else None),
            "current_exit_s": self.cue.exit_s if self.cue else None,
            "spectrum": spectrum_levels(self.engine.mixer),
            "audio": self.engine.snapshot(),
            "events": self.history[-12:],
            "error": self.error,
            "error_kind": "batch_warning"
            if self.error and self.error == self._generation_warning
            else "app_error"
            if self.error
            else None,
        }

    def stop_playback(self) -> dict[str, object]:
        """Silence both decks without cancelling the current generation batch."""
        with self._lock:
            self._skip_pending = None
            self._skip_waiting = False
            self._playback_stopped = True
            self._waiting = True
            self._invalidate_preparation()
            # Stop the callback before touching deck state or draining queued playback commands.
            self.engine.stop()
            self.engine.queue.drain()
            for index, deck in enumerate(self.engine.mixer.decks):
                deck.playing = False
                deck.position = 0
                deck.fade_remaining = 0
                deck.fade_gain = 1.0
                deck.fade_target = 1.0
                deck.loop = False
                if self.state is not None:
                    self.state.deck(DeckId.B if index else DeckId.A).playing = False
            self.engine.mixer.clock.reset()
            publish_audio_priority(None)
            self._event({"event": "playback_stopped", "generating": self._generating})
            return self.status()

    def stop(self) -> None:
        """Shut down the entire live set, including workers (application exit only)."""
        self._skip_pending = None
        self._skip_waiting = False
        self._stop.set()
        self._render_cancel.set()
        self._prepare_cancel.set()
        self._pending_start = None
        if self._thread is not None:
            self._thread.join()
        publish_audio_priority(None)
        self.engine.stop()
        if self._prepare_thread is not None:
            self._prepare_thread.join()
        if (
            self._generation_thread is not None
            and self._generation_thread is not threading.current_thread()
        ):
            self._generation_thread.join()
        self.engine.queue.drain()
        for deck in self.engine.mixer.decks:
            deck.playing = False
            deck.fade_remaining = 0
            deck.fade_gain = 1.0
            deck.fade_target = 1.0
        self.engine.mixer.clock.reset()
        self.state = None
        self.current = None
        self.current_track = None
        self.cue = None
        self.prepared = None
        self._prepared_loaded = False
        self._prepared_for = None
        self._generation_theme = None
        self._generation_done_themes.clear()
        self._generation_completed = 0
        self._startup_verified_count = 0
        self._startup_buffer_s = 0.0
        self._startup_required = 1
        self._generation_failed = 0
        self._generation_current = 0
        self._generating = False
        self._generation_batch_id = None
        self._generation_jobs = []
        self._generation_stage = "idle"
        self._generation_warning = None
        self.active_deck = 0
        self._played_hashes.clear()
        self._played_families.clear()
        self._inventory.clear()
        self._waiting = False
        self._playback_stopped = False
        self._slot_started.clear()
        self._attempt_started = None
        self._missed_for = None
        self._preparation_misses = 0
