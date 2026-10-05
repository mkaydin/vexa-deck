"""Host-side continuous DJ runner. All decoding and decisions stay off the audio callback."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vexa_audio.cues import CueMap
from vexa_audio.engine import AudioEngine
from vexa_audio.equalizer import EqualizerSettings
from vexa_audio.loader import PreloadedTrack
from vexa_audio.preview import PreviewRenderer
from vexa_audio.telemetry import spectrum_levels
from vexa_contracts import DeckId, MusicalClock, SessionState
from vexa_yue2.backends import BackendUnavailable

from .depot import Depot, DepotAsset
from .feasibility import FeasibilityFilter, FilterConfig
from .generation import TRACK_COUNT, plan_theme
from .generation import isolated_generate_theme as generate_theme
from .prepared_audio import persist_pcm, prepare_candidate
from .track_titles import fallback_title, title_for_asset, unique_title
from .transitions import TransitionReport

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Prepared:
    asset: DepotAsset
    track: PreloadedTrack
    cue: CueMap
    report: TransitionReport


class LiveSet:
    """One set owns the device and alternates A/B at measured exit cues."""

    def __init__(self, depot: Depot, engine: AudioEngine | None = None,
                 *, preview_dir: Path = Path("var/previews"),
                 min_track_duration_s: float = 0.0) -> None:
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
        self._generation_thread: threading.Thread | None = None
        self._render_cancel = threading.Event()
        self._generation_theme: str | None = None
        self._generation_done_themes: set[str] = set()
        self._generation_completed = 0
        self._generation_failed = 0
        self._generation_current = 0
        self._generation_prompt_origin: str | None = None
        self._preparing = False
        self._prepared_for: tuple[str, int] | None = None
        self._generating = False
        self._pending_start: tuple[str, str] | None = None
        self.error: str | None = None
        self._last_audio_underruns = 0
        self._equalizer_path = self.renderer.out_dir.parent / "config/equalizer.json"
        if self._equalizer_path.is_file():
            try:
                saved = json.loads(self._equalizer_path.read_text())
                self.engine.mixer.equalizer.configure(EqualizerSettings(**saved))
            except (ValueError, TypeError, OSError) as exc:
                LOGGER.warning("Ignoring invalid saved equalizer settings: %s", exc)

    def configure_equalizer(self, settings: EqualizerSettings) -> dict[str, object]:
        with self._lock:
            path = self._equalizer_path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(settings.to_dict(), indent=2) + "\n")
            temporary.replace(path)
            self.engine.mixer.equalizer.configure(settings)
            LOGGER.info("MASTER EQ %s", settings.to_dict())
            return self.engine.mixer.equalizer.snapshot()

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
            record = {"at": datetime.now(UTC).isoformat(),
                      "session_id": self.state.session_id, "theme": self.state.theme,
                      "asset_id": self.current.manifest.asset_id,
                      "title": title_for_asset(self.current), "rating": rating,
                      "note": note[:500]}
            path = self.renderer.out_dir.parent / "feedback.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._event({"event": "feedback", "rating": rating,
                                 "asset_id": record["asset_id"]})
            return record

    def _cue(self, asset: DepotAsset) -> CueMap:
        cue = CueMap.load(asset.path.with_suffix(".cues.json"))
        if not cue.valid_for(asset.path) or cue.entry_s is None or cue.exit_s is None:
            raise ValueError(f"no valid measured cues for {asset.manifest.asset_id}")
        return cue

    def _load(self, asset: DepotAsset) -> tuple[PreloadedTrack, CueMap]:
        cue = self._cue(asset)
        track = self.engine.preload(asset.path, bpm=asset.manifest.beat_grid.bpm,
                                    expected_hash=asset.manifest.content_sha256)
        persist_pcm(track, self.renderer.out_dir.parent / "prepared")
        return track, cue

    def _duration_eligible(self, asset: DepotAsset) -> bool:
        """Keep short legacy depot renders out of full-length live sets."""
        return asset.manifest.audio.duration_s >= self.min_track_duration_s

    def start(self, theme: str, *, session_id: str) -> dict[str, object]:
        with self._lock:
            if self.state is not None:
                raise RuntimeError("a live set is already running")
            self._stop.clear()
            matches = self.depot.search(theme, limit=80,
                                        min_duration_s=self.min_track_duration_s)
            for similarity, asset in matches:
                if not self._duration_eligible(asset):
                    continue
                try:
                    track, cue = self._load(asset)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.error = str(exc)
                    continue
                try:
                    self.engine.mixer.set_fallback(track)
                    self.engine.start()
                    self.engine.set_loop(True, 0)
                    self.engine.load_and_play(track, 0,
                        entry_frame=0)
                    self.current, self.cue, self.current_track = asset, cue, track
                    self.error = None
                    self.state = SessionState(session_id=session_id, theme=theme,
                        clock=MusicalClock(bpm=asset.manifest.beat_grid.bpm),
                        fallback_asset_id=asset.manifest.asset_id)
                    self.state.deck(DeckId.A).asset_id = asset.manifest.asset_id
                    self.state.deck(DeckId.A).playing = True
                    self._event({"event": "start", "asset_id": asset.manifest.asset_id,
                                 "theme_score": round(similarity, 3),
                                 "play_start_s": 0.0, "mix_entry_s": cue.entry_s,
                                 "exit_s": cue.exit_s,
                                 "file_duration_s": asset.manifest.audio.duration_s})
                    self._stop.clear()
                    self._thread = threading.Thread(target=self._run, daemon=True)
                    self._thread.start()
                    self._queue_generation(theme, session_id)
                    return self.status()
                except Exception as exc:
                    self.error = f"audio device could not start: {exc}"
                    return {"running": False, "generating": False, "reason": self.error}
            queued = self._queue_generation(theme, session_id)
            return {"running": False, "generating": queued,
                    "reason": "no full-length cueable depot match; YuE2 generation queued" if queued
                              else ("no full-length cueable depot match; "
                                    "generation already attempted")}

    def _queue_generation(self, theme: str, session_id: str, *, replenish: bool = False) -> bool:
        if theme in self._generation_done_themes and not replenish:
            return False
        if self._generating:
            if theme != self._generation_theme:
                self._generation_theme = theme
                self._render_cancel.set()
            return True
        self._generation_theme = theme
        if replenish:
            self._generation_done_themes.discard(theme)
            self._event({"event": "generation_replenish", "theme": theme})
        self._generating = True
        self.error = None
        self._generation_completed = 0
        self._generation_failed = 0
        self._generation_current = 0
        self._generation_prompt_origin = None
        if self.state is None:
            self._pending_start = (theme, session_id)
        self._generation_thread = threading.Thread(target=self._generate, daemon=True)
        self._generation_thread.start()
        return True

    def _generate(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                theme = self._generation_theme
                self._render_cancel = threading.Event()
                self._generation_completed = 0
                self._generation_failed = 0
                self._generation_current = 0
            if theme is None:
                break
            try:
                briefs = plan_theme(theme, cancel=self._render_cancel)
            except Exception as exc:
                with self._lock:
                    self.error = f"track prompt planning failed: {exc}"
                    self._generating = False
                    self._event({"event": "generation_plan_failed", "theme": theme,
                                 "reason": str(exc)})
                break
            with self._lock:
                if self._generation_theme != theme:
                    continue
                self._generating = True
                self._generation_prompt_origin = briefs[0].origin
                self._event({"event": "generation_planned", "theme": theme,
                                     "count": len(briefs), "origin": briefs[0].origin})
            attempts = 0
            last_failure = None
            setup_failed = False
            while self._generation_completed < TRACK_COUNT and attempts < TRACK_COUNT * 3:
                with self._lock:
                    if self._stop.is_set() or self._generation_theme != theme:
                        break
                    index = self._generation_completed + 1
                    brief = briefs[index - 1]
                    titles = {title_for_asset(asset).casefold()
                              for asset in getattr(self.depot, "assets", {}).values()}
                    title = unique_title(
                        brief.title or fallback_title(theme, brief.style, str(index)), titles
                    )
                    attempts += 1
                    self._generation_current = index
                    self._event({"event": "generation_started", "track": index,
                                         "attempt": attempts, "title": title,
                                         "duration_s": brief.duration_s})
                try:
                    asset = generate_theme(theme, self.depot.library,
                        duration_s=brief.duration_s, energy=brief.energy,
                        style=brief.style, prompt_origin=brief.origin, lyrics=brief.lyrics,
                        cancel=self._render_cancel, title=title,
                        **({"music_plan": brief.music_plan} if brief.music_plan else {}))
                    with self._lock:
                        if self._stop.is_set() or self._generation_theme != theme:
                            break
                        self.depot.add(asset.manifest, asset.path)
                        self._generation_completed += 1
                        self.error = None
                        self._prepared_for = None
                        self._event({"event": "generated", "track": index,
                                             "asset_id": asset.manifest.asset_id})
                        pending = self._pending_start
                        if pending is not None and pending[0] == theme:
                            self._pending_start = None
                    if not self._stop.is_set() and pending is not None and pending[0] == theme:
                        self.start(theme, session_id=pending[1])
                except Exception as exc:
                    if self._stop.is_set() or self._generation_theme != theme:
                        break
                    with self._lock:
                        self._generation_failed += 1
                        last_failure = str(exc)
                        self.error = f"generation track {index}, attempt {attempts} failed: {exc}"
                        self._event({"event": "generation_failed", "track": index,
                                             "attempt": attempts, "reason": str(exc)})
                    if isinstance(exc, BackendUnavailable):
                        setup_failed = True
                        break
            with self._lock:
                if self._generation_theme != theme:
                    continue
                if not self._stop.is_set() and self._generation_completed == TRACK_COUNT:
                    self._generation_done_themes.add(theme)
                elif not self._stop.is_set():
                    if setup_failed:
                        self.error = f"YuE2 unavailable: {last_failure}"
                    else:
                        self.error = (
                            f"batch incomplete: {self._generation_completed}/{TRACK_COUNT} "
                            f"admitted after {attempts} attempts; last failure: {last_failure}"
                        )
                    self._event({"event": "generation_batch_failed", "attempts": attempts,
                                 "completed": self._generation_completed, "reason": self.error})
                self._generating = False
                self._generation_current = 0
                break

    def steer(self, theme: str) -> dict[str, object]:
        with self._lock:
            if self.state is None:
                if self._pending_start is None:
                    raise RuntimeError("no live set")
                self._pending_start = (theme, self._pending_start[1])
                self._event({"event": "steer_pending", "theme": theme})
                self._queue_generation(theme, self._pending_start[1])
                return self.status()
            self.state.theme = theme
            self.state.generation += 1
            self.prepared = None
            self._prepared_loaded = False
            self._prepared_for = None
            self._event({"event": "steer", "theme": theme,
                                 "generation": self.state.generation})
            self._queue_generation(theme, self.state.session_id)
            return self.status()

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            try:
                underruns = self.engine.mixer.stats.underruns
                if underruns > self._last_audio_underruns:
                    self._last_audio_underruns = underruns
                    self._event({"event": "audio_underrun", "count": underruns,
                                 "generating": self._generating,
                                 "audio": self.engine.snapshot()})
                self.tick()
            except Exception as exc:
                self.error = str(exc)
                LOGGER.exception("live scheduling tick failed")

    def tick(self) -> None:
        with self._lock:
            if self.state is None or self.current is None or self.cue is None:
                return
            deck = self.engine.mixer.decks[self.active_deck]
            self.state.clock.bar = self.engine.mixer.bar
            self.state.clock.beat = self.engine.mixer.clock.beat
            position_s = deck.position / self.engine.sample_rate
            exit_s = self.cue.exit_s
            if exit_s is None:
                return
            key = (self.current.manifest.asset_id, self.state.generation)
            if (self.prepared is None and not self._preparing
                    and self._prepared_for != key and position_s < exit_s - 5.0):
                self._preparing = True
                self._prepared_for = key
                generation = self.state.generation
                self._prepare_thread = threading.Thread(target=self._prepare, args=(generation,))
                self._prepare_thread.start()
            incoming = 1 - self.active_deck
            standby = self.engine.mixer.decks[incoming]
            if (self.prepared is not None and not self._prepared_loaded
                    and standby.fade_remaining == 0
                    and (not standby.playing or standby.fade_gain == 0.0)):
                self.engine.prepare_deck(self.prepared.track, incoming)
                self.state.deck(DeckId.B if incoming else DeckId.A).asset_id = (
                    self.prepared.asset.manifest.asset_id)
                self._prepared_loaded = True
                self._event({"event": "deck_preloaded", "deck": incoming,
                                     "asset_id": self.prepared.asset.manifest.asset_id})
            if position_s >= exit_s and self.prepared is not None and self._prepared_loaded:
                self._commit(self.prepared)

    def _prepare(self, generation: int) -> None:
        try:
            with self._lock:
                if self.state is None or self.current is None or self.cue is None:
                    return
                theme, current, cue = self.state.theme, self.current, self.cue
                current_track = self.current_track
                target_bpm = self.state.clock.bpm
                played = [row["asset_id"] for row in self.history
                          if row.get("event") in {"start", "transition"}
                          and "asset_id" in row]
                recent = set(played[-4:])
                matches = self.depot.search(theme, limit=80,
                                            min_duration_s=self.min_track_duration_s)
                shortlist = [asset for _, asset in matches
                             if asset.manifest.asset_id not in recent
                             and self._duration_eligible(asset)]
                filtered = self.filter.build(state=self.state,
                    library=[asset.manifest for asset in shortlist],
                    current_asset=current.manifest,
                    recent_family_ids=(current.manifest.family_id,))
                allowed = {action.asset_id for action in filtered.transitions}
                self._event({"event": "candidate_filter", "allowed": len(allowed),
                             "rejected": len(filtered.rejections),
                             "reasons": [str(row) for row in filtered.rejections[:20]]})
            if current_track is None:
                return
            ranked: list[Prepared] = []
            for asset in shortlist:
                if asset.manifest.asset_id not in allowed:
                    continue
                try:
                    track, next_cue, report = prepare_candidate(
                        asset, current=current, current_track=current_track, current_cue=cue,
                        target_bpm=target_bpm, cache_dir=self.renderer.out_dir.parent / "prepared",
                        preview_dir=self.renderer.out_dir, cancel=self._stop,
                        name=f"live_{generation}_{current.manifest.asset_id}_{asset.manifest.asset_id}",
                    )
                    ranked.append(Prepared(asset, track, next_cue, report))
                except (OSError, ValueError, RuntimeError) as exc:
                    self._event({"event": "candidate_failed",
                                 "asset_id": asset.manifest.asset_id, "reason": str(exc)})
                    continue
            ranked.sort(key=lambda item: (-item.report.score, item.asset.manifest.asset_id))
            with self._lock:
                if (not self._stop.is_set() and self.state is not None
                        and self.state.generation == generation):
                    self.prepared = ranked[0] if ranked else None
                    self._prepared_loaded = False
                    self._event({"event": "prepared", "asset_id":
                        self.prepared.asset.manifest.asset_id if self.prepared else None,
                        "score": self.prepared.report.score if self.prepared else None,
                        "rms_jump_db": self.prepared.report.rms_jump_db if self.prepared else None,
                        "spectral_distance": (self.prepared.report.spectral_distance
                                              if self.prepared else None)})
                    if self.prepared is None and not self._generating:
                        self._queue_generation(theme, self.state.session_id, replenish=True)
        except Exception as exc:
            self.error = f"transition preparation failed: {exc}"
            LOGGER.exception("transition preparation failed")
        finally:
            with self._lock:
                self._preparing = False

    def _commit(self, next_track: Prepared) -> None:
        assert self.state is not None
        incoming = 1 - self.active_deck
        self.engine.set_loop(True, incoming)
        self.engine.crossfade_to(next_track.track, incoming,
            fade_frames=round(4.0 * self.engine.sample_rate),
            entry_frame=0)
        self.engine.set_loop(False, self.active_deck)
        outgoing_id = self.current.manifest.asset_id if self.current else None
        self.active_deck = incoming
        self.current, self.cue, self.current_track = (
            next_track.asset, next_track.cue, next_track.track
        )
        self.state.deck(DeckId.A).asset_id = (next_track.asset.manifest.asset_id
            if incoming == 0 else outgoing_id)
        self.state.deck(DeckId.B).asset_id = (next_track.asset.manifest.asset_id
            if incoming == 1 else outgoing_id)
        self._event({"event": "transition", "from": outgoing_id,
                             "asset_id": next_track.asset.manifest.asset_id,
                             "incoming_deck": incoming, "play_start_s": 0.0,
                             "score": next_track.report.score,
                             "preview": str(next_track.report.preview_path)})
        self.prepared = None
        self._prepared_loaded = False
        self._prepared_for = None

    def status(self) -> dict[str, object]:
        active = self.engine.mixer.decks[self.active_deck]
        assets = getattr(self.depot, "assets", {})
        # Only send names shown by the decks/history, not the entire depot on every poll.
        visible_ids = {event.get("asset_id") for event in self.history[-12:]}
        if self.state:
            visible_ids.update(self.state.deck(deck).asset_id for deck in (DeckId.A, DeckId.B))
        titles = {key: title_for_asset(assets[key]) for key in visible_ids if key in assets}
        return {"asset_titles": titles,
                "current_title": title_for_asset(self.current),
                "prepared_title": title_for_asset(self.prepared.asset) if self.prepared else None,
                "running": self.engine.running, "session_id":
                self.state.session_id if self.state else
                self._pending_start[1] if self._pending_start else None,
                "theme": self.state.theme if self.state else
                self._pending_start[0] if self._pending_start else None,
                "current_asset_id": self.current.manifest.asset_id if self.current else None,
                "prepared_asset_id": (
                    self.prepared.asset.manifest.asset_id if self.prepared else None
                ),
                "active_deck": self.active_deck, "generating": self._generating,
                "prepared_loaded": self._prepared_loaded,
                "generation": {"target": TRACK_COUNT, "completed": self._generation_completed,
                               "failed": self._generation_failed,
                               "current": self._generation_current,
                               "theme": self._generation_theme,
                               "prompt_origin": self._generation_prompt_origin},
                "bpm": self.state.clock.bpm if self.state else None,
                "deck_a_asset_id": self.state.deck(DeckId.A).asset_id if self.state else None,
                "deck_b_asset_id": self.state.deck(DeckId.B).asset_id if self.state else None,
                "current_position_s": round(active.position / self.engine.sample_rate, 2),
                "current_duration_s": self.current_track.duration_s if self.current_track else None,
                "prepared_duration_s": (self.prepared.track.duration_s
                                         if self.prepared else None),
                "current_exit_s": self.cue.exit_s if self.cue else None,
                "spectrum": spectrum_levels(self.engine.mixer),
                "audio": self.engine.snapshot(),
                "events": self.history[-12:], "error": self.error}

    def stop(self) -> None:
        self._stop.set()
        self._render_cancel.set()
        self._pending_start = None
        if self._thread is not None:
            self._thread.join()
        self.engine.stop()
        if self._prepare_thread is not None:
            self._prepare_thread.join()
        if (self._generation_thread is not None
                and self._generation_thread is not threading.current_thread()):
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
        self._generation_failed = 0
        self._generation_current = 0
        self._generating = False
        self.active_deck = 0
