"""The readiness gates.

``ARCHITECTURE.md:46``: an asset becomes ``ready`` only after decode, duration, loudness,
beat-grid, loop-boundary and audio-quality checks pass.

This module is deliberately blunt. A gate that passes a broken file is worse than no gate at all,
because it moves the failure from ingestion — where a human sees it — to a live set. So every
threshold is stated, every failure names itself, and nothing here silently repairs anything.

The thresholds are starting values, not gospel. ``PLAN.md`` Phase B2 exists to measure and adjust
them against real generated material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

from .analysis import (
    AnalysisError,
    AudioProbe,
    TemporalAnalysis,
    detect_temporal,
    measure_loudness,
    measure_true_peak,
    probe_audio,
)


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Starting values. See the module docstring: these are meant to be measured, not assumed."""

    #: Integrated loudness band a set is allowed to live in.
    min_lufs: float = -24.0
    max_lufs: float = -6.0
    #: True-peak ceiling. The master limiter sits below this.
    max_true_peak_dbtp: float = -1.0
    #: Acceptable loudness spread inside one track, in LU.
    max_loudness_range_lu: float = 12.0
    #: Duration band for a library asset.
    min_duration_s: float = 5.0
    max_duration_s: float = 900.0
    #: RMS below this is silence.
    silence_rms: float = 1e-4
    #: Fraction of samples at full scale that counts as clipping.
    max_clipped_sample_ratio: float = 1e-4
    #: Beat-grid confidence required to schedule against.
    min_beat_confidence: float = 0.6
    #: Key confidence required before a mode clash may refuse a transition.
    min_key_confidence: float = 0.5
    #: Sample rates the library accepts.
    allowed_sample_rates: tuple[int, ...] = (44100, 48000)
    required_channels: int = 2


@dataclass(slots=True)
class GateOutcome:
    """The verdict plus everything that produced it, so a rejection can be explained."""

    quality: QualityReport
    probe: AudioProbe | None = None
    temporal: TemporalAnalysis | None = None
    loudness_lufs: float | None = None
    true_peak_dbtp: float | None = None
    #: The manifest this analysis produced, or ``None`` when the file could not be read.
    manifest: AssetManifest | None = None
    #: Set when the file could not be read at all, as opposed to failing a check.
    fatal: str | None = None

    @property
    def admitted(self) -> bool:
        return self.fatal is None and self.quality.passed


def analyse(
    path: str | Path,
    *,
    asset_id: str,
    family_id: str | None = None,
    source_type: SourceType = SourceType.YUE2_RENDER,
    provenance: Provenance | None = None,
    approval: ApprovalState = ApprovalState.PENDING,
    thresholds: GateThresholds | None = None,
    parent_id: str | None = None,
    analyse_tempo: bool = True,
) -> GateOutcome:
    """Run every gate over an audio file and produce a manifest that reflects the result.

    Returns an outcome whose ``quality`` report is honest whether it passed or failed. The caller
    decides what to do with a failure; nothing here mutates global state or enqueues anything.
    """
    t = thresholds or GateThresholds()
    quality = QualityReport()

    try:
        probe, samples = probe_audio(path)
    except AnalysisError as exc:
        return GateOutcome(quality=quality, fatal=str(exc))

    temporal: TemporalAnalysis | None = None
    if analyse_tempo:
        try:
            temporal = detect_temporal(samples, probe.sample_rate_hz)
        except AnalysisError as exc:
            quality.flags.append(f"temporal analysis unavailable: {exc}")

    loudness = measure_loudness(samples, probe.sample_rate_hz)
    true_peak = measure_true_peak(samples, probe.sample_rate_hz)

    # -- decode -------------------------------------------------------------
    quality.decode_ok = True
    if probe.channels != t.required_channels:
        quality.decode_ok = False
        quality.flags.append(
            f"expected {t.required_channels} channels, decoded {probe.channels}"
        )
    if probe.sample_rate_hz not in t.allowed_sample_rates:
        quality.decode_ok = False
        quality.flags.append(
            f"sample rate {probe.sample_rate_hz} not in {t.allowed_sample_rates}"
        )

    # -- duration -----------------------------------------------------------
    quality.duration_ok = t.min_duration_s <= probe.duration_s <= t.max_duration_s
    if not quality.duration_ok:
        quality.flags.append(
            f"duration {probe.duration_s:.1f}s outside "
            f"[{t.min_duration_s:.1f}, {t.max_duration_s:.1f}]"
        )

    # -- loudness -----------------------------------------------------------
    quality.loudness_ok = (
        loudness.integrated_lufs >= t.min_lufs
        and loudness.integrated_lufs <= t.max_lufs
        and true_peak <= t.max_true_peak_dbtp
        and loudness.loudness_range_lu <= t.max_loudness_range_lu
    )
    if not quality.loudness_ok:
        quality.flags.append(
            f"loudness {loudness.integrated_lufs:.1f} LUFS, peak {true_peak:.1f} dBTP, "
            f"range {loudness.loudness_range_lu:.1f} LU outside policy"
        )

    # -- beat grid ----------------------------------------------------------
    if temporal is None or temporal.bpm <= 0:
        quality.beat_grid_ok = False
        quality.flags.append("no tempo detected")
    else:
        quality.beat_grid_ok = temporal.beat_confidence >= t.min_beat_confidence
        if not quality.beat_grid_ok:
            quality.flags.append(
                f"beat grid confidence {temporal.beat_confidence:.2f} below "
                f"{t.min_beat_confidence:.2f}; ambiguous, refuse automatic transitions"
            )

    # -- audio quality ------------------------------------------------------
    quality.audio_quality_ok = (
        probe.rms >= t.silence_rms and probe.clipped_sample_ratio <= t.max_clipped_sample_ratio
    )
    if probe.rms < t.silence_rms:
        quality.flags.append("file is silent")
    if probe.clipped_sample_ratio > t.max_clipped_sample_ratio:
        quality.flags.append(
            f"{probe.clipped_sample_ratio:.4%} of samples are clipped"
        )

    # -- loop boundary ------------------------------------------------------
    # Loop boundaries. This gate *verifies proposed loops*; it does not require one to exist.
    # A full track with usable sections is perfectly playable without an 8-bar loop, and
    # quarantining it would contradict ROADMAP.md:51's "prefer full tracks and verified sections".
    # What is refused is proposing a loop the grid cannot support.
    loops: list[LoopPoints] = []
    if temporal is not None and temporal.bpm > 0:
        seconds_per_bar = 240.0 / temporal.bpm
        total_bars = int(probe.duration_s / seconds_per_bar)
        if quality.beat_grid_ok and total_bars >= 16:
            # An 8-bar loop starting on bar 0 is beat-aligned by construction.
            loops.append(LoopPoints(start_bar=0, end_bar=8, beat_aligned=True))
        elif not quality.beat_grid_ok:
            quality.flags.append(
                "beat grid is ambiguous, so no loop will be proposed; sections remain usable"
            )
        elif 0 < total_bars < 16:
            quality.flags.append(
                f"only {total_bars} bars detected; too short to cut a verified 8-bar loop"
            )
    else:
        quality.flags.append("no tempo detected; sections remain the only entry points")

    # Vacuously true when no loop was proposed.
    quality.loop_boundary_ok = all(loop.beat_aligned for loop in loops)

    sections: list[SectionMarker] = []
    if temporal is not None and temporal.bpm > 0:
        seconds_per_bar = 240.0 / temporal.bpm
        total_bars = int(probe.duration_s / seconds_per_bar)
        if temporal.section_bars:
            edges = [0, *temporal.section_bars, max(total_bars, 1)]
            from itertools import pairwise

            for index, (start, end) in enumerate(pairwise(edges)):
                if end > start:
                    sections.append(
                        SectionMarker(
                            section_id=f"s{index}", kind="segment", start_bar=start, end_bar=end
                        )
                    )
        elif total_bars > 0:
            sections.append(
                SectionMarker(section_id="s0", kind="segment", start_bar=0, end_bar=total_bars)
            )

    quality.notes = {
        "decode": f"{probe.codec} {probe.sample_rate_hz} Hz {probe.channels}ch",
        "loudness": f"{loudness.integrated_lufs:.1f} LUFS, {true_peak:.1f} dBTP",
        "duration": f"{probe.duration_s:.2f}s, {probe.frames} frames",
        "quality": f"rms {probe.rms:.5f}, clipped {probe.clipped_sample_ratio:.5%}",
    }

    # Readiness follows both the gates and human sign-off. Claiming READY while unapproved would
    # be rejected by AssetManifest's own validator, so the state is derived rather than assumed.
    if not quality.passed:
        readiness = ReadinessState.QUARANTINED
    elif approval is ApprovalState.APPROVED:
        readiness = ReadinessState.READY
    else:
        readiness = ReadinessState.AWAITING_REVIEW

    manifest = AssetManifest(
        asset_id=asset_id,
        family_id=family_id or asset_id,
        source_type=source_type,
        parent_id=parent_id,
        content_sha256=_sha256(path),
        audio=AudioProperties(
            codec=probe.codec,
            sample_rate_hz=probe.sample_rate_hz,
            channels=probe.channels,
            duration_s=probe.duration_s,
            integrated_lufs=round(loudness.integrated_lufs, 2),
            true_peak_dbtp=round(true_peak, 2),
        ),
        beat_grid=(
            BeatGrid(
                bpm=round(temporal.bpm, 3),
                grid_version=1,
                confidence=temporal.beat_confidence,
            )
            if temporal and temporal.bpm > 0
            else None
        ),
        key=(
            KeyEstimate(value=temporal.key, confidence=temporal.key_confidence)
            if temporal and temporal.key
            else None
        ),
        sections=sections,
        loops=loops,
        quality=quality,
        approval=approval,
        readiness=readiness,
        provenance=provenance or Provenance(),
    )

    return GateOutcome(
        quality=quality,
        probe=probe,
        temporal=temporal,
        loudness_lufs=loudness.integrated_lufs,
        true_peak_dbtp=true_peak,
        manifest=manifest,
    )


def _sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class Quarantine:
    """Assets refused admission, with the reason, for the review UI."""

    entries: list[tuple[str, list[str]]] = field(default_factory=list)

    def add(self, asset_id: str, flags: list[str]) -> None:
        self.entries.append((asset_id, flags))


__all__ = [
    "BackendKind",
    "GateOutcome",
    "GateThresholds",
    "Quarantine",
    "analyse",
]