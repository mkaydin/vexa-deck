"""Asset contracts.

The central idea from ``ARCHITECTURE.md:33-44`` is that an asset is never a flat file but a node in
a *family*: a YuE2 render, its revisions, the sections and loops cut from it, and any separated
stems. Relationships are therefore explicit and versioned rather than implied by file layout.

Every model-produced field is an :class:`Estimate` carrying a confidence, never a bare value.
``ARCHITECTURE.md:44``: *"Treat model-produced tags as estimates with confidence, not facts."*
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ApprovalState, BackendKind, ReadinessState, SourceType
from .version import CONTRACT_REVISION, CONTRACT_VERSION


class Contract(BaseModel):
    """Base for everything that crosses a process boundary."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: int = CONTRACT_VERSION
    contract_revision: str = CONTRACT_REVISION


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Estimate(BaseModel):
    """A value the models produced, with how much to trust it."""

    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: float = Field(ge=0.0, le=1.0)

    def trustworthy(self, threshold: float) -> bool:
        return self.confidence >= threshold


class KeyEstimate(Estimate):
    """Musical key plus its confidence. Used for the harmonic compatibility filter."""

    #: CamelCase keys, e.g. ``"A minor"``. Kept as a string so an unusual estimate is never lost.
    value: str


class SectionMarker(BaseModel):
    """A detected structural section, expressed in bars so it survives tempo changes."""

    model_config = ConfigDict(extra="forbid")

    section_id: str
    #: e.g. ``intro``, ``verse``, ``chorus``, ``outro``.
    kind: str
    start_bar: int = Field(ge=0)
    end_bar: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> SectionMarker:
        if self.end_bar <= self.start_bar:
            raise ValueError(f"section {self.section_id!r} ends at or before it starts")
        return self


class LoopPoints(BaseModel):
    """An approved loop region. Must be beat-aligned to be schedulable."""

    model_config = ConfigDict(extra="forbid")

    start_bar: int = Field(ge=0)
    end_bar: int = Field(ge=0)
    beat_aligned: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> LoopPoints:
        if self.end_bar <= self.start_bar:
            raise ValueError("loop ends at or before it starts")
        return self


class BeatGrid(BaseModel):
    """Measured tempo grid. ``grid_version`` changes whenever the grid is recomputed."""

    model_config = ConfigDict(extra="forbid")

    bpm: float = Field(gt=0.0)
    grid_version: int = Field(ge=1)
    #: 0..1. A low value means the grid is ambiguous and automatic transitions should be refused
    #: (``ROADMAP.md:52``).
    confidence: float = Field(ge=0.0, le=1.0)
    time_signature: tuple[int, int] = (4, 4)

    @property
    def reliable(self) -> bool:
        return self.confidence >= 0.6


class AudioProperties(BaseModel):
    """Container and measured level. Loudness and peak are measured, never estimated."""

    model_config = ConfigDict(extra="forbid")

    codec: str
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(ge=1)
    duration_s: float = Field(gt=0.0)
    #: Integrated loudness in LUFS (ITU-R BS.1770-4).
    integrated_lufs: float | None = None
    #: Oversampled true peak in dBTP.
    true_peak_dbtp: float | None = None

    def has_headroom(self, ceiling_dbtp: float = -1.0) -> bool:
        """Whether the true peak leaves room for a master limiter and a crossfade."""
        return self.true_peak_dbtp is not None and self.true_peak_dbtp <= ceiling_dbtp


class QualityReport(BaseModel):
    """The ingestion gates from ``ARCHITECTURE.md:46``.

    All six must pass before an asset may be scheduled. ``passed`` is derived rather than stored
    so a caller cannot mark a failing asset ready by setting a flag.
    """

    model_config = ConfigDict(extra="forbid")

    decode_ok: bool = False
    duration_ok: bool = False
    loudness_ok: bool = False
    beat_grid_ok: bool = False
    loop_boundary_ok: bool = False
    audio_quality_ok: bool = False
    flags: list[str] = Field(default_factory=list)

    #: Free-text per-check notes for the review UI. Not machine-enforced.
    notes: dict[str, str] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.decode_ok
            and self.duration_ok
            and self.loudness_ok
            and self.beat_grid_ok
            and self.loop_boundary_ok
            and self.audio_quality_ok
        )

    @property
    def failures(self) -> list[str]:
        checks = {
            "decode": self.decode_ok,
            "duration": self.duration_ok,
            "loudness": self.loudness_ok,
            "beat_grid": self.beat_grid_ok,
            "loop_boundary": self.loop_boundary_ok,
            "audio_quality": self.audio_quality_ok,
        }
        return [name for name, ok in checks.items() if not ok]


class Provenance(BaseModel):
    """Everything needed to reproduce an asset (``README.md:22``).

    A quantized render is a materially different artifact from a bf16 one, so the backend and its
    quant level are recorded here rather than treated as an implementation detail.
    """

    model_config = ConfigDict(extra="forbid")

    backend: BackendKind | None = None
    model_id: str | None = None
    model_revision: str | None = None
    #: e.g. ``Q8_0``, ``BF16``. ``None`` for non-quantized backends.
    quantization: str | None = None
    decoder_id: str | None = None
    source_prompt: str | None = None
    seed: int | None = None
    license: str | None = None
    #: Content hashes of every input that fed this asset, for reproducibility.
    source_hashes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class AssetManifest(Contract):
    """One playable material.

    The ``READY`` invariant is enforced structurally: a manifest may only claim readiness when it
    is approved *and* every quality gate passed. This is the single most important invariant in
    the library, because everything downstream is allowed to assume ``READY`` means playable.
    """

    asset_id: str = Field(min_length=1)
    #: Optional display label; does not identify the audio or participate in content hashes.
    title: str | None = Field(default=None, min_length=1, max_length=80)
    family_id: str = Field(min_length=1)
    source_type: SourceType
    parent_id: str | None = None

    #: Content hash. A path is not an identity (``PLAN.md`` §6), so the scheduler binds to this.
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: int = Field(default=1, ge=1)

    audio: AudioProperties
    beat_grid: BeatGrid | None = None
    key: KeyEstimate | None = None

    sections: list[SectionMarker] = Field(default_factory=list)
    loops: list[LoopPoints] = Field(default_factory=list)

    tags: dict[str, list[Estimate]] = Field(default_factory=dict)

    quality: QualityReport = Field(default_factory=QualityReport)
    approval: ApprovalState = ApprovalState.PENDING
    readiness: ReadinessState = ReadinessState.UNANALYZED
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _ready_implies_playable(self) -> AssetManifest:
        if self.readiness is not ReadinessState.READY:
            return self
        if self.approval is not ApprovalState.APPROVED:
            raise ValueError(
                f"asset {self.asset_id!r} claims READY but approval is {self.approval.value!r}; "
                "only approved assets enter the ready library"
            )
        if not self.quality.passed:
            raise ValueError(
                f"asset {self.asset_id!r} claims READY but gates failed: "
                f"{', '.join(self.quality.failures)}"
            )
        return self

    @model_validator(mode="after")
    def _regions_inside_duration(self) -> AssetManifest:
        bars = self._bars()
        if bars is None:
            return self
        for section in self.sections:
            if section.end_bar > bars:
                raise ValueError(
                    f"section {section.section_id!r} ends at bar {section.end_bar}, "
                    f"past the end of a {bars}-bar asset"
                )
        for loop in self.loops:
            if loop.end_bar > bars:
                raise ValueError(f"loop ends at bar {loop.end_bar}, past a {bars}-bar asset")
        return self

    def _bars(self) -> int | None:
        """Total bars, or ``None`` when the tempo is unknown and bars are meaningless."""
        if self.beat_grid is None:
            return None
        seconds_per_bar = 240.0 / self.beat_grid.bpm
        return int(self.audio.duration_s / seconds_per_bar)

    def admissible(self, *, min_confidence: float = 0.6) -> bool:
        """Whether the scheduler may consider this asset.

        Requires readiness, approval, a reliable beat grid for cueing, and a section or loop
        marked as a safe entry point.
        """
        return (
            self.readiness is ReadinessState.READY
            and self.approval is ApprovalState.APPROVED
            and self.beat_grid is not None
            and self.beat_grid.reliable
            and bool(self.sections or self.loops)
        )

    def energy(self) -> float | None:
        """Mean tagged energy in 0..1, or ``None`` when the asset is untagged.

        Reads the tag *value*, not its confidence. A track tagged ``0.9`` confidence is not a
        0.9-energy track, and conflating the two would silently break every energy filter.
        """
        estimates = self.tags.get("energy")
        if not estimates:
            return None
        values: list[float] = []
        for estimate in estimates:
            try:
                values.append(float(estimate.value))
            except ValueError:
                return None
        return sum(values) / len(values)