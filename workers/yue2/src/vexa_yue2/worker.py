"""The generation worker.

Claims a job, renders it with a chosen backend, analyses the result, runs the readiness gates, and
either admits the asset to the library or quarantines it.

The invariants, in order of how much damage breaking them would do:

1. **A failed or cancelled job has no effect on playback** (``ROADMAP.md:33``). Nothing here
   touches the audio engine, and every failure path is a job state, not a crash.
2. **Nothing is admitted without passing the gates.** A render is a *candidate*, not an asset. The
   gate result decides, and a quarantined render is kept with its reasons rather than deleted —
   it is evidence about the model.
3. **A backend is chosen by measured capacity**, not by hope. A job that no backend can take is
   reported as such rather than attempted and OOM-killed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vexa_contracts import (
    ApprovalState,
    AssetManifest,
    GenerationJob,
    GenerationResult,
    JobState,
    SourceType,
)

from .backends import (
    BackendUnavailable,
    GenerationBackend,
    GenerationFailed,
    device_vram_mib,
    resolve_device,
)
from .gates import GateThresholds, analyse


@dataclass(slots=True)
class WorkerConfig:
    #: Where renders are written before admission.
    render_dir: Path = Path("var/renders")
    #: Where admitted assets live. A render is moved here only once it passes.
    library_dir: Path = Path("var/library")
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    #: Beats per second of tempo analysis; lower is faster and less precise.
    analysis_tempo: bool = True
    #: Devices this worker may claim, by name fragment. Empty means "any".
    allowed_devices: tuple[str, ...] = ()
    #: Where the built backend lives, when it is not on PATH. YuE2's C++ runtime is an external
    #: binary that is never bundled, so this points at a local build rather than naming one.
    backend_binary: Path | None = None


@dataclass(slots=True)
class JobOutcome:
    """What happened to one job. Always explicit, never a silent drop."""

    job_id: str
    state: JobState
    backend: str | None = None
    asset: AssetManifest | None = None
    result: GenerationResult | None = None
    reason: str = ""
    wall_time_s: float = 0.0

    @property
    def admitted(self) -> bool:
        return self.asset is not None and self.asset.admissible()


class BackendSelector:
    """Picks a backend that can actually run the job on a device that exists."""

    def __init__(
        self, backends: Sequence[GenerationBackend], config: WorkerConfig | None = None
    ) -> None:
        self.backends = list(backends)
        self.config = config or WorkerConfig()

    def devices(self) -> list[int]:
        if self.config.allowed_devices:
            found = [resolve_device(name) for name in self.config.allowed_devices]
            return [i for i in found if i is not None]
        return [0, 1]

    def choose(self, job: GenerationJob) -> tuple[GenerationBackend | None, int | None, str]:
        """Return ``(backend, gpu_index, reason)``. ``None`` backend means nothing can take it."""
        # Smallest first: prefer the backend that fits, not the one with headroom to spare.
        candidates = sorted(self.backends, key=lambda b: b.capability.peak_vram_mib)
        # Skip reasons are kept, because "nothing could take this job" is useless on its own.
        skipped: list[str] = []

        for gpu in self.devices():
            vram = device_vram_mib(gpu)
            if vram <= 0:
                skipped.append(f"cuda:{gpu}: no device visible")
                continue
            for backend in candidates:
                need = backend.capability.peak_vram_mib
                if need > vram:
                    skipped.append(
                        f"{backend.kind.value} needs {need} MiB, cuda:{gpu} has {vram}"
                    )
                    continue
                # A brief that asks for a symbolic plan needs a backend that can emit one.
                if job.brief.cot != "off" and not backend.capability.supports_plan_edit:
                    skipped.append(
                        f"{backend.kind.value} cannot emit a plan for cot={job.brief.cot}"
                    )
                    continue
                ok, why = backend.available()
                if not ok:
                    skipped.append(f"{backend.kind.value}: {why}")
                    continue
                return backend, gpu, f"{backend.kind.value} fits {vram} MiB on cuda:{gpu}"
        return None, None, "; ".join(skipped) or "no backend available"

    def describe(self) -> list[dict[str, object]]:
        """A routing table, for diagnostics."""
        rows = []
        for gpu in self.devices():
            vram = device_vram_mib(gpu)
            for backend in self.backends:
                ok, why = backend.available()
                rows.append(
                    {
                        "backend": backend.kind.value,
                        "quant": backend.capability.quant,
                        "needs_mib": backend.capability.peak_vram_mib,
                        "device": f"cuda:{gpu}",
                        "device_mib": vram,
                        "fits": vram >= backend.capability.peak_vram_mib and ok,
                        "plan_edit": backend.capability.supports_plan_edit,
                        "available": ok,
                        "note": why,
                    }
                )
        return rows


class GenerationWorker:
    """Runs one job at a time, end to end."""

    def __init__(
        self,
        selector: BackendSelector,
        *,
        config: WorkerConfig | None = None,
        on_event: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self.selector = selector
        self.config = config or selector.config
        self.on_event = on_event or (lambda kind, data: None)

    def _emit(self, kind: str, **data: object) -> None:
        self.on_event(kind, data)

    def run(self, job: GenerationJob) -> JobOutcome:
        """Render, analyse, gate, admit. Never raises for a job-level failure."""
        started = time.time()
        job.attempts += 1

        backend, gpu, reason = self.selector.choose(job)
        if backend is None:
            job.error = reason
            job.transition(JobState.FAILED) if job.state is JobState.QUEUED else None
            return JobOutcome(job_id=job.job_id, state=job.state, reason=reason)

        job.backend = backend.kind
        job.gpu_index = gpu
        job.transition(JobState.RUNNING)
        self._emit("job_started", job_id=job.job_id, backend=backend.kind.value, gpu=gpu)

        out_path = self.config.render_dir / f"{job.job_id}.wav"
        try:
            result = backend.render(
                job.brief, out_path, seed=job.seed, gpu_index=gpu or 0,
                on_progress=lambda line: self._emit("job_progress", job_id=job.job_id, line=line),
            )
        except BackendUnavailable as exc:
            job.error = str(exc)
            job.transition(JobState.FAILED)
            return JobOutcome(
                job_id=job.job_id, state=job.state, backend=backend.kind.value, reason=str(exc)
            )
        except GenerationFailed as exc:
            # Retryable within the job's budget; beyond that it is failed.
            job.error = str(exc)
            if job.can_retry:
                job.transition(JobState.FAILED)
            else:
                job.transition(JobState.FAILED)
            return JobOutcome(
                job_id=job.job_id, state=job.state, backend=backend.kind.value, reason=str(exc),
                wall_time_s=time.time() - started,
            )

        job.transition(JobState.ANALYZING)
        outcome = self._admit(job, result, started)
        self._emit(
            "job_finished",
            job_id=job.job_id,
            state=outcome.state.value,
            admitted=outcome.admitted,
            reason=outcome.reason,
        )
        return outcome

    def _admit(self, job: GenerationJob, result: GenerationResult, started: float) -> JobOutcome:
        """Analyse a render and decide whether it may enter the library."""
        outcome = analyse(
            result.audio_path,
            asset_id=f"{job.job_id}",
            family_id=job.family_id,
            source_type=SourceType.YUE2_RENDER,
            approval=ApprovalState.PENDING,
            thresholds=self.config.thresholds,
            analyse_tempo=self.config.analysis_tempo,
            provenance=_provenance_from(job, result),
        )
        manifest = outcome.manifest

        if outcome.fatal is not None:
            job.error = outcome.fatal
            job.transition(JobState.FAILED)
            return JobOutcome(
                job_id=job.job_id, state=job.state, result=result, reason=outcome.fatal,
                wall_time_s=time.time() - started,
            )

        if not outcome.quality.passed:
            # Kept, with reasons. A quarantined render is evidence about the model, not waste.
            job.error = "; ".join(outcome.quality.flags)
            job.transition(JobState.REVIEW_NEEDED)
            self._emit(
                "asset_quarantined", job_id=job.job_id, flags=outcome.quality.flags
            )
            return JobOutcome(
                job_id=job.job_id, state=job.state, result=result, asset=manifest,
                reason=job.error, wall_time_s=time.time() - started,
            )

        job.transition(JobState.READY)
        job.asset_id = manifest.asset_id
        return JobOutcome(
            job_id=job.job_id, state=job.state, result=result, asset=manifest,
            reason="passed every gate; awaiting human approval",
            wall_time_s=time.time() - started,
        )


def _provenance_from(job: GenerationJob, result: GenerationResult):
    """Full reproduction record (``README.md:22``).

    A quantized render is a materially different artifact from a bf16 one, so the backend and its
    quant level travel with the asset rather than being treated as an implementation detail.
    """
    from vexa_contracts import Provenance

    return Provenance(
        backend=result.backend,
        model_id=result.model_id,
        quantization=result.quantization,
        source_prompt=job.brief.style,
        seed=result.seed,
        license="CC-BY-NC-4.0",
        source_hashes=[result.content_sha256],
    )


def priority_queue(jobs: Sequence[GenerationJob]) -> list[GenerationJob]:
    """Queued jobs, live requests first, then oldest.

    A live request outranks background library preparation, which is what makes "play now" feel
    immediate even while a batch is running.
    """
    queued = [j for j in jobs if j.state is JobState.QUEUED]
    return sorted(queued, key=lambda j: (j.priority, j.created_at))