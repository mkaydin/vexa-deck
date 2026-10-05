"""Generation backends.

One protocol, several implementations, selected per job by what the hardware can hold. This is the
producer half of the pipeline; ``analysis.py`` and ``gates.py`` are the consumer half.

The split exists because of a measured constraint. YuE2's reference PyTorch pipeline peaks at
**11.18 GiB** and does not fit an 8 GB card. The quantized GGUF backends trade a little fidelity
for a memory profile that does — ``yue2.cpp`` measures 3.8 GB at a reduced context — so both cards
can generate, concurrently. ``PLAN.md`` §3.3.

Two rules apply to every backend:

* **A backend never blocks the worker on anything the worker shares.** Cancellation is a process
  kill for the C++ backends, which is why they are subprocesses rather than in-process calls.
* **The result carries its own provenance.** A Q8_0 render is a different artifact from a bf16
  one, and the manifest has to be able to say which it is.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vexa_contracts import BackendKind, GenerationBrief, GenerationJob, GenerationResult

from .runtime import native_environment


class BackendUnavailable(RuntimeError):
    """The backend cannot run here. Never fatal: another backend, or the queue, takes over."""


class GenerationFailed(RuntimeError):
    """The backend ran and failed. Retryable within the job's budget."""


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """What a backend needs, so the scheduler can route without guessing."""

    kind: BackendKind
    #: Minimum device VRAM in MiB. Measured peak, not a guess.
    peak_vram_mib: int
    #: Whether a brief can be planned, edited as ABC, and re-rendered.
    supports_plan_edit: bool
    #: The executable this backend shells out to, or None for the in-process one.
    executable: str | None = None
    quant: str | None = None


#: Measured peaks. See PLAN.md §3.2 for provenance.
CAPABILITIES: dict[BackendKind, BackendCapability] = {
    BackendKind.YUE2_CPP: BackendCapability(
        kind=BackendKind.YUE2_CPP,
        peak_vram_mib=5800,
        supports_plan_edit=True,
        executable="yue-synth",
        quant="Q8_0",
    ),
    BackendKind.AUDIO_CPP: BackendCapability(
        kind=BackendKind.AUDIO_CPP,
        peak_vram_mib=8867,
        supports_plan_edit=False,
        executable="audiocpp_cli",
        quant="Q8_0",
    ),
    BackendKind.TORCH: BackendCapability(
        kind=BackendKind.TORCH,
        peak_vram_mib=11180,
        supports_plan_edit=True,
        quant="BF16",
    ),
}


#: The project's primary and secondary devices, by name.
#:
#: nvidia-smi index 0 is the 5060 Ti while CUDA device 0 is the 4060, so any index-based default
#: silently lands on the wrong card. Every selection resolves through ``resolve_device``.
PRIMARY_GPU_NAME = "RTX 5060 Ti"
SECONDARY_GPU_NAME = "RTX 4060"


def primary_gpu() -> int:
    """Index of the 5060 Ti, or 0 if it is absent."""
    return resolve_device(PRIMARY_GPU_NAME) or 0


def secondary_gpu() -> int:
    """Index of the 4060, falling back to the primary if only one card is present."""
    found = resolve_device(SECONDARY_GPU_NAME)
    return primary_gpu() if found is None else found


def device_vram_mib(index: int) -> int:
    """Total VRAM of a CUDA device, in MiB. 0 when torch is unavailable."""
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        return 0
    return int(torch.cuda.get_device_properties(index).total_memory // (1024**2))


def resolve_device(name_fragment: str) -> int | None:
    """Find a GPU by *name*.

    ``nvidia-smi`` and torch enumerate these cards in opposite order, so an index-based lookup
    silently picks the wrong one (``PLAN.md`` §3.1).
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    for index in range(torch.cuda.device_count()):
        if name_fragment.lower() in torch.cuda.get_device_name(index).lower():
            return index
    return None


class GenerationBackend(ABC):
    """Renders one brief to one audio file."""

    @property
    @abstractmethod
    def kind(self) -> BackendKind: ...

    @property
    @abstractmethod
    def capability(self) -> BackendCapability: ...

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Whether this backend can run here, and why not if it cannot."""

    @abstractmethod
    def render(
        self,
        brief: GenerationBrief,
        out_path: Path,
        *,
        seed: int | None,
        gpu_index: int,
        on_progress: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        """Render to ``out_path``. Blocking; the worker owns the thread."""

    def plan(self, brief: GenerationBrief, out_path: Path) -> str:
        """Render only the symbolic score, as ABC text.

        This is what makes the agentic-edit workflow possible: read the score, change a chord,
        re-render. Backends without it raise, rather than pretending.
        """
        raise BackendUnavailable(f"{self.kind.value} does not expose plan-only rendering")


class SubprocessBackend(GenerationBackend):
    """Base for the C++/GGML backends, which are external executables.

    A subprocess rather than an in-process call for three reasons: cancelling a render becomes a
    process kill instead of a cooperative flag, a crash takes down only the render, and the GPU is
    released when the process exits.
    """

    #: Base seconds before a render is treated as hung and killed.
    timeout_s: float = 900.0

    def __init__(self, *, model_path: Path | None = None, vae_path: Path | None = None,
                 binary: str | None = None, workdir: Path | None = None) -> None:
        self.model_path = model_path
        self.vae_path = vae_path
        self.binary = binary
        self.workdir = workdir or Path("var/renders")
        #: Peak VRAM sampled during the last render, when the backend can report it.
        self.last_peak_vram_mib: int | None = None

    def executable(self) -> str:
        """Locate the backend binary: an explicit path, then PATH."""
        name = self.binary or self.capability.executable or ""
        found = shutil.which(name) or (name if Path(name).is_file() else None)
        if not found:
            raise BackendUnavailable(
                f"{self.kind.value}: executable {name!r} not found on PATH. "
                "Build the backend separately; model weights are never bundled."
            )
        return found

    def available(self) -> tuple[bool, str]:
        name = self.binary or self.capability.executable or ""
        if not (shutil.which(name) or Path(name).is_file()):
            return False, f"executable {name!r} not on PATH"
        if self.model_path is not None and not Path(self.model_path).exists():
            return False, f"model weights missing at {self.model_path}"
        return True, "ok"

    def _run(self, argv: list[str], *, on_progress: Callable[[str], None] | None) -> str:
        """Run the backend, streaming its log. A hang is killed, not waited on."""
        started = time.time()
        self.workdir.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.workdir,
                env=native_environment(argv[0]),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GenerationFailed(
                f"{self.kind.value} exceeded {self.timeout_s:.0f}s and was killed"
            ) from exc
        except OSError as exc:
            raise BackendUnavailable(f"{self.kind.value}: could not launch backend: {exc}") from exc

        if on_progress and completed.stderr:
            on_progress(completed.stderr.strip().splitlines()[-1] if completed.stderr else "")
        if completed.returncode != 0:
            raise GenerationFailed(
                f"{self.kind.value} exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout or '').strip()[:400]}"
            )
        self.last_wall_time_s = time.time() - started
        return completed.stdout

    last_wall_time_s: float = 0.0

    def _result(
        self, job: GenerationJob, out_path: Path, *, score: str | None = None
    ) -> GenerationResult:
        import soundfile as sf

        from vexa_yue2.loader import file_sha256

        info = sf.info(str(out_path))
        return GenerationResult(
            job_id=job.job_id,
            backend=self.kind,
            quantization=self.capability.quant,
            model_id=str(self.model_path) if self.model_path else None,
            seed=job.seed,
            audio_path=str(out_path),
            content_sha256=file_sha256(out_path),
            duration_s=float(info.frames) / float(info.samplerate),
            score_abc=score,
            wall_time_s=self.last_wall_time_s,
            peak_vram_mib=self.last_peak_vram_mib,
        )


class YueCppBackend(SubprocessBackend):
    """`yue2.cpp` — the compact default. Q8_0, and the only GGML runtime here that exposes
    plan-only rendering, which the agentic-edit workflow needs."""

    @property
    def kind(self) -> BackendKind:
        return BackendKind.YUE2_CPP

    @property
    def capability(self) -> BackendCapability:
        return CAPABILITIES[BackendKind.YUE2_CPP]

    def plan(self, brief: GenerationBrief, out_path: Path) -> str:
        """Emit the ABC score without rendering audio. Cheap, and the whole point of the
        agentic-edit loop: change a chord, then re-render."""
        request = self._request_json(brief, seed=None)
        request_path = out_path.with_suffix(".request.json")
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps(request), encoding="utf-8")

        score_path = out_path.with_suffix(".abc")
        self._run(
            [self.executable(), "--model", str(self.model_path), "--vae", str(self.vae_path),
             "--request", str(request_path), "--out", str(score_path)],
            on_progress=None,
        )
        return score_path.read_text(encoding="utf-8")

    def render(self, brief, out_path, *, seed, gpu_index, on_progress=None):
        request = self._request_json(brief, seed=seed)
        request_path = out_path.with_suffix(".request.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps(request), encoding="utf-8")

        # The GGUF backends take a JSON request file rather than command-line flags: a long lyric
        # string would otherwise hit shell length limits and quoting rules.
        self._run(
            [self.executable(), "--model", str(self.model_path), "--vae", str(self.vae_path),
             "--request", str(request_path), "--out", str(out_path), "--cuda", str(gpu_index)],
            on_progress=on_progress,
        )
        return self._result(_current_job, out_path)

    def _request_json(self, brief: GenerationBrief, *, seed: int | None) -> dict:
        return {
            "style": brief.style,
            "lyrics": brief.lyrics,
            "cot": brief.cot,
            "cfg_scale": brief.cfg_scale,
            "seed": seed,
            "target_duration_s": brief.target_duration_s,
        }


class TorchBackend(GenerationBackend):
    """The reference PyTorch pipeline.

    Highest fidelity and the only route to a full plan → edit → re-render cycle, but it needs
    ~11.2 GiB, so it only fits the 16 GB card. Kept as the reference against which the quantized
    backends are judged.
    """

    def __init__(self, checkpoint: str = "m-a-p/YuE2-3B", device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self._device = device
        self._pipeline = None
        self.last_peak_vram_mib: int | None = None

    @property
    def kind(self) -> BackendKind:
        return BackendKind.TORCH

    @property
    def capability(self) -> BackendCapability:
        return CAPABILITIES[BackendKind.TORCH]

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            from yue2 import YuE2Pipeline  # noqa: F401
        except ImportError as exc:
            return False, f"torch backend not installed ({exc})"
        if device_vram_mib(primary_gpu()) < self.capability.peak_vram_mib:
            return False, "no device with enough VRAM for the bf16 reference pipeline"
        return True, "ok"

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from yue2 import YuE2Pipeline
        except ImportError as exc:
            raise BackendUnavailable(f"yue2 is not installed: {exc}") from exc
        device = self._device or f"cuda:{primary_gpu()}"
        self._pipeline = YuE2Pipeline.from_pretrained(self.checkpoint, device=device)
        return self._pipeline

    def plan(self, brief: GenerationBrief, out_path: Path) -> str:
        pipeline = self._load()
        plan = pipeline.plan(style=brief.style, lyrics=brief.lyrics, cot=brief.cot)
        score = getattr(plan, "abc", None) or str(plan)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(score, encoding="utf-8")
        return score

    def render(self, brief, out_path, *, seed, gpu_index, on_progress=None):
        import torch

        pipeline = self._load()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(gpu_index)
        try:
            song = pipeline(
                style=brief.style, lyrics=brief.lyrics, cot=brief.cot,
                cfg_scale=brief.cfg_scale, seed=seed,
            )
            song.save(str(out_path))
        except Exception as exc:
            raise GenerationFailed(f"torch backend failed: {exc}") from exc
        if torch.cuda.is_available():
            self.last_peak_vram_mib = int(
                torch.cuda.max_memory_peak_mib(gpu_index)
            )
        self.last_wall_time_s = time.time() - started
        return self._result(_current_job, out_path, score=None)

    def _result(self, job, out_path, *, score):
        import soundfile as sf

        from vexa_yue2.loader import file_sha256

        info = sf.info(str(out_path))
        return GenerationResult(
            job_id=job.job_id,
            backend=self.kind,
            quantization=self.capability.quant,
            model_id=self.checkpoint,
            seed=job.seed,
            audio_path=str(out_path),
            content_sha256=file_sha256(out_path),
            duration_s=float(info.frames) / float(info.samplerate),
            score_abc=score,
            wall_time_s=self.last_wall_time_s,
            peak_vram_mib=self.last_peak_vram_mib,
        )


class DryRunBackend(GenerationBackend):
    """Writes a synthetic file instead of calling a model.

    Exists so the *worker* — job claiming, cancellation, analysis, gating, cache admission — can be
    tested on a machine with no weights and no GPU. It is explicitly not a generation backend for
    promotion: its output is a tone, and anything it produces is stamped so it can never be
    mistaken for a real render.
    """

    def __init__(self, *, sample_rate: int = 44100, seconds: float = 6.0) -> None:
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.last_wall_time_s = 0.0
        self.last_peak_vram_mib = None

    @property
    def kind(self) -> BackendKind:
        # Deliberately NOT reported as the real yue2.cpp. Claiming that identity would let a dry
        # run win the routing table over a real backend and be mistaken for a real render.
        return BackendKind.DRY_RUN

    @property
    def capability(self) -> BackendCapability:
        return BackendCapability(
            kind=BackendKind.DRY_RUN,
            peak_vram_mib=0,
            supports_plan_edit=True,
            quant="DRY_RUN",
        )

    def available(self) -> tuple[bool, str]:
        return True, "ok"

    def render(self, brief, out_path, *, seed, gpu_index, on_progress=None):
        import numpy as np
        import soundfile as sf

        t = np.linspace(0, self.seconds, int(self.seconds * self.sample_rate), endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 220.0 * t)
        click = np.zeros_like(tone)
        period = int(self.sample_rate * 60.0 / 120.0)
        for i in range(0, len(click), period):
            click[i : i + 400] += 0.3
        stereo = np.stack([np.clip(tone + click, -1, 1)] * 2, axis=1).astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), stereo, self.sample_rate, subtype="PCM_16")
        self.last_wall_time_s = 0.01

        from vexa_yue2.loader import file_sha256

        return GenerationResult(
            job_id=_current_job.job_id,
            backend=self.kind,
            quantization="DRY_RUN",
            model_id="dry-run",
            seed=seed,
            audio_path=str(out_path),
            content_sha256=file_sha256(out_path),
            duration_s=self.seconds,
            wall_time_s=self.last_wall_time_s,
        )


#: The job currently being rendered. The backends take their parameters explicitly, but the
#: result must be attributed to a job, and threading a context through every call signature adds
#: noise without adding safety. Set by the worker, single-threaded by construction.
_current_job: GenerationJob = GenerationJob(
    job_id="unbound", brief=GenerationBrief(style="unbound")
)