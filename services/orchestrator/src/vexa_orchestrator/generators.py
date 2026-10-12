"""Music backend catalog and isolated render adapters; no Torch imports in the host."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path

from vexa_yue2.backends import BackendUnavailable
from vexa_yue2.runtime import native_environment

ROOT = Path(__file__).resolve().parents[4]
LOGGER = logging.getLogger(__name__)
NAMES = {"yue2": "YuE2", "acestep": "ACE-Step 1.5 Turbo", "minimax": "MiniMax Music 3"}
PRESETS = {"yue2": ("q8",), "acestep": ("turbo",), "minimax": ("balanced", "quality", "compact")}
DEFAULTS = {"yue2": "q8", "acestep": "turbo", "minimax": "balanced"}
MINIMAX_STEPS = {"fast": 12, "balanced": 20, "quality": 30}
MM_QUANTS = {
    "balanced": ("Q6_K", "Q8_0"),
    "quality": ("Q8_0", "Q8_0"),
    "compact": ("Q5_K_M", "Q5_K_M"),
}


def _legacy_components(generator: str, preset: str) -> dict[str, Path]:
    if generator == "yue2":
        base = ROOT / "third_party/yue2.cpp"
        return {
            "binary": Path(os.environ.get("VEXA_YUE_BINARY", base / "build/yue-synth")),
            "model": Path(os.environ.get("VEXA_YUE_MODEL", base / "models/YuE2-3B-Q8_0.gguf")),
            "vae": Path(os.environ.get("VEXA_YUE_VAE", base / "models/YuE2-Vae-F32.gguf")),
        }
    if generator == "acestep":
        base = Path(os.environ.get("VEXA_ACE_AUDIO_SOURCE") or ROOT / "third_party/ACE-Step-1.5")
        weights = Path(os.environ.get("VEXA_ACE_AUDIO_MODELS") or ROOT / "models/acestep-audio")
        return {
            "binary": base / ".venv/bin/python",
            "handler": base / "acestep/handler.py",
            "dit": weights / "acestep-v15-turbo/model.safetensors",
            "vae": weights / "vae/diffusion_pytorch_model.safetensors",
            "embedding": weights / "Qwen3-Embedding-0.6B/model.safetensors",
            "planner": weights / "acestep-5Hz-lm-1.7B/model.safetensors",
        }
    base = Path(os.environ.get("VEXA_MINIMAX_MODELS") or ROOT / "models/minimax-music3")
    lm, dit = MM_QUANTS[preset]
    return {
        "binary": Path(
            os.environ.get("VEXA_MINIMAX_BINARY")
            or ROOT / "third_party/minimaxmusic.cpp/build/mm-synth"
        ),
        "lm_model": base / f"MiniMax-Music3-language_model-{lm}.gguf",
        "depth_model": base / "MiniMax-Music3-rvq_depth_decoder-Q8_0.gguf",
        "dit_model": base / f"MiniMax-Music3-transformer-{dit}.gguf",
        "cond_model": base / "MiniMax-Music3-condition_encoder-F32.gguf",
        "vae_model": base / "MiniMax-Music3-vocoder-F32.gguf",
    }


def components(generator: str, preset: str) -> dict[str, Path]:
    existing = _legacy_components(generator, preset)
    registry = ROOT / "var/config/model-registry.json"
    try:
        saved = json.loads(registry.read_text()).get(generator + ":" + preset, {})
        paths = {k: Path(v) for k, v in saved.get("components", {}).items()}
        valid = paths.keys() == existing.keys() and all(
            p.is_file() and p.stat().st_size for p in paths.values()
        )
        if valid and saved.get("preferred"):
            return paths
        if valid and not all(p.is_file() and p.stat().st_size for p in existing.values()):
            return paths
    except (OSError, ValueError, TypeError):
        pass
    return existing


def resolve_components(generator, preset, frozen=None):
    paths = (
        components(generator, preset)
        if frozen is None
        else {key: Path(value) for key, value in frozen.items()}
    )
    if paths.keys() != _legacy_components(generator, preset).keys():
        raise BackendUnavailable("Saved runtime components do not match this generator")
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise BackendUnavailable(f"Model/runtime component missing: {path}")
    if not os.access(paths["binary"], os.X_OK):
        raise BackendUnavailable("Model runtime is not executable")
    return paths


def selection(
    generator: str | None = None, preset: str | None = None, *, require_available: bool = False
) -> tuple[str, str]:
    generator = generator or os.environ.get("VEXA_GENERATOR") or "yue2"
    if generator not in NAMES:
        raise ValueError(f"Unknown music generator: {generator}")
    preset = preset or DEFAULTS[generator]
    if preset not in PRESETS[generator]:
        raise ValueError(f"Unsupported {NAMES[generator]} preset: {preset}")
    if require_available:
        paths = components(generator, preset)
        missing = [str(p) for p in paths.values() if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise BackendUnavailable(
                f"{NAMES[generator]} components missing: " + ", ".join(missing)
            )
        if not os.access(paths["binary"], os.X_OK):
            raise BackendUnavailable(
                f"{NAMES[generator]} executable is not runnable: {paths['binary']}"
            )
    return generator, preset


def catalog() -> list[dict]:
    result = []
    for generator, name in NAMES.items():
        variants = []
        for preset in PRESETS[generator]:
            try:
                selection(generator, preset, require_available=True)
                error = None
            except BackendUnavailable as exc:
                error = str(exc)
            variants.append(
                {
                    "id": preset,
                    "available": error is None,
                    "reason": error,
                    "paths": {k: str(v) for k, v in components(generator, preset).items()},
                }
            )
        result.append(
            {
                "id": generator,
                "name": name,
                "presets": variants,
                "default_preset": DEFAULTS[generator],
                "instrumental": True,
                "vocals": True,
                "experimental": generator == "minimax",
            }
        )
    return result


def run_renderer(
    command: list[str],
    binary: Path,
    log: Path,
    cancel: threading.Event | None,
    duration_s: float,
    *,
    timeout_s: int | None = None,
) -> None:
    from .generation import _gpu_index

    try:
        gpu = _gpu_index()
    except RuntimeError as exc:
        raise BackendUnavailable(str(exc)) from exc
    env = {**native_environment(binary), "CUDA_VISIBLE_DEVICES": gpu}
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = "1"
    if shutil.which("nice"):
        command = ["nice", "-n", "10", *command]
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Music render cancelled")
    deadline = time.monotonic() + (timeout_s or max(1800, duration_s * 30))
    with log.open("w") as output:
        try:
            process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        except OSError as exc:
            raise BackendUnavailable(f"Renderer could not launch: {exc}") from exc
        try:
            while process.poll() is None:
                if (cancel is not None and cancel.is_set()) or time.monotonic() > deadline:
                    raise RuntimeError("Music render cancelled or timed out")
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if process.returncode:
        detail = log.read_text(errors="replace")[-1500:]
        if (
            "error while loading shared libraries" in detail
            or "ModuleNotFoundError" in detail
            or "[Registry] ERROR" in detail
            or process.returncode in (126, 127)
        ):
            raise BackendUnavailable(detail)
        raise RuntimeError(f"Renderer exited {process.returncode}: {detail}")
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Music render cancelled")


def native_revision(binary: Path) -> str | None:
    source = binary.parent.parent
    if source.name == ".venv":
        source = source.parent
    if not (source / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def minimax_timings(log: Path) -> dict:
    """Read native stage measurements, keeping estimates separate from admission time."""
    if not log.is_file():
        return {}
    text = log.read_text(errors="replace")
    patterns = {
        "ar_s": r"\[AR\] \d+ frames total, ([\d.]+) s",
        "replay_s": r"\[AR\] Replay: .*?, ([\d.]+) s",
        "dit_s": r"\[DiT\] CFG=[\d.]+, \d+ windows, ([\d.]+) s",
        "vae_s": r"\[VAE\] Decode: .*?, ([\d.]+) ms",
        "total_s": r"\[Done\] ([\d.]+) s total",
    }
    result = {}
    for key, pattern in patterns.items():
        found = re.search(pattern, text)
        if found:
            result[key] = round(float(found[1]) / (1000 if key == "vae_s" else 1), 3)
    return result


def minimax_caption(
    style: str, lyrics: str, music_plan: dict | None, *, duration_s: float = 165
) -> str:
    from .minimax_prompts import fallback_caption, validate_caption

    metadata = music_plan or {}
    if metadata.get("minimax_caption"):
        return validate_caption(
            metadata["minimax_caption"],
            style=style,
            lyrics=lyrics,
            genre=metadata.get("genre", ""),
            vocal=bool(lyrics.strip()),
            duration_s=duration_s,
        )
    if style.startswith("Global Metadata\n") and all(
        section in style for section in ("\nVocal Details\n", "\nArrangement\n")
    ):
        return style
    return fallback_caption(style, lyrics, metadata, duration_s)


def render_alternative(
    generator: str,
    preset: str,
    wav: Path,
    request: Path,
    *,
    style: str,
    lyrics: str,
    duration_s: float,
    music_plan: dict | None,
    cancel=None,
    seed: int | None = None,
    timeout_s: int | None = None,
    runtime_components: dict | None = None,
) -> dict:
    generator, preset = selection(generator, preset)
    paths = resolve_components(generator, preset, runtime_components)
    seed = secrets.randbelow(2**31) if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("render seed must be a nonnegative 31-bit integer")
    metadata = {
        "generator": generator,
        "preset": preset,
        "seed": seed,
        "runtime_revision": native_revision(paths["binary"]),
    }
    runtime_receipt = paths["binary"].with_suffix(".vexa.json")
    if runtime_receipt.is_file():
        with suppress(OSError, ValueError):
            metadata["runtime_patch_sha256"] = json.loads(runtime_receipt.read_text()).get(
                "runtime_patch_sha256"
            )
    model_root = paths["lm_model"].parent if generator == "minimax" else paths["dit"].parent.parent
    receipt = model_root / (
        "vexa-download.json" if generator == "minimax" else "download-revision.json"
    )
    if receipt.is_file():
        try:
            saved = json.loads(receipt.read_text())
            metadata["model_revision"] = saved.get("model_revision", saved.get("revision"))
            metadata["model_repo"] = saved.get("model_repo", saved.get("repo"))
        except (OSError, ValueError):
            pass
    if generator == "minimax":
        # Older saved plans retain their original 30-step request. New settings are
        # frozen in each plan, independently of the model's weight precision preset.
        speed = ((music_plan or {}).get("generation_settings") or {}).get("render_speed", "quality")
        if speed not in MINIMAX_STEPS:
            raise ValueError(f"Unsupported MiniMax render speed: {speed}")
        metadata.update(render_speed=speed, denoising_steps=MINIMAX_STEPS[speed])
        payload = {
            "caption": minimax_caption(style, lyrics, music_plan, duration_s=duration_s),
            "lyrics": lyrics.strip() or "[Instrumental]",
            "duration": duration_s,
            "min_duration": max(0, duration_s - 1),
            "steps": MINIMAX_STEPS[speed],
            "seed": seed,
            "lm_seed": seed,
            "output_format": "wav16",
            "peak_clip": 0,
            **{key: path.name for key, path in paths.items() if key != "binary"},
        }
        command = [
            str(paths["binary"]),
            "--models",
            str(paths["lm_model"].parent),
            "--request",
            str(request),
            "--out",
            str(wav),
            "--max-seq",
            "8192",
        ]
    elif generator == "acestep":
        payload = {
            "caption": style,
            "lyrics": lyrics.strip() or "[Instrumental]",
            "instrumental": not bool(lyrics.strip()),
            "duration": duration_s,
            "seed": seed,
            "bpm": (music_plan or {}).get("bpm"),
            "key": (music_plan or {}).get("key", ""),
            "models": str(paths["dit"].parent.parent),
            "source": str(paths["handler"].parent.parent),
        }
        command = [
            str(paths["binary"]),
            str(ROOT / "tools/render_acestep_audio.py"),
            str(request),
            str(wav),
        ]
    else:
        raise ValueError("YuE2 uses its existing native renderer")
    request.write_text(json.dumps(payload, indent=2) + "\n")
    if generator == "minimax":
        from .minimax_prompts import FORMAT

        metadata["caption_format"] = FORMAT
        metadata["caption_origin"] = (music_plan or {}).get("caption_origin", "adapter")
    LOGGER.info(
        "Music render start generator=%s preset=%s duration=%.1f request=%s",
        generator,
        preset,
        duration_s,
        request,
    )
    run_renderer(
        command,
        paths["binary"],
        wav.with_suffix(".render.log"),
        cancel,
        duration_s,
        **({"timeout_s": timeout_s} if timeout_s is not None else {}),
    )
    if generator == "minimax":
        metadata["render_timings"] = minimax_timings(wav.with_suffix(".render.log"))
        replay = wav.with_suffix(".json")
        if replay.exists():
            replay.replace(wav.with_suffix(".replay.json"))
    return metadata
