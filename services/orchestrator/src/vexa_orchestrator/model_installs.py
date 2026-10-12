"""Persistent, allowlisted model installation jobs using pinned installer sources."""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .generators import DEFAULTS, NAMES, PRESETS, ROOT, components

TERMINAL = {"ready", "cancelled", "failed", "interrupted"}


def installer(generator):
    filename = {
        "yue2": "install_yue2",
        "minimax": "install_minimax_music",
        "acestep": "install_acestep_audio",
    }[generator]
    spec = importlib.util.spec_from_file_location(filename, ROOT / "tools" / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor(generator, preset):
    if generator not in PRESETS or preset not in PRESETS[generator]:
        raise ValueError("Unsupported generator/preset")
    module = installer(generator)
    revision = module.MODEL_REVISION
    source_revision = module.SOURCE_REVISION
    directory = ROOT / "third_party/managed" / f"{generator}-{source_revision[:12]}"
    models = (
        directory / "models"
        if generator == "yue2"
        else ROOT / "models/managed" / f"{generator}-{revision[:12]}"
    )
    patterns = (
        list(module.WEIGHTS)
        if generator == "yue2"
        else module.weights(preset)
        if generator == "minimax"
        else [
            "acestep-v15-turbo/*",
            "vae/*",
            "Qwen3-Embedding-0.6B/*",
            "acestep-5Hz-lm-1.7B/*",
            "LICENSE*",
        ]
    )
    return dict(
        generator=generator,
        preset=preset,
        model_repo=module.MODEL_REPO,
        model_revision=revision,
        source_revision=source_revision,
        source_url=getattr(module, "SOURCE_URL", getattr(module, "SOURCE", "")),
        directory=str(directory),
        models=str(models),
        patterns=patterns,
        license_url=f"https://huggingface.co/{module.MODEL_REPO}/tree/{revision}",
    )


def managed_components(spec):
    source, models = Path(spec["directory"]), Path(spec["models"])
    if spec["generator"] == "yue2":
        return dict(
            binary=source / "build/yue-synth",
            model=models / "YuE2-3B-Q8_0.gguf",
            vae=models / "YuE2-Vae-F32.gguf",
        )
    if spec["generator"] == "minimax":
        names = installer("minimax").weights(spec["preset"])[:5]
        return dict(
            zip(
                ("lm_model", "depth_model", "dit_model", "cond_model", "vae_model"),
                (models / n for n in names),
                strict=True,
            ),
            binary=source / "build/mm-synth",
        )
    return dict(
        binary=source / ".venv/bin/python",
        handler=source / "acestep/handler.py",
        dit=models / "acestep-v15-turbo/model.safetensors",
        vae=models / "vae/diffusion_pytorch_model.safetensors",
        embedding=models / "Qwen3-Embedding-0.6B/model.safetensors",
        planner=models / "acestep-5Hz-lm-1.7B/model.safetensors",
    )


def check_manifest(spec, files):
    required = [pattern for pattern in spec["patterns"] if not pattern.startswith("LICENSE")]
    if any(
        not any(fnmatch.fnmatchcase(f.rfilename, pattern) for f in files) for pattern in required
    ):
        raise InstallFailure("Pinned manifest is missing required model components")
    if any(Path(f.rfilename).is_absolute() or ".." in Path(f.rfilename).parts for f in files):
        raise InstallFailure("Invalid path in download manifest")
    if not files or any(f.size is None for f in files):
        raise InstallFailure("Cannot determine required model files and sizes")


class InstallFailure(RuntimeError):
    """A deliberately user-facing error, without third-party request credentials."""


class InstallCancelled(Exception):
    pass


class ModelInstalls:
    def __init__(self, directory=None, busy=None):
        self.directory = directory or ROOT / "var/model-installs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "jobs.json"
        self.lock = threading.RLock()
        self._owner = (self.directory / "installer.lock").open("a+")
        self._readonly = False
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._readonly = True
        self.jobs = {}
        self.busy = busy or (lambda: False)
        self.cancels = {}
        self.thread = None
        self.closed = False
        if self.path.exists():
            try:
                self.jobs = json.loads(self.path.read_text())
            except (ValueError, OSError):
                self.jobs = {}
        if not isinstance(self.jobs, dict):
            self.jobs = {}
        self.jobs = {
            key: value
            for key, value in self.jobs.items()
            if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key)
            and isinstance(value, dict)
            and "state" in value
        }
        for job in [] if self._readonly else self.jobs.values():
            if job["state"] not in TERMINAL:
                job.update(
                    state="interrupted",
                    message="Application closed during installation; resume to recover.",
                )
        self._save()

    def _save(self):
        if self._readonly:
            return
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.jobs, indent=2))
        temporary.replace(self.path)

    def _update(self, identity, **changes):
        with self.lock:
            self.jobs[identity].update(changes, updated_at=time.time())
            self._save()

    def catalog(self):
        result = []
        for generator in NAMES:
            for preset in PRESETS[generator]:
                spec = descriptor(generator, preset)
                paths = components(generator, preset)
                ready = all(p.is_file() and p.stat().st_size for p in paths.values()) and os.access(
                    paths["binary"], os.X_OK
                )
                result.append(
                    {
                        **spec,
                        "name": NAMES[generator],
                        "installed": ready,
                        "location": str(paths["binary"].parent),
                        "requirements": "git, cmake, CUDA toolkit (nvcc)"
                        if generator != "acestep"
                        else "git, uv, CUDA-compatible Python environment",
                    }
                )
        return result

    def inspect(self, generator, preset):
        from huggingface_hub import HfApi

        spec = descriptor(generator, preset)
        entries = (
            HfApi()
            .model_info(
                spec["model_repo"], revision=spec["model_revision"], files_metadata=True, timeout=15
            )
            .siblings
        )
        files = [
            f for f in entries if any(fnmatch.fnmatchcase(f.rfilename, p) for p in spec["patterns"])
        ]
        check_manifest(spec, files)
        models = Path(spec["models"])
        needed = sum(
            f.size
            for f in files
            if not (models / f.rfilename).is_file()
            or (models / f.rfilename).stat().st_size != f.size
        )
        parent = models
        while not parent.exists():
            parent = parent.parent
        requirements = ("git", "uv") if generator == "acestep" else ("git", "cmake", "nvcc")
        return dict(
            total_bytes=sum(f.size for f in files),
            incremental_bytes=needed,
            required_free_bytes=needed * 2 + 8 * 1024**3,
            free_bytes=shutil.disk_usage(parent).free,
            missing_tools=[name for name in requirements if not shutil.which(name)],
            license_url=spec["license_url"],
        )

    def list(self):
        with self.lock:
            return json.loads(json.dumps(list(self.jobs.values())))

    def get(self, identity):
        with self.lock:
            if identity not in self.jobs:
                raise ValueError("Installation job not found")
            job = dict(self.jobs[identity])
        path = self.directory / f"{identity}.log"
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 8192))
                tail = handle.read().decode("utf-8", errors="replace")
            tail = re.sub(r"https?://[^/@\s]+@", "https://[redacted]@", tail)
            tail = re.sub(
                r"(?i)(token|authorization|password)\s*[:=]\s*[^\r\n]+", r"\1=[redacted]", tail
            )
            job["log_tail"] = tail
        return job

    def start(self, generator, preset=None):
        spec = descriptor(generator, preset or DEFAULTS.get(generator))
        with self.lock:
            if self.closed or self._readonly:
                raise ValueError("Installer is shutting down or owned by another backend")
            if any(j["state"] not in TERMINAL for j in self.jobs.values()):
                raise ValueError("One installation is already active; cancel or finish it first")
            identity = uuid.uuid4().hex
            self.jobs[identity] = dict(
                id=identity,
                spec=spec,
                state="queued",
                completed_bytes=0,
                total_bytes=None,
                speed_bps=None,
                eta_s=None,
                file=None,
                message="Checking requirements and download manifest",
                updated_at=time.time(),
            )
            self._save()
            self._launch(identity)
            return self.get(identity)

    def resume(self, identity):
        with self.lock:
            if self.closed or self._readonly:
                raise ValueError("Installer is shutting down or owned by another backend")
            job = self.get(identity)
            if job["state"] == "ready" or any(
                j["state"] not in TERMINAL for j in self.jobs.values()
            ):
                raise ValueError("Finish the active job before resuming")
            self._update(identity, state="queued", message="Recovering existing verified downloads")
            self._launch(identity)
            return self.get(identity)

    def cancel(self, identity):
        self.get(identity)
        if self._readonly:
            raise ValueError("Installation belongs to another backend")
        if identity in self.cancels:
            self.cancels[identity].set()
        return self.get(identity)

    def close(self):
        self.closed = True
        for event in self.cancels.values():
            event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if (self.thread is None or not self.thread.is_alive()) and not self._owner.closed:
            self._owner.close()

    def _launch(self, identity):
        cancel = self.cancels[identity] = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(identity, cancel), daemon=True)
        self.thread.start()

    def _command(self, identity, cancel, command, cwd=None):
        if cancel.is_set():
            raise InstallCancelled()
        log = self.directory / f"{identity}.log"
        with log.open("ab") as output:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=output, stderr=output, start_new_session=True
            )
            while process.poll() is None:
                if cancel.wait(0.2):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise InstallCancelled()
            if process.returncode:
                raise InstallFailure(
                    "Runtime setup failed; see installation log. Existing sources are preserved."
                )

    def _probe(self, spec, paths):
        from vexa_yue2.runtime import native_environment

        command = (
            [str(paths["binary"]), "-c", "import torch; from acestep.handler import AceStepHandler"]
            if spec["generator"] == "acestep"
            else [str(paths["binary"])]
        )
        probe = subprocess.run(
            command,
            cwd=spec["directory"],
            env=native_environment(paths["binary"]),
            capture_output=True,
            timeout=15,
        )
        usage = b"usage" in (probe.stdout + probe.stderr).lower()
        valid = (
            probe.returncode == 0
            if spec["generator"] == "acestep"
            else probe.returncode == 1 and usage
        )
        if not valid:
            raise InstallFailure(
                "Runtime cannot start; repair dependencies before generating music"
            )

    def _run(self, identity, cancel):
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from huggingface_hub.utils import tqdm

            spec = self.get(identity)["spec"]
            if spec != descriptor(spec["generator"], spec["preset"]):
                raise InstallFailure(
                    "Saved job differs from the pinned catalog; start a new install"
                )
            self._update(identity, state="preflight")
            required = ("git", "uv") if spec["generator"] == "acestep" else ("git", "cmake", "nvcc")
            missing = [name for name in required if not shutil.which(name)]
            if missing:
                raise InstallFailure("Missing prerequisites: " + ", ".join(missing))
            files = (
                HfApi()
                .model_info(
                    spec["model_repo"],
                    revision=spec["model_revision"],
                    files_metadata=True,
                    timeout=15,
                )
                .siblings
            )
            selected = [
                f
                for f in files
                if any(fnmatch.fnmatchcase(f.rfilename, pattern) for pattern in spec["patterns"])
            ]
            check_manifest(spec, selected)
            total = sum(f.size for f in selected)
            source, models = Path(spec["directory"]), Path(spec["models"])
            source.parent.mkdir(parents=True, exist_ok=True)
            # Account for model cache/staging and an estimated runtime build/environment reserve.
            needed = sum(
                f.size
                for f in selected
                if not (models / f.rfilename).is_file()
                or (models / f.rfilename).stat().st_size != f.size
            )
            model_parent = models
            while not model_parent.exists():
                model_parent = model_parent.parent
            if (
                shutil.disk_usage(model_parent).free < needed * 2 + 8 * 1024**3
                or shutil.disk_usage(source.parent).free < 8 * 1024**3
            ):
                raise InstallFailure(
                    "Insufficient free space: allow twice the missing weight size "
                    "plus 8 GiB runtime reserve"
                )
            self._update(
                identity, total_bytes=total, message="Pinned files and disk requirements checked"
            )
            if cancel.is_set():
                raise InstallCancelled()
            if source.exists() and not (source / ".git").is_dir():
                # A cancelled clone may leave a directory. Preserve it rather than deleting data.
                preserved = source.with_name(source.name + ".incomplete-" + uuid.uuid4().hex[:8])
                source.rename(preserved)
            if not source.exists():
                self._command(
                    identity,
                    cancel,
                    ["git", "clone", "--no-checkout", spec["source_url"], str(source)],
                )
            self._command(
                identity,
                cancel,
                ["git", "-C", str(source), "checkout", "--detach", spec["source_revision"]],
            )
            models.mkdir(parents=True, exist_ok=True)
            completed = 0
            manager = self
            self._update(identity, state="downloading")
            for item in selected:
                if cancel.is_set():
                    raise InstallCancelled()
                self._update(identity, file=item.rfilename, speed_bps=None, eta_s=None)
                expected = item.lfs.sha256 if item.lfs else None
                target = models / item.rfilename

                def valid(item=item, target=target, expected=expected):
                    if not target.is_file() or target.stat().st_size != item.size:
                        return False
                    digest = hashlib.sha256() if expected else hashlib.sha1()
                    if not expected:
                        digest.update(f"blob {item.size}\0".encode())
                    with target.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            if cancel.is_set():
                                raise InstallCancelled()
                            digest.update(block)
                    return digest.hexdigest() == (expected or item.blob_id)

                class Progress(tqdm):
                    file_size = item.size
                    base_bytes = completed

                    def update(self, amount=1):
                        if cancel.is_set():
                            raise InstallCancelled()
                        result = super().update(amount)
                        now = time.monotonic()
                        if now - getattr(self, "_vexa_last", 0) > 0.25:
                            self._vexa_last = now
                            speed = self.format_dict.get("rate")
                            done = min(self.file_size, int(self.n))
                            manager._update(
                                identity,
                                completed_bytes=self.base_bytes + done,
                                speed_bps=speed,
                                eta_s=(total - self.base_bytes - done) / speed if speed else None,
                            )
                        return result

                if not target.exists():
                    legacy = next(
                        (
                            p
                            for p in components(spec["generator"], spec["preset"]).values()
                            if p.is_file() and p.as_posix().endswith(item.rfilename)
                        ),
                        None,
                    )
                    if legacy is not None and valid(target=legacy):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(legacy, target)
                        except OSError:
                            temporary = target.with_name(
                                target.name + ".adopting-" + uuid.uuid4().hex
                            )
                            try:
                                with legacy.open("rb") as src, temporary.open("xb") as dst:
                                    for block in iter(lambda: src.read(1024 * 1024), b""):
                                        if cancel.is_set():
                                            raise InstallCancelled()
                                        dst.write(block)
                                if not valid(target=temporary):
                                    raise InstallFailure("Existing weight changed during adoption")
                                temporary.replace(target)
                            finally:
                                temporary.unlink(missing_ok=True)
                if not valid():
                    if target.exists():
                        self._update(
                            identity,
                            state="waiting_resources",
                            message="Repair waits until music releases existing files",
                        )
                        while self.busy():
                            if cancel.wait(1):
                                raise InstallCancelled()
                    if shutil.disk_usage(models).free < item.size * 2 + 8 * 1024**3:
                        raise InstallFailure(
                            "Insufficient disk space to download or repair this file"
                        )
                    self._update(identity, state="downloading")
                    hf_hub_download(
                        spec["model_repo"],
                        item.rfilename,
                        revision=spec["model_revision"],
                        local_dir=models,
                        tqdm_class=Progress,
                        force_download=target.is_file(),
                    )
                    self._update(identity, state="verifying")
                    if not valid():
                        raise InstallFailure(
                            "Downloaded file failed integrity verification: " + item.rfilename
                        )
                    self._update(identity, state="downloading")
                completed += item.size
                self._update(identity, completed_bytes=completed)
            # Build stages have no byte ETA and wait for generation/audio priority.
            self._update(
                identity,
                state="waiting_resources",
                file=None,
                eta_s=None,
                message="Weights verified; runtime setup waits while music is active",
            )
            while self.busy():
                if cancel.wait(1):
                    raise InstallCancelled()
            self._update(
                identity,
                state="building",
                message="Preparing pinned runtime; detailed output in installation log",
            )
            module = installer(spec["generator"])
            script = Path(module.__file__)
            command = [sys.executable, str(script), "--directory", str(source), "--runtime-only"]
            if spec["generator"] != "yue2":
                command += ["--models", str(models)]
            if spec["generator"] == "minimax":
                command += ["--preset", spec["preset"]]
            self._command(identity, cancel, command)
            if cancel.is_set():
                raise InstallCancelled()
            paths = managed_components(spec)
            if not all(p.is_file() and p.stat().st_size for p in paths.values()) or not os.access(
                paths["binary"], os.X_OK
            ):
                raise InstallFailure("Runtime/components incomplete after setup")
            self._probe(spec, paths)
            receipt = dict(
                preferred=True,
                spec=spec,
                components={k: str(v) for k, v in paths.items()},
                files=[
                    dict(name=f.rfilename, size=f.size, sha256=f.lfs.sha256 if f.lfs else None)
                    for f in selected
                ],
            )
            # Renderer provenance reads the same pinned model receipt as CLI installations.
            model_receipt = {
                "source_revision": spec["source_revision"],
                "model_revision": spec["model_revision"],
                "revision": spec["model_revision"],
                "model_repo": spec["model_repo"],
                "repo": spec["model_repo"],
                "preset": spec["preset"],
                "files": receipt["files"],
            }
            receipt_name = (
                "download-revision.json" if spec["generator"] == "acestep" else "vexa-download.json"
            )
            receipt_path = models / receipt_name
            staging = receipt_path.with_suffix(".tmp")
            staging.write_text(json.dumps(model_receipt, indent=2))
            staging.replace(receipt_path)
            registry = self.directory.parent / "config/model-registry.json"
            registry.parent.mkdir(parents=True, exist_ok=True)
            data = json.loads(registry.read_text()) if registry.exists() else {}
            data[spec["generator"] + ":" + spec["preset"]] = receipt
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(registry)
            self._update(
                identity,
                state="ready",
                message="Verified models and runtime ready for a new batch",
                eta_s=0,
            )
        except InstallCancelled:
            self._update(
                identity,
                state="cancelled",
                message="Cancelled; verified files and resumable downloads retained",
            )
        except Exception as error:
            # URLs/auth tokens from third-party errors must not leak into persisted API messages.
            kind = type(error).__name__
            message = (
                str(error)
                if isinstance(error, InstallFailure)
                else f"{kind}: installation failed; check network, credentials and prerequisites"
            )
            self._update(identity, state="failed", message=message, eta_s=None)
        finally:
            if self.closed and not self._owner.closed:
                self._owner.close()
