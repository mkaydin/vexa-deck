"""Run expensive audio work outside the interpreter that owns PortAudio."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

from vexa_yue2.backends import BackendUnavailable

ROOT = Path(__file__).resolve().parents[4]
LOGGER = logging.getLogger(__name__)


def worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    sources = [
        ROOT / member / "src"
        for member in (
            "packages/contracts",
            "services/audio",
            "services/orchestrator",
            "workers/yue2",
            "services/planner",
            "services/laya",
        )
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in sources), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    return env


@contextmanager
def gpu_job_lock():
    """Serialize planner/YuE2 jobs across backend processes on the same machine."""
    import fcntl

    path = ROOT / "var/gpu-jobs.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_job(
    action: str, payload: dict, *, cancel: threading.Event | None = None, timeout_s: float = 300
) -> dict:
    """Exchange paths and small metadata; never pickle whole decoded audio buffers."""
    directory = ROOT / "var/jobs"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{action}-", dir=directory) as temporary:
        job = Path(temporary) / "job.json"
        result = Path(temporary) / "result.json"
        log = ROOT / "var/logs" / f"offline-{Path(temporary).name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Offline audio job start action=%s log=%s", action, log)
        job.write_text(json.dumps({"action": action, "payload": payload}), encoding="utf-8")
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "vexa_orchestrator.offline_worker", str(job), str(result)],
                env=worker_environment(),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_s
            last_progress = ""
            progress_at = 0.0
            try:
                while process.poll() is None:
                    if action == "plan" and time.monotonic() - progress_at > 5:
                        progress_at = time.monotonic()
                        lines = [
                            line
                            for line in log.read_text(errors="replace").splitlines()
                            if "ACE-Step " in line or "Prompt writer progress" in line
                        ]
                        if lines and lines[-1] != last_progress:
                            last_progress = lines[-1]
                            LOGGER.info("Music planner progress: %s", last_progress)
                    if cancel is not None and cancel.is_set():
                        raise RuntimeError(f"{action} cancelled")
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{action} timed out")
                    if cancel is None:
                        time.sleep(0.05)
                    else:
                        cancel.wait(0.05)
            finally:
                if process.poll() is None:
                    # The group includes yue-synth/rubberband; Stop leaves no GPU job orphaned.
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
            if not result.exists():
                raise RuntimeError(
                    f"{action} worker exited {process.returncode}: {log.read_text()[-1500:]}"
                )
            body = json.loads(result.read_text())
            if not body.get("ok"):
                error = body.get("error", "unknown offline worker failure")
                if body.get("unavailable"):
                    raise BackendUnavailable(error)
                raise ValueError(error)
            if process.returncode:
                raise RuntimeError(f"{action} worker exited {process.returncode}")
            LOGGER.info("Offline audio job completed action=%s log=%s", action, log)
            return body["result"]
