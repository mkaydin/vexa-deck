"""Offline work fails visibly and cancellation cleans up its whole process group."""

import threading
from pathlib import Path

import pytest
from vexa_orchestrator.background import run_job
from vexa_orchestrator.depot import Depot
from vexa_yue2.backends import BackendUnavailable


def test_missing_yue2_component_survives_process_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_YUE_BINARY", str(tmp_path / "missing"))
    with pytest.raises(BackendUnavailable, match="component missing"):
        run_job(
            "generate",
            {"theme": "rap", "library": str(tmp_path), "style": "Boom bap rap, dry breakbeats"},
        )


def test_worker_validation_failure_has_the_actual_reason():
    with pytest.raises(ValueError, match="unknown offline action"):
        run_job("invalid", {})


def test_cancellation_terminates_descendant_render_process(monkeypatch, tmp_path):
    import subprocess

    import vexa_orchestrator.background as background

    real_popen = subprocess.Popen
    child_pid_path = tmp_path / "child.pid"
    marker = tmp_path / "orphan-survived"
    script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',"
        f"{
            (
                'import time; from pathlib import Path; time.sleep(3); '
                + f'Path({str(marker)!r}).touch()'
            )!r
        }]); "
        f"open({str(child_pid_path)!r},'w').write(str(child.pid)); time.sleep(10)"
    )

    def spawn(*args, **kwargs):
        return real_popen([background.sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(background.subprocess, "Popen", spawn)
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            run_job("generate", {}, cancel=cancel)
        assert child_pid_path.exists()
        # The descendant is absent or a reaped/zombie process; it cannot write the marker.
        status = Path(f"/proc/{child_pid_path.read_text()}/status")
        if status.exists():
            state = next(
                line for line in status.read_text().splitlines() if line.startswith("State:")
            )
            assert "Z" in state
        assert not marker.exists()
    finally:
        timer.cancel()


def test_legacy_instrumental_batch_is_not_selected_for_vocal_rap(tmp_path):

    from types import SimpleNamespace

    manifest = SimpleNamespace(
        asset_id="legacy",
        family_id="legacy",
        admissible=lambda: True,
        provenance=SimpleNamespace(source_prompt="1990s underground rap hip hop music"),
        tags={"instrumental": [SimpleNamespace(value="true")]},
        audio=SimpleNamespace(duration_s=165),
    )
    depot = Depot(tmp_path)
    depot.add(manifest, tmp_path / "legacy.wav")
    assert depot.search("90s underground rap") == []
    assert depot.search("90s underground rap instrumental")
