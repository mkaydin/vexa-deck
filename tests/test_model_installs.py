import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from vexa_orchestrator import model_installs as module
from vexa_orchestrator.model_installs import ModelInstalls, descriptor


def test_catalog_pins_all_supported_presets():
    for generator, preset in [
        ("yue2", "q8"),
        ("minimax", "balanced"),
        ("minimax", "quality"),
        ("minimax", "compact"),
        ("acestep", "turbo"),
    ]:
        spec = descriptor(generator, preset)
        assert len(spec["model_revision"]) == len(spec["source_revision"]) == 40
        assert spec["patterns"] and "managed" in spec["directory"]
    with pytest.raises(ValueError):
        descriptor("shell-command", "anything")


def test_restart_preserves_job_and_marks_interrupted(tmp_path):
    manager = ModelInstalls(tmp_path)
    manager.jobs["job"] = dict(id="job", state="downloading", completed_bytes=123)
    manager._save()
    manager.close()
    restored = ModelInstalls(tmp_path)
    assert restored.get("job")["state"] == "interrupted"
    assert restored.get("job")["completed_bytes"] == 123
    with pytest.raises(ValueError):
        restored.get("../../file")


def fixture_job(tmp_path, monkeypatch, corrupt=False):
    manager = ModelInstalls(tmp_path / "jobs")
    source, models = tmp_path / "source", tmp_path / "models"
    source.mkdir()
    (source / ".git").mkdir()
    spec = dict(
        generator="minimax",
        preset="balanced",
        directory=str(source),
        models=str(models),
        model_repo="test/model",
        model_revision="a" * 40,
        source_revision="b" * 40,
        patterns=["*.gguf"],
    )
    monkeypatch.setattr(module, "descriptor", lambda *args: spec)
    content = b"fixture-weights"
    metadata = SimpleNamespace(
        rfilename="test.gguf",
        size=len(content),
        lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()),
    )
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info", lambda *a, **kw: SimpleNamespace(siblings=[metadata])
    )
    calls = []

    def download(repo, filename, **kwargs):
        calls.append(filename)
        models.mkdir(exist_ok=True)
        (models / filename).write_bytes(b"corrupt" if corrupt else content)
        return str(models / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/tool/" + name)
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3)
    )
    monkeypatch.setattr(
        module, "installer", lambda name: SimpleNamespace(__file__="/fixture/installer.py")
    )
    binary = source / "build/mm-synth"

    def command(*args, **kw):
        binary.parent.mkdir(exist_ok=True)
        binary.write_bytes(b"fixture-runtime")
        binary.chmod(0o755)

    monkeypatch.setattr(manager, "_command", command)
    monkeypatch.setattr(manager, "_probe", lambda *args: None)
    monkeypatch.setattr(
        module, "managed_components", lambda spec: dict(binary=binary, weights=models / "test.gguf")
    )
    manager.jobs["job"] = dict(id="job", spec=spec, state="queued", completed_bytes=0)
    return manager, calls, content


def test_download_verify_register_and_reuse(tmp_path, monkeypatch):
    manager, calls, content = fixture_job(tmp_path, monkeypatch)
    manager._run("job", threading.Event())
    assert manager.get("job")["state"] == "ready"
    registry = json.loads((tmp_path / "config/model-registry.json").read_text())
    assert registry["minimax:balanced"]["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    manager._run("job", threading.Event())
    assert calls == ["test.gguf"]


def test_corruption_never_ready(tmp_path, monkeypatch):
    manager, _, _ = fixture_job(tmp_path, monkeypatch, corrupt=True)
    manager._run("job", threading.Event())
    assert manager.get("job")["state"] == "failed"
    assert not (tmp_path / "config/model-registry.json").exists()


def test_cancel_and_low_disk(tmp_path, monkeypatch):
    manager, _, _ = fixture_job(tmp_path, monkeypatch)
    cancelled = threading.Event()
    cancelled.set()
    manager._run("job", cancelled)
    assert manager.get("job")["state"] == "cancelled"
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    manager._run("job", threading.Event())
    assert manager.get("job")["state"] == "failed" and "space" in manager.get("job")["message"]


def test_cancel_waiting_runtime_preserves_verified_files(tmp_path, monkeypatch):
    manager, _, content = fixture_job(tmp_path, monkeypatch)
    cancel = threading.Event()

    def busy():
        cancel.set()
        return True

    manager.busy = busy
    manager._run("job", cancel)
    assert manager.get("job")["state"] == "cancelled"
    assert (tmp_path / "models/test.gguf").read_bytes() == content
    assert not (tmp_path / "config/model-registry.json").exists()


def test_other_backend_cannot_start_parallel_installs(tmp_path, monkeypatch):
    owner = ModelInstalls(tmp_path)
    monkeypatch.setattr(owner, "_launch", lambda identity: None)
    job = owner.start("minimax", "balanced")
    second = ModelInstalls(tmp_path)
    assert second.get(job["id"])["state"] == "queued"
    with pytest.raises(ValueError, match="owned"):
        second.start("yue2", "q8")
    second.close()
    owner.close()


def test_runtime_probe_failure_is_not_ready(tmp_path, monkeypatch):
    manager, _, _ = fixture_job(tmp_path, monkeypatch)

    def bad_probe(*args):
        raise RuntimeError("Runtime cannot start")

    monkeypatch.setattr(manager, "_probe", bad_probe)
    manager._run("job", threading.Event())
    assert manager.get("job")["state"] == "failed"
    assert not (tmp_path / "config/model-registry.json").exists()


def test_missing_required_component_rejected_before_download():
    from vexa_orchestrator.model_installs import check_manifest

    with pytest.raises(RuntimeError, match="missing required"):
        check_manifest(
            {"patterns": ["backbone.gguf", "vae.gguf", "LICENSE*"]},
            [SimpleNamespace(rfilename="backbone.gguf", size=10)],
        )


def test_probe_checks_native_loading_without_inference(tmp_path):
    binary = tmp_path / "runtime"
    binary.write_text("#!/bin/sh\necho Usage: runtime\nexit 1\n")
    binary.chmod(0o755)
    manager = ModelInstalls(tmp_path / "jobs")
    manager._probe(dict(generator="minimax", directory=str(tmp_path)), {"binary": binary})
    binary.write_text("#!/bin/sh\necho missing-library\nexit 127\n")
    with pytest.raises(RuntimeError, match="cannot start"):
        manager._probe(dict(generator="minimax", directory=str(tmp_path)), {"binary": binary})
    manager.close()


def test_network_failure_can_resume_without_leaking_credentials(tmp_path, monkeypatch):
    manager, _, _ = fixture_job(tmp_path, monkeypatch)
    import huggingface_hub

    successful = huggingface_hub.hf_hub_download
    attempts = []

    def flaky(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("https://host/private?token=do-not-expose")
        return successful(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", flaky)
    manager._run("job", threading.Event())
    assert manager.get("job")["state"] == "failed"
    assert "do-not-expose" not in manager.get("job")["message"]
    manager.resume("job")
    manager.thread.join(timeout=2)
    assert manager.get("job")["state"] == "ready"
    manager.close()
