"""Native launch environments survive checkout moves and report setup failures."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from vexa_orchestrator.generation import generate_theme
from vexa_yue2.backends import BackendUnavailable
from vexa_yue2.runtime import native_environment


def test_bundled_libraries_follow_binary_location_and_preserve_environment(tmp_path):
    build = tmp_path / "remounted" / "build"
    (build / "lib").mkdir(parents=True)
    binary = build / "yue-synth"
    binary.touch()
    inherited = {"LD_LIBRARY_PATH": f"/custom/lib::{build}", "CUDA_VISIBLE_DEVICES": "GPU-test"}
    environment = native_environment(binary, inherited)
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(build),
        str(build / "lib"),
        "/custom/lib",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert inherited["LD_LIBRARY_PATH"] == f"/custom/lib::{build}"


def test_loader_failure_is_not_treated_as_retryable_audio_rejection(monkeypatch, tmp_path):
    binary = tmp_path / "yue-synth"
    model = tmp_path / "model.gguf"
    vae = tmp_path / "vae.gguf"
    for variable, path in (
        ("VEXA_YUE_BINARY", binary),
        ("VEXA_YUE_MODEL", model),
        ("VEXA_YUE_VAE", vae),
    ):
        path.touch()
        monkeypatch.setenv(variable, str(path))
    monkeypatch.setattr("vexa_orchestrator.generation._gpu_index", lambda: "GPU-primary")

    def fail_launch(cmd, *, stdout, stderr, env):
        assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(tmp_path)
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-primary"
        request = json.loads(Path(cmd[cmd.index("--request") + 1]).read_text())
        assert request["lyrics"] == "[Verse] Original rhythmic rap lyrics"
        stdout.write("error while loading shared libraries: libggml.so.0: not found")
        return SimpleNamespace(returncode=127, poll=lambda: 127)

    monkeypatch.setattr("vexa_orchestrator.generation.subprocess.Popen", fail_launch)
    with pytest.raises(BackendUnavailable, match=r"libggml\.so\.0"):
        generate_theme(
            "rap",
            tmp_path / "library",
            style="Underground boom bap rap",
            lyrics="[Verse] Original rhythmic rap lyrics",
        )


def test_missing_component_fails_before_attempting_render(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_YUE_BINARY", str(tmp_path / "missing-yue-synth"))
    with pytest.raises(BackendUnavailable, match="component missing"):
        generate_theme("house", tmp_path / "library", style="Instrumental house, no vocals")
    assert not (tmp_path / "library").exists()
