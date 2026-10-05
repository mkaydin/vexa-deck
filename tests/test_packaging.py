"""Release boundaries, reproducibility and safe installer/update/uninstall workflows."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from package_app import BLOCKED_PARTS, BLOCKED_SUFFIXES, build  # noqa: E402


def run(*arguments, cwd=ROOT, check=True):
    return subprocess.run(
        ["/usr/bin/python3", *map(str, arguments)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture
def extracted(tmp_path):
    archive = build(ROOT, tmp_path / "release")
    with tarfile.open(archive) as bundle:
        bundle.extractall(tmp_path / "extracted", filter="data")
    return next((tmp_path / "extracted").iterdir())


def test_archive_is_reproducible_and_contains_no_models_media_or_secrets(tmp_path):
    first = build(ROOT, tmp_path / "one")
    second = build(ROOT, tmp_path / "two")
    assert first.read_bytes() == second.read_bytes()
    assert (
        first.with_name(first.name + ".sha256")
        .read_text()
        .startswith(hashlib.sha256(first.read_bytes()).hexdigest())
    )
    with tarfile.open(first) as bundle:
        members = bundle.getmembers()
        assert all(member.isfile() for member in members)
        relative = [Path(*Path(member.name).parts[1:]) for member in members]
        assert all(not set(path.parts) & BLOCKED_PARTS for path in relative)
        assert all(path.suffix.lower() not in BLOCKED_SUFFIXES for path in relative)
        assert Path(".env.example") in relative and Path(".env") not in relative
        assert Path("apps/desktop/assets/video/renderer.js") in relative
        assert Path("third_party/video-art/LICENSE.txt") in relative
        assert Path("LICENSE") in relative
        release = next(
            member for member in members if member.name.endswith("packaging/release.json")
        )
        inventory = json.load(bundle.extractfile(release))
        for member in members:
            key = str(Path(*Path(member.name).parts[1:]))
            if key == "packaging/release.json":
                continue
            assert (
                hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
                == (inventory["files"][key]["sha256"])
            )


def test_install_update_and_uninstall_preserve_user_data(extracted, tmp_path):
    prefix = tmp_path / "installed app"
    data = tmp_path / "desktop data"
    binary = tmp_path / "command bin"
    options = ["--prefix", prefix, "--data-home", data, "--bin-home", binary, "--skip-deps"]
    script = extracted / "tools/install_app.py"
    preview = run(script, *options, "--dry-run")
    assert not prefix.exists() and "source_files" in preview.stdout
    run(script, *options)
    assert (prefix / "vexa-videos").is_dir()
    assert (prefix / "models").is_dir()
    assert (data / "applications/vexa-deck.desktop").is_file()
    launcher = binary / "vexa-deck"
    assert os.access(launcher, os.X_OK)
    assert str(prefix) in launcher.read_text()
    protected = {
        prefix / ".env": "VEXA_CPU_THREADS=1\n",
        prefix / "var/config/console.json": '{"theme":"violet"}',
        prefix / "vexa-videos/user.mp4": "user video",
        prefix / "models/user.gguf": "user weights",
        prefix / "assets/library/user.wav": "user audio",
    }
    for path, value in protected.items():
        path.write_text(value)
    run(script, *options)
    assert all(path.read_text() == value for path, value in protected.items())
    run(script, "--uninstall", "--prefix", prefix, "--dry-run")
    assert launcher.exists()
    run(script, "--uninstall", "--prefix", prefix)
    assert not launcher.exists() and not (data / "applications/vexa-deck.desktop").exists()
    assert all(path.read_text() == value for path, value in protected.items())


def test_release_tampering_is_rejected(extracted, tmp_path):
    (extracted / "apps/desktop/main.py").write_text("changed source")
    result = run(
        extracted / "tools/install_app.py",
        "--prefix",
        tmp_path / "install",
        "--skip-deps",
        check=False,
    )
    assert result.returncode != 0 and "inventory" in result.stderr
    assert not (tmp_path / "install").exists()


def test_unrelated_install_directory_is_not_overwritten(extracted, tmp_path):
    target = tmp_path / "unrelated"
    target.mkdir()
    (target / "important.txt").write_text("keep")
    result = run(extracted / "tools/install_app.py", "--prefix", target, "--skip-deps", check=False)
    assert result.returncode != 0
    assert (target / "important.txt").read_text() == "keep"


def test_extracted_preview_launcher_runs_without_backend_dependencies(extracted, tmp_path):
    screenshot = tmp_path / "preview.png"
    environment = dict(os.environ)
    environment.update(QT_QPA_PLATFORM="offscreen", QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu")
    result = subprocess.run(
        [
            "bash",
            str(extracted / "apps/desktop/run.sh"),
            "--preview",
            "--screenshot",
            str(screenshot),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert screenshot.stat().st_size > 1000
