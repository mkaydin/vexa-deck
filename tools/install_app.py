"""Per-user Linux install/update and shortcut removal; preserves models, media and preferences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from install_desktop import install as install_desktop
from package_app import ROOT, inventory, release_files


def defaults():
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")).expanduser()
    binary = Path(os.environ.get("XDG_BIN_HOME", Path.home() / ".local/bin")).expanduser()
    return data, binary


def verify_release(paths):
    release = ROOT / "packaging/release.json"
    if not release.exists():
        return
    recorded = json.loads(release.read_text())["files"]
    current = inventory(ROOT, paths)["files"]
    if recorded != current:
        raise RuntimeError("Release inventory does not match source files; extract a clean archive")


def install(args):
    prefix = args.prefix.expanduser().resolve()
    data_home = args.data_home.expanduser().resolve()
    bin_home = args.bin_home.expanduser().resolve()
    if prefix == ROOT or prefix.is_relative_to(ROOT):
        raise ValueError(
            "Install outside the source checkout; use install_desktop.py for a dev shortcut"
        )
    paths = release_files()
    verify_release(paths)
    receipt_path = prefix / ".vexa-install.json"
    if prefix.exists() and any(prefix.iterdir()) and not receipt_path.is_file():
        raise ValueError(f"Refusing to overwrite an unrelated nonempty directory: {prefix}")
    command = ["uv", "sync", "--locked", "--no-dev", "--extra", "core"]
    if not args.system_qt:
        command += ["--extra", "desktop"]
    if not args.minimal:
        command += ["--extra", "local-planner"]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "prefix": str(prefix),
                    "source_files": len(paths),
                    "dependencies": command if not args.skip_deps else "skipped",
                    "desktop_data": str(data_home),
                    "command_directory": str(bin_home),
                    "model_weights": "separate: scripts/install-models.sh",
                },
                indent=2,
            )
        )
        return
    uv = shutil.which("uv")
    if not args.skip_deps and uv is None:
        candidate = Path.home() / ".local/bin/uv"
        uv = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
        if uv is None:
            raise RuntimeError(
                "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
            )
    launcher = bin_home / "vexa-deck"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    # Only replace shortcuts/icons registered by our own installer.
    if (
        launcher.exists()
        and previous.get("launcher") != str(launcher)
        and (not launcher.is_file() or "apps/desktop/run.sh" not in launcher.read_text())
    ):
        raise ValueError(
            f"Command already exists and is not owned by this installation: {launcher}"
        )
    desktop = data_home / "applications/vexa-deck.desktop"
    if (
        desktop.exists()
        and previous.get("desktop") != str(desktop)
        and (not desktop.is_file() or "StartupWMClass=vexa-deck" not in desktop.read_text())
    ):
        raise ValueError(f"Menu entry already exists; use a different --data-home: {desktop}")
    record = inventory(ROOT, paths)
    for source in paths:
        destination = prefix / source.relative_to(ROOT)
        if destination.is_symlink() or not destination.resolve().is_relative_to(prefix):
            raise ValueError(f"Refusing to replace a symlink: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o755 if source.suffix == ".sh" else 0o644)
    for directory in ("var/logs", "var/config", "assets/library", "models", "vexa-videos"):
        (prefix / directory).mkdir(parents=True, exist_ok=True)
    environment = prefix / ".env"
    if not environment.exists():
        shutil.copyfile(prefix / ".env.example", environment)
        environment.chmod(0o600)
    # Record copied files before dependencies run, so an interrupted install can be retried.
    record.update(
        prefix=str(prefix),
        launcher=str(launcher),
        desktop=str(desktop),
        icons=previous.get("icons", {}),
    )
    for key in ("desktop_sha256", "launcher_sha256"):
        if key in previous:
            record[key] = previous[key]
    receipt_path.write_text(json.dumps(record, indent=2) + "\n")
    if not args.skip_deps:
        command[0] = uv
        print("Installing locked Python runtime dependencies...", flush=True)
        subprocess.run(command, cwd=prefix, check=True)
        backend = prefix / ".venv/bin/python"
        subprocess.run(
            [str(backend), "-c", "import vexa_orchestrator, vexa_audio, fastapi, uvicorn"],
            cwd=prefix,
            check=True,
        )
        gui_python = "/usr/bin/python3" if args.system_qt else str(backend)
        subprocess.run(
            [
                gui_python,
                "-c",
                "import PySide6.QtWidgets, PySide6.QtMultimedia, PySide6.QtWebEngineWidgets",
            ],
            check=True,
        )
    installed_desktop = install_desktop(data_home, app_root=prefix)
    bin_home.mkdir(parents=True, exist_ok=True)
    script = (
        "#!/usr/bin/env bash\n"
        f'exec /usr/bin/bash {shlex.quote(str(prefix / "apps/desktop/run.sh"))} "$@"\n'
    )
    launcher.write_text(script)
    launcher.chmod(0o755)
    for size in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        icon = data_home / "icons/hicolor" / f"{size}x{size}/apps/vexa-deck.png"
        record["icons"][str(icon)] = hashlib.sha256(icon.read_bytes()).hexdigest()
    record["desktop_sha256"] = hashlib.sha256(installed_desktop.read_bytes()).hexdigest()
    record["launcher_sha256"] = hashlib.sha256(launcher.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Installed VEXA//DECK to {prefix}")
    print(f"Launch: {launcher}  (add {bin_home} to PATH if needed)")
    print(f"Install music models: bash {prefix / 'scripts/install-models.sh'}")
    if args.skip_deps:
        print("Dependencies skipped: preview requires system Qt; live playback requires .venv.")


def uninstall(args):
    prefix = args.prefix.expanduser().resolve()
    receipt_path = prefix / ".vexa-install.json"
    if not receipt_path.exists():
        raise ValueError(f"No installation receipt: {receipt_path}")
    record = json.loads(receipt_path.read_text())
    if record.get("prefix") != str(prefix):
        raise ValueError("Installation has moved; its old shortcut paths cannot be removed safely")
    managed = {
        record["desktop"]: record.get("desktop_sha256"),
        record["launcher"]: record.get("launcher_sha256"),
        **record.get("icons", {}),
    }
    for filename, digest in managed.items():
        path = Path(filename)
        if digest and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            print(f"Remove {path}")
            if not args.dry_run:
                path.unlink()
    print(f"Application menu/command removed. App files, models and user data remain in {prefix}.")


def main():
    data_home, bin_home = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=data_home / "vexa-deck")
    parser.add_argument("--data-home", type=Path, default=data_home)
    parser.add_argument("--bin-home", type=Path, default=bin_home)
    parser.add_argument("--system-qt", action="store_true", help="use distribution Python/PySide6")
    parser.add_argument("--minimal", action="store_true", help="omit Torch/local text planner deps")
    parser.add_argument(
        "--skip-deps", action="store_true", help="copy/integrate only; no downloads"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show actions without writing/downloading"
    )
    parser.add_argument(
        "--uninstall", action="store_true", help="remove owned menu/command/icons only"
    )
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("This installer targets Linux; other platforms have not been packaged")
    try:
        uninstall(args) if args.uninstall else install(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Installation failed: {exc}\n")


if __name__ == "__main__":
    main()
