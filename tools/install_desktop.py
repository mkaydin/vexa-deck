"""Install VEXA//DECK's Linux menu launcher and icons for the current user."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)


def exec_argument(value: str) -> str:
    """Quote an argument for the desktop entry Exec key (not for a shell)."""
    value = value.replace("%", "%%")
    for character in ("\\", '"', "`", "$"):
        value = value.replace(character, "\\" + character)
    return '"' + value + '"'


def entry_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def install(data_home: Path, *, app_root: Path = ROOT) -> Path:
    for size in ICON_SIZES:
        source = app_root / "assets/icons" / f"vexa-icon-{size}.png"
        destination = data_home / "icons/hicolor" / f"{size}x{size}/apps/vexa-deck.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    applications = data_home / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    launcher = applications / "vexa-deck.desktop"
    launcher.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=VEXA//DECK\n"
        "Comment=Continuous music sets in a dark cyberpunk console\n"
        f"Exec=/usr/bin/bash {exec_argument(str(app_root / 'apps/desktop/run.sh'))}\n"
        f"Path={entry_value(str(app_root))}\n"
        "Icon=vexa-deck\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Audio;Music;\n"
        "Keywords=DJ;Music;Mix;Vexa;\n"
        "StartupWMClass=vexa-deck\n"
        "StartupNotify=false\n",
        encoding="utf-8",
    )
    for program, target in (
        ("update-desktop-database", applications),
        ("gtk-update-icon-cache", data_home / "icons/hicolor"),
    ):
        executable = shutil.which(program)
        if executable:
            arguments = [executable, str(target)]
            if program == "gtk-update-icon-cache":
                arguments = [executable, "--force", "--ignore-theme-index", str(target)]
            result = subprocess.run(arguments, capture_output=True, text=True, check=False)
            if result.returncode:
                print(f"Cache refresh: {result.stderr.strip()}")
    print(f"Installed launcher: {launcher}")
    print(f"Installed icons: {data_home / 'icons/hicolor'}")
    return launcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-home",
        type=Path,
        default=Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")),
        help="desktop data directory (default: XDG_DATA_HOME or ~/.local/share)",
    )
    arguments = parser.parse_args()
    install(arguments.data_home.expanduser().resolve())


if __name__ == "__main__":
    main()
