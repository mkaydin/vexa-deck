#!/usr/bin/env bash
# Print prerequisites by default. --apply explicitly runs the package-manager command.
set -euo pipefail
vexa_apply=0
vexa_build=0
for argument in "$@"; do
    case "$argument" in
        --apply) vexa_apply=1 ;;
        --with-build-tools) vexa_build=1 ;;
        -h|--help)
            echo 'Usage: install-system-deps.sh [--with-build-tools] [--apply]'
            echo 'Prints commands unless --apply is given. CUDA toolkit and GPU driver are separate.'
            exit 0 ;;
        *) echo "Unknown option: $argument" >&2; exit 2 ;;
    esac
done
if command -v apt-get >/dev/null; then
    vexa_cmd=(apt-get install -y python3 git libportaudio2 libsndfile1 ffmpeg libegl1
        libopengl0 libxcb-cursor0 libxkbcommon0 libwayland-client0 fonts-dejavu-core)
    if [[ "$vexa_build" == 1 ]]; then vexa_cmd+=(build-essential cmake ninja-build); fi
elif command -v pacman >/dev/null; then
    vexa_cmd=(pacman -S --needed python git portaudio libsndfile ffmpeg mesa libglvnd
        xcb-util-cursor libxkbcommon wayland ttf-dejavu)
    if [[ "$vexa_build" == 1 ]]; then vexa_cmd+=(base-devel cmake ninja); fi
elif command -v dnf >/dev/null; then
    vexa_cmd=(dnf install -y python3 git portaudio libsndfile ffmpeg-free mesa-libEGL
        libglvnd-opengl xcb-util-cursor libxkbcommon wayland dejavu-sans-mono-fonts)
    if [[ "$vexa_build" == 1 ]]; then vexa_cmd+=(gcc gcc-c++ make cmake ninja-build); fi
else
    echo 'Unsupported package manager. Install Python, Git, PortAudio, FFmpeg and Qt/Wayland libraries.' >&2
    exit 1
fi
printf 'System dependencies: '
printf '%q ' "${vexa_cmd[@]}"
printf '\n'
if [[ "$vexa_apply" == 1 ]]; then
    if [[ "$EUID" != 0 ]]; then vexa_cmd=(sudo "${vexa_cmd[@]}"); fi
    "${vexa_cmd[@]}"
fi
