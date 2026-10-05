#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

# Parse .env as data; never source it as shell code. Explicit environment values win.
if [[ -f .env && -x .venv/bin/python && "${VEXA_ENV_LOADED:-0}" != 1 ]]; then
    exec .venv/bin/python tools/launch_env.py "$repo_root/apps/desktop/run.sh" "$@"
fi
needs_backend=1
for argument in "$@"; do
    if [[ "$argument" == --preview ]]; then needs_backend=0; fi
done

# Keep background analysis/BLAS pools from monopolizing the audio host's CPU.
export VEXA_CPU_THREADS="${VEXA_CPU_THREADS:-2}"
export OMP_NUM_THREADS="$VEXA_CPU_THREADS"
export MKL_NUM_THREADS="$VEXA_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$VEXA_CPU_THREADS"
export NUMBA_NUM_THREADS="$VEXA_CPU_THREADS"
export VEXA_LOG_DIR="${VEXA_LOG_DIR:-$repo_root/var/logs}"
export VEXA_RUN_ID="${VEXA_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$VEXA_LOG_DIR"
exec > >(tee -a "$VEXA_LOG_DIR/$VEXA_RUN_ID-launcher.log") 2>&1
printf '[%s] VEXA run=%s repo=%s log_dir=%s\n' "$(date -u +%FT%TZ)" \
    "$VEXA_RUN_ID" "$repo_root" "$VEXA_LOG_DIR"

if [[ "$needs_backend" == 1 && ! -x .venv/bin/python ]]; then
    echo "Backend environment missing. Run: bash scripts/install.sh or uv sync --locked --extra core --extra desktop --extra local-planner" >&2
    exit 1
fi

# Resolve source paths at launch so a relocated installation still imports its backend.
if [[ -x .venv/bin/python ]]; then
    workspace_sources="$(.venv/bin/python apps/desktop/backend_runtime.py)"
    export PYTHONPATH="${workspace_sources}${PYTHONPATH:+:$PYTHONPATH}"
    printf '[%s] Backend workspace source paths resolved from %s\n' "$(date -u +%FT%TZ)" "$repo_root"
fi
if [[ "$needs_backend" == 1 && -z "${VEXA_API_URL:-}" ]]; then
    if ! .venv/bin/python -c 'import vexa_orchestrator, vexa_contracts, vexa_audio, uvicorn, fastapi'; then
        echo "Backend imports failed. Run: uv sync --locked --inexact --extra core --extra local-planner" >&2
        exit 1
    fi
fi

printf '[%s] Music planner=%s model=%s (GPU released before YuE2)\n' \
    "$(date -u +%FT%TZ)" "${VEXA_MUSIC_PLANNER:-acestep}" \
    "${VEXA_ACE_MODEL:-$repo_root/models/acestep-5Hz-lm-1.7B}"

gui_modules='import PySide6.QtWidgets, PySide6.QtMultimedia, PySide6.QtWebEngineWidgets'
if .venv/bin/python -c "$gui_modules" >/dev/null 2>&1; then
    gui_python=.venv/bin/python
elif /usr/bin/python3 -c "$gui_modules" >/dev/null 2>&1; then
    gui_python=/usr/bin/python3
else
    echo "PySide6 Widgets, Multimedia and WebEngine are required. Install the Linux PySide6 packages including QtWebEngine, or add PySide6 to the uv environment." >&2
    exit 1
fi

printf '[%s] GUI python=%s args=%s\n' "$(date -u +%FT%TZ)" "$gui_python" "$*"
if "$gui_python" apps/desktop/main.py "$@"; then
    exit_code=0
else
    exit_code=$?
fi
printf '[%s] GUI exited status=%s log=%s\n' "$(date -u +%FT%TZ)" \
    "$exit_code" "$VEXA_LOG_DIR/$VEXA_RUN_ID-launcher.log"
exit "$exit_code"
