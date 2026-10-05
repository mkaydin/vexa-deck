#!/usr/bin/env bash
set -euo pipefail
vexa_source="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$vexa_source"
vexa_component="${1:-all}"
case "$vexa_component" in
    -h|--help)
        echo "Usage: bash scripts/install-models.sh [all|yue2|planner|writer]"
        echo "Downloads multi-GB weights. Requires CUDA build tools for yue2 and Ollama for writer."
        exit 0 ;;
    all|yue2|planner|writer) ;;
    *) echo "Unknown model component: $vexa_component" >&2; exit 2 ;;
esac
if [[ ! -x .venv/bin/python ]]; then
    echo "Run scripts/install.sh first (full local-planner dependencies required)." >&2
    exit 1
fi
export PYTHONPATH="$(.venv/bin/python apps/desktop/backend_runtime.py)${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f .env && "${VEXA_ENV_LOADED:-0}" != 1 ]]; then
    exec .venv/bin/python tools/launch_env.py "$vexa_source/scripts/install-models.sh" "$@"
fi
if [[ "$vexa_component" == all || "$vexa_component" == yue2 ]]; then
    .venv/bin/python tools/install_yue2.py
fi
if [[ "$vexa_component" == all || "$vexa_component" == planner ]]; then
    .venv/bin/python tools/install_music_planner.py
fi
if [[ "$vexa_component" == all || "$vexa_component" == writer ]]; then
    if ! command -v ollama >/dev/null; then
        echo "Install Ollama first: https://ollama.com/download/linux" >&2
        exit 1
    fi
    .venv/bin/python tools/install_local_lyric_writer.py
fi
