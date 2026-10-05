#!/usr/bin/env bash
set -euo pipefail
vexa_source="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$vexa_source/tools/package_app.py" "$@"
