"""JSON Schema export.

``ARCHITECTURE.md:106`` requires the contracts to be versioned from the start; exporting them as
JSON Schema makes that reviewable outside Python. The GUI (TypeScript) and any external tool can
validate against these without importing pydantic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .version import CONTRACT_REVISION, CONTRACT_VERSION

_CACHE: dict[str, type[BaseModel]] | None = None


def registry() -> dict[str, type[BaseModel]]:
    """Every model defined in a contracts module, keyed by class name.

    Imported lazily so the package's own ``__init__`` can re-export from here without a cycle.
    """
    global _CACHE
    if _CACHE is None:
        from . import asset, decision, jobs, requests, session

        found: dict[str, type[BaseModel]] = {}
        for module in (asset, session, decision, requests, jobs):
            for name in dir(module):
                obj = getattr(module, name)
                if not isinstance(obj, type) or not issubclass(obj, BaseModel):
                    continue
                if name.startswith("_") or obj.__module__ != module.__name__:
                    continue
                found[name] = obj
        _CACHE = dict(sorted(found.items()))
    return _CACHE


def schema_bundle() -> dict[str, Any]:
    """All contract schemas plus a manifest identifying the version they came from."""
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_revision": CONTRACT_REVISION,
        "schemas": {name: model.model_json_schema() for name, model in registry().items()},
    }


def export_schemas(path: str | Path) -> Path:
    """Write the bundle to ``path`` as JSON. Returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema_bundle(), indent=2, sort_keys=True), encoding="utf-8")
    return target


if __name__ == "__main__":  # pragma: no cover - manual invocation
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("contracts.schema.json")
    written = export_schemas(out)
    print(f"wrote {len(registry())} contract schemas to {written}")