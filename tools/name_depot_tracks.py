"""Give existing generated tracks stable titles without touching audio, IDs or cue files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vexa_orchestrator.depot import Depot
from vexa_orchestrator.track_titles import clean_title, title_for_asset, unique_title

ROOT = Path(__file__).resolve().parents[1]


def name_tracks(library: Path) -> list[dict]:
    depot = Depot(library)
    seen = {
        clean_title(asset.manifest.title).casefold()
        for asset in depot.assets.values()
        if asset.manifest.title and not asset.manifest.asset_id.startswith("live-")
    }
    # Reserve explicit titles before assigning any missing names.
    for asset in depot.assets.values():
        raw = json.loads(asset.path.with_suffix(".json").read_text())
        if raw.get("title"):
            seen.add(clean_title(raw["title"]).casefold())
    changed = []
    for identity, asset in sorted(depot.assets.items()):
        if not identity.startswith("live-"):
            continue
        path = asset.path.with_suffix(".json")
        raw = json.loads(path.read_text())
        if clean_title(raw.get("title")):
            continue
        title = unique_title(title_for_asset(asset), seen)
        raw["title"] = title
        # Preserve all existing manifest keys exactly; replace only the metadata file.
        temporary = path.with_suffix(".title.tmp")
        temporary.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
        changed.append({"asset_id": identity, "title": title})
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "assets/library")
    args = parser.parse_args()
    changes = name_tracks(args.library)
    print(json.dumps({"named": len(changes), "tracks": changes}, indent=2, ensure_ascii=False))
