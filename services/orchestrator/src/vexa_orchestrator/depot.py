"""Verified local YuE2 depot and a deterministic theme shortlist.

Retrieval is based on recorded prompts and tags. A theme with no meaningful match returns no
asset, allowing the live runner to request generation instead of pretending a random track fits.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vexa_contracts import AssetManifest

from .track_titles import hydrate_title
from .vocal_intent import has_vocal_delivery, vocal_intent

_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "the",
        "with",
        "for",
        "of",
        "in",
        "to",
        "music",
        "track",
        "instrumental",
        "no",
        "vocals",
        "vocal",
        "lyrics",
        "lyric",
        "without",
        "original",
        "like",
        "vibe",
        "theme",
        "playlist",
    }
)
_ALIASES = {
    "jazzy": "jazz",
    "rainy": "rain",
    "soaked": "rain",
    "chilled": "chill",
    "energetic": "energy",
    "darkness": "dark",
    "technoid": "techno",
}
_GENRES = frozenset(
    {
        "jazz",
        "house",
        "techno",
        "trance",
        "hardstyle",
        "disco",
        "ambient",
        "downtempo",
        "chillout",
        "hip",
        "hop",
        "rap",
        "trap",
        "phonk",
        "pop",
        "drum",
        "bass",
        "rock",
        "metal",
        "classical",
        "funk",
        "soul",
    }
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_ALIASES.get(word, word) for word in words if word not in _STOP}


@dataclass(frozen=True, slots=True)
class DepotAsset:
    manifest: AssetManifest
    path: Path
    words: frozenset[str]
    vibe: dict[str, object]


class Depot:
    def __init__(self, library: Path) -> None:
        self.library = library
        self.assets: dict[str, DepotAsset] = {}
        self.document_frequency: Counter[str] = Counter()
        self.reload()

    def reload(self) -> None:
        assets: dict[str, DepotAsset] = {}
        frequency: Counter[str] = Counter()
        for path in sorted(self.library.glob("*.wav")):
            manifest_path = path.with_suffix(".json")
            if not manifest_path.exists():
                continue
            try:
                manifest = AssetManifest.model_validate_json(manifest_path.read_text())
                if not manifest.admissible():
                    continue
                tags = " ".join(tag.value for group in manifest.tags.values() for tag in group)
                words = frozenset(
                    _tokens(
                        " ".join((manifest.asset_id, manifest.provenance.source_prompt or "", tags))
                    )
                )
                vibe_path = path.with_suffix(".vibe.json")
                vibe = json.loads(vibe_path.read_text()) if vibe_path.exists() else {}
                if vibe and vibe.get("source_sha256") != manifest.content_sha256:
                    vibe = {}
            except (OSError, ValueError):
                continue
            hydrate_title(manifest, path)
            assets[manifest.asset_id] = DepotAsset(manifest, path, words, vibe)
            frequency.update(words)
        self.assets = assets
        self.document_frequency = frequency

    def add(self, manifest: AssetManifest, path: Path, vibe: dict | None = None) -> None:
        if not manifest.admissible():
            raise ValueError("only ready, gated assets enter the depot")
        words = frozenset(
            _tokens(
                " ".join(
                    (
                        manifest.asset_id,
                        manifest.provenance.source_prompt or "",
                        " ".join(tag.value for group in manifest.tags.values() for tag in group),
                    )
                )
            )
        )
        hydrate_title(manifest, path)
        self.assets[manifest.asset_id] = DepotAsset(manifest, path, words, vibe or {})
        self.document_frequency.update(words)

    def search(
        self,
        theme: str,
        *,
        target_energy: float | None = None,
        exclude_families: set[str] | None = None,
        limit: int = 12,
        min_duration_s: float = 0.0,
    ) -> list[tuple[float, DepotAsset]]:
        query = _tokens(theme)
        if not query:
            return []
        total = max(1, len(self.assets))
        requested_genres = query & _GENRES
        intent = vocal_intent(theme)
        # Rap without a vocal override remains a vocal request, matching generation.
        vocal = intent is True or (intent is None and bool(
            re.search(r"\brap\b|hip[ -]?hop|boom[ -]?bap|\btrap\b|\bdrill\b", theme.lower())
        ))
        results: list[tuple[float, DepotAsset]] = []
        for asset in self.assets.values():
            if asset.manifest.audio.duration_s < min_duration_s:
                continue
            if asset.manifest.family_id in (exclude_families or set()):
                continue
            if requested_genres and not requested_genres.issubset(asset.words):
                continue
            recorded = {tag.value.lower() for tag in asset.manifest.tags.get("instrumental", [])}
            source = asset.manifest.provenance.source_prompt or ""
            if intent is False:
                # Unknown legacy tracks do not satisfy an explicit no-lyrics request.
                # These tags describe generation intent, not measured vocal separation.
                if "false" in recorded:
                    continue
                if "true" not in recorded and (
                    vocal_intent(source) is not False or has_vocal_delivery(source)
                ):
                    continue
            elif vocal and ("true" in recorded or (
                "false" not in recorded and vocal_intent(source) is False
            )):
                continue
            common = query & asset.words
            if not common:
                continue
            matched = sum(
                math.log(1 + total / (1 + self.document_frequency[word])) for word in common
            )
            possible = sum(
                math.log(1 + total / (1 + self.document_frequency[word])) for word in query
            )
            score = matched / possible if possible else 0.0
            if target_energy is not None and asset.manifest.energy() is not None:
                score *= 1.0 - 0.25 * abs(asset.manifest.energy() - target_energy)
            if score >= 0.18:
                results.append((score, asset))
        return sorted(results, key=lambda row: (-row[0], row[1].manifest.asset_id))[:limit]
