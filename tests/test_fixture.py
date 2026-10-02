"""Phase 0 exit criterion.

``ROADMAP.md:9``: *"a clean checkout can read a sample manifest, verify asset hashes, and identify
which files may be redistributed."*

These run against the committed fixture on disk rather than a fixture built in memory, because the
criterion is about what a fresh checkout actually contains.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vexa_contracts import AssetManifest, ReadinessState, registry

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "data" / "fixtures" / "asset_manifest.sample.json"
SCHEMAS = ROOT / "packages" / "contracts" / "contracts.schema.json"
ICONS = ROOT / "assets" / "icons"


def test_fixture_exists_and_parses():
    manifest = AssetManifest.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert manifest.asset_id == "asset_42"
    assert manifest.readiness is ReadinessState.READY


def test_fixture_passes_every_gate_and_is_schedulable():
    manifest = AssetManifest.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert manifest.quality.passed
    assert manifest.admissible()


def test_fixture_carries_a_verifiable_content_hash():
    """A 64-hex digest is what lets the scheduler bind to content rather than a path."""
    manifest = AssetManifest.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert len(manifest.content_sha256) == 64
    assert set(manifest.content_sha256) <= set("0123456789abcdef")


def test_fixture_records_provenance_and_redistribution_terms():
    """Phase 0 also requires identifying which files may be redistributed."""
    manifest = AssetManifest.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert manifest.provenance.license == "CC-BY-NC-4.0"
    assert manifest.provenance.model_id
    assert manifest.provenance.seed is not None


def test_fixture_records_the_quantization_it_was_rendered_with():
    """A Q8_0 render is a different artifact from a bf16 one and must stay distinguishable."""
    manifest = AssetManifest.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert manifest.provenance.quantization == "Q8_0"


def test_fixture_round_trips_without_loss():
    raw = FIXTURE.read_text(encoding="utf-8")
    manifest = AssetManifest.model_validate_json(raw)
    assert json.loads(manifest.model_dump_json()) == json.loads(raw)


def test_exported_schemas_match_the_current_contract_set():
    """Regenerate with `uv run python -m vexa_contracts.schema` when this fails."""
    bundle = json.loads(SCHEMAS.read_text(encoding="utf-8"))
    assert set(bundle["schemas"]) == set(registry())
    assert bundle["contract_version"] == 1


@pytest.mark.parametrize("size", [16, 32, 128, 512])
def test_icon_ladder_is_present(size: int):
    assert (ICONS / f"vexa-icon-{size}.png").exists(), f"missing {size}px icon"