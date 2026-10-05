"""Protect the analysis report from plausible defaults and stale source audio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_analysis import audit
from extract_vibe import EXTRACTOR_VERSION, chords_and_tuning


def test_tuning_is_converted_from_fractional_semitones(monkeypatch, tmp_path):
    import extract_vibe
    import librosa

    monkeypatch.setattr(extract_vibe, "_mono", lambda _: (np.ones(4096), 48000))
    monkeypatch.setattr(
        librosa.feature, "chroma_cqt", lambda **_: np.ones((12, 4), dtype=float)
    )
    seen = {}

    def tuning(**kwargs):
        seen.update(kwargs)
        return 0.25

    monkeypatch.setattr(librosa, "estimate_tuning", tuning)
    _, hz = chords_and_tuning(tmp_path / "unused.wav")
    assert seen["sr"] == 48000
    assert seen["y"].shape == (4096,)
    assert hz == pytest.approx(440.0 * 2 ** (0.25 / 12))


def test_audit_rejects_stale_sidecar(tmp_path):
    import hashlib

    wav = tmp_path / "track.wav"
    wav.write_bytes(b"source audio")
    digest = hashlib.sha256(wav.read_bytes()).hexdigest()
    wav.with_suffix(".json").write_text(json.dumps({"content_sha256": digest}))
    sidecar = {
        "source_sha256": digest,
        "extractor_version": EXTRACTOR_VERSION,
        "rhythm": "unknown",
        "onset_grid_alignment": None,
        "tuning_hz": None,
        "sub_bass_fraction": 0.1,
        "presence_fraction": 0.2,
    }
    wav.with_suffix(".vibe.json").write_text(json.dumps(sidecar))
    report, errors = audit(tmp_path)
    assert report["valid"] and not errors

    wav.write_bytes(b"changed audio")
    report, errors = audit(tmp_path)
    assert not report["valid"]
    assert any("hash does not match audio" in error for error in errors)
