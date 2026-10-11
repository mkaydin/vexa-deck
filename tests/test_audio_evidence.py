"""Physical signal regressions: raw defects cannot be hidden by mastering or intent tags."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf
from vexa_audio.cues import CUE_VERSION, CueMap
from vexa_contracts import GenerationSettings
from vexa_orchestrator.audio_calibration import evaluate_annotations
from vexa_orchestrator.audio_evidence import EvidenceSession, inspect_audio, load_evidence, sha256
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.generation import TrackBrief, admit_render, generate_theme
from vexa_orchestrator.generation_jobs import GenerationJobs
from vexa_orchestrator.generation_reports import track_report
from vexa_orchestrator.generation_runner import GenerationRunner, recovered_asset
from vexa_orchestrator.live import LiveSet
from vexa_orchestrator.semantic_audio import observe
from vexa_yue2.analysis import AnalysisError, TemporalAnalysis, measure_true_peak, probe_audio
from vexa_yue2.backends import BackendUnavailable
from vexa_yue2.master import master


def tone(seconds=12, sr=48000, amplitude=0.1, channels=2):
    mono = amplitude * np.sin(2 * np.pi * 1000 * np.arange(round(sr * seconds)) / sr)
    return np.repeat(mono[:, None], channels, axis=1)


def write(path, samples, sr=48000):
    sf.write(path, samples, sr, subtype="FLOAT")
    return path


@pytest.mark.parametrize(
    "kind, reason",
    [
        ("silent", "silent_audio"),
        ("clipped", "severe_source_clipping"),
        ("dc", "excessive_dc_offset"),
        ("dropout", "internal_digital_dropout"),
        ("nonfinite", "nonfinite_samples"),
        ("short", "duration_mismatch"),
    ],
)
def test_raw_defects_reject_before_mastering_and_preserve_original(tmp_path, kind, reason):
    samples = tone()
    if kind == "silent":
        samples[:] = 0
    elif kind == "clipped":
        samples[1000:10000] = 1
    elif kind == "dc":
        samples += 0.15
    elif kind == "dropout":
        samples[4 * 48000 : 7 * 48000] = 0
    elif kind == "nonfinite":
        samples[100] = np.nan
    else:
        samples = samples[:48000]
    path = write(tmp_path / "test.wav", samples)
    original = path.read_bytes()
    session = EvidenceSession(path, duration_s=12, requested={})
    with pytest.raises(ValueError, match=reason):
        session.begin()
    assert path.read_bytes() == original == path.with_suffix(".raw.wav").read_bytes()
    report = load_evidence(path)
    assert report["final"] is None and report["status"] == "rejected"
    assert reason in report["admission"]["reason_codes"]
    assert not path.with_suffix(".json").exists()
    assert "NaN" not in path.with_suffix(".evidence.json").read_text()


def test_one_bounded_pass_repairs_format_dc_and_edges_then_remeasures(tmp_path):
    path = write(tmp_path / "repair.wav", tone(sr=32000, channels=1) + 0.01, sr=32000)
    samples, sr = sf.read(path, always_2d=True)
    samples[0] = samples[-1] = 0.1
    write(path, samples, sr)
    session = EvidenceSession(path, duration_s=12, requested={})
    session.begin()
    session.mastered(master(path))
    report = load_evidence(path)
    ops = [step["operation"] for step in report["corrections"]]
    assert {
        "mono_to_stereo",
        "resample",
        "dc_removal",
        "boundary_fade",
        "loudness_and_peak",
    } <= set(ops)
    assert report["raw"]["channels"] == 1 and report["final"]["channels"] == 2
    assert report["final"]["sample_rate_hz"] == 48000
    assert max(abs(x) for x in report["final"]["dc_offset_by_channel"]) < 1e-4
    assert report["raw"]["source_sha256"] == sha256(path.with_suffix(".raw.wav"))
    assert report["final"]["source_sha256"] == sha256(path)
    assert report["raw"]["source_sha256"] != report["final"]["source_sha256"]


def test_out_of_phase_stereo_cannot_hide_peak_or_silence(tmp_path):
    samples = tone(amplitude=0.7)
    samples[:, 1] *= -1
    path = write(tmp_path / "stereo.wav", samples)
    report = inspect_audio(path)
    assert report["rms"] > 0.4 and report["true_peak_dbtp"] > -4
    assert probe_audio(path)[0].rms > 0.4
    result = master(path)
    assert result.output_lufs == pytest.approx(-14, abs=0.2)
    assert result.output_peak_dbtp <= -1.5 + 0.01


def test_polyphase_peak_detects_intersample_overshoot_and_master_leaves_headroom(tmp_path):
    samples = 0.85 * np.sin(2 * np.pi * 0.25 * np.arange(48000 * 2) + np.pi / 4)
    assert measure_true_peak(samples, 48000) > 20 * np.log10(np.max(np.abs(samples))) + 1
    path = write(tmp_path / "peak.wav", np.stack([samples, -samples], axis=1))
    report = master(path, target_lufs=-6, ceiling_dbtp=-12)
    assert (
        report.limited and measure_true_peak(sf.read(path, always_2d=True)[0], 48000) <= -12 + 0.01
    )


def test_decode_gate_refuses_nonfinite_samples(tmp_path):
    samples = tone(seconds=1)
    samples[10] = np.inf
    path = write(tmp_path / "bad.wav", samples)
    with pytest.raises(AnalysisError, match="nonfinite"):
        probe_audio(path)


def admitted_fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_CLAP_MODEL", "")
    path = write(tmp_path / "live-123456abcdef.wav", tone(seconds=30))
    monkeypatch.setattr(
        "vexa_yue2.gates.detect_temporal",
        lambda *a: TemporalAnalysis(120, 0.9, "A minor", 0.7, []),
    )
    monkeypatch.setattr(
        "vexa_orchestrator.generation.analyse_cues",
        lambda path, **kw: CueMap(
            sha256(path),
            CUE_VERSION,
            120,
            0.9,
            tuple(np.arange(0, 30, 0.5)),
            0.9,
            (),
            4,
            24,
            "beat",
        ),
    )
    return admit_render(path, theme="techno instrumental", duration_s=30, energy=0.9)


def test_admission_depot_reports_and_recovery_use_hash_bound_evidence(monkeypatch, tmp_path):
    asset = admitted_fixture(monkeypatch, tmp_path)
    evidence = load_evidence(asset.path, source_sha256=asset.manifest.content_sha256)
    assert evidence["admission"]["playable"] and evidence["admission"]["mixable"]
    assert evidence["admission"]["constraints_satisfied"] is None
    assert evidence["observations"]["status"] == "unavailable"
    assert asset.manifest.energy() is None and "instrumental" not in asset.manifest.tags
    depot = Depot(tmp_path)
    assert depot.assets[asset.manifest.asset_id].vibe["sub_bass_fraction"] is not None
    assert depot.search("techno no lyrics")
    assert recovered_asset(asset.path)
    evidence["final"]["source_sha256"] = "0" * 64
    asset.path.with_suffix(".evidence.json").write_text(json.dumps(evidence))
    assert recovered_asset(asset.path) is None
    assert not Depot(tmp_path).assets
    with pytest.raises(ValueError, match="evidence"):
        depot.add(asset.manifest, asset.path)


def test_render_evidence_history_survives_retry_and_changed_report_is_not_trusted(
    monkeypatch, tmp_path
):
    asset = admitted_fixture(monkeypatch, tmp_path)
    jobs = GenerationJobs(tmp_path / "jobs.sqlite3")
    snapshot = {
        "generator": "yue2",
        "preset": "q8",
        "library": str(tmp_path),
        "writer": {"mode": "template", "model": "none"},
        "controls": GenerationSettings(track_count=1).model_dump(),
    }
    batch = jobs.open_batch("techno instrumental", snapshot, 1)
    jobs.update(
        batch, 0, state="rejected", asset_path=str(asset.path), asset_id=asset.manifest.asset_id
    )
    jobs.attempt(
        batch,
        0,
        "rendering",
        "failed",
        "fixture rejection",
        evidence=GenerationRunner.evidence_reference(asset.path),
    )
    jobs.reset_slot(batch, 0)
    report = track_report(jobs, batch, 0)
    assert len(report["evidence_history"]) == 1 and report["audio_evidence"] is None
    asset.path.with_suffix(".evidence.json").write_text("{}")
    report = track_report(jobs, batch, 0)
    assert report["evidence_history"] == [] and report["warnings"]


def test_missing_semantic_model_is_unknown_not_instrumental(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_CLAP_MODEL", str(tmp_path / "not-installed"))
    result = observe(tmp_path / "does-not-exist.wav", {"genre": "phonk"})
    assert result["status"] == "unavailable"
    assert result["vocal_mode"] is None and result["confidence"] is None
    assert result["genre"] is None and result["vocal_occupancy"] is None


def test_playable_free_time_audio_is_preserved_but_not_admitted_for_beat_mixing(
    monkeypatch, tmp_path
):
    path = write(tmp_path / "free.wav", tone(seconds=12))
    monkeypatch.setattr(
        "vexa_yue2.gates.detect_temporal", lambda *a: TemporalAnalysis(0, 0, None, 0)
    )
    with pytest.raises(ValueError, match="quality_gate_failed"):
        admit_render(path, theme="ambient", duration_s=12, energy=0.3)
    admission = load_evidence(path)["admission"]
    assert admission["playable"] and not admission["mixable"]
    assert path.exists() and not path.with_suffix(".json").exists()


def test_unexpected_analysis_error_is_persisted_and_never_published(monkeypatch, tmp_path):
    path = write(tmp_path / "failed.wav", tone(seconds=12))
    monkeypatch.setattr(
        "vexa_orchestrator.generation.master",
        lambda *a: (_ for _ in ()).throw(RuntimeError("disk fixture failure")),
    )
    with pytest.raises(RuntimeError, match="disk fixture"):
        admit_render(path, theme="techno", duration_s=12, energy=0.3)
    report = load_evidence(path)
    assert report["status"] == "analysis_failed" and report["raw"]
    assert not path.with_suffix(".json").exists()


def test_calibration_requires_independent_held_out_content_and_never_enables_a_gate(tmp_path):
    annotations = []
    for index in range(80):
        vocal = index % 2 == 0
        path = tmp_path / f"label-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "final": {"source_sha256": f"{index:064x}"},
                    "observations": {
                        "status": "observed",
                        "model_hashes": {"weights": "fixture"},
                        "voice_margin": 0.1 if vocal else -0.1,
                    },
                }
            )
        )
        annotations.append(
            {
                "evidence_path": str(path),
                "vocals_present": vocal,
                "split": "calibration" if index < 40 else "validation",
            }
        )
    assert evaluate_annotations(annotations[:3])["status"] == "insufficient_labels"
    result = evaluate_annotations(annotations)
    assert result["status"] == "candidate" and result["validation"]["false_positive_rate"] == 0
    assert not result["hard_gate_enabled"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_annotations([*annotations, annotations[0]])
    changed = json.loads((tmp_path / "label-79.json").read_text())
    changed["observations"]["model_hashes"] = {"weights": "different"}
    (tmp_path / "label-79.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="pinned"):
        evaluate_annotations(annotations)


def test_reserved_render_cannot_overwrite_a_published_or_raw_track(tmp_path):
    path = tmp_path / "live-123456abcdef.wav"
    path.write_bytes(b"playing audio")
    with pytest.raises(ValueError, match="overwrite"):
        generate_theme("techno", tmp_path, generator="minimax", render_id=path.stem, style="techno")
    assert path.read_bytes() == b"playing audio"


def test_live_reads_small_progress_artifact_without_changing_playback(tmp_path):
    from types import SimpleNamespace

    live = LiveSet(SimpleNamespace(library=tmp_path, assets={}), preview_dir=tmp_path / "previews")
    live._generating = True
    live._generation_stage = "started"
    live._generation_artifact = tmp_path / "live-123456abcdef.wav"
    progress = live._generation_artifact.with_suffix(".progress.json")
    progress.write_text(
        json.dumps({"asset_id": live._generation_artifact.stem, "stage": "analyzing"})
    )
    assert live.status()["generation"]["stage"] == "analyzing"
    assert live.state is None and not live._stop.is_set()
    progress.write_text(json.dumps({"asset_id": "stale", "stage": "rejected"}))
    assert live.status()["generation"]["stage"] == "started"
    progress.write_text("invalid json")
    assert live.status()["generation"]["stage"] == "started"
    progress.write_text("[]")
    assert live.status()["generation"]["stage"] == "started"


def test_restart_preserves_evidence_of_a_render_that_crashed_before_publication(
    monkeypatch, tmp_path
):
    asset = admitted_fixture(monkeypatch, tmp_path)
    asset.path.with_suffix(".json").unlink()
    jobs = GenerationJobs(tmp_path / "jobs.sqlite3")
    snapshot = {
        "generator": "yue2",
        "preset": "q8",
        "library": str(tmp_path),
        "writer": {"mode": "template", "model": "none"},
    }
    batch = jobs.open_batch("techno", snapshot, 1)
    jobs.update(
        batch,
        0,
        state="rendering",
        brief=asdict(TrackBrief("techno", 165, 0.5, "fixture")),
        render_attempts=1,
        asset_id=asset.manifest.asset_id,
        asset_path=str(asset.path),
    )

    def renderer(*args, **kwargs):
        raise BackendUnavailable("fixture renderer offline")

    GenerationRunner(
        jobs,
        batch,
        tmp_path,
        planner=lambda *a, **kw: pytest.fail("cached plan"),
        renderer=renderer,
        admitted=lambda *a: None,
        event=lambda *a: None,
        cancel=threading.Event(),
    ).run()
    attempts = jobs.attempts(batch, 0)
    assert attempts[0]["stage"] == "recovery" and attempts[0]["evidence"]
    report = track_report(jobs, batch, 0)
    assert report["evidence_history"][0]["asset_id"] == asset.manifest.asset_id
