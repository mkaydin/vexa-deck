"""Listening labels must describe exact audio; coverage and taste cannot fake release gates."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from vexa_orchestrator.audio_calibration import evaluate_annotations, evaluate_genre_annotations
from vexa_orchestrator.audio_evidence import sha256
from vexa_orchestrator.listening_eval import build_bundle, corpus, scorecard, validate_items
from vexa_orchestrator.semantic_audio import GENRES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_music import add_transitions, score_bundle


def fixture(tmp_path, name="one", amplitude=0.1):
    path = tmp_path / f"{name}.wav"
    time = np.arange(8000) / 8000
    samples = amplitude * np.sin(2 * np.pi * 440 * time)
    sf.write(path, np.stack([samples, -samples], axis=1), 8000, subtype="FLOAT")
    return {
        "id": name,
        "kind": "track",
        "case_id": "techno",
        "theme": "techno",
        "generator": "minimax",
        "preset": "balanced",
        "seed": 42,
        "family_id": name,
        "split": "validation",
        "audio_path": str(path),
        "technical": {"playable": True, "mixable": False},
    }


def test_matrix_is_matched_and_reserves_independent_seed_families():
    matrix = corpus([("yue2", "q8"), ("acestep", "turbo"), ("minimax", "balanced")])
    assert len(matrix["items"]) == 72
    validate_items(matrix["items"])
    for case in {row["case_id"] for row in matrix["items"]}:
        rows = [row for row in matrix["items"] if row["case_id"] == case]
        assert len({json.dumps(row["controls"], sort_keys=True) for row in rows}) == 3
        assert {row["split"] for row in rows} == {"calibration", "validation"}
        assert {row["requested_duration_s"] for row in rows} == {150}


@pytest.mark.parametrize("field", ["family_id", "source_sha256"])
def test_sibling_or_content_leakage_is_rejected(tmp_path, field):
    one, two = fixture(tmp_path), fixture(tmp_path, "two")
    one[field] = two[field] = "same"
    one["split"] = "calibration"
    with pytest.raises(ValueError, match="leaks"):
        validate_items([one, two])


def test_bundle_preserves_sources_and_matches_levels_without_repair(tmp_path):
    one, two = fixture(tmp_path), fixture(tmp_path, "two", amplitude=0.6)
    before = [Path(row["audio_path"]).read_bytes() for row in (one, two)]
    bundle = build_bundle({"items": [one, two]}, tmp_path / "review", base=tmp_path)
    assert [Path(row["audio_path"]).read_bytes() for row in (one, two)] == before
    assert bundle["items"][0]["review_lufs"] == pytest.approx(
        bundle["items"][1]["review_lufs"], abs=0.01
    )
    assert all(row["review_sha256"] for row in bundle["items"])
    page = (tmp_path / "review/index.html").read_text()
    assert "minimax" not in page and "balanced" not in page
    assert "genre_match" in page and "taste" in page
    pending = score_bundle(tmp_path / "review", tmp_path / "review/annotations.json")
    assert pending["reviews"] == 0 and pending["release_passed"] is False
    assert pending["groups"][0]["mean_ratings"]["taste"] is None
    with pytest.raises(ValueError, match="already exists"):
        build_bundle({"items": [one]}, tmp_path / "review", base=tmp_path)


def test_unplayable_source_retained_as_failure_and_never_scored(tmp_path):
    row = fixture(tmp_path)
    sf.write(row["audio_path"], np.full((100, 2), np.nan), 8000, subtype="FLOAT")
    bundle = build_bundle({"items": [row]}, tmp_path / "review", base=tmp_path)
    assert bundle["items"][0]["listening_error"]
    assert "review_audio" not in bundle["items"][0]
    report = scorecard(bundle, {"bundle_id": bundle["bundle_id"], "reviews": []})
    assert report["pending_items"] == ["one"]
    assert not report["render_matrix_complete"]


def test_taste_does_not_override_failed_admission_and_hashes_bind_labels(tmp_path):
    row = fixture(tmp_path)
    row["technical"]["playable"] = False
    bundle = build_bundle({"items": [row]}, tmp_path / "review", base=tmp_path)
    label = {
        "id": "one",
        "reviewer": "listener",
        "taste": 5,
        "audio_quality": 2,
        "genre_match": 1,
        "vocals_present": False,
        "heard_genre": "house",
        "listened_full": True,
        "defects": ["distortion"],
        "review_sha256": bundle["items"][0]["review_sha256"],
    }
    labels = {"bundle_id": bundle["bundle_id"], "reviews": [label]}
    result = scorecard(bundle, labels)
    assert result["groups"][0]["mean_ratings"]["taste"] == 5
    assert result["groups"][0]["playable_rate"] == 0
    assert result["promotion"] == "none" and not result["release_passed"]
    label["review_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        scorecard(bundle, labels)


@pytest.mark.parametrize("change", ["audio", "metadata"])
def test_score_refuses_changed_bundle(tmp_path, change):
    bundle = build_bundle({"items": [fixture(tmp_path)]}, tmp_path / "review", base=tmp_path)
    if change == "audio":
        (tmp_path / "review" / bundle["items"][0]["review_audio"]).write_bytes(b"changed")
    else:
        bundle["items"][0]["split"] = "calibration"
        (tmp_path / "review/bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="changed"):
        score_bundle(tmp_path / "review", tmp_path / "review/annotations.json")


def test_transition_cannot_connect_calibration_and_validation_audio(tmp_path):
    one, two = fixture(tmp_path), fixture(tmp_path, "two")
    one.update(source_sha256="one", split="calibration")
    two.update(source_sha256="two", split="validation")
    report = {
        "theme": "techno",
        "source": "one",
        "source_title": "one",
        "candidates": [{"source_sha256": "one", "incoming_sha256": "two"}],
    }
    with pytest.raises(ValueError, match="crosses"):
        add_transitions({"items": [one, two]}, report)


def observed(tmp_path, index, split, genre="techno"):
    path = tmp_path / f"evidence-{index}.json"
    path.write_text(
        json.dumps(
            {
                "final": {"source_sha256": str(index)},
                "observations": {
                    "status": "observed",
                    "voice_margin": 0.1,
                    "model_hashes": {"weights": "pinned"},
                    "genre_ranking": [
                        {"label": g, "cosine": 0.9 if g == genre else 0.1} for g in GENRES
                    ],
                },
            }
        )
    )
    return {
        "evidence_path": str(path),
        "split": split,
        "heard_genre": genre,
        "vocals_present": True,
        "family_id": f"family-{index}",
    }


def test_genre_metrics_use_human_held_out_labels_not_target_intent(tmp_path):
    calibration = observed(tmp_path, 0, "calibration")
    held_out = observed(tmp_path, 1, "validation")
    held_out["heard_genre"] = "house"
    result = evaluate_genre_annotations([calibration, held_out])
    assert result["top1_accuracy"] == 0 and result["held_out_tracks"] == 1
    assert not result["hard_gate_enabled"]
    assert evaluate_genre_annotations([])["status"] == "insufficient_labels"
    held_out["family_id"] = calibration["family_id"]
    with pytest.raises(ValueError, match="family"):
        evaluate_genre_annotations([calibration, held_out])
    with pytest.raises(ValueError, match="family"):
        evaluate_annotations([calibration, held_out])


def test_evidence_hash_cannot_certify_replaced_audio(tmp_path):
    row = fixture(tmp_path)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"evidence_version": 1, "final": {"source_sha256": "changed"}}))
    row["report_path"] = str(report)
    with pytest.raises(ValueError, match="evidence hash"):
        build_bundle({"items": [row]}, tmp_path / "review", base=tmp_path)
    row.pop("report_path")
    row["source_sha256"] = sha256(Path(row["audio_path"]))
    Path(row["audio_path"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="source audio changed"):
        build_bundle({"items": [row]}, tmp_path / "review", base=tmp_path)


def test_many_retries_of_two_drafts_do_not_qualify_vocal_calibration(tmp_path):
    labels = []
    for index in range(80):
        split = "calibration" if index < 40 else "validation"
        row = observed(tmp_path, index, split)
        row["vocals_present"] = index % 2 == 0
        row["family_id"] = f"{split}-{row['vocals_present']}"
        path = Path(row["evidence_path"])
        report = json.loads(path.read_text())
        report["observations"]["voice_margin"] = 0.3 if row["vocals_present"] else -0.3
        path.write_text(json.dumps(report))
        labels.append(row)
    result = evaluate_annotations(labels)
    assert result["counts"]["validation"] == {"vocal": 20, "instrumental": 20}
    assert result["family_counts"]["validation"] == {"vocal": 1, "instrumental": 1}
    assert result["status"] == "insufficient_labels"


def test_historical_and_matched_supply_metrics_are_separate(tmp_path):
    one, two = fixture(tmp_path), fixture(tmp_path, "two", amplitude=0.3)
    one.update(historical=True, wall_s=500)
    two.update(historical=False, wall_s=1)
    bundle = build_bundle({"items": [one, two]}, tmp_path / "review", base=tmp_path)
    result = scorecard(bundle, {"bundle_id": bundle["bundle_id"], "reviews": []})
    groups = {row["cohort"]: row for row in result["groups"]}
    assert groups["historical"]["mean_generation_realtime_factor"] == 500
    assert groups["matrix"]["mean_generation_realtime_factor"] == 1
    assert all(group["audible_defects"] is None for group in groups.values())
