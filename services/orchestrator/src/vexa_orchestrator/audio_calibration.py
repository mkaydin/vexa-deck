"""Held-out vocal-margin evaluation; never promotes a similarity model into a hard gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def metrics(rows: list[dict], threshold: float) -> dict:
    tp = sum(row["voice_margin"] >= threshold and row["vocals_present"] for row in rows)
    fp = sum(row["voice_margin"] >= threshold and not row["vocals_present"] for row in rows)
    fn = sum(row["voice_margin"] < threshold and row["vocals_present"] for row in rows)
    tn = len(rows) - tp - fp - fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def evaluate_annotations(annotations: list[dict]) -> dict:
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("annotations must be a nonempty list")
    rows, identities, seen, families = [], set(), set(), {}
    for annotation in annotations:
        if type(annotation.get("vocals_present")) is not bool:
            raise ValueError("vocals_present must be a human annotated boolean")
        if annotation.get("split") not in {"calibration", "validation"}:
            raise ValueError("split must be calibration or validation")
        family = annotation.get("family_id")
        if family:
            if family in families and families[family] != annotation["split"]:
                raise ValueError("musical family leaks between calibration/validation")
            families[family] = annotation["split"]
        path = Path(annotation["evidence_path"])
        evidence = json.loads(path.read_text())
        observations = evidence["observations"]
        if observations.get("status") != "observed" or not observations.get("model_hashes"):
            raise ValueError("only model-identified observed reports can be evaluated")
        margin = observations.get("voice_margin")
        if not isinstance(margin, (int, float)) or not math.isfinite(margin):
            raise ValueError("voice margin is missing or nonfinite")
        source_hash = evidence["final"]["source_sha256"]
        if source_hash in seen:
            raise ValueError("duplicate audio content leaks between calibration/validation")
        seen.add(source_hash)
        identities.add(json.dumps(observations["model_hashes"], sort_keys=True))
        rows.append({**annotation, "voice_margin": margin, "family_id": family or source_hash})
    if len(identities) != 1:
        raise ValueError("reports must use exactly one pinned model identity")
    calibration = [row for row in rows if row["split"] == "calibration"]
    validation = [row for row in rows if row["split"] == "validation"]
    counts = {
        split: {
            label: sum(row["vocals_present"] == (label == "vocal") for row in group)
            for label in ("vocal", "instrumental")
        }
        for split, group in (("calibration", calibration), ("validation", validation))
    }
    result = {
        "version": 1,
        "mode": "evaluation_only",
        "hard_gate_enabled": False,
        "model_hashes": json.loads(next(iter(identities))),
        "counts": counts,
        "family_counts": {
            split: {
                label: len(
                    {
                        row["family_id"]
                        for row in group
                        if row["vocals_present"] == (label == "vocal")
                    }
                )
                for label in ("vocal", "instrumental")
            }
            for split, group in (("calibration", calibration), ("validation", validation))
        },
        "annotation_sha256": hashlib.sha256(
            json.dumps(annotations, sort_keys=True).encode()
        ).hexdigest(),
    }
    if any(count < 20 for group in result["family_counts"].values() for count in group.values()):
        return {
            **result,
            "status": "insufficient_labels",
            "threshold": None,
            "reason": "At least 20 independent track families per vocal class in each split "
            "are required; retries of one draft do not supply independent labels.",
        }
    candidates = []
    for threshold in sorted({row["voice_margin"] for row in calibration}):
        score = metrics(calibration, threshold)
        if score["false_positive_rate"] <= 0.05 and score["recall"] >= 0.8:
            candidates.append((score["recall"], score["precision"], threshold, score))
    if not candidates:
        return {**result, "status": "not_qualified", "threshold": None}
    _, _, threshold, calibration_scores = max(candidates)
    held_out = metrics(validation, threshold)
    passed = held_out["false_positive_rate"] <= 0.05 and held_out["recall"] >= 0.8
    return {
        **result,
        "status": "candidate" if passed else "not_qualified",
        "threshold": threshold,
        "calibration": calibration_scores,
        "validation": held_out,
        "reason": "Manual multi-genre review is required before any admission-policy promotion.",
    }


def evaluate_genre_annotations(annotations: list[dict]) -> dict:
    """Evaluate the fixed genre catalog on human labels; learn no thresholds on validation.

    Requested phrases vary across renders. Only the fixed catalog is ranked here, so adding
    a target synonym does not silently change the class set in a comparison.
    """
    from collections import Counter

    from .semantic_audio import GENRES

    rows, identities, seen, families = [], set(), set(), {}
    for annotation in annotations:
        genre, split = annotation["heard_genre"], annotation["split"]
        if genre not in GENRES or split not in {"calibration", "validation"}:
            raise ValueError("invalid human genre label or split")
        evidence = json.loads(Path(annotation["evidence_path"]).read_text())
        observations = evidence["observations"]
        if observations.get("status") != "observed" or not observations.get("model_hashes"):
            raise ValueError("only pinned observed genre reports can be evaluated")
        digest = evidence["final"]["source_sha256"]
        if digest in seen:
            raise ValueError("duplicate audio content in genre evaluation")
        seen.add(digest)
        family = annotation.get("family_id", digest)
        if family in families and families[family] != split:
            raise ValueError("musical family leaks between calibration/validation")
        families[family] = split
        identities.add(json.dumps(observations["model_hashes"], sort_keys=True))
        rankings = {r["label"]: r["cosine"] for r in observations["genre_ranking"]}
        if not set(GENRES) <= set(rankings) or any(
            not isinstance(rankings[g], (int, float)) or not math.isfinite(rankings[g])
            for g in GENRES
        ):
            raise ValueError("fixed genre catalog scores are missing or nonfinite")
        ranked = sorted(GENRES, key=lambda g: (-rankings[g], g))
        rows.append(
            {"split": split, "human": genre, "predicted": ranked[0], "top3": genre in ranked[:3]}
        )
    if len(identities) > 1:
        raise ValueError("genre reports must use one pinned model identity")
    counts = {
        split: dict(Counter(r["human"] for r in rows if r["split"] == split))
        for split in ("calibration", "validation")
    }
    held_out = [r for r in rows if r["split"] == "validation"]
    return {
        "version": 1,
        "mode": "evaluation_only",
        "hard_gate_enabled": False,
        "status": "evaluated" if held_out else "insufficient_labels",
        "model_hashes": json.loads(next(iter(identities))) if identities else None,
        "counts": counts,
        "held_out_tracks": len(held_out),
        "top1_accuracy": sum(r["human"] == r["predicted"] for r in held_out) / len(held_out)
        if held_out
        else None,
        "top3_accuracy": sum(r["top3"] for r in held_out) / len(held_out) if held_out else None,
        "confusion": [
            {"human": human, "predicted": predicted, "count": count}
            for (human, predicted), count in sorted(
                Counter((r["human"], r["predicted"]) for r in held_out).items()
            )
        ],
        "reason": "Fixed-catalog ranking diagnostic, not a calibrated genre admission gate.",
    }
