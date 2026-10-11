"""Control-plane forecasts. Unknown latency is never replaced with a made-up ETA."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np


def latency_key(snapshot: dict) -> str:
    controls, writer = snapshot.get("controls", {}), snapshot.get("writer", {})
    return json.dumps(
        {
            "generator": snapshot.get("generator"),
            "preset": snapshot.get("preset"),
            "render_speed": controls.get("render_speed", "quality"),
            "duration_s": controls.get("duration_s", 165),
            "vocal_mode": controls.get("vocal_mode", "auto"),
            "tempo_policy": controls.get("tempo_policy", "hint"),
            "bpm": controls.get("bpm"),
            "writer": [writer.get("mode"), writer.get("model")],
        },
        sort_keys=True,
    )


class LatencyHistory:
    """Small persistent history, written outside the callback. Failed work stays visible."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS latency "
                "(kind TEXT, profile TEXT, seconds REAL, success INTEGER, at REAL)"
            )

    def record(self, kind: str, profile: str, seconds: float, *, success: bool):
        if not math.isfinite(seconds) or seconds < 0:
            return
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO latency VALUES(?,?,?,?,?)",
                (kind, profile, seconds, int(success), time.time()),
            )
            db.execute(
                "DELETE FROM latency WHERE rowid NOT IN "
                "(SELECT rowid FROM latency ORDER BY rowid DESC LIMIT 2000)"
            )

    def estimate(self, kind: str, profile: str) -> dict:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT seconds,success FROM latency WHERE kind=? AND profile=? "
                "ORDER BY rowid DESC LIMIT 30",
                (kind, profile),
            ).fetchall()
        successful = [row[0] for row in rows if row[1]]
        overhead = sum(row[0] for row in rows if not row[1]) / len(successful) if successful else 0
        return {
            "samples": len(successful),
            "failed_samples": sum(not row[1] for row in rows),
            "confidence": "unknown"
            if not successful
            else "provisional"
            if len(successful) < 3
            else "measured",
            "failure_overhead_s": round(overhead, 2),
            "median_s": round(float(np.median(successful)) + overhead, 2) if successful else None,
            "p90_s": round(float(np.quantile(successful, 0.9)) + overhead, 2)
            if successful
            else None,
        }


def forecast(
    *,
    current_remaining_s: float,
    prepared_span_s: float,
    candidate_spans: list[tuple[str, str, float]],
    supply: dict,
    preparation: dict,
    required: int = 0,
    ready: int = 0,
    elapsed_s: float = 0,
    fade_s: float = 4,
    safety_s: float = 15,
) -> dict:
    """Spans include the fade tail. Subtract each overlap; never double count a render/family.

    Candidate spans are a conditional inventory estimate. Only resident prepared PCM
    counts as secured supply. Matching the current track does not prove a whole chain.
    """
    secured = max(0, current_remaining_s) + max(0, prepared_span_s - fade_s)
    seen_hashes, seen_families = set(), set()
    extra = 0.0
    for content_hash, family, seconds in candidate_spans:
        if content_hash in seen_hashes or family in seen_families:
            continue
        seen_hashes.add(content_hash)
        seen_families.add(family)
        extra += max(0, seconds - fade_s)
    median, upper = supply.get("median_s"), supply.get("p90_s")
    prep_upper = preparation.get("p90_s")
    missing = max(0, required - ready)
    # Cold-start uncertainty is explicit. A single sample is a provisional observation.
    eta = max(0, missing * median - elapsed_s) if median is not None else None
    eta_upper = max(0, missing * upper - elapsed_s) if upper is not None else None
    overdue = bool(missing and upper is not None and elapsed_s > upper + safety_s)
    if overdue:
        eta = eta_upper = None
    threshold = upper + (prep_upper or 0) + safety_s if upper is not None else None
    return {
        "secured_s": round(secured, 2),
        "conditional_inventory_s": round(secured + extra, 2),
        "candidate_count": len(seen_hashes),
        "overlap_s": fade_s,
        "safety_margin_s": safety_s,
        "supply_latency": supply,
        "preparation_latency": preparation,
        "recommended_buffer_s": threshold,
        "startup_eta_s": round(eta, 2) if eta is not None else None,
        "startup_eta_upper_s": round(eta_upper, 2) if eta_upper is not None else None,
        "eta_overdue": overdue,
        "shortfall": secured < threshold if threshold is not None else None,
        "assumption": "candidate chain not yet rendered; only prepared PCM is secured",
    }
