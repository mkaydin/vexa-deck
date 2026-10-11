"""Content-bound listening comparisons. Human taste never changes admission policy."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from vexa_yue2.analysis import measure_loudness, measure_true_peak

from .audio_evidence import atomic_json, sha256

VERSION = 1
RATINGS = ("genre_match", "audio_quality", "taste")
DEFECTS = ("click", "dropout", "distortion", "level_jump", "beat_clash", "vocal_collision")
# Siblings across generators share a split; seeds are never tuned on validation results.
CASES = (
    ("techno", "dark melodic techno", "instrumental", "auto", 128),
    ("house", "warm deep house", "instrumental", "auto", 122),
    ("memphis", "dusty 1990s Memphis phonk", "instrumental", "auto", 88),
    ("drift", "aggressive drift phonk", "instrumental", "auto", 150),
    ("boombap", "1990s boom bap, original English rap about city nights", "rap", "en", 92),
    ("jazz", "rainy jazz bar, original English singing about a late train", "singing", "en", 90),
    ("ambient", "spacious free-time ambient", "instrumental", "auto", None),
    ("mixed", "Anatolian psychedelia and house, original Turkish singing", "singing", "tr", 118),
)


def corpus(generators: list[tuple[str, str]], seeds=(42, 1042, 2042)) -> dict:
    from .generators import selection

    if len(set(seeds)) != len(seeds) or len(seeds) < 3:
        raise ValueError("use at least three distinct seeds")
    if not generators or len(set(generators)) != len(generators):
        raise ValueError("use distinct generator/preset pairs")
    rows = []
    for generator, preset in generators:
        selection(generator, preset)
        for case, theme, voice, language, bpm in CASES:
            for index, seed in enumerate(seeds):
                rows.append(
                    {
                        "id": f"{case}-{generator}-{preset}-{seed}",
                        "kind": "track",
                        "case_id": case,
                        "theme": theme,
                        "family_id": f"{case}-{seed}",
                        "split": "calibration" if index == 0 else "validation",
                        "generator": generator,
                        "preset": preset,
                        "seed": seed,
                        "requested_duration_s": 150,
                        "controls": {
                            "track_count": 1,
                            "duration_s": 150,
                            "seed": seed,
                            "vocal_mode": voice,
                            "language": language,
                            "bpm": bpm,
                            "plan_attempts": 3,
                            "render_attempts": 3,
                        },
                    }
                )
    return {"version": VERSION, "purpose": "rendered-comparison", "items": rows}


def validate_items(items: list[dict]):
    if not items:
        raise ValueError("a nonempty comparison is required")
    ids, families, contents = set(), {}, {}
    for row in items:
        identity = row["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,120}", identity) or identity in ids:
            raise ValueError("item IDs must be unique safe filenames")
        ids.add(identity)
        split, family = row["split"], row["family_id"]
        if split not in {"calibration", "validation"} or not family:
            raise ValueError("each item needs a split and musical family")
        if family in families and families[family] != split:
            raise ValueError("musical family leaks between calibration and validation")
        families[family] = split
        if row.get("kind") not in {"track", "transition"}:
            raise ValueError("kind must be track or transition")
        for digest in [row.get("source_sha256"), *row.get("parent_sha256", [])]:
            if not digest:
                continue
            if digest in contents and contents[digest] != split:
                raise ValueError("audio content leaks between calibration and validation")
            contents[digest] = split


def import_reports(roots: list[Path]) -> dict:
    """Import evidence and legacy benchmarks by neighboring files, surviving remounts.

    Legacy reports stay legacy; absent audio, seeds and evidence remain coverage gaps.
    """
    items, seen = [], set()
    paths = sorted({p for root in roots for p in root.rglob("*.evidence.json")})
    paths += sorted({p for root in roots for p in root.rglob("*.report.json")})
    for path in paths:
        report = json.loads(path.read_text())
        is_evidence = path.name.endswith(".evidence.json")
        suffix = ".evidence.json" if is_evidence else ".report.json"
        audio = path.with_name(path.name.removesuffix(suffix) + ".wav")
        if not audio.is_file():
            continue
        digest = sha256(audio)
        if digest in seen:
            continue
        seen.add(digest)
        brief_path = audio.with_suffix(".brief.json")
        brief = json.loads(brief_path.read_text()) if brief_path.exists() else report
        request_path = audio.with_suffix(".request.json")
        request = json.loads(request_path.read_text()) if request_path.exists() else {}
        measured = report.get("final") or report.get("raw") or {}
        if is_evidence and measured.get("source_sha256") != digest:
            raise ValueError(f"evidence does not describe current audio: {path}")
        case = (
            brief.get("theme")
            or brief.get("case")
            or report.get("requested", {}).get("genre")
            or "unknown"
        )
        # The native request identifies MiniMax even when cue rejection prevented a brief.
        generator, preset = brief.get("generator"), brief.get("preset")
        if not generator and "lm_model" in request:
            generator = "minimax"
            lm = Path(request["lm_model"]).name
            preset = next(
                (
                    name
                    for quant, name in (
                        ("Q6_K", "balanced"),
                        ("Q8_0", "quality"),
                        ("Q5_K_M", "compact"),
                    )
                    if quant in lm
                ),
                None,
            )
        draft_identity = request.get("caption") or (brief.get("music_plan") or {}).get(
            "creative_draft"
        )
        family = (
            "legacy-draft-"
            + hashlib.sha256(json.dumps(draft_identity, sort_keys=True).encode()).hexdigest()
            if draft_identity
            else f"legacy-{digest}"
        )
        item = {
            "id": f"sample-{digest[:16]}",
            "kind": "track",
            "case_id": case,
            "theme": case,
            "title": brief.get("title") or audio.stem,
            "family_id": family,
            "split": "validation",
            "generator": generator,
            "preset": preset,
            "seed": brief.get("seed", request.get("seed")),
            "audio_path": str(audio.resolve()),
            "source_sha256": digest,
            "report_path": str(path.resolve()),
            "requested_duration_s": brief.get("requested_duration_s")
            or report.get("duration_target_s"),
            "historical": True,
            "technical": report.get("admission")
            if is_evidence
            else {"playable": report.get("technical_quality_passed"), "mixable": None},
            "corrections": report.get("corrections", []),
            "observations": report.get("observations", {}),
            "requested": report.get(
                "requested",
                {
                    "bpm": brief.get("requested_bpm"),
                    "vocal_mode": (brief.get("music_plan") or {})
                    .get("track_plan", {})
                    .get("intent", {})
                    .get("vocal_mode"),
                },
            ),
            "status": report.get("status"),
            "historical_listener_feedback": brief.get("listener_feedback"),
            "provenance": {
                key: brief.get(key)
                for key in (
                    "model_revision",
                    "runtime_revision",
                    "runtime_patch_sha256",
                    "prompt_origin",
                )
            },
            "wall_s": report.get("wall_s"),
            "request_path": str(request_path.resolve()) if request else None,
            "request_sha256": sha256(request_path) if request else None,
        }
        items.append(item)
    return {"version": VERSION, "purpose": "historical-listening-baseline", "items": items}


def build_bundle(manifest: dict, output: Path, *, base: Path) -> dict:
    """Level-match copies with one linear gain; retain defects and every failed sample."""
    validate_items(manifest["items"])
    if (output / "bundle.json").exists():
        raise ValueError("bundle already exists; choose a new output to preserve annotations")
    rows, available = [], set()
    for item in manifest["items"]:
        row = dict(item)
        if item.get("audio_path"):
            path = (base / item["audio_path"]).resolve()
            if path.is_relative_to(output.resolve()):
                raise ValueError("review output must be separate from source audio")
            digest = sha256(path)
            if item.get("source_sha256") and item["source_sha256"] != digest:
                raise ValueError(f"source audio changed: {item['id']}")
            row.update(audio_path=str(path), source_sha256=digest)
            if item.get("report_path"):
                report_path = (base / item["report_path"]).resolve()
                row.update(report_path=str(report_path), report_sha256=sha256(report_path))
                evidence = json.loads(report_path.read_text())
                if "evidence_version" in evidence:
                    measure = evidence.get("final") or evidence.get("raw") or {}
                    if measure.get("source_sha256") != digest:
                        raise ValueError(f"audio evidence hash mismatch: {item['id']}")
            try:
                samples, sr = sf.read(path, always_2d=True, dtype="float32")
                if not len(samples) or not np.isfinite(samples).all():
                    raise ValueError("empty or nonfinite samples")
                level = measure_loudness(samples, sr).integrated_lufs
                peak = measure_true_peak(samples, sr)
                if not math.isfinite(level) or not math.isfinite(peak) or level <= -70:
                    raise ValueError("silence or unmeasurable level")
                row.update(
                    duration_s=len(samples) / sr, source_lufs=level, source_true_peak_dbtp=peak
                )
                available.add(item["id"])
                del samples
            except (RuntimeError, ValueError) as exc:
                row["listening_error"] = str(exc)
        rows.append(row)
    validate_items(rows)
    # Match whole-file integrated level without nonlinear repair. Quietest peak-limited
    # item sets the shared level, so high-crest-factor sources are not distorted.
    target = min(
        [
            -18.0,
            *[
                r["source_lufs"] - 2 - r["source_true_peak_dbtp"]
                for r in rows
                if r["id"] in available
            ],
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "media").mkdir(exist_ok=True)
    (output / "evidence").mkdir(exist_ok=True)
    for index, row in enumerate(rows, 1):
        row["review_id"] = f"blind-{index:03}"
        for source_key, digest_key, review_key in (
            ("report_path", "report_sha256", "review_report"),
            ("request_path", "request_sha256", "review_request"),
        ):
            if row.get(source_key):
                source = Path(row[source_key])
                row[digest_key] = sha256(source)
                destination = output / "evidence" / f"{row['id']}-{review_key}.json"
                shutil.copyfile(source, destination)
                if sha256(destination) != row[digest_key]:
                    raise ValueError("source metadata changed while creating review snapshot")
                row[review_key] = str(destination.relative_to(output))
        if row["id"] not in available:
            continue
        samples, sr = sf.read(row["audio_path"], always_2d=True, dtype="float32")
        gain = target - row["source_lufs"]
        normalized = samples * 10 ** (gain / 20)
        destination = output / "media" / f"{row['review_id']}.wav"
        sf.write(destination, normalized, sr, subtype="PCM_24")
        row.update(
            review_audio=f"media/{destination.name}",
            review_sha256=sha256(destination),
            review_gain_db=gain,
            review_lufs=measure_loudness(normalized, sr).integrated_lufs,
        )
        if sha256(Path(row["audio_path"])) != row["source_sha256"]:
            raise ValueError("source changed during bundle preparation")
    payload = {
        "version": VERSION,
        "items": rows,
        "target_lufs": target,
        "normalization": "linear gain only; no limiting or defect repair",
        "release_passed": False,
        "promotion": "none",
    }
    payload["bundle_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    atomic_json(output / "bundle.json", payload)
    annotations = {"version": VERSION, "bundle_id": payload["bundle_id"], "reviews": []}
    atomic_json(output / "annotations.json", annotations)
    (output / "index.html").write_text(review_html(payload))
    return payload


def scorecard(bundle: dict, annotations: dict) -> dict:
    """Score human labels independently of technical gates; incomplete coverage fails closed."""
    if annotations.get("bundle_id") != bundle["bundle_id"]:
        raise ValueError("annotations belong to a different bundle")
    items = {r["id"]: r for r in bundle["items"]}
    aliases = {r["review_id"]: r for r in bundle["items"] if r.get("review_id")}
    validate_items(list(items.values()))
    reviews, seen = [], set()
    for label in annotations.get("reviews", []):
        row = aliases.get(label["id"]) or items.get(label["id"])
        key = (row["id"] if row else label["id"], label.get("reviewer"))
        if not row or not key[1] or key in seen:
            raise ValueError("unknown item, missing reviewer, or duplicate review")
        if not row.get("review_sha256") or label.get("review_sha256") != row["review_sha256"]:
            raise ValueError("review audio hash mismatch")
        if row.get("listening_error") or not row.get("review_audio"):
            raise ValueError("cannot score audio that was not available for listening")
        seen.add(key)
        for name in RATINGS:
            value = label.get(name)
            if value is not None and (type(value) is not int or not 1 <= value <= 5):
                raise ValueError("ratings must be integer 1-5 or null")
        if label.get("vocals_present") is not None and type(label["vocals_present"]) is not bool:
            raise ValueError("vocals_present must be boolean or null")
        if not set(label.get("defects", [])) <= set(DEFECTS):
            raise ValueError("unknown defect label")
        if label.get("listened_full") is not True:
            raise ValueError("full item listening is required for submitted reviews")
        if label.get("heard_genre") is not None:
            from .semantic_audio import GENRES

            if label["heard_genre"] not in GENRES:
                raise ValueError("heard_genre must be an observer genre or null")
        reviews.append({**label, "id": row["id"]})
    groups = defaultdict(list)
    for row in items.values():
        cohort = "historical" if row.get("historical") else "matrix"
        if row["kind"] == "transition":
            cohort = "transition"
        groups[(row.get("generator"), row.get("preset"), row["kind"], cohort)].append(row)
    summaries = []
    for (generator, preset, kind, cohort), rows in sorted(
        groups.items(), key=lambda pair: str(pair[0])
    ):
        ids = {r["id"] for r in rows}
        labels = [r for r in reviews if r["id"] in ids]

        def rate(field, group=rows):
            return sum((r.get("technical") or {}).get(field) is True for r in group) / len(group)

        summaries.append(
            {
                "generator": generator,
                "preset": preset,
                "kind": kind,
                "cohort": cohort,
                "samples": len(rows),
                "reviewed_samples": len({r["id"] for r in labels}),
                "playable_rate": rate("playable"),
                "mixable_rate": rate("mixable"),
                "unknown_admission": sum(
                    (r.get("technical") or {}).get("playable") is None for r in rows
                ),
                "correction_rate": sum(bool(r.get("corrections")) for r in rows) / len(rows),
                "mean_ratings": {
                    name: _mean([r[name] for r in labels if r.get(name) is not None])
                    for name in RATINGS
                },
                "mean_generation_realtime_factor": _mean(
                    [
                        r["wall_s"] / r["duration_s"]
                        for r in rows
                        if r.get("wall_s") is not None and r.get("duration_s")
                    ]
                ),
                "audible_defects": sum(bool(r.get("defects")) for r in labels) if labels else None,
            }
        )
    tracks = [r for r in items.values() if r["kind"] == "track"]
    by_content = defaultdict(list)
    for row in tracks:
        if row.get("source_sha256"):
            by_content[row["source_sha256"]].append(row["id"])
    duplicates = {digest: ids for digest, ids in by_content.items() if len(ids) > 1}
    expected = {
        (c[0], g, p, s)
        for g, p in (("yue2", "q8"), ("acestep", "turbo"), ("minimax", "balanced"))
        for c in CASES
        for s in (42, 1042, 2042)
    }
    present = {
        (r["case_id"], r.get("generator"), r.get("preset"), r.get("seed"))
        for r in tracks
        if not r.get("historical")
        and r.get("source_sha256") not in duplicates
        and r.get("duration_s", 0) >= 148
        and r.get("duration_s", 0) <= 182
        and r.get("review_audio")
    }
    return {
        "version": VERSION,
        "bundle_id": bundle["bundle_id"],
        "groups": summaries,
        "reviews": len(reviews),
        "pending_items": sorted(set(items) - {r["id"] for r in reviews}),
        "missing_matrix_cells": [list(cell) for cell in sorted(expected - present)],
        "duplicate_sources": duplicates,
        "samples": [
            {
                key: row.get(key)
                for key in (
                    "id",
                    "review_id",
                    "title",
                    "case_id",
                    "generator",
                    "preset",
                    "seed",
                    "actual_seed",
                    "duration_s",
                    "wall_s",
                    "status",
                    "state",
                    "historical",
                    "technical",
                    "corrections",
                    "transition",
                    "provenance",
                )
            }
            for row in items.values()
        ],
        "render_matrix_complete": expected <= present,
        "release_passed": False,
        "promotion": "none",
        "reason": "Planning, failure replay, source-bound device soak, held-out calibration "
        "and human review are separate release gates; this scorecard cannot promote models.",
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def review_html(bundle: dict) -> str:
    from .semantic_audio import GENRES

    # Model names, admission scores and previous opinions are absent from blind cards.
    cards = []
    public = []
    for index, row in enumerate(bundle["items"], 1):
        public.append({"id": row["review_id"], "review_sha256": row.get("review_sha256")})
        if not row.get("review_audio"):
            cards.append(
                f"<section><h2>Sample {index}</h2><p>Audio unavailable for review.</p></section>"
            )
            continue
        theme = html.escape(
            row.get("theme", row["case_id"]).split("\n[Vexa preferences]\n", 1)[0].strip()
        )
        requested = row.get("requested") or row.get("controls") or {}
        targets = []
        if row.get("requested_duration_s"):
            targets.append(f"Duration target: {row['requested_duration_s']:g} seconds")
        if requested.get("bpm"):
            targets.append(f"Tempo target: {requested['bpm']:g} BPM")
        if requested.get("vocal_mode") and requested["vocal_mode"] != "auto":
            targets.append(f"Voice target: {requested['vocal_mode']}")
        target_text = html.escape(" · ".join(targets))
        ratings = "".join(
            f'<label>{name.replace("_", " ")} <select data-field="{name}">'
            '<option value="">Unrated</option>'
            + "".join(f"<option>{n}</option>" for n in range(1, 6))
            + "</select></label>"
            for name in RATINGS
        )
        defects = "".join(
            f'<label><input type="checkbox" data-defect="{d}">{d.replace("_", " ")}</label>'
            for d in DEFECTS
        )
        genres = '<option value="">Unknown / outside catalog</option>' + "".join(
            f"<option>{html.escape(genre)}</option>" for genre in GENRES
        )
        cards.append(f'''<section data-index="{index - 1}"><h2>Sample {index} · {row["kind"]}</h2>
<p>{theme}</p><p>{target_text}</p>
<audio controls preload="none" src="{row["review_audio"]}"></audio>
<div>{ratings}</div><label>Human voice <select data-field="vocals_present">
<option value="">Unknown</option><option value="true">Present</option>
<option value="false">Absent</option>
</select></label><div>{defects}</div><label>Notes <textarea data-field="notes"></textarea></label>
<label>Heard genre <select data-field="heard_genre">{genres}</select></label>
<label><input type="checkbox" data-field="listened_full">I listened to the full item</label>
</section>''')
    data = json.dumps({"version": VERSION, "bundle_id": bundle["bundle_id"], "items": public})
    return (
        """<!doctype html><meta charset="utf-8"><title>VEXA listening review</title>
<style>body{background:#111719;color:#e3eded;font:16px system-ui;
max-width:900px;margin:40px auto;padding:20px}
section{border:1px solid #446;padding:20px;margin:18px 0}label{display:inline-block;margin:10px}
audio,textarea{width:100%}select,button,input,textarea{font:inherit}button{padding:12px}</style>
<h1>VEXA listening review</h1><p>Compare on the same device and volume.
Ratings: 1 poor, 5 excellent.
Genre match and technical defects are separate from personal taste. Model identities are hidden.
Downloaded labels must be scored against this exact bundle. Save your labels before closing.</p>
<label>Reviewer <input id="reviewer" required></label>
<button id="export">Download annotations</button>
<p id="status" role="status"></p>"""
        + "".join(cards)
        + """<script>
const data = """
        + data
        + """;
document.querySelector('#export').onclick = () => {
 const reviewer = document.querySelector('#reviewer').value.trim();
 if (!reviewer) {
  document.querySelector('#status').textContent = 'Enter a reviewer name.'; return;
 }
 const reviews = [];
 document.querySelectorAll('section[data-index]').forEach(card => {
  if (!card.querySelector('[data-field="listened_full"]').checked) return;
  const item = data.items[Number(card.dataset.index)];
  const row = {...item, reviewer, listened_full:true, defects:[]};
  card.querySelectorAll('[data-field]').forEach(field => {
   const key = field.dataset.field;
   if (key === 'listened_full') return;
   if (key === 'notes') row[key] = field.value;
   else if (key === 'heard_genre') row[key] = field.value || null;
   else if (key === 'vocals_present') row[key] = field.value ? field.value === 'true' : null;
   else row[key] = field.value ? Number(field.value) : null;
  });
  card.querySelectorAll('[data-defect]:checked').forEach(field => {
   row.defects.push(field.dataset.defect);
  });
  reviews.push(row);
 });
 const body = {version:data.version,bundle_id:data.bundle_id,reviews};
 const url = URL.createObjectURL(new Blob([JSON.stringify(body,null,2)],{type:'application/json'}));
 const link = document.createElement('a'); link.href=url;
 link.download='annotations.json'; link.click();
 URL.revokeObjectURL(url);
 document.querySelector('#status').textContent = `${reviews.length} reviews exported.`;
};</script>"""
    )
