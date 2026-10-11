"""P7: bounded render matrix, blind listening bundles and content-bound human scorecards."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
from backend_runtime import workspace_pythonpath  # noqa: E402

for source in workspace_pythonpath(ROOT).split(":"):
    sys.path.insert(0, source)

from vexa_orchestrator.audio_evidence import atomic_json as _atomic_json  # noqa: E402
from vexa_orchestrator.audio_evidence import sha256  # noqa: E402
from vexa_orchestrator.listening_eval import (  # noqa: E402
    build_bundle,
    corpus,
    import_reports,
    scorecard,
    validate_items,
)


def atomic_json(path: Path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, body)


def render_matrix(manifest: dict, output: Path, limit: int, selected: list[str]) -> dict:
    from vexa_contracts import AssetManifest, GenerationSettings
    from vexa_orchestrator.background import run_job
    from vexa_orchestrator.depot import DepotAsset
    from vexa_orchestrator.generation_jobs import GenerationJobs
    from vexa_orchestrator.generation_runner import GenerationRunner
    from vexa_orchestrator.generation_settings import conditioning_theme
    from vexa_orchestrator.generation_v2 import plan_track, writer_settings

    validate_items(manifest["items"])
    if set(selected) - {r["id"] for r in manifest["items"]}:
        raise ValueError("unknown selected matrix ID")
    library = output.resolve() / "library"
    library.mkdir(parents=True, exist_ok=True)
    worklist_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    saved = output / "results.json"
    result = (
        json.loads(saved.read_text())
        if saved.exists()
        else {"version": 1, "worklist_sha256": worklist_hash, "items": manifest["items"]}
    )
    if result["worklist_sha256"] != worklist_hash:
        raise ValueError("worklist changed; use a separate run directory")
    cancel = threading.Event()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, lambda *_: cancel.set())
    jobs = GenerationJobs(output / "generation/jobs.sqlite3")
    writer = writer_settings()
    attempted = 0

    def renderer(theme, directory, **kwargs):
        kwargs.pop("cancel", None)
        response = run_job(
            "generate",
            {"theme": theme, "library": str(directory), **kwargs},
            cancel=cancel,
            timeout_s=kwargs["render_timeout_s"] + 900,
        )
        return DepotAsset(
            AssetManifest.model_validate(response["manifest"]),
            Path(response["path"]),
            frozenset(),
            {},
        )

    try:
        for index, item in enumerate(result["items"]):
            if cancel.is_set() or attempted >= limit:
                break
            if selected and item["id"] not in selected:
                continue
            if item.get("state") in {"ready", "rejected", "unavailable"}:
                continue
            controls = GenerationSettings.model_validate(item["controls"])
            theme = conditioning_theme(item["theme"], controls)
            snapshot = {
                "generator": item["generator"],
                "preset": item["preset"],
                "library": str(library),
                "writer": writer,
                "controls": controls.model_dump(),
            }
            batch = item.get("batch_id") or jobs.open_batch(theme, snapshot, 1)
            if jobs.batch(batch)["snapshot"] != snapshot:
                raise ValueError("writer or settings changed on resume; use a new run directory")
            item.update(batch_id=batch, state="running")
            atomic_json(saved, result)
            started = time.monotonic()
            print(f"Rendering {item['id']} (bounded to {limit} cells)", flush=True)
            GenerationRunner(
                jobs,
                batch,
                library,
                planner=plan_track,
                renderer=renderer,
                admitted=lambda *_: None,
                event=lambda event: print(event["event"], flush=True),
                cancel=cancel,
            ).run()
            row = jobs.tracks(batch)[0]
            item.update(
                state=row["state"],
                error=row["error"],
                wall_s=item.get("wall_s", 0) + time.monotonic() - started,
                attempts=jobs.attempts(batch, 0),
            )
            path = Path(row["asset_path"]) if row["asset_path"] else None
            if path and path.is_file():
                item.update(audio_path=str(path), source_sha256=sha256(path))
                evidence = path.with_suffix(".evidence.json")
                if evidence.exists():
                    report = json.loads(evidence.read_text())
                    item.update(
                        report_path=str(evidence),
                        technical=report["admission"],
                        corrections=report["corrections"],
                        observations=report.get("observations", {}),
                    )
                brief = path.with_suffix(".brief.json")
                if brief.exists():
                    metadata = json.loads(brief.read_text())
                    item.update(
                        title=metadata.get("title"),
                        actual_seed=metadata.get("seed"),
                        provenance={
                            key: metadata.get(key)
                            for key in (
                                "runtime_revision",
                                "runtime_patch_sha256",
                                "model_revision",
                            )
                        },
                    )
            result["items"][index] = item
            atomic_json(saved, result)
            attempted += 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result


def score_bundle(directory: Path, labels_path: Path) -> dict:
    from vexa_orchestrator.audio_calibration import evaluate_annotations

    bundle = json.loads((directory / "bundle.json").read_text())
    expected_id = bundle.pop("bundle_id")
    digest = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if digest != expected_id:
        raise ValueError("bundle metadata changed")
    bundle["bundle_id"] = expected_id
    for item in bundle["items"]:
        if item.get("review_audio"):
            path = (directory / item["review_audio"]).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or sha256(path) != item["review_sha256"]
            ):
                raise ValueError("review audio changed or escaped the bundle")
        for path_key, digest_key in (
            ("review_report", "report_sha256"),
            ("review_request", "request_sha256"),
        ):
            if item.get(path_key):
                path = (directory / item[path_key]).resolve()
                if not path.is_relative_to(directory.resolve()) or sha256(path) != item[digest_key]:
                    raise ValueError("observer or generator report changed")
    annotations = json.loads(labels_path.read_text())
    report = scorecard(bundle, annotations)
    voice, used = [], set()
    items = {r["id"]: r for r in bundle["items"]}
    items.update({r["review_id"]: r for r in bundle["items"] if r.get("review_id")})
    for label in annotations.get("reviews", []):
        item = items[label["id"]]
        if label.get("vocals_present") is None or item["kind"] != "track":
            continue
        if item.get("observations", {}).get("status") != "observed":
            continue
        if item["id"] in used:
            raise ValueError("resolve multiple vocal annotations per sample before calibration")
        used.add(item["id"])
        voice.append(
            {
                "evidence_path": str((directory / item["review_report"]).resolve()),
                "split": item["split"],
                "family_id": item["family_id"],
                "vocals_present": label["vocals_present"],
            }
        )
    atomic_json(directory / "vocal-annotations.json", voice)
    report["vocal_calibration"] = (
        evaluate_annotations(voice)
        if voice
        else {
            "status": "insufficient_labels",
            "hard_gate_enabled": False,
            "reason": "No human vocal labels paired with pinned observed evidence.",
        }
    )
    from vexa_orchestrator.audio_calibration import evaluate_genre_annotations

    genre = []
    for label in annotations.get("reviews", []):
        item = items[label["id"]]
        if label.get("heard_genre") and item.get("observations", {}).get("status") == "observed":
            genre.append(
                {
                    "evidence_path": str((directory / item["review_report"]).resolve()),
                    "split": item["split"],
                    "family_id": item["family_id"],
                    "heard_genre": label["heard_genre"],
                }
            )
    report["genre_evaluation"] = evaluate_genre_annotations(genre)
    atomic_json(directory / "scorecard.json", report)
    return report


def add_transitions(manifest: dict, report: dict) -> dict:
    """Append current shared-DSP clips with both parents confined to one split."""
    parents = {r["source_sha256"]: r["split"] for r in manifest["items"] if r.get("source_sha256")}
    for row in report["candidates"]:
        hashes = [row["source_sha256"], row["incoming_sha256"]]
        splits = {parents[d] for d in hashes if d in parents}
        if len(splits) > 1:
            raise ValueError("transition crosses calibration/validation parents")
        path = Path(row["preview"])
        digest = sha256(path)
        manifest["items"].append(
            {
                "id": f"transition-{digest[:16]}",
                "kind": "transition",
                "case_id": report["theme"],
                "theme": report["theme"],
                "title": f"{report['source_title']} to {row['title']}",
                "family_id": f"transition-{report['source']}",
                "split": next(iter(splits), "validation"),
                "parent_sha256": hashes,
                "source_sha256": digest,
                "audio_path": str(path.resolve()),
                "transition": row,
                "technical": {"playable": None, "mixable": None},
            }
        )
    validate_items(manifest["items"])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    matrix = sub.add_parser("matrix", help="write a worklist; no models run")
    matrix.add_argument(
        "--generators", nargs="+", default=["yue2:q8", "acestep:turbo", "minimax:balanced"]
    )
    matrix.add_argument("--out", type=Path, required=True)
    imp = sub.add_parser("import", help="import existing full audio, including failed renders")
    imp.add_argument("reports", type=Path, nargs="+")
    imp.add_argument("--out", type=Path, required=True)
    render = sub.add_parser("render", help="run bounded production jobs in a separate depot")
    render.add_argument("manifest", type=Path)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--limit", type=int, required=True)
    render.add_argument("--ids", nargs="+", default=[])
    bundle = sub.add_parser("bundle", help="write audio copies and a local blind review page")
    bundle.add_argument("manifest", type=Path)
    bundle.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score", help="score exported human labels; never promote a policy")
    score.add_argument("bundle", type=Path)
    score.add_argument("annotations", type=Path)
    transitions = sub.add_parser("transitions", help="append newly rendered shared-DSP clips")
    transitions.add_argument("manifest", type=Path)
    transitions.add_argument("report", type=Path)
    transitions.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "matrix":
        body = corpus([tuple(value.split(":")) for value in args.generators])
        atomic_json(args.out, body)
        print(f"Planned {len(body['items'])} cells; no audio generated.")
    elif args.command == "import":
        body = import_reports(args.reports)
        atomic_json(args.out, body)
        print(f"Imported {len(body['items'])} content-distinct tracks.")
    elif args.command == "render":
        if args.limit < 1:
            parser.error("--limit must be positive")
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        render_matrix(json.loads(args.manifest.read_text()), args.out, args.limit, args.ids)
    elif args.command == "bundle":
        body = build_bundle(
            json.loads(args.manifest.read_text()), args.out, base=args.manifest.resolve().parent
        )
        print(f"Built {len(body['items'])} review cards: {args.out / 'index.html'}")
    elif args.command == "score":
        body = score_bundle(args.bundle, args.annotations)
        print(f"Scored {body['reviews']} reviews; {len(body['pending_items'])} items pending.")
    elif args.command == "transitions":
        atomic_json(
            args.out,
            add_transitions(
                json.loads(args.manifest.read_text()), json.loads(args.report.read_text())
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
