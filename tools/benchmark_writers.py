"""Compare pinned writers on 10 themes x 3 variations, with resumable wave reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
from backend_runtime import workspace_pythonpath  # noqa: E402

for source in workspace_pythonpath(ROOT).split(":"):
    sys.path.insert(0, source)


def main():
    from dotenv import load_dotenv
    from vexa_orchestrator.background import run_job
    from vexa_orchestrator.writer_eval import CORPUS, CORPUS_HASH, scorecard
    from vexa_orchestrator.writers import pin_local

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b", "qwen3.5:9b"])
    parser.add_argument("--include-cloud", action="store_true")
    parser.add_argument("--variations", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--cases", nargs="*", choices=[row[0] for row in CORPUS])
    parser.add_argument("--output", type=Path, default=ROOT / "var/reports/writer-benchmark")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    models = [*args.models]
    if args.include_cloud and "nemotron-3-super:cloud" not in models:
        models.append("nemotron-3-super:cloud")
    summaries = {}
    for model in models:
        profile = pin_local(
            {
                "mode": "remote" if model.endswith(":cloud") else "local",
                "model": model,
                "base": "http://127.0.0.1:11434" if model.endswith(":cloud") else "",
            }
        )
        directory = args.output / model.replace(":", "-").replace("/", "_")
        directory.mkdir(parents=True, exist_ok=True)
        rows, pending = [], []
        for case, (key, _theme, _groups) in enumerate(CORPUS):
            if args.cases and key not in args.cases:
                continue
            for variation in range(args.variations):
                path = directory / f"{key}-{variation}.json"
                previous = json.loads(path.read_text()) if path.is_file() else None
                if (
                    previous
                    and previous.get("profile") == profile
                    and previous.get("corpus_hash") == CORPUS_HASH
                ):
                    rows.append(previous)
                else:
                    pending.append((case, variation))
        while pending:
            wave, pending = pending[:3], pending[3:]
            result = run_job(
                "benchmark_writer",
                {"profile": profile, "cases": wave, "output": str(directory.resolve())},
                timeout_s=1800,
            )
            rows.extend(result["rows"])
            summaries[model] = scorecard(rows)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "scorecard.json").write_text(
                json.dumps(
                    {
                        "corpus_hash": CORPUS_HASH,
                        "models": summaries,
                        "audio_generated": False,
                        "semantic_scores_are_text_proxies": True,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps({"model": model, **summaries[model]}, ensure_ascii=False), flush=True)
            if any(r.get("unavailable") for r in result["rows"]):
                break
        summaries[model] = scorecard(rows)
    (args.output / "scorecard.json").write_text(
        json.dumps(
            {
                "corpus_hash": CORPUS_HASH,
                "models": summaries,
                "audio_generated": False,
                "semantic_scores_are_text_proxies": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
