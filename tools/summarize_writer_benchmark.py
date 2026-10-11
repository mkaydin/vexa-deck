"""Build a model scorecard from saved, versioned requests without calling any model."""

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
    from vexa_orchestrator.writer_eval import scorecard

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cards, hashes = {}, set()
    for folder in sorted(args.directory.iterdir()):
        if not folder.is_dir():
            continue
        rows = []
        for path in sorted(folder.glob("*-?.json")):
            row = json.loads(path.read_text())
            if "accepted" in row and "profile" in row:
                rows.append(row)
        if not rows:
            continue
        model = rows[0]["profile"]["model"]
        card = scorecard(rows)
        hashes.update(r["corpus_hash"] for r in rows)
        requests = [
            request
            for row in rows
            for attempt in row["attempts"]
            for request in attempt.get("requests", [])
        ]
        card.update(
            profile=rows[0]["profile"],
            provider_calls=len(requests),
            input_tokens_recorded=sum(
                r.get("prompt_eval_count") or r.get("prompt_tokens") or 0 for r in requests
            ),
            output_tokens_recorded=sum(
                r.get("eval_count") or r.get("completion_tokens") or 0 for r in requests
            ),
            per_theme={
                case: {
                    "accepted": sum(r["accepted"] for r in rows if r["case"] == case),
                    "requests": sum(r["case"] == case for r in rows),
                }
                for case in sorted({r["case"] for r in rows})
            },
        )
        cards[model] = card
    if len(hashes) != 1:
        raise ValueError("scorecard must compare one consistent corpus/prompt/filter version")
    report = {
        "corpus_hash": next(iter(hashes)),
        "models": cards,
        "audio_generated": False,
        "semantic_scores_are_text_proxies": True,
        "gpu_memory_scope": "total local RTX 5060 Ti device usage sampled every 0.5 seconds; "
        "includes desktop; cloud model memory is not measured",
    }
    (args.directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
