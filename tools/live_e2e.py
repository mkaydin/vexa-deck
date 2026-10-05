"""Exercise theme → API → real audio device → two-deck transitions.

This test makes sound only when --audible is supplied. It leaves the depot untouched and stops
the audio stream even if an assertion fails.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from vexa_audio.engine import AudioEngine, EngineConfig
from vexa_orchestrator.api import create_app
from vexa_orchestrator.depot import Depot
from vexa_orchestrator.live import LiveSet

ROOT = Path(__file__).resolve().parent.parent


def run(theme: str, *, transitions: int, timeout_s: float, audible: bool,
        report_path: Path) -> dict[str, object]:
    if not audible:
        raise ValueError("pass --audible to confirm that this test will play through the speakers")
    engine = AudioEngine(EngineConfig())
    live = LiveSet(Depot(ROOT / "assets/library"), engine)
    app = create_app(live=live)
    started_at = datetime.now(UTC)
    observed: set[tuple[str, str | None]] = set()
    records: list[dict[str, object]] = []
    started = time.monotonic()
    try:
        with TestClient(app) as client:
            response = client.post("/sessions", json={"theme": theme})
            response.raise_for_status()
            session = response.json()
            if not session["can_start_now"]:
                raise RuntimeError(f"depot did not start immediately: {session}")
            print(f"playing {theme!r} as {session['session_id']}", flush=True)
            while time.monotonic() - started < timeout_s:
                response = client.get("/live")
                response.raise_for_status()
                status = response.json()
                if status["error"]:
                    raise RuntimeError(status["error"])
                for event in status["events"]:
                    if event["event"] not in {"start", "prepared", "transition"}:
                        continue
                    identity = (event["event"], event.get("asset_id"))
                    if identity in observed:
                        continue
                    observed.add(identity)
                    row = {**event, "elapsed_s": round(time.monotonic() - started, 2)}
                    records.append(row)
                    print(json.dumps(row), flush=True)
                completed = [row for row in records if row["event"] == "transition"]
                if len(completed) >= transitions:
                    active_name = "deck_a" if status["active_deck"] == 0 else "deck_b"
                    inactive_name = "deck_b" if status["active_deck"] == 0 else "deck_a"
                    active = status["audio"][active_name]
                    inactive = status["audio"][inactive_name]
                    if (active["playing"] and active["gain"] >= 0.99
                            and not active["fading"] and inactive["gain"] <= 0.01
                            and not inactive["fading"]):
                        break
                time.sleep(0.25)
            else:
                raise TimeoutError(
                    f"only {len(completed)} of {transitions} crossfades completed"
                )

            audio = status["audio"]
            stats = audio["stats"]
            if not status["running"] or stats["callbacks"] < 10:
                raise AssertionError("output device did not run callbacks")
            if stats["underruns"] or stats["queue_dropped"]:
                raise AssertionError(f"audio health failed: {stats}")
            if stats["silence_blocks"] > max(100, stats["callbacks"] * 0.02):
                raise AssertionError(f"too many silent output blocks: {stats}")
            if len(completed) >= 2 and status["active_deck"] != 0:
                raise AssertionError("decks did not alternate A → B → A")
            report = {
                "started_at": started_at.isoformat(), "theme": theme,
                "session_id": session["session_id"],
                "elapsed_s": round(time.monotonic() - started, 2),
                "transitions": len(completed), "events": records,
                "audio": audio, "current_asset_id": status["current_asset_id"],
                "device": asdict(engine.status()),
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            return report
    finally:
        live.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default="dark melodic techno")
    parser.add_argument("--transitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--audible", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "var/reports/live-e2e.json")
    args = parser.parse_args()
    report = run(args.theme, transitions=args.transitions, timeout_s=args.timeout,
                 audible=args.audible, report_path=args.report)
    print(json.dumps({"result": "PASS", "transitions": report["transitions"],
                      "elapsed_s": report["elapsed_s"], "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
