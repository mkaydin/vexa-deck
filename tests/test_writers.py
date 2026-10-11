"""Writer capability checks, immutable batch selections and comparable telemetry."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vexa_orchestrator.api import create_app
from vexa_orchestrator.generation import _ollama_prompts
from vexa_orchestrator.live import LiveSet
from vexa_orchestrator.writer_eval import scorecard, text_evidence
from vexa_orchestrator.writers import WriterProfile, installed_local, preflight
from vexa_yue2.backends import BackendUnavailable


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "nemotron-3-super:cloud"},
        {"provider": "openai"},
        {"context_length": 100000},
        {"creativity": float("nan")},
        {"base": "http://user:secret@localhost"},
        {"mode": "auto"},
    ],
)
def test_invalid_writer_profiles_fail_before_a_batch(changes):
    with pytest.raises(ValidationError):
        WriterProfile.model_validate({"mode": "local", "model": "fixture", **changes})


def test_private_manifest_identity_is_pinned(monkeypatch, tmp_path):
    monkeypatch.setattr("vexa_orchestrator.writers.ROOT", tmp_path)
    path = tmp_path / "models/ollama-writer/manifests/registry.ollama.ai/library/qwen3.5/9b"
    path.parent.mkdir(parents=True)
    path.write_text('{"layers":[{"size":123}]}')
    assert installed_local("qwen3.5:9b") == {
        "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": 123,
    }
    assert installed_local("other:tag") is None


def test_preflight_rejects_changed_digest_and_reports_cloud_capabilities(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    monkeypatch.setattr(
        "vexa_orchestrator.writers.httpx.get",
        lambda *a, **kw: Response(
            {"models": [{"name": "test:cloud", "digest": "actual", "remote_host": "cloud"}]}
        ),
    )
    monkeypatch.setattr(
        "vexa_orchestrator.writers.httpx.post",
        lambda *a, **kw: Response({"capabilities": ["completion", "thinking"]}),
    )
    settings = {"model": "test:cloud", "provider": "ollama", "digest": "old"}
    with pytest.raises(BackendUnavailable, match="digest changed"):
        preflight(settings, "http://fixture")
    settings["digest"] = "actual"
    identity = preflight(settings, "http://fixture")
    assert identity["schema"] == "json_text" and identity["thinking_control"]


def test_stream_records_tokens_and_obeys_bounded_local_options(monkeypatch):
    captured = {}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"message": {"content": "{}"}, "done": False})
            yield json.dumps({"done": True, "prompt_eval_count": 42, "eval_count": 7})

    def stream(*args, **kw):
        captured.update(kw["json"])
        return Stream()

    monkeypatch.setattr("vexa_orchestrator.generation.httpx.stream", stream)
    metrics = {}
    assert (
        _ollama_prompts(
            "http://fixture",
            "qwen:9b",
            [],
            1,
            None,
            structured=True,
            schema={"type": "object"},
            options={"num_ctx": 8192, "num_predict": 4000},
            metrics=metrics,
            thinking=None,
            keep_alive=60,
        )
        == "{}"
    )
    assert "think" not in captured
    assert captured["options"]["num_ctx"] == 8192 and captured["keep_alive"] == 60
    assert metrics["prompt_eval_count"] == 42 and metrics["eval_count"] == 7
    assert metrics["ttft_s"] is not None


def test_writer_change_keeps_active_snapshot_and_decks(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    depot = SimpleNamespace(library=tmp_path, assets={})
    live = LiveSet(depot, preview_dir=tmp_path / "previews")
    old = live.writer_profile()
    batch = live._jobs.open_batch(
        "techno", {"writer": old, "generator": "yue2", "preset": "q8", "library": str(tmp_path)}, 2
    )
    marker = live.engine.mixer.decks[0]
    live._generating = True
    live.configure_writer(WriterProfile(mode="remote", model="fixture", base="http://fixture"))
    assert live._jobs.batch(batch)["snapshot"]["writer"] == old
    assert live.writer_profile()["model"] == "fixture"
    assert live.engine.mixer.decks[0] is marker and not live._render_cancel.is_set()
    restarted = LiveSet(depot, preview_dir=tmp_path / "previews")
    assert restarted.writer_profile()["model"] == "fixture"
    live._generating = False
    live.stop()
    restarted.stop()


def test_api_writer_validation_and_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    monkeypatch.setattr("vexa_orchestrator.live.installed_local", lambda model: None)
    live = LiveSet(SimpleNamespace(library=tmp_path, assets={}), preview_dir=tmp_path / "previews")
    with TestClient(create_app(live=live)) as client:
        assert client.get("/generation/writers").status_code == 200
        assert (
            client.post(
                "/generation/writer", json={"mode": "local", "model": "missing"}
            ).status_code
            == 503
        )
        response = client.post(
            "/generation/writer", json={"mode": "template", "model": "none", "provider": "none"}
        )
        assert response.status_code == 200
        assert response.json()["selected"]["context_length"] == 8192


def test_scorecard_percentiles_include_slow_tail_and_semantics_exclude_backend_text():
    rows = [
        {
            "accepted": True,
            "wall_s": float(i),
            "attempts": [{"accepted": True}],
            "device_peak_mib": 7400,
            "brief": {"title": str(i)},
            "evidence": {"semantic_proxy_coverage": 1, "section_trigram_overlap": 0},
        }
        for i in range(1, 7)
    ]
    assert scorecard(rows)["p95_wall_s"] == 6
    brief = SimpleNamespace(
        music_plan={
            "creative_draft": {
                "production_notes": "Generic music",
                "instruments": ["synth"],
                "section_descriptions": {"s01": "More music"},
            },
            "track_plan": {"intent": {"language": "unknown"}},
        },
        lyrics="",
        style="cowbell 808 bass",
    )
    assert text_evidence(brief, (r"cowbell", r"808"))["semantic_proxy_coverage"] == 0
