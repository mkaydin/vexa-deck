"""A theme starts a bounded ten-track YuE2 batch with explicit progress."""

import json
from types import SimpleNamespace

import pytest
from vexa_contracts import MusicalClock, SessionState
from vexa_orchestrator.generation import TRACK_COUNT, _gpu_index, plan_theme
from vexa_orchestrator.live import LiveSet
from vexa_yue2.backends import BackendUnavailable


@pytest.fixture(autouse=True)
def remote_planner_for_legacy_tests(monkeypatch):
    monkeypatch.setenv("VEXA_MUSIC_PLANNER", "ollama")


def test_theme_plan_has_ten_distinct_long_track_briefs(monkeypatch):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    briefs = plan_theme("rain-soaked jazz")
    assert len(briefs) == TRACK_COUNT
    assert len({brief.style for brief in briefs}) == TRACK_COUNT
    assert all("rain-soaked jazz" in brief.style for brief in briefs)
    assert all("no vocals" in brief.style.lower() for brief in briefs)
    assert all(165 <= brief.duration_s <= 180 for brief in briefs)


def test_yue2_uses_5060_uuid_across_different_cuda_index_order(monkeypatch):
    monkeypatch.delenv("VEXA_YUE_GPU", raising=False)
    output = (
        "0, NVIDIA GeForce RTX 5060 Ti, GPU-primary\n1, NVIDIA GeForce RTX 4060, GPU-secondary\n"
    )
    monkeypatch.setattr(
        "vexa_orchestrator.generation.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    assert _gpu_index() == "GPU-primary"
    monkeypatch.setenv("VEXA_YUE_GPU", "1")
    assert _gpu_index() == "GPU-secondary"


def test_llm_plan_supplies_distinct_prompts_and_records_origin(monkeypatch):
    monkeypatch.setenv("VEXA_LLM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("VEXA_LLM_MODEL", "test-model")
    styles = [
        f"Warm house track {index}, electric piano, drums, bass and evolving pads."
        for index in range(TRACK_COUNT)
    ]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            import json

            return {"choices": [{"message": {"content": json.dumps(styles)}}]}

    monkeypatch.setattr(
        "vexa_orchestrator.generation.httpx.post", lambda *args, **kwargs: Response()
    )
    briefs = plan_theme("warm house")
    assert [brief.origin for brief in briefs] == ["llm:test-model"] * TRACK_COUNT
    assert all("Instrumental, no vocals" in brief.style for brief in briefs)


def test_local_ollama_stream_releases_model_after_prompt_call(monkeypatch):
    from vexa_orchestrator.generation import _ollama_prompts

    request = {}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"message": {"content": '{"tracks": ["A jazz track"]}'}, "done": True})

    def fake_stream(method, url, **kwargs):
        request.update(kwargs["json"])
        assert method == "POST" and url.endswith("/api/chat")
        return Stream()

    monkeypatch.setattr("vexa_orchestrator.generation.httpx.stream", fake_stream)
    raw = _ollama_prompts(
        "http://127.0.0.1:11434", "gemma4:e4b", [{"role": "user", "content": "jazz"}], 1, None
    )
    assert json.loads(raw)["tracks"] == ["A jazz track"]
    assert request["keep_alive"] == 0
    assert request["format"]["properties"]["tracks"]["maxItems"] == 1


def test_live_batch_admits_ten_tracks_and_reports_progress(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    submitted = []
    admitted = []
    depot = SimpleNamespace(
        library=tmp_path, add=lambda manifest, path: admitted.append(manifest.asset_id)
    )
    live = LiveSet(depot, preview_dir=tmp_path / "previews")
    live.state = SessionState(
        session_id="test",
        theme="dark techno",
        clock=MusicalClock(bpm=128),
        fallback_asset_id="existing",
    )

    def fake_generate(
        theme, library, *, duration_s, energy, style, prompt_origin, cancel, lyrics="", title=""
    ):
        submitted.append((theme, duration_s, energy, style))
        assert prompt_origin == "local"
        return SimpleNamespace(
            manifest=SimpleNamespace(asset_id=f"test-{len(submitted)}"),
            path=library / f"test-{len(submitted)}.wav",
        )

    monkeypatch.setattr("vexa_orchestrator.live.generate_theme", fake_generate)
    assert live._queue_generation("dark techno", "test")
    live._generation_thread.join(timeout=3)
    assert not live._generation_thread.is_alive()
    assert len(submitted) == len(admitted) == TRACK_COUNT
    assert live.status()["generation"]["completed"] == TRACK_COUNT
    assert live.status()["generating"] is False
    live.stop()


def test_failed_render_is_retried_until_ten_tracks_are_admitted(monkeypatch, tmp_path):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    attempts = []
    admitted = []
    depot = SimpleNamespace(
        library=tmp_path, add=lambda manifest, _path: admitted.append(manifest.asset_id)
    )
    live = LiveSet(depot, preview_dir=tmp_path / "previews")
    live.state = SessionState(
        session_id="test",
        theme="warm house",
        clock=MusicalClock(bpm=122),
        fallback_asset_id="existing",
    )

    def fake_generate(theme, library, **_kwargs):
        attempts.append(theme)
        if len(attempts) == 1:
            raise ValueError("early YuE2 end token")
        return SimpleNamespace(
            manifest=SimpleNamespace(asset_id=f"test-{len(attempts)}"),
            path=library / f"test-{len(attempts)}.wav",
        )

    monkeypatch.setattr("vexa_orchestrator.live.generate_theme", fake_generate)
    assert live._queue_generation("warm house", "test")
    live._generation_thread.join(timeout=3)
    assert not live._generation_thread.is_alive()
    assert len(attempts) == TRACK_COUNT + 1
    assert len(admitted) == TRACK_COUNT
    assert live.status()["generation"]["failed"] == 1
    live.stop()


def test_full_length_live_set_skips_short_legacy_depot_tracks(monkeypatch, tmp_path):
    from vexa_orchestrator.depot import Depot

    depot = Depot(tmp_path)
    short = SimpleNamespace(manifest=SimpleNamespace(audio=SimpleNamespace(duration_s=45.0)))
    depot.search = lambda *args, **kwargs: [(1.0, short)]
    live = LiveSet(depot, min_track_duration_s=148)
    queued = []
    monkeypatch.setattr(
        live,
        "_queue_generation",
        lambda theme, session_id: queued.append((theme, session_id)) or True,
    )
    monkeypatch.setattr(
        live, "_load", lambda asset: (_ for _ in ()).throw(AssertionError("short asset loaded"))
    )

    result = live.start("dark techno", session_id="test")

    assert not result["running"]
    assert result["generating"]
    assert queued == [("dark techno", "test")]


@pytest.mark.parametrize(
    "error, expected_attempts",
    [
        (BackendUnavailable("libggml.so.0: cannot open shared object file"), 1),
        (ValueError("generated audio failed quality gates: clipping"), TRACK_COUNT * 3),
    ],
)
def test_batch_preserves_failure_and_skips_permanent_setup_retries(
    monkeypatch, tmp_path, error, expected_attempts
):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    live = LiveSet(SimpleNamespace(library=tmp_path), preview_dir=tmp_path / "previews")
    attempts = []

    def fail_render(*args, **kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr("vexa_orchestrator.live.generate_theme", fail_render)
    try:
        assert live._queue_generation("warm house", "test")
        live._generation_thread.join(timeout=3)
        assert not live._generation_thread.is_alive()
        assert len(attempts) == expected_attempts
        assert str(error) in live.error
        assert live.status()["generating"] is False
        assert live.history[-1]["event"] == "generation_batch_failed"
    finally:
        live.stop()


def test_default_prompt_model_is_requested_nemotron_cloud(monkeypatch):
    from vexa_orchestrator.generation import _llm_connection

    for name in ("VEXA_LLM_MODEL", "VEXA_LLM_BASE_URL", "VEXA_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    base, model, _, provider = _llm_connection()
    assert (base, model, provider) == ("http://127.0.0.1:11434", "nemotron-3-super:cloud", "ollama")


def test_cloud_request_does_not_send_unsupported_schema(monkeypatch):
    from vexa_orchestrator.generation import _ollama_prompts

    request = {}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"message": {"content": '{"tracks":[]}'}, "done": True})

    def stream(*args, **kwargs):
        request.update(kwargs["json"])
        return Stream()

    monkeypatch.setattr("vexa_orchestrator.generation.httpx.stream", stream)
    _ollama_prompts("http://localhost:11434", "nemotron-3-super:cloud", [], 10, None)
    assert "format" not in request
    assert request["model"] == "nemotron-3-super:cloud"


def test_pop_drift_is_repaired_and_original_rap_lyrics_are_preserved(monkeypatch):
    monkeypatch.delenv("VEXA_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("VEXA_LLM_MODEL", raising=False)
    monkeypatch.delenv("VEXA_LLM_BASE_URL", raising=False)
    calls = []
    lyrics = "[Verse 1]\n" + "Original rhythmic verse with a late night city image.\n" * 40

    def prompts(base, model, messages, count, cancel):
        calls.append(list(messages))
        style = (
            "Glossy pop anthem with melodic singing and a bright synth chorus."
            if len(calls) == 1
            else "1990s boom bap hip hop, swung breakbeats, dusty jazz samples, dry rap vocals."
        )
        return json.dumps({"tracks": [{"style": style, "lyrics": lyrics}]})

    monkeypatch.setattr("vexa_orchestrator.generation._ollama_prompts", prompts)
    brief = plan_theme("90s underground rap", count=1)[0]
    assert len(calls) == 2
    assert "failed validation" in calls[1][-1]["content"]
    assert brief.lyrics == lyrics.strip()
    assert "no vocals" not in brief.style.lower()
    assert brief.style.startswith("1990s underground boom bap hip hop")


def test_cloud_failure_is_visible_instead_of_silent_house_fallback(monkeypatch):
    monkeypatch.delenv("VEXA_LLM_PROVIDER", raising=False)

    def unavailable(*args, **kwargs):
        raise RuntimeError("Ollama cloud sign-in required")

    monkeypatch.setattr("vexa_orchestrator.generation._ollama_prompts", unavailable)
    with pytest.raises(RuntimeError, match="sign-in required"):
        plan_theme("90s underground rap")


def test_instrumental_rap_keeps_genre_without_inventing_vocals(monkeypatch):
    monkeypatch.setenv("VEXA_LLM_PROVIDER", "none")
    briefs = plan_theme("90s underground rap instrumental")
    assert all("boom bap hip hop" in item.style for item in briefs)
    assert all("No vocals" in item.style and item.lyrics == "" for item in briefs)


def test_rap_plan_rejects_lyrics_too_short_for_full_length():
    from vexa_orchestrator.generation import _parse_track_plan

    plan = {
        "tracks": [
            {
                "style": "1990s underground boom bap rap, dusty jazz chops and dry drums.",
                "lyrics": "[Verse 1]\n" + "A short original rhythmic rap line.\n" * 16,
            }
        ]
    }
    with pytest.raises(ValueError, match="40 lyric lines"):
        _parse_track_plan(json.dumps(plan), "90s underground rap", 1, "nemotron-3-super:cloud")
