import numpy as np
import pytest
from test_continuity import ready_for_restart
from vexa_audio.queue import Command, CommandKind


def advance(live, blocks=40):
    outputs = []
    for _ in range(blocks):
        block = np.zeros((256, 2), dtype="float32")
        live.engine.mixer.render(block, len(block))
        outputs.append(block)
        live.tick()
    return np.concatenate(outputs)


def setup(tmp_path, monkeypatch, candidate=True):
    live, asset = ready_for_restart(tmp_path)
    live._waiting = False
    live.engine.mixer.decks[0].playing = True
    live.engine.mixer.decks[0].position = 0
    monkeypatch.setattr("vexa_orchestrator.live.publish_audio_priority", lambda reason: None)
    live.depot.search = lambda *a, **kw: [(1, asset)] if candidate else []
    # Keep ordinary subsequent transition preparation out of this isolated handoff test.
    monkeypatch.setattr(live, "_prepare", lambda *a: None)
    return live, asset


@pytest.mark.parametrize("active", [0, 1])
def test_skip_fades_then_starts_at_zero_generation_unchanged(tmp_path, monkeypatch, active):
    live, asset = setup(tmp_path, monkeypatch)
    if active:
        live.engine.mixer.decks[1].track = live.current_track
        live.engine.mixer.decks[1].playing = True
        live.engine.mixer.decks[0].playing = False
        live.active_deck = 1
    result = live.skip_track()
    assert result["continuity"]["skipping"]
    live.skip_track()
    assert len(live.engine.queue) == 1  # repeated request advances once
    output = advance(live, blocks=18)
    assert live.current is asset and live.active_deck == 1 - active
    assert live._generating and not live._render_cancel.is_set()
    assert not live._skip_pending and not live._waiting
    event = next(e for e in live.history if e["event"] == "skip_completed")
    assert event["play_start_s"] == 0
    # Existing constant PCM lets us distinguish a fade from an abrupt hard cut.
    assert np.max(np.abs(np.diff(output[:, 0]))) < 0.003
    assert live.engine.mixer.decks[1 - active].position < len(output)
    assert not live.engine.mixer.decks[active].playing
    assert asset.manifest.content_sha256 in live._played_hashes


def test_skip_empty_waits_then_ready_asset_resumes(tmp_path, monkeypatch):
    live, asset = setup(tmp_path, monkeypatch, candidate=False)
    live.skip_track()
    output = advance(live, blocks=18)
    assert live._skip_waiting and live._waiting and not np.any(output[-256:])
    assert live._generating and not live._render_cancel.is_set()
    live.depot.search = lambda *a, **kw: [(1, asset)]
    live._next_skip_check = 0
    live.tick()
    assert not live._skip_waiting and live.current is asset
    live.engine.mixer.render(np.zeros((256, 2), dtype="float32"), 256)
    assert live.engine.mixer.decks[live.active_deck].position == 256


def test_stop_wins_over_queued_skip(tmp_path, monkeypatch):
    live, _ = setup(tmp_path, monkeypatch)
    live.skip_track()
    live.stop_playback()
    output = advance(live)
    assert not np.any(output) and live._playback_stopped
    assert live._skip_pending is None and live.engine.mixer._skip_pending is None
    assert live._generating and not live._render_cancel.is_set()


def test_skip_queue_full_and_fade_rejected(tmp_path, monkeypatch):
    live, _ = setup(tmp_path, monkeypatch)
    live.engine.mixer.decks[0].fade_remaining = 10
    with pytest.raises(RuntimeError, match="fade"):
        live.skip_track()
    live.engine.mixer.decks[0].fade_remaining = 0
    while not live.engine.queue.full:
        live.engine.queue.put(Command(CommandKind.PAUSE))
    with pytest.raises(RuntimeError, match="queue is full"):
        live.skip_track()
    assert live._skip_pending is None


def test_skip_changed_source_never_plays(tmp_path, monkeypatch):
    live, asset = setup(tmp_path, monkeypatch)
    with asset.path.open("ab") as f:
        f.write(b"changed")
    live.skip_track()
    advance(live, blocks=18)
    assert live.current is not asset and live._skip_waiting
