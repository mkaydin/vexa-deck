"""Cue reports and previews must reflect source audio and real time positions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from vexa_audio.cues import CueMap, analyse_cues
from vexa_audio.loader import PreloadedTrack
from vexa_audio.preview import PreviewRenderer, PreviewSpec
from vexa_orchestrator.transitions import score_rendered_transition


def test_cues_are_hash_bound_and_silence_has_no_cue(tmp_path):
    sr = 22050
    path = tmp_path / "click.wav"
    samples = np.zeros(sr * 32, dtype=np.float32)
    for beat, frame in enumerate(range(0, len(samples), sr // 2)):
        samples[frame : frame + 150] = 0.65 if beat % 4 == 0 else 0.25
    sf.write(path, np.stack([samples, samples], axis=1), sr)
    cues = analyse_cues(path, expected_bpm=120)
    assert cues.beat_confidence >= 0.6
    assert len(cues.beat_times_s) > 30
    assert cues.entry_s is not None
    assert cues.exit_s is not None
    report = tmp_path / "click.cues.json"
    cues.save(report)
    assert CueMap.load(report).valid_for(path)
    assert replace(cues, version=2).valid_for(path)  # Previously admitted cache remains usable.
    assert not replace(cues, version=1).valid_for(path)

    sf.write(path, np.zeros((sr * 6, 2), dtype=np.float32), sr)
    assert not CueMap.load(report).valid_for(path)
    assert analyse_cues(path).entry_s is None


@pytest.mark.parametrize("beat_end_s, usable", [(100, True), (60, False)])
def test_long_outro_needs_a_bounded_measured_overlap(tmp_path, beat_end_s, usable):
    sr = 22050
    path = tmp_path / "outro.wav"
    samples = np.zeros(sr * 120, dtype=np.float32)
    for beat, frame in enumerate(range(0, sr * beat_end_s, sr // 2)):
        samples[frame : frame + 150] = 0.65 if beat % 4 == 0 else 0.25
    sf.write(path, np.stack([samples, samples], axis=1), sr)
    cue = analyse_cues(path, expected_bpm=120)
    assert cue.beat_confidence >= 0.6
    assert (cue.exit_s is not None) == usable
    if usable:
        assert 90 <= cue.exit_s < 100 - 6
        assert cue.entry_s is not None
        assert cue.beat_times_s[-1] >= cue.exit_s + 6
    assert sf.info(path).duration == 120
    assert cue.valid_for(path)


def test_tracker_timestamps_without_onsets_cannot_support_an_exit(monkeypatch, tmp_path):
    import librosa

    sr = 22050
    path = tmp_path / "bridged-outro.wav"
    samples = np.zeros(sr * 120, dtype=np.float32)
    # Real pulses stop halfway through; a tracker must not invent a usable outro.
    for frame in range(0, sr * 60, sr // 2):
        samples[frame : frame + 150] = 0.5
    sf.write(path, np.stack([samples, samples], axis=1), sr)
    frames = librosa.time_to_frames(np.arange(0.5, 119, 0.5), sr=sr, hop_length=512)
    monkeypatch.setattr(librosa.beat, "beat_track", lambda **kwargs: (120.0, frames))
    cue = analyse_cues(path, expected_bpm=120)
    assert cue.beat_confidence >= 0.6
    assert cue.exit_s is None


def test_transition_preview_places_source_at_zero_and_fades_each_deck(tmp_path):
    sr = 8000
    a = np.full((sr * 8, 2), 0.2, dtype=np.float32)
    b = np.full((sr * 8, 2), -0.2, dtype=np.float32)
    outgoing = PreloadedTrack("a" * 64, a, sr, 2, samples_per_bar=sr)
    incoming = PreloadedTrack("b" * 64, b, sr, 2, samples_per_bar=sr)
    renderer = PreviewRenderer(sample_rate=sr, out_dir=tmp_path)
    preview = renderer.render_transition(
        PreviewSpec("pair", lead_in_s=1, fade_s=1, tail_s=1),
        outgoing,
        incoming,
        out_name="pair",
        entry_bar=2,
        incoming_entry_bar=1,
    )
    audio, _ = sf.read(preview.path, always_2d=True)
    assert audio[0, 0] == pytest.approx(0.2, abs=0.001)
    assert audio[int(1.5 * sr), 0] == pytest.approx(0.0, abs=0.002)
    assert audio[-1, 0] == pytest.approx(-0.2, abs=0.001)


def test_rendered_score_uses_measured_cues_and_audio(tmp_path):
    sr = 8000
    t = np.arange(sr * 12) / sr
    wave = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    quiet = (0.025 * np.sin(2 * np.pi * 1800 * t)).astype(np.float32)
    a = PreloadedTrack("a" * 64, np.stack([wave, wave], axis=1), sr, 2)
    b = PreloadedTrack("b" * 64, np.stack([wave, wave], axis=1), sr, 2)
    c = PreloadedTrack("c" * 64, np.stack([quiet, quiet], axis=1), sr, 2)
    cue = CueMap("a" * 64, 2, 120.0, 1.0, (), 0.0, (), 1.0, 7.0, "beat")
    renderer = PreviewRenderer(sample_rate=sr, out_dir=tmp_path)
    same = score_rendered_transition(renderer, a, b, cue, cue, name="same", lead_s=2.0, fade_s=2.0)
    changed = score_rendered_transition(
        renderer, a, c, cue, cue, name="changed", lead_s=2.0, fade_s=2.0
    )
    assert same.preview_path.exists()
    assert same.score > changed.score
    assert changed.rms_jump_db > same.rms_jump_db
