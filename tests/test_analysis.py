"""Analysis and gate behaviour.

The loudness measurements are checked against the BS.1770-4 reference signals rather than against
whatever the implementation happens to produce — a loudness meter that is self-consistently wrong
is worse than no meter, because the gate would admit material that clips in the mix.
"""

from __future__ import annotations

import numpy as np
import pytest
from vexa_contracts import ApprovalState, SourceType
from vexa_yue2.analysis import (
    REFERENCE_RATE,
    measure_loudness,
    measure_true_peak,
    probe_audio,
)
from vexa_yue2.gates import GateThresholds, analyse

SR = REFERENCE_RATE


def sine(freq: float, seconds: float, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def write_wav(path, samples: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf

    sf.write(str(path), samples, sr, subtype="PCM_16")


def click_track(seconds: float, bpm: float = 120.0, sr: int = SR, bed_dbfs: float = -22.0):
    """A click track over a quiet bed.

    A bare sine has no onset structure, so no beat tracker should be confident about it. Any
    test that needs a trustworthy grid has to supply real percussive onsets.
    """
    bed = sine_rms_dbfs(bed_dbfs, freq=200.0, seconds=seconds, sr=sr)
    click = np.zeros_like(bed)
    period = int(sr * 60.0 / bpm)
    for index in range(0, len(click), period):
        click[index : index + 400] += 0.6
    return np.clip(bed + click, -1.0, 1.0)

def sine_rms_dbfs(dbfs: float, freq: float = 1000.0, seconds: float = 10.0, sr: int = SR):
    """A sine whose **RMS** is ``dbfs`` dBFS.

    BS.1770 calibration is specified on RMS, not peak: a sine at -23 dBFS RMS must read
    -23.0 LUFS. Using peak amplitude here would be off by the 3.01 dB crest factor.
    """
    peak = 10 ** (dbfs / 20.0) * np.sqrt(2.0)
    return sine(freq, seconds, amplitude=peak, sr=sr)


def test_a_1khz_tone_at_minus_23dbfs_rms_measures_minus_23_lufs():
    """The BS.1770-4 calibration signal. Without this the meter is just self-consistent."""
    measured = measure_loudness(sine_rms_dbfs(-23.0), SR)
    assert measured.integrated_lufs == pytest.approx(-23.0, abs=0.15)


def test_a_1khz_tone_at_minus_20dbfs_rms_measures_minus_20_lufs():
    measured = measure_loudness(sine_rms_dbfs(-20.0), SR)
    assert measured.integrated_lufs == pytest.approx(-20.0, abs=0.15)


def test_loudness_scales_linearly_with_amplitude():
    """A 6 dB amplitude increase is a 6 LU increase."""
    quiet = measure_loudness(sine_rms_dbfs(-26.0, seconds=6.0), SR).integrated_lufs
    loud = measure_loudness(sine_rms_dbfs(-20.0, seconds=6.0), SR).integrated_lufs
    assert loud - quiet == pytest.approx(6.0, abs=0.2)


def test_a_silent_signal_does_not_produce_a_plausible_reading():
    measured = measure_loudness(np.zeros(SR * 5), SR)
    assert measured.integrated_lufs < -60.0


def test_quiet_noise_is_gated_out_rather_than_reported():
    """The absolute gate at -70 LUFS exists so a digital-silence tail cannot drag the average."""
    loud = sine_rms_dbfs(-16.0, seconds=8.0)
    padded = np.concatenate([loud, np.zeros(SR * 8)])
    measured = measure_loudness(padded, SR)
    assert measured.integrated_lufs == pytest.approx(-16.0, abs=0.5)


def test_stereo_content_is_measured_from_the_channel_mean():
    """BS.1770 sums channel powers; identical channels must not read 3 LU hot."""
    left = sine(1000.0, 8.0, amplitude=10 ** (-23 / 20))
    both = np.stack([left, left], axis=1)
    mono = measure_loudness(left, SR).integrated_lufs
    stereo = measure_loudness(both, SR).integrated_lufs
    assert stereo == pytest.approx(mono, abs=0.2)


# --- True peak --------------------------------------------------------------


def test_a_full_scale_sine_reads_about_zero_dbfs_tp():
    measured = measure_true_peak(sine(1000.0, 2.0, amplitude=1.0), SR)
    assert measured == pytest.approx(0.0, abs=0.2)


def test_quiet_material_reads_the_expected_peak():
    measured = measure_true_peak(sine(1000.0, 2.0, amplitude=0.1), SR)
    assert measured == pytest.approx(-20.0, abs=0.2)


# --- Decode -----------------------------------------------------------------


def test_probe_reports_container_facts(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, np.stack([sine(440.0, 3.0)] * 2, axis=1))
    probe, _ = probe_audio(path)
    assert probe.sample_rate_hz == SR
    assert probe.channels == 2
    assert probe.duration_s == pytest.approx(3.0, abs=0.01)


def test_probe_detects_clipping(tmp_path):
    path = tmp_path / "hot.wav"
    write_wav(path, sine(440.0, 2.0, amplitude=1.5))
    probe, _ = probe_audio(path)
    assert probe.clipped_sample_ratio > 0.0


def test_probe_detects_silence(tmp_path):
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(SR * 2))
    probe, _ = probe_audio(path)
    assert probe.rms < 1e-4


def test_undecodable_file_is_an_error_not_a_crash(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"this is not audio")
    from vexa_yue2.analysis import AnalysisError

    with pytest.raises(AnalysisError):
        probe_audio(path)


# --- Gates ------------------------------------------------------------------


def test_a_healthy_asset_passes_every_gate(tmp_path):
    path = tmp_path / "good.wav"
    # A click track at 120 BPM over a quiet bed. This gives a genuinely detectable tempo, which a
    # bare sine does not.
    seconds = 40  # 20 bars at 120 BPM: long enough for a verified 8-bar loop
    bed = sine_rms_dbfs(-22.0, freq=200.0, seconds=seconds)
    click = np.zeros_like(bed)
    for index in range(0, len(click), int(SR * 0.5)):  # 120 BPM
        click[index : index + 400] += 0.6
    mixed = np.clip(bed + click, -1.0, 1.0)
    write_wav(path, np.stack([mixed] * 2, axis=1))

    outcome = analyse(path, asset_id="good", source_type=SourceType.YUE2_RENDER)
    assert outcome.fatal is None
    assert outcome.quality.decode_ok
    assert outcome.quality.duration_ok
    assert outcome.quality.audio_quality_ok, outcome.quality.flags
    assert outcome.quality.loudness_ok, outcome.quality.flags


def test_silence_is_quarantined_not_admitted(tmp_path):
    path = tmp_path / "silent.wav"
    write_wav(path, np.zeros((SR * 10, 2)))
    outcome = analyse(path, asset_id="silent")
    assert not outcome.quality.audio_quality_ok
    assert any("silent" in f for f in outcome.quality.flags)
    assert not outcome.admitted


def test_clipping_is_caught_by_the_gate(tmp_path):
    path = tmp_path / "clipped.wav"
    write_wav(path, sine(440.0, 10.0, amplitude=1.4))
    outcome = analyse(path, asset_id="clipped")
    assert not outcome.quality.audio_quality_ok
    assert any("clipped" in f for f in outcome.quality.flags)


def test_a_too_short_file_is_refused(tmp_path):
    path = tmp_path / "blip.wav"
    write_wav(path, sine(440.0, 1.0, amplitude=0.2))
    outcome = analyse(path, asset_id="blip")
    assert not outcome.quality.duration_ok
    assert not outcome.admitted


def test_mono_is_refused_because_the_library_is_stereo_only(tmp_path):
    path = tmp_path / "mono.wav"
    write_wav(path, sine(440.0, 10.0, amplitude=0.2))
    outcome = analyse(path, asset_id="mono")
    assert not outcome.quality.decode_ok
    assert any("channels" in f for f in outcome.quality.flags)


def test_an_unreadable_file_reports_fatal_rather_than_failing_a_gate(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"nope")
    outcome = analyse(path, asset_id="broken")
    assert outcome.fatal is not None
    assert not outcome.admitted


def test_beat_grid_refuses_to_promise_loops_when_the_tempo_is_ambiguous(tmp_path):
    """ROADMAP.md:52 — an ambiguous grid must not be papered over with a fake loop."""
    path = tmp_path / "ambient.wav"
    noise = np.random.default_rng(0).normal(0, 0.2, (SR * 20, 2))
    write_wav(path, noise)
    outcome = analyse(path, asset_id="ambient")
    assert outcome.manifest is not None
    assert outcome.manifest.loops == []


def test_tempo_alone_does_not_create_a_verified_loop(tmp_path):
    path = tmp_path / "click.wav"
    write_wav(path, np.stack([click_track(40.0)] * 2, axis=1))
    outcome = analyse(path, asset_id="click")
    assert outcome.manifest is not None
    assert outcome.quality.beat_grid_ok
    assert outcome.manifest.loops == []


def test_thresholds_are_adjustable_without_touching_the_logic(tmp_path):
    """Raising the loudness floor must refuse material a looser policy admits."""
    path = tmp_path / "tone.wav"
    write_wav(path, np.stack([sine_rms_dbfs(-15.0, freq=440.0, seconds=12.0)] * 2, axis=1))
    loose = analyse(path, asset_id="t", thresholds=GateThresholds(min_lufs=-20.0))
    strict = analyse(path, asset_id="t", thresholds=GateThresholds(min_lufs=-10.0))
    assert loose.quality.loudness_ok
    assert not strict.quality.loudness_ok
    assert any("outside policy" in f for f in strict.quality.flags)


def test_a_passing_analysis_still_needs_human_approval(tmp_path):
    """Gates passing is not approval. PRODUCT.md:27 keeps those two separate on purpose."""
    path = tmp_path / "tone.wav"
    write_wav(path, np.stack([sine_rms_dbfs(-15.0, freq=440.0, seconds=40.0)] * 2, axis=1))
    outcome = analyse(path, asset_id="t", approval=ApprovalState.PENDING)
    assert outcome.admitted
    assert outcome.manifest is not None
    assert outcome.manifest.quality.passed
    assert not outcome.manifest.admissible(), "gates are not approval"


def test_an_approved_and_passing_manifest_becomes_schedulable(tmp_path):
    path = tmp_path / "music.wav"
    write_wav(path, np.stack([click_track(40.0)] * 2, axis=1))
    outcome = analyse(path, asset_id="t", approval=ApprovalState.APPROVED)
    assert outcome.manifest is not None
    assert outcome.manifest.quality.passed, outcome.quality.flags
    assert outcome.manifest.admissible()
