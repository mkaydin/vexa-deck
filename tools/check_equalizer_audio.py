"""Measure EQ adjustments during two-deck playback on PortAudio, with speaker output muted."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from vexa_audio.engine import AudioEngine
from vexa_audio.equalizer import EqualizerSettings

ROOT = Path(__file__).resolve().parents[1]


def check():
    engine = AudioEngine()
    original_render = engine.mixer.render

    def muted_render(output, frames):
        original_render(output, frames)
        # Exercise actual mixer/DSP with tones, then silence before sending to the device.
        output.fill(0)

    engine.mixer.render = muted_render
    with tempfile.TemporaryDirectory(prefix="vexa-eq-audio-") as directory:
        t = np.arange(44100 * 12) / 44100
        for index, frequency in enumerate((60, 1000)):
            signal = 0.05 * np.sin(2 * np.pi * frequency * t)
            path = Path(directory) / f"deck-{index}.wav"
            sf.write(path, np.column_stack((signal, signal)), 44100)
            track = engine.preload(path)
            if index == 0:
                engine.load_and_play(track, 0)
            else:
                engine.crossfade_to(track, 1, fade_frames=44100 * 6)
        try:
            engine.start()
            for step in range(80):
                gains = (6, 2, -3, 1, 4) if step % 2 else (-6, -2, 3, -1, -4)
                engine.mixer.equalizer.configure(EqualizerSettings(step % 8 != 0, gains, -6))
                time.sleep(0.1)
            snapshot = engine.snapshot()
        finally:
            engine.stop()
    stats = snapshot["stats"]
    report = {
        "speaker_output": "muted",
        "device_opened": True,
        "updates": 80,
        "block_deadline_ms": engine.config.blocksize * 1000 / engine.sample_rate,
        "stats": stats,
        "equalizer": snapshot["equalizer"],
    }
    destination = ROOT / "var/reports/equalizer-audio-qa.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert stats["callbacks"] > 100
    assert stats["underruns"] == stats["output_underflows"] == stats["queue_dropped"] == 0
    assert stats["fade_blocks"] > 20
    assert stats["worst_callback_ms"] < report["block_deadline_ms"]


if __name__ == "__main__":
    check()
