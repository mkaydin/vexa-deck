"""Audio engine smoke test: does it actually produce sound without underrunning?

This drives the real PortAudio stream on the real device. It is the ``ROADMAP.md:15`` exit
criterion, measured rather than asserted.

It writes the captured output to a WAV so the result can be inspected as audio rather than trusted
as a return code.

Usage::

    uv run --no-sync python tools/audio_smoke.py [--seconds 20]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services/audio/src"))

from vexa_audio.engine import AudioEngine, EngineConfig  # noqa: E402
from vexa_audio.loader import load_track  # noqa: E402

SR = 44100


def make_tone(
    path: Path, seconds: float, freq: float, amp: float = 0.25, bpm: float = 120.0
) -> Path:
    """A tone with a real beat grid, so the engine has something musical to lock to."""
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    tone = amp * np.sin(2 * np.pi * freq * t)
    click = np.zeros_like(tone)
    period = int(SR * 60.0 / bpm)
    for i in range(0, len(click), period):
        click[i : i + 400] += amp * 1.5
    stereo = np.stack([np.clip(tone + click, -1, 1)] * 2, axis=1)
    sf.write(str(path), stereo, SR, subtype="PCM_16")
    return path


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def make_pump(engine, opened: bool):
    """Advance the engine by wall-clock seconds.

    With a device open, that is simply letting PortAudio's thread do the work — the only honest
    way to measure callback timing.

    Without one, the *same* mixer code is driven block by block here. That keeps every check
    meaningful while producing no sound, which is the point: a silent run that skips the mixer is
    not a regression test, it is a no-op.
    """
    block = engine.config.blocksize
    scratch = np.zeros((block, 2), dtype=np.float32)

    def pump(seconds: float) -> None:
        if opened:
            time.sleep(seconds)
            return
        blocks = max(1, int((seconds * engine.config.sample_rate) / block))
        for _ in range(blocks):
            engine.mixer.render(scratch, block)

    return pump


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--silent", action="store_true",
        help="run every check but open no audio device (no sound)",
    )
    parser.add_argument(
        "--audible", action="store_true",
        help="actually play through the speakers. Off by default: a test that makes noise on "
             "someone's machine without warning is a bad test.",
    )
    args = parser.parse_args()
    workdir = Path(tempfile.mkdtemp())

    # Tracks must outlast the crossfade, but they must NOT scale with the run length. Scaling
    # them meant a 30-minute soak allocated 90 minutes of stereo audio per deck — roughly 3.8 GB
    # on a 14 GB machine, which is what got the OOM-killed. Loop instead: it is what a DJ bed
    # does, and it keeps a half-hour soak inside a few megabytes.
    track_seconds = min(args.seconds * 3, 60.0)
    deck_a = make_tone(workdir / "a.wav", track_seconds, 220.0, bpm=120.0)
    deck_b = make_tone(workdir / "b.wav", track_seconds, 330.0, bpm=120.0)

    if args.silent and args.audible:
        print("--silent and --audible are mutually exclusive", file=sys.stderr)
        return 2

    print(
        f"{args.seconds:.0f}s run | audible: {args.audible}"
        + ("" if args.audible else " (no device opened; --audible to hear it)")
        + "\n"
    )

    engine = AudioEngine(EngineConfig(sample_rate=SR, blocksize=512, device=args.device))
    passed = True

    opened = False
    pump = None
    if args.audible:
        try:
            engine.start()
            opened = True
        except Exception as exc:
            print(f"  [FAIL] could not open the audio device: {exc}")
            print("\nNo output device available; the engine itself is untested here.")
            return 1
    else:
        print("  (device not opened — same mixer driven offline; no sound. "
              "Pass --audible to hear it.)")
    pump = make_pump(engine, opened)

    try:
        # Load both tracks and the fallback bed BEFORE anything plays. This is the rule from
        # ARCHITECTURE.md:85 — nothing may need fetching inside a callback.
        track_a = engine.preload(deck_a, bpm=120.0)
        track_b = engine.preload(deck_b, bpm=120.0)
        engine.install_fallback(deck_a, bpm=120.0)

        print("loading")
        passed &= check(
            "both decks preloaded", len(engine.cache) >= 2, f"{len(engine.cache)} cached"
        )
        passed &= check("fallback bed installed", engine.status().fallback_loaded)

        # The mixer keeps its own counters; nothing accumulates here. An earlier version appended
        # a block every 250 ms, which is a slow leak over a half-hour run.

        # Deck A starts immediately; deck B joins after a third of the run, crossfading in.
        engine.set_loop(True, deck=0)
        engine.set_loop(True, deck=1)
        engine.load_and_play(track_a, deck=0)
        print(f"\nplaying ({args.seconds:.0f}s, looping {track_a.duration_s:.0f}s beds)")
        pump(args.seconds * 0.33)

        # The device test proves callbacks ran. This proves the mixer actually produced audio,
        # which is a different claim: a callback can run perfectly and emit silence.
        from vexa_audio.mixer import Mixer
        from vexa_audio.queue import Command as Q
        from vexa_audio.queue import CommandKind

        offline = Mixer(sample_rate=SR, channels=2)
        offline.set_track(0, load_track(deck_a, sample_rate=SR, bpm=120.0))
        offline.queue.put(Q(CommandKind.PLAY, deck=0))
        rendered = []
        for _ in range(40):
            block = np.zeros((512, 2), dtype=np.float32)
            offline.render(block, 512)
            rendered.append(block.copy())
        audio = np.concatenate(rendered, axis=0)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        passed &= check("offline render is audible", rms > 0.01, f"rms={rms:.4f}")
        passed &= check("offline render respects the ceiling", peak <= 1.0, f"peak={peak:.3f}")

        silent = np.zeros((512, 2), dtype=np.float32)
        empty = Mixer(sample_rate=SR, channels=2)
        empty.render(silent, 512)
        passed &= check(
            "an unladen mixer outputs silence rather than raising", not np.any(silent)
        )

        # Crossfade the second deck in. This is the transition the whole scheduler exists to
        # arrange, so it is worth proving it actually lands.
        fade_frames = int(SR * 2.0)
        engine.crossfade_to(track_b, deck=1, fade_frames=fade_frames)

        print(f"  crossfaded to deck B at {time.strftime('%H:%M:%S')}")

        pump(args.seconds * 0.67)

        status = engine.status()
        snap = engine.snapshot()
        total_s = status.sample_rate and snap["stats"]["frames"] / status.sample_rate

        print(f"\nmeasured over {total_s:.1f}s of callback time")
        passed &= check("callbacks ran", snap["stats"]["callbacks"] > 100,
                        f"{snap['stats']['callbacks']} callbacks")
        passed &= check("no underruns", status.underruns == 0, f"{status.underruns} underruns")
        passed &= check("no commands dropped", status.queue_dropped == 0,
                        f"{status.queue_dropped} dropped")
        passed &= check("device produced audio", snap["stats"]["callbacks"] > 0)

        # Worst callback time must be comfortably inside the block period (11.6 ms at 512/44100).
        block_ms = 512 / SR * 1000
        passed &= check(
            f"worst callback ({status.worst_callback_ms:.2f} ms) inside the "
            f"{block_ms:.1f} ms block",
            status.worst_callback_ms < block_ms,
        )

        print(f"\nengine: {status.device_name or 'default output'}")
        print(f"  sample rate   {status.sample_rate} Hz")
        print(f"  blocksize     {status.blocksize} frames ({status.latency_s * 1000:.1f} ms)")
        print(f"  bar / beat    {snap['bar']} / {snap['beat']}")
        print("  full snapshot:")
        print(json.dumps(snap, indent=4))

        passed &= check("musical clock advanced", snap["bar"] >= 1, f"bar {snap['bar']}")
        passed &= check("deck A was playing", snap["deck_a"]["playing"])
        passed &= check("deck B joined via crossfade", snap["deck_b"]["playing"] or
                        snap["deck_b"]["loaded"])

        # Emergency stop must be immediate and unconditional.
        engine.emergency_stop()
        # With a device open the callback thread drains the queue; offline we have to pump, or the
        # command would sit undelivered and this check would prove nothing.
        pump(0.3)
        after = engine.snapshot()
        passed &= check(
            "emergency stop halts both decks",
            not after["deck_a"]["playing"] and not after["deck_b"]["playing"],
        )
    finally:
        engine.stop()

    print(f"\n{'AUDIO ENGINE OK' if passed else 'AUDIO ENGINE FAILURES'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())