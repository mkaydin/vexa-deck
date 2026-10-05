"""Benchmark ACE-Step 2B Turbo in its own environment, outside the live depot.

Download: .venv/bin/python tools/check_acestep_audio.py --download
Render: third_party/ACE-Step-1.5/.venv/bin/python tools/check_acestep_audio.py
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/ACE-Step-1.5"
CHECKPOINTS = ROOT / "models/acestep-audio"
CASES = {
    "phonk": (
        140,
        "F minor",
        "Instrumental drift phonk, gritty distorted 808 bass, "
        "metallic cowbell melody, hard clipped drums, syncopated hi-hats. "
        "Eight-bar rhythmic intro, evolving bass drops, breakdown and eight-bar DJ outro. "
        "Entirely instrumental, no voice samples, vocals, speech or humming.",
    ),
    "hiphop": (
        90,
        "D minor",
        "Instrumental 1990s underground boom bap hip-hop, dusty "
        "vinyl texture, chopped jazz piano, warm bass, swinging kick and snare. "
        "Eight-bar drum intro, contrasting sections, breakdown, eight-bar drum outro. "
        "No vocals, speech, humming or vocal samples.",
    ),
    "techno": (
        128,
        "A minor",
        "Instrumental dark melodic techno, driving four-on-the-floor "
        "kick, rolling bass, hypnotic analog arpeggios, atmospheric pads, evolving "
        "percussion, tension-building breakdown. Eight-bar rhythmic intro and outro. "
        "No vocals, speech, humming or vocal samples.",
    ),
    "vocal": (
        90,
        "D minor",
        "1990s underground boom bap rap, dusty jazz piano, warm bass, "
        "swinging drums, clear English rap vocal, rainy city atmosphere. "
        "Instrumental drum intro and outro, verses and repeated chorus.",
    ),
}
LYRICS = """[Verse 1]
Clock on the wall says the shift is done
Rain on the pavement hides the sun
Steel in my hands and dust in my coat
I carry the words that I never wrote
Bus rolling past with a window of light
Home is a spark at the edge of the night
Steps keep time with the railway hum
One more street and the day is done
[Chorus]
Rain keeps falling, I keep walking
City lights and the sidewalks talking
Through the grey there is light to find
Leave that factory noise behind
[Verse 2]
Corner shop closes, the shutters come down
Last train sings through the sleeping town
Coins in my pocket, a key in my palm
After the thunder I reach for the calm
Steam from the kettle, a chair by the door
Boots leave the weight of the day on the floor
Tomorrow can wait till the morning is clear
Tonight all the people I love are here
[Chorus]
Rain keeps falling, I keep walking
City lights and the sidewalks talking
Through the grey there is light to find
Leave that factory noise behind
"""


def download() -> None:
    """Fetch pinned audio components and reuse the installed text LM."""
    os.environ["HF_HOME"] = str(ROOT / "models/hf-cache")
    from huggingface_hub import HfApi, snapshot_download

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info("ACE-Step/Ace-Step1.5")
    snapshot_download(
        info.id,
        revision=info.sha,
        local_dir=CHECKPOINTS,
        allow_patterns=[
            "acestep-v15-turbo/*",
            "vae/*",
            "Qwen3-Embedding-0.6B/*",
            "README.md",
            "LICENSE*",
        ],
    )
    lm = CHECKPOINTS / "acestep-5Hz-lm-1.7B"
    if not lm.exists():
        lm.symlink_to(ROOT / "models/acestep-5Hz-lm-1.7B", target_is_directory=True)
    (CHECKPOINTS / "download-revision.json").write_text(
        json.dumps({"repo": info.id, "revision": info.sha}, indent=2) + "\n"
    )


def main() -> None:
    """Generate reproducible cases and persist timing, memory and audio measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--duration", type=float, default=165)
    parser.add_argument("--gpu-name", default="NVIDIA GeForce RTX 5060 Ti")
    parser.add_argument("--output", type=Path, default=ROOT / "var/reports/acestep-2b-turbo")
    args = parser.parse_args()
    if args.download:
        download()
        return
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    )
    matches = [
        row.split(",", 1)[1].strip()
        for row in devices.splitlines()
        if row.split(",", 1)[0].strip() == args.gpu_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one GPU named {args.gpu_name!r}: {devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = matches[0]
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(CHECKPOINTS)
    os.environ["HF_HOME"] = str(ROOT / "models/hf-cache")
    sys.path.insert(0, str(UPSTREAM))
    import numpy as np
    import soundfile as sf
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "initializing",
        "model": "acestep-v15-turbo",
        "thinking": False,
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cases": [],
        "human_listening": "pending",
        "live_audio_test": "pending",
    }
    report_path = args.output / "report.json"

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run with GPU access outside the sandbox")
        report["gpu"] = torch.cuda.get_device_name(0)
        if report["gpu"] != args.gpu_name:
            raise RuntimeError(f"Wrong GPU selected: {report['gpu']}")
        report["torch"] = torch.__version__
        with (ROOT / "var/gpu-jobs.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            started = time.monotonic()
            handler = AceStepHandler()
            status, ok = handler.initialize_service(
                str(UPSTREAM),
                "acestep-v15-turbo",
                device="cuda",
                offload_to_cpu=True,
                offload_dit_to_cpu=False,
                use_flash_attention=False,
                compile_model=False,
            )
            if not ok:
                raise RuntimeError(status)
            report["initialization_s"] = time.monotonic() - started
            for name in CASES if args.case == "all" else [args.case]:
                bpm, key, caption = CASES[name]
                params = GenerationParams(
                    caption=caption,
                    lyrics=LYRICS if name == "vocal" else "[Instrumental]",
                    instrumental=name != "vocal",
                    vocal_language="en" if name == "vocal" else "unknown",
                    bpm=bpm,
                    keyscale=key,
                    timesignature="4",
                    duration=args.duration,
                    seed=42,
                    inference_steps=8,
                    thinking=False,
                    use_cot_metas=False,
                    use_cot_caption=False,
                    use_cot_language=False,
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.monotonic()
                result = generate_music(
                    handler,
                    None,
                    params,
                    GenerationConfig(
                        batch_size=1, use_random_seed=False, seeds=[42], audio_format="wav"
                    ),
                    save_dir=str(args.output / name),
                )
                torch.cuda.synchronize()
                entry = {
                    "name": name,
                    "wall_s": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "params": asdict(params),
                    "success": result.success,
                    "outputs": [],
                }
                report["cases"].append(entry)
                if not result.success:
                    raise RuntimeError(result.error)
                for audio in result.audios:
                    samples, sr = sf.read(audio["path"], always_2d=True)
                    entry["outputs"].append(
                        {
                            "path": audio["path"],
                            "sample_rate": sr,
                            "duration_s": len(samples) / sr,
                            "channels": samples.shape[1],
                            "finite": bool(np.isfinite(samples).all()),
                            "peak_dbfs": float(20 * np.log10(max(np.abs(samples).max(), 1e-12))),
                            "rms_dbfs": float(
                                20 * np.log10(max(np.sqrt(np.mean(samples**2)), 1e-12))
                            ),
                        }
                    )
                report["status"] = "running"
                save()
                print(json.dumps(entry), flush=True)
            report["status"] = "generated"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
