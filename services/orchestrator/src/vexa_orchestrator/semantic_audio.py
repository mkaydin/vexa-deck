"""Optional local CLAP observations. Similarity scores are not vocal probabilities."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_evidence import sha256

VERSION = 1
DEFAULT_MODEL = Path(__file__).resolve().parents[4] / "models/audio-observers/clap-htsat-unfused"
GENRES = (
    "techno",
    "house",
    "Memphis phonk",
    "drift phonk",
    "boom bap hip hop",
    "trap",
    "pop",
    "jazz",
    "ambient",
    "rock",
    "classical",
    "disco",
    "drum and bass",
)
VOICE = ("Instrumental music with no human voice.", "Music with a human voice singing or rapping.")


@lru_cache(maxsize=1)
def _load_local(directory: str):
    import torch
    from transformers import ClapModel, ClapProcessor

    torch.set_num_threads(1)
    model = (
        ClapModel.from_pretrained(directory, local_files_only=True, weights_only=True)
        .to("cpu")
        .eval()
    )
    processor = ClapProcessor.from_pretrained(directory, local_files_only=True)
    weights = sorted(Path(directory).glob("*.safetensors")) or sorted(Path(directory).glob("*.bin"))
    if not weights:
        raise ValueError("local CLAP weights are missing")
    identity = {file.name: sha256(file) for file in [Path(directory) / "config.json", *weights]}
    return model, processor, identity


def _features(value):
    # Transformers releases return either tensors or output objects.
    if hasattr(value, "pooler_output"):
        value = value.pooler_output
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def observe(path: Path, requested: dict, *, cancel=None) -> dict:
    """CPU-only, local-only, three eight-second windows. No hidden model downloads."""
    directory = os.environ.get("VEXA_CLAP_MODEL", str(DEFAULT_MODEL)).strip()
    unknown = {
        "analyzer": "clap",
        "version": VERSION,
        "calibrated": False,
        "vocal_mode": None,
        "genre": None,
        "language": None,
        "vocal_occupancy": None,
        "confidence": None,
    }
    if not directory:
        return {**unknown, "status": "unavailable", "reason": "VEXA_CLAP_MODEL is not configured"}
    if not Path(directory).is_dir():
        return {**unknown, "status": "unavailable", "reason": "local CLAP directory is missing"}
    from .audio_priority import deferred_reason

    reason = deferred_reason()
    if reason:
        return {**unknown, "status": "deferred", "reason": reason}
    try:
        import torch
        from scipy.signal import resample_poly

        model, processor, identity = _load_local(str(Path(directory).resolve()))
        sr = int(processor.feature_extractor.sampling_rate)
        info = sf.info(path)
        width = min(8.0, info.duration)
        starts = sorted(
            {0.0, max(0.0, (info.duration - width) / 2), max(0.0, info.duration - width)}
        )
        target = requested.get("genre") or "music"
        genre_labels = list(dict.fromkeys([target, *GENRES]))
        labels = [f"The sound of {name} music." for name in genre_labels] + list(VOICE)
        instruments = list(
            dict.fromkeys(
                requested.get("include_instruments", []) + requested.get("exclude_instruments", [])
            )
        )[:12]
        labels += [f"Music with {name}." for name in instruments]
        with torch.inference_mode():
            text = _features(
                model.get_text_features(
                    **processor(text=labels, return_tensors="pt", padding=True, truncation=True)
                )
            )
            windows, embeddings = [], []
            for start in starts:
                reason = deferred_reason()
                if reason:
                    return {**unknown, "status": "deferred", "reason": reason}
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("semantic analysis cancelled")
                samples, original_sr = sf.read(
                    path,
                    start=round(start * info.samplerate),
                    frames=round(width * info.samplerate),
                    always_2d=True,
                    dtype="float32",
                )
                from math import gcd

                divisor = gcd(sr, original_sr)
                audio = resample_poly(samples.mean(axis=1), sr // divisor, original_sr // divisor)
                embedding = _features(
                    model.get_audio_features(
                        **processor(audio=audio, sampling_rate=sr, return_tensors="pt")
                    )
                )
                scores = (embedding @ text.T)[0].cpu().numpy()
                windows.append(
                    {
                        "start_s": start,
                        "end_s": start + width,
                        "cosine_similarity": dict(
                            zip(labels, scores.astype(float).tolist(), strict=True)
                        ),
                        "voice_margin": float(
                            scores[len(genre_labels) + 1] - scores[len(genre_labels)]
                        ),
                    }
                )
                embeddings.append(embedding[0].cpu().numpy())
        values = np.mean([list(window["cosine_similarity"].values()) for window in windows], axis=0)
        ranking = sorted(
            zip(genre_labels, values[: len(genre_labels)].astype(float).tolist(), strict=True),
            key=lambda row: -row[1],
        )
        aggregate = np.mean(embeddings, axis=0)
        aggregate /= max(float(np.linalg.norm(aggregate)), 1e-9)
        return {
            **unknown,
            "status": "observed",
            "model_hashes": identity,
            "device": "cpu",
            "windows": windows,
            "genre_ranking": [{"label": name, "cosine": score} for name, score in ranking],
            "audio_embedding": aggregate.astype(float).tolist(),
            "voice_margin": float(np.mean([w["voice_margin"] for w in windows])),
            "warning": "Uncalibrated cosine rankings; do not certify genre, vocals or language.",
        }
    except Exception as exc:
        if cancel is not None and cancel.is_set():
            raise
        return {**unknown, "status": "unavailable", "reason": str(exc)}
