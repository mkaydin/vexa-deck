"""Versioned ingestion evidence and bounded repairs, outside the playback process."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from vexa_yue2.analysis import measure_loudness, measure_true_peak

EVIDENCE_VERSION = 1
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidencePolicy:
    """Provisional technical limits. Semantic observations never act as hard gates."""

    version: str = "technical-v1"
    duration_tolerance_s: float = 2.0
    max_raw_clipped_ratio: float = 0.001
    max_dc_offset: float = 0.1
    dc_repair_threshold: float = 0.001
    silence_rms: float = 1e-4
    digital_silence_rms: float = 1e-7
    max_internal_digital_silence_s: float = 2.0
    boundary_fade_s: float = 0.005
    boundary_click_threshold: float = 0.05


POLICY = EvidencePolicy()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_json(value):
    """Silence is unknown loudness, never JSON Infinity or a plausible zero LUFS."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def atomic_json(path: Path, body: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite_json(body), indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def inspect_audio(path: Path) -> dict:
    """Whole-file safety metrics plus bounded one-second level/spectral windows."""
    started = time.monotonic()
    report = {"source_sha256": sha256(path), "decode_ok": False}
    try:
        info = sf.info(path)
        samples, sr = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as exc:
        return {**report, "error": str(exc), "reason_codes": ["decode_failed"]}
    report.update(
        decode_ok=True,
        finite_samples=bool(np.isfinite(samples).all()),
        sample_rate_hz=sr,
        channels=samples.shape[1],
        frames=len(samples),
        duration_s=len(samples) / sr,
        codec=info.format,
        subtype=info.subtype,
    )
    if not report["finite_samples"] or not len(samples):
        return {**report, "reason_codes": ["nonfinite_samples" if len(samples) else "empty_audio"]}
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak = float(np.max(np.abs(samples)))
    windows = []
    silent = []
    low_power = high_power = total_power = 0.0
    # Analyze each channel's power: phase cancellation must not turn stereo into silence.
    for start in range(0, len(samples), sr):
        segment = samples[start : start + sr]
        level = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
        end_s = (start + len(segment)) / sr
        windows.append(
            {"start_s": start / sr, "end_s": end_s, "rms_dbfs": 20 * math.log10(max(level, 1e-12))}
        )
        if level <= POLICY.digital_silence_rms:
            if silent and abs(silent[-1]["end_s"] - start / sr) < 1e-6:
                silent[-1]["end_s"] = end_s
            else:
                silent.append({"start_s": start / sr, "end_s": end_s})
        spectrum = np.fft.rfft(segment * np.hanning(len(segment))[:, None], axis=0)
        power = np.square(np.abs(spectrum)).sum(axis=1)
        freqs = np.fft.rfftfreq(len(segment), 1 / sr)
        total_power += float(power.sum())
        low_power += float(power[freqs < 120].sum())
        high_power += float(power[(freqs >= 1000) & (freqs < 6000)].sum())
    duration = report["duration_s"]
    internal = [
        span
        for span in silent
        if span["start_s"] >= min(10.0, duration * 0.1)
        and span["end_s"] <= duration - min(10.0, duration * 0.1)
    ]
    loudness = measure_loudness(samples, sr)
    report.update(
        rms=rms,
        sample_peak_dbfs=20 * math.log10(max(peak, 1e-12)),
        clipped_sample_ratio=float(np.mean(np.abs(samples) >= 0.999)),
        dc_offset_by_channel=np.mean(samples, axis=0).astype(float).tolist(),
        integrated_lufs=loudness.integrated_lufs,
        loudness_range_lu=loudness.loudness_range_lu,
        true_peak_dbtp=measure_true_peak(samples, sr),
        boundary_amplitude={
            "start": float(np.max(np.abs(samples[0]))),
            "end": float(np.max(np.abs(samples[-1]))),
        },
        digital_silence_spans=silent,
        internal_digital_silence_spans=internal,
        level_curve=windows,
        spectral_balance={
            "sub_bass_fraction": low_power / total_power if total_power else None,
            "presence_fraction": high_power / total_power if total_power else None,
        },
        reason_codes=[],
        elapsed_s=time.monotonic() - started,
        measurement_method="vexa numpy/scipy; summed channel K-weighted power; 4x polyphase peaks",
    )
    return finite_json(report)


def safety_reasons(report: dict, duration_s: float, *, raw: bool) -> list[str]:
    reasons = list(report.get("reason_codes", []))
    if not report.get("decode_ok") or reasons:
        return reasons or ["decode_failed"]
    if abs(report["duration_s"] - duration_s) > POLICY.duration_tolerance_s:
        reasons.append("duration_mismatch")
    if report["channels"] not in (1, 2):
        reasons.append("unsupported_channels")
    if report["rms"] < POLICY.silence_rms:
        reasons.append("silent_audio")
    if raw and report["clipped_sample_ratio"] > POLICY.max_raw_clipped_ratio:
        reasons.append("severe_source_clipping")
    if max(abs(value) for value in report["dc_offset_by_channel"]) > POLICY.max_dc_offset:
        reasons.append("excessive_dc_offset")
    if any(
        span["end_s"] - span["start_s"] >= POLICY.max_internal_digital_silence_s
        for span in report["internal_digital_silence_spans"]
    ):
        reasons.append("internal_digital_dropout")
    return reasons


def correct_boundaries_and_format(path: Path, raw_report: dict) -> list[dict]:
    """One pass: mono/rate normalization, small DC removal and 5 ms boundary fades."""
    from scipy.signal import resample_poly

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    corrections = []
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
        corrections.append({"operation": "mono_to_stereo", "channels": 2})
    if sr not in (44100, 48000):
        divisor = math.gcd(sr, 48000)
        samples = resample_poly(samples, 48000 // divisor, sr // divisor, axis=0)
        corrections.append({"operation": "resample", "from_hz": sr, "to_hz": 48000})
        sr = 48000
    offsets = np.mean(samples, axis=0)
    if np.max(np.abs(offsets)) > POLICY.dc_repair_threshold:
        samples = samples - offsets
        corrections.append({"operation": "dc_removal", "offsets": offsets.astype(float).tolist()})
    count = min(round(sr * POLICY.boundary_fade_s), len(samples) // 2)
    for edge, index in (("start", 0), ("end", -1)):
        if count and np.max(np.abs(samples[index])) > POLICY.boundary_click_threshold:
            ramp = np.linspace(0, 1, count)[:, None]
            if edge == "start":
                samples[:count] *= ramp
            else:
                samples[-count:] *= ramp[::-1]
            corrections.append(
                {"operation": "boundary_fade", "edge": edge, "duration_s": count / sr}
            )
    if corrections:
        temporary = path.with_suffix(".correction.tmp.wav")
        # Float format preserves above-full-scale peaks until the bounded mastering pass.
        sf.write(temporary, samples, sr, subtype="FLOAT")
        temporary.replace(path)
    return corrections


class EvidenceSession:
    def __init__(self, path: Path, *, duration_s: float, requested: dict):
        self.path = path
        self.report_path = path.with_suffix(".evidence.json")
        self.body = {
            "evidence_version": EVIDENCE_VERSION,
            "asset_id": path.stem,
            "policy": asdict(POLICY),
            "requested": requested,
            "duration_target_s": duration_s,
            "status": "raw_check",
            "raw": None,
            "final": None,
            "corrections": [],
            "observations": {},
            "admission": {
                "playable": False,
                "constraints_satisfied": None,
                "mixable": False,
                "reason_codes": [],
                "warnings": [],
                "capabilities": {
                    "full_track_playback": False,
                    "beat_synchronized_mix": False,
                    "stem_mix": False,
                    "verified_vocal_safe_overlap": False,
                },
            },
        }

    def save(self):
        atomic_json(self.report_path, self.body)
        progress = {"asset_id": self.path.stem, "stage": self.body["status"]}
        atomic_json(self.path.with_suffix(".progress.json"), progress)
        LOGGER.info(
            "Audio evidence asset=%s stage=%s reasons=%s",
            self.path.stem,
            self.body["status"],
            self.body["admission"]["reason_codes"],
        )

    def begin(self):
        raw_path = self.path.with_suffix(".raw.wav")
        if raw_path.exists():
            raise ValueError("reserved raw render already exists; refusing to overwrite evidence")
        shutil.copyfile(self.path, raw_path)
        self.body["raw_path"] = raw_path.name
        self.save()
        self.body["raw"] = inspect_audio(raw_path)
        self.save()
        reasons = safety_reasons(self.body["raw"], self.body["duration_target_s"], raw=True)
        if reasons:
            self.reject(reasons)
        self.body["status"] = "correcting"
        self.body["corrections"] = correct_boundaries_and_format(self.path, self.body["raw"])
        self.save()

    def mastered(self, report):
        if report is not None:
            self.body["corrections"].append({"operation": "loudness_and_peak", **report.as_dict()})
        self.body["status"] = "analyzing"
        self.body["final"] = inspect_audio(self.path)
        self.save()
        reasons = safety_reasons(self.body["final"], self.body["duration_target_s"], raw=False)
        if reasons:
            self.reject(reasons)

    def reject(self, reasons: list[str], *, detail=""):
        self.body["status"] = "rejected"
        self.body["admission"]["reason_codes"] = list(dict.fromkeys(reasons))
        self.body["admission"]["detail"] = detail
        self.save()
        raise ValueError(
            "audio evidence rejected: " + ", ".join(reasons) + (f": {detail}" if detail else "")
        )


def load_evidence(path: Path, *, source_sha256: str | None = None) -> dict | None:
    """Metadata-only read for host/UI. Recovery separately hashes the audio itself."""
    try:
        body = json.loads(path.with_suffix(".evidence.json").read_text())
        if body["evidence_version"] != EVIDENCE_VERSION or body["asset_id"] != path.stem:
            return None
        if not isinstance(body.get("admission"), dict) or not isinstance(
            body.get("observations"), dict
        ):
            return None
        if any(type(body["admission"].get(key)) is not bool for key in ("playable", "mixable")):
            return None
        if any(
            body.get(key) is not None and not isinstance(body[key], dict)
            for key in ("raw", "final")
        ):
            return None
        if source_sha256 and (body.get("final") or {}).get("source_sha256") != source_sha256:
            return None
        return body
    except (OSError, ValueError, KeyError, TypeError):
        return None
