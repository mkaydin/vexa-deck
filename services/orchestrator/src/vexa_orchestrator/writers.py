"""Writer profiles, pinned local identity and provider capabilities for small requests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from vexa_yue2.backends import BackendUnavailable

ROOT = Path(__file__).resolve().parents[4]
CANDIDATES = (
    ("qwen3.5:9b", "Qwen 3.5 9B", "local"),
    ("gemma4:e4b", "Gemma 4 E4B", "local"),
    ("nemotron-3-super:cloud", "Nemotron 3 Super / cloud", "remote"),
)


class WriterProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: str = Field(pattern="^(local|remote|template)$")
    model: str = Field(min_length=1, max_length=160, pattern=r"^[\w./:-]+$")
    provider: str = Field(default="ollama", pattern="^(ollama|openai|none)$")
    base: str = ""
    digest: str | None = None
    context_length: int = Field(default=8192, ge=4096, le=32768)
    max_tokens: int = Field(default=4000, ge=512, le=6000)
    creativity: float = Field(default=0.35, ge=0, le=1)
    lyric_creativity: float = Field(default=0.6, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2147483000)

    @model_validator(mode="after")
    def coherent_provider(self):
        if self.mode == "local" and (self.provider != "ollama" or self.model.endswith(":cloud")):
            raise ValueError("local writer requires a local Ollama model")
        if self.mode == "remote" and self.provider == "none":
            raise ValueError("remote writer requires a provider")
        if self.base:
            parsed = urlsplit(self.base)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("writer endpoint must be an HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("writer endpoint must not contain credentials, query or fragment")
        elif self.mode == "remote":
            raise ValueError("remote writer requires an endpoint")
        return self


def installed_local(model: str) -> dict | None:
    """Read project manifests without starting or loading a model in the audio host."""
    depot = ROOT / "models/ollama-writer/manifests"
    for path in depot.rglob("*") if depot.is_dir() else ():
        if not path.is_file():
            continue
        parts = path.relative_to(depot).parts
        name = "/".join(parts[2:-1]) + ":" + parts[-1]
        if name == model:
            raw = path.read_bytes()
            body = json.loads(raw)
            return {
                "digest": hashlib.sha256(raw).hexdigest(),
                "size_bytes": sum(layer.get("size", 0) for layer in body.get("layers", [])),
            }
    return None


def pin_local(settings: dict) -> dict:
    profile = WriterProfile.model_validate(settings).model_dump()
    if profile["mode"] == "local" and not profile["base"] and not profile["digest"]:
        manifest = installed_local(profile["model"])
        if manifest:
            profile["digest"] = manifest["digest"]
    return profile


def catalog() -> list[dict]:
    rows = []
    for model, label, mode in CANDIDATES:
        manifest = installed_local(model) if mode == "local" else None
        rows.append(
            {
                "model": model,
                "name": label,
                "mode": mode,
                "available": bool(manifest) if mode == "local" else None,
                "digest": manifest["digest"] if manifest else None,
                "size_bytes": manifest["size_bytes"] if manifest else None,
                "reason": None
                if manifest
                else "cloud access checked when used"
                if mode == "remote"
                else "install in the private writer depot",
            }
        )
    return rows


def preflight(settings: dict, base: str) -> dict:
    if settings["provider"] != "ollama":
        return {
            "model": settings["model"],
            "digest": None,
            "schema": "json_text",
            "thinking_control": False,
        }
    response = httpx.get(f"{base}/api/tags", timeout=10)
    response.raise_for_status()
    row = next(
        (
            r
            for r in response.json().get("models", [])
            if r["name"] in {settings["model"], settings["model"] + ":latest"}
        ),
        None,
    )
    if row is None:
        raise BackendUnavailable(f"Writer {settings['model']} is not installed at the endpoint")
    if settings.get("digest") and row.get("digest") != settings["digest"]:
        raise BackendUnavailable("writer model digest changed; select the updated model explicitly")
    show = httpx.post(f"{base}/api/show", json={"model": settings["model"]}, timeout=10)
    show.raise_for_status()
    capabilities = show.json().get("capabilities", row.get("capabilities", []))
    cloud = settings["model"].endswith(":cloud") or bool(row.get("remote_host"))
    return {
        "model": settings["model"],
        "digest": row.get("digest"),
        "schema": "json_text" if cloud else "local_json_schema",
        "thinking_control": "thinking" in capabilities,
        "capabilities": capabilities,
    }


def complete(settings: dict, messages: list[dict], schema: dict, stage: str) -> str:
    """Uniform output text and comparable telemetry, without implicit provider fallback."""
    from .generation import _ollama_prompts

    metrics = {"stage": stage, "model": settings["model"]}
    started = time.monotonic()
    try:
        temperature = (
            settings.get("lyric_creativity", 0.6)
            if stage == "lyrics"
            else settings.get("creativity", 0.35)
        )
        if settings["provider"] == "ollama":
            metrics["seed_requested"] = settings.get("seed", 42)
            identity = settings.get("_identity", {})
            return _ollama_prompts(
                settings["base"],
                settings["model"],
                messages,
                1,
                None,
                structured=True,
                schema=schema,
                options={
                    "num_ctx": settings.get("context_length", 8192),
                    "num_predict": settings.get("max_tokens", 4000),
                    "temperature": temperature,
                    "seed": settings.get("seed", 42),
                },
                metrics=metrics,
                keep_alive=settings.get("_keep_alive", 60),
                thinking=False if identity.get("thinking_control", True) else None,
                schema_supported=identity.get("schema", "local_json_schema") != "json_text",
            )
        response = httpx.post(
            f"{settings['base']}/chat/completions",
            timeout=180,
            headers={"Authorization": f"Bearer {os.environ.get('VEXA_LLM_API_KEY', '')}"},
            json={
                "model": settings["model"],
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": settings.get("max_tokens", 4000),
            },
        )
        response.raise_for_status()
        body = response.json()
        metrics.update(body.get("usage", {}))
        return body["choices"][0]["message"]["content"]
    finally:
        metrics["wall_s"] = time.monotonic() - started
        settings.setdefault("_requests", []).append(metrics)
