"""ACE-Step's music-specific text LM, isolated from the live audio interpreter.

Uses the upstream inspiration protocol, not generic JSON/chat prompting or audio codes.
Protocol: https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/llm_inference.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import httpx

from .music_plan import arrangement, tempo_hint, track_query, validate_metadata, vocal_language

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]
MODEL_NAME = "acestep-5Hz-lm-1.7B"
INSTRUCTION = "Format the user's input into a more detailed and specific musical description:"


def compact_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit].rsplit(" ", 1)[0].rstrip(".,;")


def parse_sample(raw: str) -> dict:
    """ACE-Step returns YAML-like metadata inside <think>, then free-form lyrics."""
    before, separator, lyrics = raw.partition("</think>")
    if not separator:
        raise ValueError("ACE-Step response truncated before metadata ended")
    metadata = {}
    field = None
    for line in before.replace("<think>", "").splitlines():
        match = re.match(
            r"^(bpm|caption|duration|genres|keyscale|language|timesignature):\s*(.*)", line
        )
        if match:
            field, value = match.groups()
            metadata[field] = "" if value.strip() in ("|", ">", "|-", ">-") else value.strip()
        elif field and line.strip():
            metadata[field] += " " + line.strip()
    lyrics = re.sub(r"^#\s*Lyrics?\s*\n", "", lyrics.strip(), flags=re.I)
    lyrics = re.sub(r"<\|im_end\|>.*$", "", lyrics, flags=re.S).strip()
    if re.search(r"<\|audio_code_", lyrics):
        raise ValueError("planner returned audio codes instead of lyrics")
    if "caption" in metadata:
        metadata["caption"] = metadata["caption"].strip().strip("\"'")
    metadata["lyrics"] = lyrics
    if not metadata.get("caption"):
        raise ValueError("ACE-Step response has no musical description")
    return metadata


class TextPlanner:
    def __init__(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            LogitsProcessorList,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        model_path = Path(os.environ.get("VEXA_ACE_MODEL") or ROOT / "models" / MODEL_NAME)
        if not (model_path / "vexa-download.json").exists():
            raise RuntimeError(
                "ACE-Step planner missing; run .venv/bin/python tools/install_music_planner.py"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("ACE-Step requires the configured RTX 5060 Ti CUDA device")
        torch.set_num_threads(1)
        LOGGER.info("ACE-Step load model=%s GPU=%s", model_path, torch.cuda.get_device_name(0))
        if "5060 Ti" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Music planning must use the RTX 5060 Ti")
        self.torch = torch
        self.LogitsProcessorList = LogitsProcessorList
        self.StoppingCriteria = StoppingCriteria
        self.StoppingCriteriaList = StoppingCriteriaList
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda:0")
            .eval()
        )
        from .vendor.acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor

        self.constrained = MetadataConstrainedLogitsProcessor(
            self.tokenizer,
            enabled=True,
            debug=False,
            skip_genres=True,
            max_duration=180,
        )
        self.max_tokens = int(os.environ.get("VEXA_ACE_MAX_TOKENS", "1024"))

    def sample(
        self,
        query: str,
        *,
        instrumental: bool,
        seed: int,
        metadata: dict | None = None,
        caption: str = "",
        lyrics: str = "",
    ) -> dict:
        self.torch.manual_seed(seed)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"# Instruction\n{INSTRUCTION}\n\n"},
                {
                    "role": "user",
                    "content": f"# Caption\n{caption}\n\n# Lyric\n{lyrics or '[Instrumental]'}\n",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        processor = self.constrained
        processor.reset()
        processor.set_generation_phase("understand")
        processor.set_user_metadata(metadata or {})
        processor.set_skip_genres(True)
        prefix_length = inputs.input_ids.shape[1]

        class AdvanceFSM:
            def __call__(self, input_ids, scores):
                if input_ids.shape[1] > prefix_length:
                    processor.update_state(input_ids[0, -1].item())
                return processor(input_ids, scores)

        end_reasoning = self.tokenizer.encode("</think>", add_special_tokens=False)

        class MetadataComplete(self.StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return input_ids[0, -len(end_reasoning) :].tolist() == end_reasoning

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                logits_processor=self.LogitsProcessorList([AdvanceFSM()]),
                stopping_criteria=self.StoppingCriteriaList([MetadataComplete()]),
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=0.65,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        if len(generated) >= self.max_tokens:
            raise ValueError("ACE-Step text plan reached its token limit")
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        diagnostic = ROOT / "var/plans/raw" / f"ace-{time.time_ns()}.txt"
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(raw, encoding="utf-8")
        return parse_sample(raw)


@contextmanager
def local_writer_server():
    """Use a private Ollama worker bound to the same single GPU as ACE-Step/YuE2."""
    configured = os.environ.get("VEXA_ACE_WRITER_URL", "").strip()
    if configured:
        yield configured.rstrip("/")
        return
    from .generation import _gpu_index

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env.update(
        OLLAMA_HOST=f"127.0.0.1:{port}",
        CUDA_VISIBLE_DEVICES=_gpu_index(),
        OLLAMA_MODELS=str(ROOT / "models/ollama-writer"),
        OLLAMA_NUM_PARALLEL="1",
        OLLAMA_MAX_LOADED_MODELS="1",
        OLLAMA_KEEP_ALIVE="0",
        OLLAMA_VULKAN="0",
    )
    log = ROOT / "var/logs" / f"lyric-writer-{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        process = subprocess.Popen(
            ["ollama", "serve"], env=env, stdout=output, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"private lyric writer exited: {log.read_text()[-1000:]}")
                try:
                    response = httpx.get(f"{base}/api/tags", timeout=1)
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("private lyric writer startup timed out") from None
                    time.sleep(0.2)
            LOGGER.info(
                "ACE-Step private lyric writer ready GPU=%s log=%s",
                env["CUDA_VISIBLE_DEVICES"],
                log,
            )
            yield base
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def write_local_briefs(theme: str, count: int):
    """Use a local general LLM for coherent, original theme-specific lyrics."""
    from .generation import _plan_remote_theme
    from .music_plan import lyric_lines

    model = os.environ.get("VEXA_ACE_WRITER_MODEL") or "gemma4:e4b"
    if model.endswith(":cloud"):
        raise ValueError("the local ACE-Step lyric writer cannot be a cloud model")
    with local_writer_server() as base:
        response = httpx.get(f"{base}/api/tags", timeout=10)
        response.raise_for_status()
        names = {row["name"] for row in response.json().get("models", [])}
        if model not in names and f"{model}:latest" not in names:
            raise RuntimeError(
                f"Local lyric writer {model} missing; run "
                ".venv/bin/python tools/install_local_lyric_writer.py"
            )
        keys = {"VEXA_LLM_PROVIDER": "ollama", "VEXA_LLM_MODEL": model, "VEXA_LLM_BASE_URL": base}
        previous = {name: os.environ.get(name) for name in keys}
        LOGGER.info("ACE-Step local lyric writer model=%s theme=%s tracks=%d", model, theme, count)
        try:
            os.environ.update(keys)
            briefs = []
            for index in range(count):
                LOGGER.info("ACE-Step lyric writing track=%d/%d", index + 1, count)
                brief = _plan_remote_theme(theme, count=1, track_index=index)[0]
                if brief.lyrics and any(old.lyrics == brief.lyrics for old in briefs):
                    raise ValueError(f"local lyric writer repeated track {index + 1}")
                briefs.append(brief)
                LOGGER.info(
                    "ACE-Step local lyrics accepted track=%d/%d lines=%d",
                    index + 1,
                    count,
                    len(lyric_lines(brief.lyrics)),
                )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        # An explicit unload completes before ACE-Step claims CUDA memory.
        response = httpx.post(
            f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60
        )
        response.raise_for_status()
        if response.json().get("error"):
            raise RuntimeError(response.json()["error"])
        LOGGER.info("ACE-Step lyric writer released model=%s before musical planning", model)
        return briefs


def plan_in_worker(theme: str, count: int) -> dict:
    # Imported here only; the audio host never imports torch/transformers.
    from .generation import ENERGY_ARC, ROLES, TRACK_DURATIONS, _parse_track_plan, _theme_profile

    started = time.monotonic()
    genre, palette, vocal = _theme_profile(theme)
    written = write_local_briefs(theme, count)
    planner = TextPlanner()
    briefs = []
    for index in range(count):
        query = track_query(
            theme,
            genre=genre,
            palette=palette,
            vocal=vocal,
            duration_s=TRACK_DURATIONS[index],
            energy=ENERGY_ARC[index],
            index=index,
            role=ROLES[index],
        )
        failure = ""
        for attempt in range(3):
            LOGGER.info(
                "ACE-Step planning track=%d/%d attempt=%d theme=%s",
                index + 1,
                count,
                attempt + 1,
                theme,
            )
            try:
                sample = planner.sample(
                    query + (f"\nCorrect previous failure: {failure}." if failure else ""),
                    instrumental=not vocal,
                    seed=1701 + index * 31 + attempt,
                    caption=written[index].style,
                    lyrics=written[index].lyrics,
                    metadata={
                        "language": vocal_language(theme) if vocal else "unknown",
                        "duration": str(round(TRACK_DURATIONS[index])),
                        "bpm": str(round(tempo_hint(theme, genre))),
                        "timesignature": "4",
                    },
                )
                metadata = validate_metadata(sample, theme, genre)
                sample["lyrics"] = written[index].lyrics
                if vocal and sample.get("language", "").strip() != vocal_language(theme):
                    raise ValueError("planner changed the requested vocal language")
                refinement_accepted = True
                try:
                    _parse_track_plan(
                        json.dumps(
                            {
                                "tracks": [
                                    {"style": sample["caption"], "lyrics": written[index].lyrics}
                                ]
                            }
                        ),
                        theme,
                        1,
                        MODEL_NAME,
                    )
                    if vocal and "boom bap" in genre and re.search(
                        r"singing|sung|croon", sample["caption"], re.I
                    ):
                        raise ValueError("musical refinement introduced singing into spoken rap")
                except ValueError as exc:
                    refinement_accepted = False
                    LOGGER.warning(
                        "ACE-Step optional caption rejected: %s; keeping writer style", exc
                    )
                refinement = compact_text(sample["caption"], 300) if refinement_accepted else ""
                form = arrangement(
                    TRACK_DURATIONS[index], metadata["bpm"], vocal=vocal, rap="hip hop" in genre
                )
                form_text = "; ".join(f"{row['section']} {row['bars']} bars" for row in form)
                style = (
                    f"{compact_text(written[index].style, 850)}. "
                    f"{refinement}. "
                    f"Target {metadata['bpm']:g} BPM, {metadata['key']}, "
                    f"4/4. Arrangement: {form_text}. "
                    f"{ROLES[index]}; energy {ENERGY_ARC[index]:.2f}. "
                    "Keep the rhythmic intro/outro instrumental and continue to the final bar."
                )
                # Use the same genre/vocal/length gate as the existing remote planner.
                brief = _parse_track_plan(
                    json.dumps({"tracks": [{"style": style, "lyrics": sample["lyrics"]}]}),
                    theme,
                    1,
                    MODEL_NAME,
                )[0]
                musical_plan = {
                    **metadata,
                    "requested_duration_s": TRACK_DURATIONS[index],
                    "planner_duration_s": sample.get("duration"),
                    "sections": form,
                    "role": ROLES[index],
                    "theme": theme,
                    "language": vocal_language(theme) if vocal else "unknown",
                    "instrumental": not vocal,
                    "genre": genre,
                    "query": query,
                    "lyric_origin": written[index].origin,
                    "musical_caption": sample["caption"],
                    "refinement_accepted": refinement_accepted,
                    "lyric_writer_style": written[index].style,
                }
                row = asdict(brief)
                row.update(
                    duration_s=TRACK_DURATIONS[index],
                    energy=ENERGY_ARC[index],
                    origin=(
                        f"local:{os.environ.get('VEXA_ACE_WRITER_MODEL', 'gemma4:e4b')}"
                        f"+{MODEL_NAME}"
                    ),
                    music_plan=musical_plan,
                    title=getattr(written[index], "title", "") or brief.title,
                )
                if any(old["lyrics"] == row["lyrics"] for old in briefs) and vocal:
                    raise ValueError("planner repeated another track's lyrics")
                briefs.append(row)
                LOGGER.info(
                    "ACE-Step accepted track=%d bpm=%s key=%s lyric_lines=%d",
                    index + 1,
                    metadata["bpm"],
                    metadata["key"],
                    len(brief.lyrics.splitlines()),
                )
                break
            except ValueError as exc:
                failure = str(exc)
                LOGGER.warning("ACE-Step rejected track=%d reason=%s", index + 1, failure)
        else:
            raise ValueError(f"ACE-Step track {index + 1} failed after three attempts: {failure}")
    report = ROOT / "var/plans" / f"ace-{time.time_ns()}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "theme": theme,
        "model": MODEL_NAME,
        "elapsed_s": time.monotonic() - started,
        "tracks": briefs,
    }
    report.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("ACE-Step plan saved %s; worker exit releases GPU before YuE2", report)
    return {"tracks": briefs, "report": str(report)}
