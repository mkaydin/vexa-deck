"""Render one ACE-Step request in its dedicated Python environment, outside PortAudio."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main():
    request, output = map(Path, sys.argv[1:])
    payload = json.loads(request.read_text())
    source = Path(payload["source"])
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = payload["models"]
    os.environ["ACESTEP_PROJECT_ROOT"] = str(
        Path(__file__).resolve().parents[1] / "var/cache/acestep"
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        str(source),
        "acestep-v15-turbo",
        device="cuda",
        offload_to_cpu=True,
        offload_dit_to_cpu=False,
        use_flash_attention=False,
        compile_model=False,
    )
    if not ok:
        raise RuntimeError(status)
    params = GenerationParams(
        caption=payload["caption"],
        lyrics=payload["lyrics"],
        instrumental=payload["instrumental"],
        duration=payload["duration"],
        bpm=payload.get("bpm"),
        keyscale=payload.get("key", ""),
        seed=payload["seed"],
        timesignature="4",
        inference_steps=8,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
    )
    with tempfile.TemporaryDirectory(prefix="ace-render-", dir=output.parent) as directory:
        result = generate_music(
            handler,
            None,
            params,
            GenerationConfig(
                batch_size=1, use_random_seed=False, seeds=[payload["seed"]], audio_format="wav"
            ),
            save_dir=directory,
        )
        if not result.success or len(result.audios) != 1:
            raise RuntimeError(result.error or "ACE-Step returned no single track")
        shutil.copyfile(result.audios[0]["path"], output)


if __name__ == "__main__":
    main()
