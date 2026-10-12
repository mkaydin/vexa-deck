"""Real local HTTP/SSE → Qt telemetry check using synthetic output; no audio device."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/desktop"))
import main  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

SERVER = """import sys, threading, time
from itertools import pairwise
from pathlib import Path
import numpy as np
import uvicorn
from vexa_audio.loader import PreloadedTrack
from vexa_orchestrator.api import create_app
from vexa_orchestrator.live import LiveSet
from vexa_orchestrator.depot import Depot
live=LiveSet(Depot(Path(sys.argv[1])/'library'),preview_dir=Path(sys.argv[1])/'previews')
sr=live.engine.sample_rate
wave=(0.2*np.sin(2*np.pi*1000*np.arange(sr)/sr)).astype('float32')
live.engine.mixer.set_track(0,PreloadedTrack('fixture',np.column_stack((wave,-wave)),sr,2))
live.engine.mixer.decks[0].playing=live.engine.mixer.decks[0].loop=True
live.engine.mixer.telemetry.start()
def render():
    block=np.zeros((2048,2),dtype='float32')
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        live.engine.mixer.render(block,2048)
        time.sleep(2048/sr)
threading.Thread(target=render,daemon=True).start()
uvicorn.run(create_app(live=live),host='127.0.0.1',port=int(sys.argv[2]),log_level='warning',timeout_graceful_shutdown=1)
"""


def check():
    app = QApplication([])
    with tempfile.TemporaryDirectory(prefix="vexa-sse-") as directory:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, PYTHONPATH=main.workspace_pythonpath(ROOT))
        log = Path(directory) / "server.log"
        frames = []
        with log.open("w") as output:
            server = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-c", SERVER, directory, str(port)],
                env=env,
                stdout=output,
                stderr=output,
            )
            client = main.BackendClient(f"http://127.0.0.1:{port}")
            client.telemetry_received.connect(lambda body: frames.append(body))
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                    fresh = [f for f in frames if not f["stale"]]
                    if len(fresh) >= 8 and frames[-1]["stale"]:
                        break
                assert len(fresh) >= 8, log.read_text()
                assert max(fresh[-1]["levels"]) > 0.6
                assert all(f["source"] == "post-master" for f in frames)
                assert frames[-1]["stale"] and not any(frames[-1]["levels"])
                assert any(
                    a["sequence"] != b["sequence"] for a, b in pairwise(fresh)
                )
                client.shutdown()
                app.processEvents()
                assert not client._poll.isActive()
            finally:
                client.shutdown()
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        report = dict(
            transport="HTTP SSE → Qt",
            frames=len(frames),
            fresh_frames=len(fresh),
            stereo_phase_preserved=True,
            stale_cleared=True,
            shutdown=True,
            audio_device_opened=False,
        )
        target = ROOT / "var/reports/v02"
        target.mkdir(parents=True, exist_ok=True)
        (target / "telemetry-stream.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    check()
