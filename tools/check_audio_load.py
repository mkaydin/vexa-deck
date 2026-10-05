"""Measure device underflows during CPU contention; plays a quiet 14-second test tone.

Run on the audio host: .venv/bin/python tools/check_audio_load.py
This simulates CPU helpers, not a full YuE2 inference job. Report: var/reports/audio-cpu-load.json.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from vexa_audio.engine import AudioEngine
from vexa_audio.loader import PreloadedTrack

os.chdir(Path(__file__).resolve().parents[1])

# Quiet test tone; two independent matrix workers simulate generation's CPU helper pools.
engine = AudioEngine()
t = np.arange(engine.sample_rate * 2, dtype=np.float32) / engine.sample_rate
samples = np.repeat((0.015 * np.sin(t * 2 * np.pi * 220))[:, None], 2, axis=1)
engine.start()
engine.load_and_play(PreloadedTrack('load-check', samples, engine.sample_rate, 2, duration_s=2))
engine.set_loop(True)
workers = []
try:
    time.sleep(2)
    before = engine.snapshot()
    worker_code = ('import numpy as np,time; a=np.ones((900,900)); '
                   'end=time.monotonic()+12;\nwhile time.monotonic()<end: np.dot(a,a)')
    for _ in range(2):
        workers.append(subprocess.Popen(['nice', '-n', '10', sys.executable, '-c', worker_code],
            env={**os.environ, 'OPENBLAS_NUM_THREADS':'2', 'OMP_NUM_THREADS':'2'},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(12)
    after = engine.snapshot()
    report = {'before':before['stats'], 'after':after['stats'], 'device':after['device'],
              'load_seconds':12, 'workers':2,
              'underflows_during_load':after['stats']['output_underflows']-before['stats']['output_underflows']}
    path = Path('var/reports/audio-cpu-load.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
finally:
    engine.stop()
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
        worker.wait()
