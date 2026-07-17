#!/usr/bin/env python3
"""Benchmark streaming decode speed for a model: feed stdin PCM (s16le/16k/mono)
through the real OnlineASR loop in --min-chunk increments and report per-pass wall
time + effective streaming RTF, so you can tell whether it keeps up live.

  cat rec.s16le | ./gpu-server.sh --bench large-v3            # via gpu-server.sh (sets CUDA libs)
  cat rec.s16le | ./gpu-server.sh --bench large-v3-turbo 1.0  # MODEL [MIN_CHUNK] [BEAM]

Each re-decode reprocesses the growing buffer (until it trims at segment
boundaries), so per-pass MAX reflects the worst case. Live-viable ==
per-pass stays under MIN_CHUNK (the time until the next chunk arrives).
"""
import sys
import time

import numpy as np
from faster_whisper import WhisperModel

from streaming import OnlineASR, SAMPLE_RATE


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "large-v3"
    min_chunk = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    beam = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    audio = np.frombuffer(sys.stdin.buffer.read(), dtype=np.int16).astype(np.float32) / 32768.0
    dur = len(audio) / SAMPLE_RATE

    m = WhisperModel(model, device="cuda", compute_type="int8")
    # Warm up CUDA kernels so the first timed pass isn't inflated by one-time JIT.
    list(m.transcribe(audio[:SAMPLE_RATE], language="ru", beam_size=beam, word_timestamps=True)[0])

    online = OnlineASR(m, "ru", "", beam, 18.0)
    step = int(min_chunk * SAMPLE_RATE)
    times = []
    for i in range(0, len(audio), step):
        online.insert_audio(audio[i:i + step])
        t = time.time()
        online.process()
        times.append(time.time() - t)
    online.finish()

    total = sum(times)
    s = sorted(times)
    over = sum(1 for x in times if x >= min_chunk)
    print(f"model={model}  audio={dur:.1f}s  passes={len(times)}  min_chunk={min_chunk}s  beam={beam}")
    print(f"per-pass wall: median={s[len(s)//2]:.2f}s  max={s[-1]:.2f}s")
    print(f"effective streaming RTF (sum/audio) = {total/dur:.2f}  ->  {'LIVE-VIABLE' if total < dur else 'TOO SLOW for live'}")
    print(f"passes over the {min_chunk}s budget: {over}/{len(times)}  ({'keeps up' if over == 0 else 'falls behind'})")


if __name__ == "__main__":
    main()
