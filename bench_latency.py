#!/usr/bin/env python3
"""Measure real streaming latency: replay stdin PCM at 1x real-time through the
OnlineASR loop and report, per committed word, how long after it was spoken it
appeared (confirmation delay). Deterministic from a recording — no live mic.

  cat rec.s16le | ./gpu-server.sh --latency large-v3-turbo        # MODEL [MIN_CHUNK] [BEAM]

This is the GPU-side algorithmic latency (chunk accumulation + decode +
LocalAgreement's 2-pass confirmation). Real end-to-end delay adds the LAN round
trip (a few ms) and mic capture buffering on top.
"""
import sys
import time

import numpy as np
from faster_whisper import WhisperModel

from streaming import OnlineASR, SAMPLE_RATE


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepdml/faster-whisper-large-v3-turbo-ct2"
    min_chunk = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    beam = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    audio = np.frombuffer(sys.stdin.buffer.read(), dtype=np.int16).astype(np.float32) / 32768.0
    dur = len(audio) / SAMPLE_RATE

    m = WhisperModel(model, device="cuda", compute_type="int8")
    list(m.transcribe(audio[:SAMPLE_RATE], language="ru", beam_size=beam, word_timestamps=True)[0])  # warm up

    online = OnlineASR(m, "ru", "", beam, 18.0)
    step = int(min_chunk * SAMPLE_RATE)
    lat = []                      # (delay_seconds, word_text)
    t0 = time.time()
    fed = 0
    for i in range(0, len(audio), step):
        chunk = audio[i:i + step]
        fed += len(chunk)
        # Pace to 1x: don't feed chunk N until real time reaches its capture moment.
        target = t0 + fed / SAMPLE_RATE
        slack = target - time.time()
        if slack > 0:
            time.sleep(slack)
        online.insert_audio(chunk)
        committed = online.process()
        now = time.time()
        for _ws, we, text in committed:
            lat.append((now - (t0 + we), text))   # shown_at - spoken_at
    for _ws, we, text in online.finish():
        lat.append((time.time() - (t0 + we), text))

    if not lat:
        print(f"model={model}: no words committed")
        return
    delays = sorted(d for d, _ in lat)
    n = len(delays)
    pct = lambda q: delays[min(n - 1, int(q * n))]
    slowest = sorted(lat, reverse=True)[:5]
    print(f"model={model}  min_chunk={min_chunk}s  beam={beam}  words={n}  audio={dur:.1f}s")
    print(f"confirmation delay (spoken -> shown):  median={pct(0.5):.2f}s  p90={pct(0.9):.2f}s  max={delays[-1]:.2f}s")
    print("slowest words: " + ", ".join(f'"{t.strip()}"={d:.1f}s' for d, t in slowest))


if __name__ == "__main__":
    main()
