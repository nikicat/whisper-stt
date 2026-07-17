#!/usr/bin/env python3
"""CUDA smoke test: load the model on the GPU and run one transcribe pass on a
synthetic buffer — no audio file needed. Run via `./gpu-server.sh --selftest`
so CTranslate2's CUDA/cuDNN libs are on the path.

If int8 kernels are missing on this GPU the model load or first encode raises;
pass a different compute type as the 2nd arg to try a fallback:
    ./gpu-server.sh --selftest deepdml/faster-whisper-large-v3-turbo-ct2 float32
"""

import sys
import time

import numpy as np


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepdml/faster-whisper-large-v3-turbo-ct2"
    compute = sys.argv[2] if len(sys.argv) > 2 else "int8"

    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    print(f"CTranslate2 sees {n} CUDA device(s)")
    if n == 0:
        print("FAIL: no CUDA device visible — check driver / LD_LIBRARY_PATH")
        sys.exit(1)

    from faster_whisper import WhisperModel

    t = time.time()
    m = WhisperModel(model, device="cuda", compute_type=compute)
    print(f"loaded {model} on cuda/{compute} in {time.time() - t:.1f}s")

    # One encode+decode pass exercises the CUDA kernels (the encoder always runs
    # on the full 30s window regardless of content).
    t = time.time()
    segs, _ = m.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"), language="ru")
    list(segs)
    dt = time.time() - t
    print(f"encode+decode pass OK in {dt:.2f}s")
    print(f"per-pass ~{dt:.2f}s  ->  turbo streaming is {'realtime' if dt < 1.0 else 'BORDERLINE (raise --min-chunk)'}")
    print("CUDA is working — wire up ./stream-remote.sh from the laptop.")


SAMPLE_RATE = 16000

if __name__ == "__main__":
    main()
