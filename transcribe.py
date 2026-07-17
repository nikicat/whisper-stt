#!/usr/bin/env python3
"""Realtime EN+RU speech-to-text tuned for Russian speech with English tech terms.

Pipeline: ffmpeg (PipeWire/pulse mic) -> webrtcvad utterance segmentation ->
faster-whisper large-v3-turbo (int8, language forced to Russian).

Capture and transcription run on separate threads so a slow decode never drops
audio: the VAD loop keeps reading the mic while a worker transcribes finished
utterances from a queue.
"""

import argparse
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000          # webrtcvad + Whisper both want 16 kHz mono
FRAME_MS = 30                # webrtcvad accepts 10/20/30 ms frames
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2  # 16-bit mono -> 960 bytes


def load_glossary(path: Path) -> str:
    """Turn glossary.txt into a comma-joined initial_prompt (biases spelling)."""
    if not path.exists():
        return ""
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    # A short natural lead-in + the term list reads better to Whisper than a
    # bare CSV. Kept well under the ~224-token prompt budget by the glossary size.
    return ("Расшифровка технической речи. Термины: " + ", ".join(terms) + ".") if terms else ""


def start_ffmpeg(source: str) -> subprocess.Popen:
    """Spawn ffmpeg reading the pulse/PipeWire source as raw s16le PCM on stdout."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "pulse", "-i", source,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


def read_exact(reader, n: int) -> bytes:
    """Read exactly n bytes, looping over short reads. Returns <n bytes only at
    true EOF. (An unbuffered pipe / ssh stdin can hand back partial frames.)"""
    buf = b""
    while len(buf) < n:
        chunk = reader.read(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf


def transcriber(model, work_q, args, initial_prompt):
    """Worker: pull finished utterances (float32 audio) off the queue and print."""
    while True:
        item = work_q.get()
        if item is None:
            return
        audio, stamp = item
        segments, _info = model.transcribe(
            audio,
            language=args.language,
            initial_prompt=initial_prompt or None,
            beam_size=args.beam_size,
            vad_filter=True,                 # second-stage cleanup on the chunk
            condition_on_previous_text=False,  # avoids runaway repetition in dictation
            temperature=0.0,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            print(f"[{stamp}] {text}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepdml/faster-whisper-large-v3-turbo-ct2",
                   help="HF repo or size name (e.g. large-v3-turbo, medium, small)")
    p.add_argument("--language", default="ru", help="Forced language (ru). Use '' or 'auto' to auto-detect.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--compute-type", default="int8", help="int8 is best for this AVX512-VNNI CPU")
    p.add_argument("--cpu-threads", type=int, default=0, help="0 = ctranslate2 default (all cores)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--glossary", default=str(Path(__file__).parent / "glossary.txt"))
    p.add_argument("--source", default="default", help="pulse/PipeWire source name ('default' or a name from --list-mics)")
    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                   help="0 = most sensitive, 3 = most aggressive at rejecting non-speech")
    p.add_argument("--silence-ms", type=int, default=700, help="trailing silence that closes an utterance")
    p.add_argument("--min-utterance-ms", type=int, default=350, help="drop blips shorter than this")
    p.add_argument("--max-utterance-s", type=int, default=20, help="force a cut for long monologues")
    p.add_argument("--preroll-ms", type=int, default=300, help="audio kept before speech onset (avoids clipped words)")
    p.add_argument("--list-mics", action="store_true", help="list pulse sources and exit")
    args = p.parse_args()

    if args.list_mics:
        subprocess.run(["pactl", "list", "short", "sources"])
        return

    if args.language in ("", "auto"):
        args.language = None

    initial_prompt = load_glossary(Path(args.glossary))

    print(f"Loading model '{args.model}' ({args.device}/{args.compute_type})...", file=sys.stderr, flush=True)
    model = WhisperModel(
        args.model, device=args.device, compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
    )
    print("Model ready. Listening — speak into the mic (Ctrl-C to stop).", file=sys.stderr, flush=True)

    work_q: queue.Queue = queue.Queue()
    worker = threading.Thread(target=transcriber, args=(model, work_q, args, initial_prompt), daemon=True)
    worker.start()

    vad = webrtcvad.Vad(args.vad_aggressiveness)
    proc = start_ffmpeg(args.source)

    preroll_frames = max(1, args.preroll_ms // FRAME_MS)
    silence_frames = max(1, args.silence_ms // FRAME_MS)
    max_frames = args.max_utterance_s * 1000 // FRAME_MS
    min_frames = args.min_utterance_ms // FRAME_MS

    ring = deque(maxlen=preroll_frames)  # pre-speech buffer
    voiced: list[bytes] = []
    triggered = False
    num_silence = 0

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def emit(frames: list[bytes]):
        if len(frames) < min_frames:
            return
        pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        stamp = time.strftime("%H:%M:%S")
        work_q.put((pcm, stamp))

    try:
        while not stop.is_set():
            frame = read_exact(proc.stdout, FRAME_BYTES)
            if len(frame) < FRAME_BYTES:  # ffmpeg died / EOF
                err = proc.stderr.read().decode(errors="replace")
                if err.strip():
                    print(f"\nffmpeg: {err.strip()}", file=sys.stderr)
                break
            speech = vad.is_speech(frame, SAMPLE_RATE)
            if not triggered:
                ring.append(frame)
                if speech:
                    triggered = True
                    voiced = list(ring)
                    ring.clear()
                    num_silence = 0
            else:
                voiced.append(frame)
                num_silence = 0 if speech else num_silence + 1
                if num_silence >= silence_frames or len(voiced) >= max_frames:
                    emit(voiced)
                    triggered, voiced, num_silence = False, [], 0
                    ring.clear()
    finally:
        proc.terminate()
        work_q.put(None)
        worker.join(timeout=5)
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
