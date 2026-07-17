#!/usr/bin/env python3
"""Live word-by-word streaming STT (EN+RU) using LocalAgreement-2.

Unlike transcribe.py (which prints a whole utterance when you pause), this prints
words progressively as they are *confirmed*. Confirmation uses LocalAgreement-2
(Macháček et al., the policy behind ufal/whisper_streaming): Whisper is re-run on a
growing audio buffer every `--min-chunk` seconds, and a word is committed only once
two consecutive runs agree on it. Unconfirmed words stay tentative and may change.

Because each step re-decodes the buffer, streaming needs the model to run
comfortably faster than realtime. On this CPU `large-v3-turbo` is borderline for
long unbroken speech; `--model small` or `medium` streams more smoothly. The buffer
is reset at every silence (VAD), so latency never grows without bound.
"""

import argparse
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

from transcribe import SAMPLE_RATE, FRAME_MS, FRAME_BYTES, load_glossary, start_ffmpeg, read_exact


class HypothesisBuffer:
    """LocalAgreement-2 confirmation over word-level (start, end, text) tuples."""

    def __init__(self):
        self.committed_in_buffer: list[tuple] = []  # recent committed words (for overlap dedup)
        self.buffer: list[tuple] = []               # previous run's unconfirmed tail
        self.new: list[tuple] = []                  # current run's candidate words
        self.last_committed_time = 0.0

    def insert(self, words, offset):
        """Add a fresh hypothesis (word timestamps relative to buffer start)."""
        words = [(a + offset, b + offset, t) for a, b, t in words]
        self.new = [(a, b, t) for a, b, t in words if a > self.last_committed_time - 0.1]
        if not self.new or not self.committed_in_buffer:
            return
        # Drop words that repeat the committed tail (n-gram overlap at the seam).
        if abs(self.new[0][0] - self.last_committed_time) < 1.0:
            cn, nn = len(self.committed_in_buffer), len(self.new)
            for i in range(1, min(cn, nn, 5) + 1):
                prev = " ".join(self.committed_in_buffer[-j][2] for j in range(i, 0, -1))
                cur = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                if prev == cur:
                    del self.new[:i]
                    break

    def flush(self):
        """Commit the longest prefix where this run agrees with the previous one."""
        committed = []
        while self.new and self.buffer:
            if self.new[0][2] == self.buffer[0][2]:
                w = self.new.pop(0)
                self.buffer.pop(0)
                committed.append(w)
                self.last_committed_time = w[1]
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        # Bound the dedup memory.
        if len(self.committed_in_buffer) > 50:
            self.committed_in_buffer = self.committed_in_buffer[-50:]
        return committed

    def flush_tail(self):
        """At an utterance end, commit whatever is still tentative."""
        tail, self.buffer = self.buffer, []
        return tail


class OnlineASR:
    """Holds the rolling audio buffer for one utterance and runs LocalAgreement."""

    def __init__(self, model, language, base_prompt, beam_size, max_buffer_s):
        self.model = model
        self.language = language
        self.base_prompt = base_prompt
        self.beam_size = beam_size
        self.max_buffer_samples = int(max_buffer_s * SAMPLE_RATE)
        self.reset()

    def reset(self):
        self.audio = np.zeros(0, dtype=np.float32)
        self.time_offset = 0.0            # seconds of audio already trimmed away
        self.hyp = HypothesisBuffer()
        self.committed: list[tuple] = []  # all committed words this utterance

    def insert_audio(self, chunk: np.ndarray):
        self.audio = np.concatenate((self.audio, chunk))

    def _prompt(self):
        # Glossary only, and only if explicitly enabled. Do NOT feed committed
        # text back as a prompt: on a rolling re-decode the model tends to
        # *continue* the prompt, which turns a hallucinated repeat into a loop.
        return self.base_prompt or None

    def _run(self):
        segments, _ = self.model.transcribe(
            self.audio,
            language=self.language,
            initial_prompt=self._prompt(),
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,            # strip silence in the buffer -> no silence hallucinations
            no_repeat_ngram_size=3,     # kill "директор директор ..." loops
            temperature=0.0,
        )
        words, seg_ends = [], []
        for s in segments:
            seg_ends.append(s.end)
            for w in (s.words or []):
                words.append((w.start, w.end, w.word))
        return words, seg_ends

    def process(self):
        """Re-decode the buffer, commit agreed words, trim on segment boundaries."""
        words, seg_ends = self._run()
        self.hyp.insert(words, self.time_offset)
        committed = self.hyp.flush()
        self.committed.extend(committed)
        self._trim(seg_ends)
        return committed

    def _trim(self, seg_ends):
        """Drop audio up to the last completed segment boundary before the last
        committed word, so the buffer (and re-decode cost) stays bounded."""
        if not self.committed:
            # Safety valve: never let a silent/garbled buffer grow unbounded.
            if len(self.audio) > self.max_buffer_samples:
                self._cut_to(self.time_offset + self.max_buffer_samples / SAMPLE_RATE / 2)
            return
        last_word_end = self.committed[-1][1]
        abs_ends = [e + self.time_offset for e in seg_ends]
        cut = None
        for e in abs_ends[:-1]:            # keep the final (in-progress) segment
            if e <= last_word_end:
                cut = e
        if cut is None and len(self.audio) > self.max_buffer_samples:
            cut = last_word_end
        if cut is not None:
            self._cut_to(cut)

    def _cut_to(self, t):
        n = int((t - self.time_offset) * SAMPLE_RATE)
        if 0 < n < len(self.audio):
            self.audio = self.audio[n:]
            self.time_offset = t

    def finish(self):
        return self.hyp.flush_tail()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepdml/faster-whisper-large-v3-turbo-ct2",
                   help="turbo streams on the edge here; try 'small'/'medium' for smoother live output")
    p.add_argument("--language", default="ru", help="forced language; '' or 'auto' to auto-detect")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--beam-size", type=int, default=1, help="1 (greedy) keeps re-decodes fast enough to stream")
    p.add_argument("--glossary", default="",
                   help="term glossary to bias spelling via initial_prompt. OFF by default in "
                        "streaming: the prompt tends to leak/loop on rolling re-decodes. Turbo "
                        "handles the terms fine without it. Enable with e.g. --glossary glossary.txt")
    p.add_argument("--source", default="default")
    p.add_argument("--stdin", action="store_true",
                   help="read raw s16le 16kHz mono PCM from stdin instead of the mic "
                        "(used on the GPU box: laptop pipes mic audio over ssh)")
    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--min-chunk", type=float, default=1.0, help="seconds of new audio between re-decodes (higher = less CPU, more lag)")
    p.add_argument("--silence-ms", type=int, default=600, help="trailing silence that ends an utterance")
    p.add_argument("--preroll-ms", type=int, default=300)
    p.add_argument("--max-buffer-s", type=float, default=18.0, help="hard cap on buffer length (bounds re-decode cost)")
    p.add_argument("--list-mics", action="store_true")
    args = p.parse_args()

    if args.list_mics:
        import subprocess
        subprocess.run(["pactl", "list", "short", "sources"])
        return
    if args.language in ("", "auto"):
        args.language = None

    base_prompt = load_glossary(Path(args.glossary)) if args.glossary else ""

    print(f"Loading model '{args.model}' ({args.device}/{args.compute_type})...", file=sys.stderr, flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=args.cpu_threads)
    online = OnlineASR(model, args.language, base_prompt, args.beam_size, args.max_buffer_s)
    print("Model ready. Listening — words appear as they're confirmed (Ctrl-C to stop).\n", file=sys.stderr, flush=True)

    vad = webrtcvad.Vad(args.vad_aggressiveness)
    if args.stdin:
        proc, reader = None, sys.stdin.buffer
    else:
        proc = start_ffmpeg(args.source)
        reader = proc.stdout

    preroll = deque(maxlen=max(1, args.preroll_ms // FRAME_MS))
    silence_frames = max(1, args.silence_ms // FRAME_MS)
    min_chunk_samples = int(args.min_chunk * SAMPLE_RATE)

    in_utterance = False
    num_silence = 0
    pending = []            # float32 frames not yet handed to process()
    pending_samples = 0
    line_open = False       # is there an unfinished printed line?

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def to_f32(frame: bytes) -> np.ndarray:
        return np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0

    def emit(words):
        nonlocal line_open
        if not words:
            return
        if not line_open:
            sys.stdout.write(f"[{time.strftime('%H:%M:%S')}]")
            line_open = True
        sys.stdout.write("".join(w[2] for w in words))  # word text carries its leading space
        sys.stdout.flush()

    def end_line():
        nonlocal line_open
        if line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            line_open = False

    try:
        while not stop.is_set():
            frame = read_exact(reader, FRAME_BYTES)
            if len(frame) < FRAME_BYTES:
                break
            speech = vad.is_speech(frame, SAMPLE_RATE)

            if not in_utterance:
                preroll.append(frame)
                if speech:
                    in_utterance, num_silence = True, 0
                    pending = [to_f32(f) for f in preroll]
                    pending_samples = sum(len(x) for x in pending)
                    preroll.clear()
                continue

            pending.append(to_f32(frame))
            pending_samples += FRAME_BYTES // 2
            num_silence = 0 if speech else num_silence + 1

            utter_end = num_silence >= silence_frames
            if pending_samples >= min_chunk_samples or utter_end:
                online.insert_audio(np.concatenate(pending))
                pending, pending_samples = [], 0
                emit(online.process())

            if utter_end or len(online.audio) > online.max_buffer_samples * 1.5:
                emit(online.finish())
                end_line()
                online.reset()
                in_utterance, num_silence = False, 0
    finally:
        if proc:
            proc.terminate()
        if in_utterance:
            emit(online.finish())
        end_line()
        print("Stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
