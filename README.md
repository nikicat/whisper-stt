# whisper-stt

Realtime EN+RU speech-to-text tuned for **Russian speech peppered with English tech
terms** ("сделай commit и запусти deploy на staging"). Runs on CPU using
`faster-whisper` `large-v3-turbo` with int8 — a good fit for this laptop's
AVX-512-VNNI Tiger Lake CPU.

## Why these choices

- **`large-v3-turbo`** — full multilingual, near-`large-v2` quality, ~8× faster
  decoding than `large-v3`. The only high-quality model with a realtime shot on CPU.
- **`language=ru` forced** — auto-detect flips to English on a run of tech terms and
  mangles the surrounding Russian. Forcing Russian keeps the sentence intact while
  Whisper still emits the English terms in Latin script.
- **glossary → `initial_prompt`** — biases spelling of *your* terms. Edit
  `glossary.txt` (one term per line) to taste.
- **ffmpeg → PipeWire capture** — no PortAudio/ALSA fragility.
- **webrtcvad segmentation on a separate thread from decoding** — a slow transcription
  never drops mic audio.

## Setup

```fish
cd ~/src/whisper-stt
uv sync            # creates .venv with Python 3.12 + deps
```

First run downloads the turbo model (~1.6 GB) into the HuggingFace cache.

## Run

```fish
uv run python transcribe.py                 # listen on the default mic
uv run python transcribe.py --list-mics     # pick a specific source
uv run python transcribe.py --source alsa_input.pci-0000_00_1f.3-...  # use it
```

Speak; each utterance is printed when you pause (~0.7 s of silence closes it):

```
[14:32:05] сделай rebase на main и запушь в staging
```

## Tuning

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `deepdml/faster-whisper-large-v3-turbo-ct2` | try `medium`/`small` if turbo feels slow |
| `--silence-ms` | `700` | lower = snappier cuts, higher = fewer mid-sentence splits |
| `--vad-aggressiveness` | `2` | `3` rejects more noise, `0` is most sensitive |
| `--beam-size` | `5` | `1` is faster, slightly less accurate |
| `--max-utterance-s` | `20` | forces a cut during long monologues |
| `--language` | `ru` | use `auto` to auto-detect per utterance |

## If turbo isn't fast enough

- Drop to `--model medium` (still solid RU) or `--model small` (faster, weaker RU).
- Lower `--beam-size 1`.
- The Iris Xe iGPU can offload the encoder via OpenVINO (more setup) — worth it only
  if CPU turbo proves too slow in practice.

## Live word-by-word streaming — `streaming.py`

`transcribe.py` prints a whole utterance when you pause. `streaming.py` prints words
*progressively* as they're confirmed, using LocalAgreement-2 (the policy behind
`ufal/whisper_streaming`, implemented in-repo — no extra dependency).

```fish
uv run python streaming.py --model small     # best quality that streams on CPU
uv run python streaming.py --model base       # snappier fallback
```

**Why not turbo here?** Streaming re-runs the model on a rolling buffer every
`--min-chunk` seconds, and Whisper's *encoder* always processes a fixed 30s window —
turbo only shrinks the *decoder*, so each re-encode still costs a full large-encoder
pass (~7-8s on this CPU → hopelessly behind). Streaming on CPU needs a small encoder;
`base`/`small` keep up, turbo/medium/large do not.

## Realtime turbo streaming via a remote GPU

A CUDA GPU makes the fixed 30s encoder pass cheap (~0.2-0.4s on a GTX 1080), so
**turbo streams in realtime**. Capture stays on the laptop; inference runs on the GPU
box. Audio is piped over SSH (no ports, no extra protocol); text streams back:

```
[laptop] ffmpeg mic → PCM ──ssh stdin──▶ [GPU] streaming.py --stdin --device cuda → text ──stdout──▶ [laptop]
```

**On the GPU box (Arch):**
```fish
# 1. NVIDIA driver — verify the GPU is visible:
nvidia-smi                 # if Pascal was dropped by the current driver,
                           # install nvidia-580xx-dkms (AUR) instead of `nvidia`
# 2. Clone + install with CUDA runtime libs (CUDA 12 + cuDNN 9 wheels; no system toolkit):
git clone <repo> ~/src/whisper-stt && cd ~/src/whisper-stt
uv sync --extra cuda
# 3. Smoke-test CUDA locally before wiring SSH:
ffmpeg -i some.wav -ac 1 -ar 16000 -f s16le - | ./gpu-server.sh --model large-v3-turbo
```

**From the laptop:**
```fish
./stream-remote.sh user@gpu-box --model large-v3-turbo --beam-size 5
```
`gpu-server.sh` points CTranslate2 at the wheel-provided CUDA libs and forces
`--compute-type int8` (Pascal's DP4A path; avoid `float16`, which is crippled on the
1080). Ctrl-C on the laptop closes the pipe and the remote shuts down cleanly.

## Recording & offline replay

`stream-remote.sh` saves each session as a paired set under `recordings/<timestamp>.`:
`.s16le` (the exact audio sent to the GPU, raw PCM — crash-safe, no header to
finalize on Ctrl-C) and `.log` (a config header + the transcript). Disable with
`RECORD=0 ./stream-remote.sh ...`.

Ctrl-C exits cleanly: `ssh`/`tee` ignore SIGINT so only `ffmpeg` stops, the remote
flushes its last words over EOF, and the pipeline drains without broken-pipe errors.

Replay a capture through the pipeline offline to compare configs (no mic needed —
raw PCM feeds `--stdin` directly):

```fish
./replay.sh recordings/20260718-0057.s16le --model small     # local CPU
cat recordings/20260718-0057.s16le | ssh gpu-host './gpu-server.sh'   # on the GPU (turbo)
ffmpeg -f s16le -ar 16000 -ac 1 -i recordings/20260718-0057.s16le out.wav  # to listen
```

Replay feeds audio as fast as it reads, so it's for comparing *transcription
quality* across settings, not live latency.

## Possible upgrades

- **Type into the focused window**: pipe finalized text through `wtype`/`ydotool`.
- **Persistent server** (reconnect, multiple clients, type-into-app): swap the SSH
  pipe for a small WebSocket server wrapping the same `OnlineASR`.
