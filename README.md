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

## Possible upgrades

- **Word-by-word streaming** (text appearing as you speak, not on pause): wrap the
  model with `ufal/whisper_streaming`'s LocalAgreement policy.
- **Type into the focused window**: pipe finalized text through `wtype`/`ydotool`.
