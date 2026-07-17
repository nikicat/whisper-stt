#!/usr/bin/env bash
# Replay a recorded PCM capture through the streaming pipeline to validate or tune
# offline — no mic, no ffmpeg (raw .s16le feeds --stdin directly).
#
#   ./replay.sh recordings/FILE.s16le [streaming.py args...]
#   ./replay.sh recordings/FILE.s16le --model small --vad-aggressiveness 3
#
# Replay on the GPU box instead (turbo) — cd first, ssh starts in the home dir:
#   cat recordings/FILE.s16le | ssh gpu-host 'cd ~/src/whisper-stt && ./gpu-server.sh'
# Listen to it:
#   ffmpeg -f s16le -ar 16000 -ac 1 -i recordings/FILE.s16le recordings/FILE.wav
#
# NOTE: replay feeds audio as fast as it reads, so utterance timing differs from a
# live run — good for comparing transcription quality across configs, not latency.
set -uo pipefail
file="${1:?usage: replay.sh recordings/FILE.s16le [streaming.py args...]}"; shift || true
exec uv run python streaming.py --stdin "$@" < "$file"
