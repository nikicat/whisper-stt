#!/usr/bin/env bash
# Runs ON THE LAPTOP. Captures the mic and pipes raw PCM over ssh to the GPU box,
# which runs gpu-server.sh and streams text back to this terminal.
#
#   ./stream-remote.sh user@gpu-host [extra streaming.py args...]
#   ./stream-remote.sh gpu-host --model large-v3-turbo --beam-size 5
#
# No `ssh -t`: a pty would corrupt the binary PCM on stdin. Ctrl-C ends ffmpeg,
# which closes the pipe; the remote sees EOF and shuts down cleanly.
set -uo pipefail
host="${1:?usage: stream-remote.sh user@gpu-host [args...]}"; shift || true

ffmpeg -hide_banner -loglevel error -f pulse -i default -ac 1 -ar 16000 -f s16le - \
  | ssh "$host" "cd ~/src/whisper-stt && ./gpu-server.sh $*"
