#!/usr/bin/env bash
# Runs ON THE LAPTOP. Captures the mic and pipes raw PCM over ssh to the GPU box,
# which runs gpu-server.sh and streams text back to this terminal.
#
#   ./stream-remote.sh user@gpu-host [extra streaming.py args...]
#   ./stream-remote.sh gpu-host --model large-v3-turbo --beam-size 5
#
# Repo location ON THE GPU BOX defaults to ~/src/whisper-stt; override if you
# cloned elsewhere:  REMOTE_DIR=/opt/whisper-stt ./stream-remote.sh gpu-host
#
# No `ssh -t`: a pty would corrupt the binary PCM on stdin. Ctrl-C ends ffmpeg,
# which closes the pipe; the remote sees EOF and shuts down cleanly.
set -uo pipefail
host="${1:?usage: stream-remote.sh user@gpu-host [args...]}"; shift || true

# \$HOME is escaped so it expands on the GPU box, not the laptop.
remote_dir="${REMOTE_DIR:-\$HOME/src/whisper-stt}"

ffmpeg -hide_banner -loglevel error -f pulse -i default -ac 1 -ar 16000 -f s16le - \
  | ssh "$host" "cd $remote_dir && ./gpu-server.sh $*"
