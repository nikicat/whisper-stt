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

# Also save the exact audio sent to the GPU as raw PCM (crash-safe: no header to
# finalize on Ctrl-C, and re-feedable directly via ./replay.sh). Disable: RECORD=0.
rec_args=()
if [ "${RECORD:-1}" != "0" ]; then
  mkdir -p recordings
  rec="recordings/$(date +%Y%m%d-%H%M%S).s16le"
  rec_args=(-ac 1 -ar 16000 -f s16le "$rec")
  echo "recording -> $rec" >&2
fi

# Dual output from one capture: output 1 -> ssh pipe, output 2 -> recording file.
ffmpeg -hide_banner -loglevel error -f pulse -i default \
  -ac 1 -ar 16000 -f s16le - \
  "${rec_args[@]}" \
  | ssh "$host" "cd $remote_dir && ./gpu-server.sh $*"
