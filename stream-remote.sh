#!/usr/bin/env bash
# Runs ON THE LAPTOP. Captures the mic, pipes raw PCM over ssh to the GPU box
# (gpu-server.sh), and streams text back — while saving a paired recording +
# transcript log per session under recordings/ for later offline evaluation.
#
#   ./stream-remote.sh user@gpu-host [extra streaming.py args...]
#   ./stream-remote.sh gpu-host --model large-v3-turbo --beam-size 5
#
# Repo location ON THE GPU BOX defaults to ~/src/whisper-stt; override if you
# cloned elsewhere:  REMOTE_DIR=/opt/whisper-stt ./stream-remote.sh gpu-host
# Skip saving audio+log:  RECORD=0 ./stream-remote.sh gpu-host
#
# Clean Ctrl-C: ssh and tee IGNORE SIGINT, so only ffmpeg stops on Ctrl-C. ffmpeg
# closes its output, the remote sees EOF and flushes its last words, then ssh/tee
# drain and exit — no broken-pipe spew. No `ssh -t` (a pty would corrupt the
# binary PCM on stdin).
set -uo pipefail
host="${1:?usage: stream-remote.sh user@gpu-host [args...]}"; shift || true

# \$HOME is escaped so it expands on the GPU box, not the laptop.
remote_dir="${REMOTE_DIR:-\$HOME/src/whisper-stt}"

# Per-session paired files: <ts>.s16le (exact audio sent) + <ts>.log (transcript
# with a config header). Raw PCM is crash-safe and re-feedable via ./replay.sh.
rec_args=()
log=/dev/null
if [ "${RECORD:-1}" != "0" ]; then
  mkdir -p recordings
  ts="$(date +%Y%m%d-%H%M%S)"
  rec="recordings/$ts.s16le"
  log="recordings/$ts.log"
  rec_args=(-ac 1 -ar 16000 -f s16le "$rec")
  { echo "# session $ts"; echo "# host=$host"; echo "# args=$*"; echo "# audio=$rec"; } > "$log"
  echo "recording -> $rec   log -> $log" >&2
fi

trap 'printf "\nstopped.\n" >&2' INT   # runs after the pipeline drains

# Dual output from one capture: output 1 -> ssh pipe, output 2 -> recording file.
# The subshells set SIGINT to ignore so Ctrl-C reaches only ffmpeg.
ffmpeg -hide_banner -loglevel error -f pulse -i default \
    -ac 1 -ar 16000 -f s16le - \
    "${rec_args[@]}" \
  | ( trap '' INT; ssh "$host" "cd $remote_dir && ./gpu-server.sh $*" ) \
  | ( trap '' INT; tee -a "$log" )
