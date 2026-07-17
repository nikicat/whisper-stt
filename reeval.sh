#!/usr/bin/env bash
# reeval.sh — re-evaluate a saved recording through the model to compare configs.
# Runs on the GPU box (turbo) by default; --local runs on this laptop's CPU.
# The transcript is printed AND saved next to the recording as
# <recording>.<label>.txt, so you can diff two runs instead of eyeballing them.
#
#   GPU_HOST=nb-home ./reeval.sh recordings/X.s16le                      # baseline turbo
#   GPU_HOST=nb-home ./reeval.sh recordings/X.s16le --glossary glossary.txt
#   GPU_HOST=nb-home ./reeval.sh recordings/X.s16le --model medium --beam-size 5
#   ./reeval.sh --local recordings/X.s16le --model small                # CPU here
#   LABEL=glossary GPU_HOST=nb-home ./reeval.sh recordings/X.s16le --glossary glossary.txt
#
# Compare two runs:
#   ./reeval.sh recordings/X.s16le                    # -> recordings/X.baseline.txt
#   LABEL=glossary ./reeval.sh recordings/X.s16le --glossary glossary.txt
#   diff recordings/X.baseline.txt recordings/X.glossary.txt
set -uo pipefail

local=0
if [ "${1:-}" = "--local" ]; then local=1; shift; fi
file="${1:?usage: [GPU_HOST=host] reeval.sh [--local] recordings/FILE.s16le [streaming args...]}"; shift || true
[ -f "$file" ] || { echo "no such recording: $file" >&2; exit 1; }

# Output label: explicit $LABEL, else derived from the args, else 'baseline'.
if   [ -n "${LABEL:-}" ]; then label="$LABEL"
elif [ "$#" -gt 0 ];      then label="$(printf '%s' "$*" | tr -cs 'A-Za-z0-9' '-' | sed 's/^-//; s/-$//')"
else label="baseline"; fi
out="${file%.s16le}.$label.txt"

echo "reeval: $file  [$label]  ->  $out" >&2
echo "# reeval  args=$*  local=$local  file=$file" > "$out"

if [ "$local" = 1 ]; then
  uv run python streaming.py --stdin "$@" < "$file" | tee -a "$out"
else
  host="${GPU_HOST:?set GPU_HOST=<gpu-ssh-host> (or pass --local to run on this CPU)}"
  remote_dir="${REMOTE_DIR:-\$HOME/src/whisper-stt}"   # \$HOME expands on the GPU box
  ( trap '' INT; ssh "$host" "cd $remote_dir && ./gpu-server.sh $*" ) < "$file" | tee -a "$out"
fi
