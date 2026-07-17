#!/usr/bin/env bash
# Runs ON THE GPU BOX. Reads raw s16le/16k/mono PCM from stdin (piped over ssh
# from the laptop's mic) and streams transcribed text back on stdout.
#
# Point CTranslate2 at the CUDA/cuDNN libs from the `cuda` extra wheels if present;
# otherwise fall back to whatever is on the default library path (e.g. system
# `cuda`/`cudnn` packages).
set -uo pipefail
cd "$(dirname "$0")"

LIBS="$(uv run python -c 'import os,nvidia.cublas.lib,nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__)+":"+os.path.dirname(nvidia.cudnn.lib.__file__))' 2>/dev/null || true)"
[ -n "$LIBS" ] && export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# int8 uses the 1080's DP4A path (Pascal FP16 is crippled, so avoid float16).
exec uv run python streaming.py --stdin --device cuda --compute-type int8 "$@"
