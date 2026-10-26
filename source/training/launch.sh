#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python ]]; then
    export PYTHON=/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python
  else
    export PYTHON=python
  fi
fi
if [[ $# -eq 0 ]]; then
  set -- train
fi
exec "$PYTHON" -u run.py "$@"
