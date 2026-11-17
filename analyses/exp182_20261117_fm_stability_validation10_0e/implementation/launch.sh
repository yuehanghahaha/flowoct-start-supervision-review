#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PY=/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e/environment/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/runs/matplotlib"
mkdir -p logs runs
case "${1:-}" in
  check)
    "$PY" -B -u check.py synthetic
    "$PY" -B -u check.py a800
    ;;
  core|extended)
    "$PY" -B -u run.py "$1" 2>&1 | tee -a "logs/$1.log"
    ;;
  *)
    printf 'Usage: bash launch.sh {check|core|extended}\n' >&2
    exit 2
    ;;
esac
