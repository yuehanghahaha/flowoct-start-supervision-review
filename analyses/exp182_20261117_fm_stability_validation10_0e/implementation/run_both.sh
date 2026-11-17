#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PY=/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e/environment/bin/python
export MPLCONFIGDIR="$PWD/runs/matplotlib"
mkdir -p logs runs
if [[ ! -f verification/a800_preflight.json ]]; then
  printf 'Missing A800 preflight\n' >&2
  exit 2
fi
bash launch.sh core >logs/core_supervisor.log 2>&1 &
core_pid=$!
bash launch.sh extended >logs/extended_supervisor.log 2>&1 &
extended_pid=$!
printf '%s core_pid=%s extended_pid=%s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$core_pid" "$extended_pid" > runs/launch.txt
core_exit=0
extended_exit=0
wait "$core_pid" || core_exit=$?
wait "$extended_pid" || extended_exit=$?
printf '%s core_exit=%s extended_exit=%s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$core_exit" "$extended_exit" > runs/exit.txt
if (( core_exit == 0 && extended_exit == 0 )); then
  "$PY" -B -u report.py --require-complete >logs/report.log 2>&1
else
  "$PY" -B -u report.py >logs/report.log 2>&1
  exit 1
fi
