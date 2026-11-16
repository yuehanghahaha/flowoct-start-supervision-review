#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python=/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e/environment/bin/python
session=exp164_startproj_fullcds
mkdir -p logs runs/train
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "Training session already exists: $session"
  exit 0
fi
printf -v command 'cd %q && set -o pipefail && %q -B -u run.py train 2>&1 | tee -a logs/train_console.log; result=${PIPESTATUS[0]}; printf "%%s\\n" "$result" > runs/train/exit_code.txt; exit "$result"' "$PWD" "$python"
tmux new-session -d -s "$session" bash -c "$command"
echo "Training launched/resumed in $session; original experiments continue."
