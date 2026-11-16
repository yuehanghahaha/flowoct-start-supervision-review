#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
session=exp148_octstart_nocond
python=${EXP148_PYTHON:-/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp102_20261106_start_alltproj_150e/.venv/bin/python}
cd "$root"
if [[ ${1:-} == worker ]]; then
    mkdir -p logs
    exec "$python" -B -u run.py auto >>logs/train_console.log 2>&1
fi
if tmux has-session -t "=$session" 2>/dev/null; then
    printf 'Session %s exists. View it with tmux attach -t %s\n' "$session" "$session"
    exit 0
fi
test -x "$python"
tmux new-session -d -s "$session" -c "$root" \
    -e "EXP148_PYTHON=$python" -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    bash ./launch.sh worker
printf 'Started session %s. Follow with tail -f %s/logs/train_console.log\n' "$session" "$root"
