#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python=${XOCT_ABLATION_PYTHON:-/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e/environment/bin/python}
session=$(basename "$PWD")
session=${session:0:6}_xoct_ablation
if [[ ${1:-} == worker ]]; then
    mkdir -p logs
    exec "$python" -B -u run.py train >>logs/train_console.log 2>&1
fi
if tmux has-session -t "=$session" 2>/dev/null; then
    printf 'Session %s is active\n' "$session"
    exit 0
fi
test -f verification/preflight.json
tmux new-session -d -s "$session" -c "$PWD" \
    -e "XOCT_ABLATION_PYTHON=$python" -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    bash ./launch.sh worker
printf 'Started %s\n' "$session"
