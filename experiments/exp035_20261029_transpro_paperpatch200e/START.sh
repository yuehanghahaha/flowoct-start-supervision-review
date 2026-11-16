#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PYTHON_BIN="${TRANSPRO_PYTHON:-/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python}"
"$PYTHON_BIN" -B -u run.py train 2>&1 | tee -a logs/train_console.log
