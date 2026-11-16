#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${TRANSPRO_PYTHON:-/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Set TRANSPRO_PYTHON to an existing Python environment containing torch 2.5.1 / torchvision 0.20.1.' >&2
  exit 1
fi
mkdir -p logs runtime_deps
# Only two missing imports are needed in headless mode. Keep them inside this experiment.
"$PYTHON_BIN" -m pip install --no-deps --target "$PWD/runtime_deps" torchsummary==1.5.1 dominate==2.9.1 2>&1 | tee logs/dependency_setup.log
"$PYTHON_BIN" -B -u run.py prepare 2>&1 | tee logs/prepare_console.log
"$PYTHON_BIN" -B -u run.py check 2>&1 | tee logs/check_console.log
printf '%s\n' 'Setup/check finished. No formal training was started.'
