#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="$PWD/weights"
export MPLCONFIGDIR="$PWD/runs/matplotlib"
case "${1:-status}" in
  preflight) exec environment/bin/python -B preflight.py ;;
  train) exec environment/bin/python -B run.py train ;;
  resume) exec environment/bin/python -B run.py resume ;;
  status) exec environment/bin/python -B run.py status ;;
  *) printf 'Usage: bash START.sh {preflight|train|resume|status}\n'; exit 2 ;;
esac
